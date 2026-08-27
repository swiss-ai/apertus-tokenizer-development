#!/usr/bin/env python
"""Compute bytes-per-token on a multilingual FineWeb2 sample, drawn so that
each language family's share of the sampled bytes matches its share of total
FineWeb2 bytes (the natural distribution, before any parity-aware reweighting).

The per-family GB table is the one used by gen_ratio_variants.py to derive the
parity-aware ratios. English, code, and math are not part of FineWeb2 proper
and are excluded.

Output: {tokenizer_name: {bytes_per_token, n_bytes, n_tokens, ...}}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fw2_prop")

# Natural FineWeb2 share per family (GB on disk in the fw2.0.1-quality_33-filterrobots
# snapshot). Source: ~/pa_tokenizers_branch/gen_ratio_variants.py FAMILY_FW2_GB.
FAMILY_FW2_GB = {
    "slavic":                   2196.0,
    "romance":                  1917.0,
    "sinotibetan":              1292.0,
    "germanic":                 1057.0,
    "isolates_and_singletons":   787.0,
    "austronesian":              169.0,
    "turkic":                    160.0,
    "semitic":                   139.0,
    "uralic":                    130.0,
    "iranian":                    99.0,
    "austroasiatic":              90.0,
    "indoaryan":                  24.0,
    "taikadai":                   24.0,
    "baltic":                     13.0,
    "dravidian":                   6.6,
    "nigercongo_bantu":            0.71,
    "celtic":                      0.63,
    "cushitic":                    0.30,
    "nigercongo_voltaniger":       0.08,
    "nigercongo_other":            0.03,
    "mande":                       0.003,
    "berber":                      0.003,
}

EXCLUDED_GROUPS = {"english", "code", "math"}


def family_to_parquets(grouped_config_path: str) -> dict[str, list[str]]:
    with open(grouped_config_path) as f:
        cfg = json.load(f)
    out = {}
    for entry in cfg["languages"]:
        name = entry["name"]
        if name in EXCLUDED_GROUPS:
            continue
        out[name] = list(entry["input"])
    return out


def sample_text_from_parquet(path: str, target_bytes: int, rng: random.Random) -> tuple[str, int]:
    """Read up to target_bytes of text from a parquet 'text' column.
    Returns (concatenated_text, actual_byte_count)."""
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        log.warning(f"cannot open {path}: {e}")
        return "", 0
    # Stream row-groups in random order until we hit target
    n_rg = pf.num_row_groups
    rg_order = list(range(n_rg))
    rng.shuffle(rg_order)
    bufs: list[str] = []
    got = 0
    for rg in rg_order:
        if got >= target_bytes:
            break
        try:
            t = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        except Exception as e:
            log.warning(f"row-group read failed {path}#{rg}: {e}")
            continue
        rng.shuffle(t)
        for s in t:
            if got >= target_bytes:
                break
            if not s:
                continue
            b = s.encode("utf-8")
            bufs.append(s)
            got += len(b)
    return "\n".join(bufs), got


def build_sample(grouped_config_path: str, total_target_bytes: int, seed: int = 0):
    rng = random.Random(seed)
    fam_to_pq = family_to_parquets(grouped_config_path)
    # Restrict the family share table to families we actually have parquets for
    fam_share = {f: FAMILY_FW2_GB[f] for f in fam_to_pq if f in FAMILY_FW2_GB}
    missing = [f for f in fam_to_pq if f not in FAMILY_FW2_GB]
    if missing:
        log.warning(f"families with no FW2 GB entry, excluded: {missing}")
    total_gb = sum(fam_share.values())
    log.info(f"Including {len(fam_share)} families, total FW2 size = {total_gb:.1f} GB")

    samples: list[tuple[str, str, int]] = []  # (family, lang_path, byte_count)
    full_text_parts: list[str] = []

    for fam, gb in sorted(fam_share.items(), key=lambda kv: -kv[1]):
        share = gb / total_gb
        fam_bytes_target = int(total_target_bytes * share)
        if fam_bytes_target <= 0:
            continue
        pqs = fam_to_pq[fam]
        per_lang = max(1, fam_bytes_target // len(pqs))
        log.info(f"  family={fam} share={share*100:.2f}% target_bytes={fam_bytes_target} "
                 f"across {len(pqs)} langs ({per_lang}/lang)")
        for pq_path in pqs:
            text, got = sample_text_from_parquet(pq_path, per_lang, rng)
            if got > 0:
                samples.append((fam, pq_path, got))
                full_text_parts.append(text)
    return "\n".join(full_text_parts), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", required=True,
                    help="JSON dict {name: {class, path}}")
    ap.add_argument("--grouped-config",
                    default="/users/cmeister747/pa_tokenizers_branch/configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
                    help="parity-aware grouped config (any tuned variant — all share the same family→parquet mapping)")
    ap.add_argument("--target-bytes", type=int, default=5_000_000,
                    help="approximate total bytes to sample (default 5 MB)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log.info(f"Building proportional FW2 sample, target {args.target_bytes/1e6:.2f} MB")
    text, samples = build_sample(args.grouped_config, args.target_bytes, args.seed)
    n_bytes = len(text.encode("utf-8"))
    log.info(f"Sample built: {n_bytes/1e6:.3f} MB across {len(samples)} files")

    with open(args.tokenizer_config) as f:
        toks_cfg = json.load(f)

    per_tok: dict = {}
    unloadable: list = []
    for name, spec in toks_cfg.items():
        path = spec["path"]
        # A roster entry may be a hub reference whose repo has no tokenizer.json (e.g. GLM ->
        # zai-org/glm-4-9b-chat, which 404s). Skip it and record the name: an entry we cannot
        # load is an explicit, reported omission, never a substitution with something else.
        try:
            if path.endswith("tokenizer.json"):
                tok = Tokenizer.from_file(path)
            else:
                tok = Tokenizer.from_pretrained(path)
        except Exception as e:
            unloadable.append({"name": name, "path": path, "error": f"{type(e).__name__}: {e}"[:200]})
            log.warning(f"  {name}: NOT MEASURED, could not load {path} ({type(e).__name__})")
            continue
        enc = tok.encode(text)
        n_tokens = len(enc.ids)
        if n_tokens == 0:
            raise RuntimeError(f"{name}: encoded to zero tokens")
        bpt = n_bytes / n_tokens
        per_tok[name] = {
            "compression_bytes_per_token": bpt,
            "n_tokens": n_tokens,
            "n_bytes": n_bytes,
        }
        log.info(f"  {name}: {bpt:.4f} b/t  (n_tokens={n_tokens})")
    if unloadable:
        log.warning(f"{len(unloadable)} roster entr(ies) could not be loaded and are absent from "
                    f"the output: {[u['name'] for u in unloadable]}")

    out = {
        "fineweb2_proportional_compression": {
            "provenance": {
                "grouped_config": args.grouped_config,
                "target_bytes": args.target_bytes,
                "actual_bytes": n_bytes,
                "n_files_sampled": len(samples),
                "seed": args.seed,
                "tokenizer_config": args.tokenizer_config,
                "n_tokenizers_measured": len(per_tok),
                "not_measured_unloadable": unloadable,
                "family_shares": {f: FAMILY_FW2_GB[f] for f in family_to_parquets(args.grouped_config) if f in FAMILY_FW2_GB},
            },
            "per_tokenizer": per_tok,
        }
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
