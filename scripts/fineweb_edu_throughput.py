#!/usr/bin/env python
"""Standalone single-thread encode throughput (MB/s) per tokenizer.

Measures how fast each tokenizer encodes English text, on the same FineWeb-Edu
snippet used by fineweb_edu_compression.py (same shard, first N rows). Encoding
goes through the HF `tokenizers` Rust backend (`Tokenizer.encode_batch`,
add_special_tokens=False). Threading is pinned to one core via
RAYON_NUM_THREADS=1 (set before importing tokenizers), so the number is a
reproducible per-core encode rate, not a machine-wide one.

Throughput = input_bytes / encode_time, reported as MB/s with MB = 10**6 bytes.
The reported time is the minimum over several timed repetitions (after a warmup),
which is the standard low-noise estimator for a deterministic workload.

Usage:
    python scripts/fineweb_edu_throughput.py \
        --output results/report_finewebedu_throughput.json \
        [--max-rows 1000] [--repeats 7]
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Pin to one core BEFORE tokenizers/rayon initializes its thread pool.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import pyarrow.parquet as pq
from tokenizers import Tokenizer

# Same shard and loader as scripts/fineweb_edu_compression.py.
DEFAULT_SHARD = ("/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/"
                 "fineweb-edu/data/CC-MAIN-2013-20/train-00000-of-00014.parquet")
DATASET_NAME = "FineWeb-Edu (HuggingFaceFW/fineweb-edu local mirror)"

# The tokenizer set comes from --tokenizer-config (see load_roster). A hardcoded 5-entry
# TOKENIZERS list lived here until 2026-07-17; it pinned Apertus and the candidate folders by
# absolute path, so this producer silently covered a different set from every other step in
# run_report.sh. Encode throughput excludes special tokens.


def load_tokenizer(path: str, ignore_merges: bool = None) -> Tokenizer:
    """Load a tokenizer.json, optionally overriding model.ignore_merges.

    ignore_merges=None leaves the file's value as-is. Otherwise the model dict
    is patched and reloaded from a temp file (the on-disk artifact is untouched).
    """
    if ignore_merges is None:
        return Tokenizer.from_file(path)
    d = json.load(open(path))
    if d.get("model", {}).get("type") != "BPE":
        raise ValueError(f"ignore_merges override only valid for BPE models: {path}")
    d["model"]["ignore_merges"] = ignore_merges
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, tmp)
    tmp.close()
    return Tokenizer.from_file(tmp.name)


def load_snippet(shard: str, max_rows: int):
    pf = pq.ParquetFile(shard)
    t = pf.read_row_group(0)
    col = "text" if "text" in t.column_names else next(
        (c for c in t.column_names if t.schema.field(c).type == "string"), None)
    if col is None:
        raise ValueError(f"no string/text column in {shard}; columns={t.column_names}")
    docs = [d for d in t.column(col).to_pylist()[:max_rows] if d]
    return docs, col


def load_roster(cfg_path):
    """Read the report tokenizer config and return [(name, tokenizer.json path), ...].

    Replaces a hardcoded 5-entry TOKENIZERS list (2026-07-17) so this producer covers whatever
    the roster covers, like every other step in run_report.sh. Only entries with a local
    tokenizer.json are measurable: a hub-ref entry has no file to time, so it is skipped and
    reported by name rather than silently dropped.
    """
    cfg = json.load(open(cfg_path))
    roster, skipped = [], []
    for name, entry in cfg.items():
        p = entry.get("path", "")
        if p.endswith(".json") and os.path.exists(p):
            roster.append((name, p))
        else:
            skipped.append(name)
    if skipped:
        print(f"note: {len(skipped)} entr(ies) have no local tokenizer.json and are not timed: "
              f"{skipped}", flush=True)
    if not roster:
        raise SystemExit(f"ERROR: no local tokenizers found in {cfg_path}")
    return roster, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer-config", required=True,
                    help="Report tokenizer config; entries with a local tokenizer.json are timed.")
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--max-rows", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--warmups", type=int, default=2)
    args = ap.parse_args(argv)
    roster, skipped = load_roster(args.tokenizer_config)

    docs, text_col = load_snippet(args.shard, args.max_rows)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    mb = total_bytes / 1e6

    provenance = {
        "dataset": DATASET_NAME,
        "shard": args.shard,
        "text_column": text_col,
        "n_docs": len(docs),
        "total_bytes": total_bytes,
        "metric": "encode throughput MB/s (MB = 10**6 bytes input UTF-8)",
        "backend": "tokenizers.Tokenizer.encode_batch, add_special_tokens=False",
        "threading": f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')} (single core)",
        "timing": f"min of {args.repeats} repeats after {args.warmups} warmups",
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
    }

    def measure(tok):
        for _ in range(args.warmups):
            tok.encode_batch(docs, add_special_tokens=False)
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            tok.encode_batch(docs, add_special_tokens=False)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best

    provenance["tokenizer_config"] = args.tokenizer_config
    provenance["n_tokenizers_timed"] = len(roster)
    provenance["not_timed_no_local_file"] = skipped

    per_tok = {}
    for name, path in roster:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing tokenizer: {path}")
        model = json.load(open(path)).get("model", {})
        model_type = model.get("type")
        shipped = model.get("ignore_merges", False)
        tok = load_tokenizer(path)
        best = measure(tok)
        entry = {
            "encode_mb_per_s": mb / best,
            "min_encode_time_s": best,
            "vocab_size": tok.get_vocab_size(),
            "model_type": model_type,
            "shipped_ignore_merges": shipped,
        }
        line = f"  {name:22s} {mb / best:8.2f} MB/s  (type={model_type}, im={shipped}, vocab {tok.get_vocab_size()})"
        # For BPE tokenizers that ship ignore_merges=False, also report the ignore_merges=True
        # variant (verified to yield identical token ids). `ignore_merges` is a BPE-only field, so
        # a non-BPE model (e.g. UnigramLM) has no such variant and is not an error: guard on the
        # model type, not on `shipped`, which defaults to False for models that lack the key.
        if model_type == "BPE" and not shipped:
            best_im = measure(load_tokenizer(path, ignore_merges=True))
            entry["encode_mb_per_s_ignore_merges"] = mb / best_im
            line += f"  ->  ignore_merges=True {mb / best_im:8.2f} MB/s ({best / best_im:.2f}x)"
        per_tok[name] = entry
        print(line)

    out = {"finewebedu_eng_throughput": {"provenance": provenance,
                                         "per_tokenizer": per_tok}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output} (corpus {mb:.2f} MB, {len(docs)} docs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
