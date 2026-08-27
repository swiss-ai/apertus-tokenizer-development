#!/usr/bin/env python
"""Resolve extrinsic (downstream-LM) results for the report from tokenizer-lm.

The report-name -> slug map is `configs/extrinsic_slug_map.json`. This module:
  1. loads that map,
  2. content-verifies each slug against the report tokenizer (vocab hash) so a wrong/sibling
     slug is flagged rather than trusted blindly,
  3. reads each metric value straight from the tokenizer-lm result JSONs (no transcription),
  4. marks every (tokenizer, metric) as a value / "pending" (run mapped but file absent) /
     "na" (no run mapped at all).

Also exposes the 5-tokenizer 10B-vs-20B stability table (balanced mixture) for the appendix.

Run-name = full-128k-{slug}[-continue|-mathcode-scratch].

HISTORY (2026-07-16): the slug map used to be parsed out of FEEDBACK.md's §2 markdown table.
FEEDBACK.md is Apertus documentation and is now archived, so reading data out of it was both a
dependency on an archived file and a parser over prose. The table (33 rows) was extracted to JSON
and the parser deleted. The old parser produced 31 entries (33 minus 2 group-shorthand rows whose
names contain "/"); the JSON keeps 29 (it also drops 2 non-roster shorthand rows whose slugs were
all null). The 2 extra drops are not roster keys, so build_records() -- which does
declared.get(name, (None, None)) -- produces byte-identical output either way (verified: 0
differences across every roster key). The roots and STABILITY_SET below were always hand-transcribed
from FEEDBACK.md §1/§3d, never parsed; they are plain constants and are unaffected.
"""
import argparse
import glob
import hashlib
import json
import os
import statistics

# --- cross-repo roots (configs/paths.json) ---
# These used to be hardcoded module constants. Lifted to configs/paths.json 2026-07-16 so the
# cross-repo paths live in one reviewable place. Each key may be overridden by an env var of the
# same name upper-cased (tokenizer_lm_root -> TOKENIZER_LM_ROOT). The BOOTSTRAP panel below is the
# LIVE 37-run bootstrap_mathcode_panel_all.json, NOT the superseded 19-run
# bootstrap_mathcode_panel.json (fixed 2026-07-16 as DEFECT-1: the old panel gave every candidate
# "pending" CIs while only the Apertus v1 baseline got intervals; point values are read from
# LM_EVAL, never from here, so they were unaffected).
_DEFAULT_PATHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "paths.json")


def _load_paths(path):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: paths config not found: {path}")
    with open(path) as f:
        cfg = json.load(f)

    def resolve(key):
        env = os.environ.get(key.upper())
        if env:
            return env
        if key not in cfg:
            raise SystemExit(f"ERROR: {path} missing required key {key!r}")
        return cfg[key]

    tlm = resolve("tokenizer_lm_root")
    return {
        "TLM": tlm,
        "CKPT": resolve("checkpoints"),
        "SLUG_DIR": resolve("tokenizer_store"),
        "LM_EVAL": f"{tlm}/results/lm_eval",
        "FLORES": f"{tlm}/results/flores",
        "MC_EVAL": f"{tlm}/results/mc_eval",
        "CORR": f"{tlm}/results/correlations",
        "BOOTSTRAP": os.path.join(tlm, resolve("bootstrap_panel")),
    }


# Module-level roots, loaded once from the default config. main() can override via --paths-config
# by calling _apply_paths() before build_records().
_P = _load_paths(_DEFAULT_PATHS)
TLM = _P["TLM"]
CKPT = _P["CKPT"]
SLUG_DIR = _P["SLUG_DIR"]
LM_EVAL = _P["LM_EVAL"]
FLORES = _P["FLORES"]
MC_EVAL = _P["MC_EVAL"]
CORR = _P["CORR"]
BOOTSTRAP = _P["BOOTSTRAP"]


def _apply_paths(path):
    """Reload roots from a non-default paths config (called by main() for --paths-config)."""
    global TLM, CKPT, SLUG_DIR, LM_EVAL, FLORES, MC_EVAL, CORR, BOOTSTRAP
    p = _load_paths(path)
    TLM, CKPT, SLUG_DIR = p["TLM"], p["CKPT"], p["SLUG_DIR"]
    LM_EVAL, FLORES, MC_EVAL, CORR, BOOTSTRAP = p["LM_EVAL"], p["FLORES"], p["MC_EVAL"], p["CORR"], p["BOOTSTRAP"]

# trained-31 FLORES languages (eval_runner.py:49 BELEBELE_TRAINED_LANGS)
TRAINED_LANGS = {
    "arb_Arab", "ben_Beng", "bul_Cyrl", "cat_Latn", "ces_Latn", "zho_Hans", "cmn_Hans",
    "deu_Latn", "ell_Grek", "eng_Latn", "fin_Latn", "fra_Latn", "heb_Hebr", "hin_Deva",
    "hrv_Latn", "hun_Latn", "ind_Latn", "ita_Latn", "jpn_Jpan", "kor_Hang", "nld_Latn",
    "pol_Latn", "por_Latn", "ron_Latn", "rus_Cyrl", "slk_Latn", "spa_Latn", "tam_Taml",
    "tha_Thai", "tur_Latn", "ukr_Cyrl", "vie_Latn",
}

# 10B-vs-20B stability set (FEEDBACK.md §3d; display -> slug). 20B = "{slug}-continue".
STABILITY_SET = [
    ("gpt4o-balanced", "gpt4o-balanced-bpe"),
    ("rightalign-balanced", "rightalign-balanced-bpe"),
    ("claude-balanced-nfc", "claude-balanced-nfc-bpe"),
    ("llama3", "llama3"),
    ("gpt4o-code", "gpt4o-code-bpe"),
]


# ---------- low-level helpers ----------
def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _vocab_hash(tok_json):
    d = _load(tok_json) if tok_json and os.path.exists(tok_json) else None
    m = (d or {}).get("model")
    if not m or not m.get("vocab"):
        return None
    # Hash the learned BPE vocab only, excluding added/special tokens (the ids in
    # added_tokens). Two tokenizers that differ solely in the NAMES of reserved
    # special slots at the same ids (e.g. <SPECIAL_25> vs <email-pii>) then hash
    # equal, since text tokenization is identical. Byte-fallback tokens live in
    # model.vocab (not added_tokens), so they stay in the hash.
    vocab = m["vocab"]
    if isinstance(vocab, dict):
        added_ids = {t.get("id") for t in (d.get("added_tokens") or [])}
        payload = {tok: i for tok, i in vocab.items() if i not in added_ids}
    else:
        payload = vocab  # Unigram: list of [token, score] pairs; hash as-is
    return hashlib.md5(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _meta_val_bpb(run):
    metas = sorted(glob.glob(f"{CKPT}/{run}/meta_*.json"))
    if not metas:
        return None
    d = _load(metas[-1])  # final = the run's full token budget
    return (d or {}).get("val_bpb")


def _bpb_stats(vals):
    """Compute {mean, stdev, se, ci_lo, ci_hi, n} for a per-language BPB vector.
    SE/CI are the across-language normal-approx 95% CI for the macro-mean estimate:
    mean ± 1.96 · stdev/√n. Captures language-to-language variation; does NOT include
    within-language sentence-sampling error (per-sentence BPB is not in the source
    JSONs). Stdev is the unbiased (ddof=1) across-language standard deviation, a
    direct measure of how much per-language BPB varies."""
    if not vals:
        return None
    n = len(vals)
    mean = statistics.mean(vals)
    if n >= 2:
        sd = statistics.stdev(vals)
        se = sd / (n ** 0.5)
        return {"mean": round(mean, 4), "stdev": round(sd, 4),
                "se": round(se, 5), "ci_lo": round(mean - 1.96 * se, 4),
                "ci_hi": round(mean + 1.96 * se, 4), "n": n}
    return {"mean": round(mean, 4), "stdev": None, "se": None,
            "ci_lo": None, "ci_hi": None, "n": n}


def _flores(run):
    """Return (all_stats, trained_stats) dicts (or (None, None)) with macro-mean +
    across-language stdev/SE/95% CI for the all-214 and trained-31 FLORES sets."""
    d = _load(f"{FLORES}/{run}.json") or _load(f"{FLORES}/{run}_flores.json")
    pl = (((d or {}).get("scores") or {}).get("flores200") or {}).get("per_language") if d else None
    if not pl:
        return None, None
    allv = [v["bpb"] for v in pl.values() if isinstance(v, dict) and v.get("bpb") is not None]
    trv = [v["bpb"] for k, v in pl.items() if k in TRAINED_LANGS and isinstance(v, dict) and v.get("bpb") is not None]
    return _bpb_stats(allv), _bpb_stats(trv)


def _blimp_code(run):
    """Returns (blimp_optB, code_bpb, optA_only). BLiMP is read as **Option-B (BOS/empty-context)
    scoring only**, for comparability across runs (the main *_blimp_code_bpb.json file is Option-A for
    some runs and Option-B for others — mixing them is not comparable). Source preference:
    {run}_blimp_optB.json; else the main file iff it is itself Option-B (metadata has
    empty_context_loglikelihood). If only Option-A exists, blimp_optB is None and optA_only=True so the
    caller can flag it rather than silently use the incomparable Option-A number. code_bpb is
    scoring-independent and always read from the main file."""
    main = _load(f"{LM_EVAL}/{run}_blimp_code_bpb.json")
    code_bpb = (((main or {}).get("scores") or {}).get("code_bpb") or {}).get("summary", {}).get("mean_bpb") if main else None
    optb = _load(f"{LM_EVAL}/{run}_blimp_optB.json")
    blimp = None
    if optb:
        blimp = optb.get("scores", {}).get("blimp", {}).get("acc,none")
    elif main and "empty_context_loglikelihood" in (main.get("metadata") or {}):
        blimp = main.get("scores", {}).get("blimp", {}).get("acc,none")
    main_blimp = (main or {}).get("scores", {}).get("blimp", {}).get("acc,none") if main else None
    optA_only = blimp is None and main_blimp is not None
    return blimp, code_bpb, optA_only


def _macro_acc(d, prefix, score_key):
    sc = (d or {}).get("scores", {})
    vals = [v[score_key] for k, v in sc.items() if k.startswith(prefix) and isinstance(v, dict) and score_key in v]
    return round(statistics.mean(vals), 4) if vals else None


def _multiblimp(run):
    return _macro_acc(_load(f"{LM_EVAL}/{run}_multiblimp.json"), "multiblimp_", "acc,none")


def _mgsm(run):
    return _macro_acc(_load(f"{LM_EVAL}/{run}_mgsm.json"), "mgsm_direct_", "exact_match,flexible-extract")


def _belebele(run):
    return _macro_acc(_load(f"{LM_EVAL}/{run}_belebele.json"), "belebele_", "acc,none")


def _mc_math(run):
    d = _load(f"{MC_EVAL}/{run}_full_limit500.json")
    ds = (d or {}).get("datasets")
    if not ds:
        return None, None
    parts = {}
    for k in ("gsm8k", "math", "pythonio"):
        e = ds.get(k) or {}
        acc = e.get("accuracy")
        if acc is None and isinstance(e.get("full"), dict):
            acc = e["full"].get("accuracy")
        if acc is not None:
            parts[k] = round(acc, 4)
    mean = round(statistics.mean(parts.values()), 4) if parts else None
    return mean, parts


def _gsm8k(run):
    d = _load(f"{LM_EVAL}/{run}_gsm8k_limit500.json") or _load(f"{LM_EVAL}/{run}_gsm8k.json")
    return ((d or {}).get("scores", {}).get("gsm8k_cot", {}).get("exact_match,flexible-extract"))


def _code_gen(run):
    d = _load(f"{LM_EVAL}/{run}_code_gen.json")
    sc = (d or {}).get("scores", {})
    return (sc.get("humaneval", {}).get("pass@1,create_test"),
            sc.get("mbpp", {}).get("pass_at_1,none"))


def _boot_ci(mc_slug, metric):
    d = _load(BOOTSTRAP)
    e = ((d or {}).get("per_run") or {}).get(mc_slug, {}).get(metric)
    if not e:
        return None
    return {"mean": e.get("mean"), "lo": e.get("lo"), "hi": e.get("hi"), "n_pass": e.get("n_pass"), "n": e.get("n")}


# ---------- slug map ----------
def load_slugmap(path):
    """Load configs/extrinsic_slug_map.json -> {report_name: (standard_slug, mathcode_slug)}.

    A null slug means no run exists or is planned for that axis; callers render it "na". That is
    a different state from "pending" (a slug IS declared but its result file is absent), and the
    two must not be conflated -- see _status().

    Aborts on a missing or malformed map rather than returning an empty dict: an empty map would
    silently mark every tokenizer "na", which looks like a legitimate "nothing was run" result.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: slug map not found: {path}")
    with open(path) as f:
        doc = json.load(f)
    raw = doc.get("slug_map")
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"ERROR: {path} has no non-empty 'slug_map' object")
    out = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or not {"standard_slug", "mathcode_slug"} <= set(entry):
            raise SystemExit(
                f"ERROR: {path}: entry {name!r} must have 'standard_slug' and 'mathcode_slug' "
                f"(use null for 'no run'); got {entry!r}")
        out[name] = (entry["standard_slug"], entry["mathcode_slug"])
    return out


def _status(val, has_run):
    """A metric is a value if read; 'pending' if a run is mapped but the file is absent;
    'na' if no run is mapped for this tokenizer at all."""
    if val is not None and not (isinstance(val, dict) and not val):
        return val
    return "pending" if has_run else "na"


def _slug_hash_index():
    """{vocab_hash: slug_dir_name} over all tokenizer-lm slug dirs (skips _archive)."""
    idx = {}
    for d in sorted(glob.glob(f"{SLUG_DIR}/*")):
        b = os.path.basename(d)
        if not os.path.isdir(d) or b.startswith("_archive"):
            continue
        h = _vocab_hash(os.path.join(d, "tokenizer.json"))
        if h:
            idx[h] = b
    return idx


def build_records(slug_map_path, report_config):
    declared = load_slugmap(slug_map_path)
    cfg = _load(report_config) or {}
    slug_idx = _slug_hash_index()
    recs = {}
    for name, tcfg in cfg.items():
        p = tcfg.get("path", "")
        rep_hash = _vocab_hash(p) if p.endswith(".json") and os.path.exists(p) else None
        decl_std, decl_mc = declared.get(name, (None, None))

        # If no standard slug is declared but the exact tokenizer exists in tokenizer-lm,
        # fall back to the content-matched slug so its (queued) standard evals read "pending"
        # rather than "not run". The mathcode slug is left as declared (it may be a sibling).
        if not decl_std and rep_hash and rep_hash in slug_idx:
            decl_std = slug_idx[rep_hash]

        # content-verify the (possibly back-filled) slugs against the report tokenizer
        std_match = bool(rep_hash) and decl_std and _vocab_hash(f"{SLUG_DIR}/{decl_std}/tokenizer.json") == rep_hash
        mc_match = bool(rep_hash) and decl_mc and _vocab_hash(f"{SLUG_DIR}/{decl_mc}/tokenizer.json") == rep_hash
        # HF reference tokenizers have no local tokenizer.json to hash (rep_hash is None), so the
        # content-verify cannot run. For these, an explicitly-declared FEEDBACK slug is an intentional
        # human mapping (e.g. Apertus == the 'apertus' slug, the same Mistral-Nemo tokenizer); treat it
        # as matched rather than mislabeling it [proxy].
        if rep_hash is None:
            std_match = std_match or bool(decl_std)
            mc_match = mc_match or bool(decl_mc)

        std = f"full-128k-{decl_std}" if decl_std else None
        mc = f"full-128k-{decl_mc}-mathcode-scratch" if decl_mc else None
        has_std, has_mc = bool(decl_std), bool(decl_mc)

        val_bpb = _meta_val_bpb(std) if std else None
        fl_all_s, fl_tr_s = _flores(std) if std else (None, None)
        fl_all = fl_all_s["mean"] if isinstance(fl_all_s, dict) else None
        fl_tr = fl_tr_s["mean"] if isinstance(fl_tr_s, dict) else None
        blimp, code_bpb, blimp_optA_only = _blimp_code(std) if std else (None, None, False)
        mblimp = _multiblimp(std) if std else None
        mgsm = _mgsm(std) if std else None
        belebele = _belebele(std) if std else None
        mc_mean, mc_parts = _mc_math(mc) if mc else (None, None)
        gsm = _gsm8k(mc) if mc else None
        heval, mbpp = _code_gen(mc) if mc else (None, None)
        mbpp_ci = _boot_ci(decl_mc, "mbpp") if decl_mc else None
        gsm_ci = _boot_ci(decl_mc, "gsm8k_flex") if decl_mc else None
        heval_ci = _boot_ci(decl_mc, "humaneval") if decl_mc else None

        recs[name] = {
            "declared_std_slug": decl_std, "declared_mc_slug": decl_mc,
            "std_matched": bool(std_match), "mc_matched": bool(mc_match),
            "val_bpb": _status(val_bpb, has_std),
            "flores_all": _status(fl_all, has_std),
            "flores_trained": _status(fl_tr, has_std),
            "flores_all_stats": _status(fl_all_s, has_std),
            "flores_trained_stats": _status(fl_tr_s, has_std),
            "code_bpb": _status(code_bpb, has_std),
            "blimp": _status(blimp, has_std),
            "blimp_optA_only": blimp_optA_only,
            "multiblimp": _status(mblimp, has_std),
            "mgsm": _status(mgsm, has_std),
            "belebele": _status(belebele, has_std),
            "mc_math": _status(mc_mean, has_mc),
            "mc_math_parts": _status(mc_parts, has_mc),
            "gsm8k": _status(gsm, has_mc),
            "humaneval": _status(heval, has_mc),
            "mbpp": _status(mbpp, has_mc),
            "mbpp_ci": _status(mbpp_ci, has_mc),
            "gsm8k_ci": _status(gsm_ci, has_mc),
            "humaneval_ci": _status(heval_ci, has_mc),
        }
    return recs


def build_stability():
    """5-tokenizer all-task table at 10B (full-128k-{slug}) and 20B ({slug}-continue)."""
    out = []
    for disp, slug in STABILITY_SET:
        for budget, run in (("10B", f"full-128k-{slug}"), ("20B", f"full-128k-{slug}-continue")):
            fl_all_s, fl_tr_s = _flores(run)
            fl_all = fl_all_s["mean"] if isinstance(fl_all_s, dict) else None
            fl_tr = fl_tr_s["mean"] if isinstance(fl_tr_s, dict) else None
            blimp, code_bpb, _ = _blimp_code(run)
            heval, mbpp = _code_gen(run)
            out.append({"tokenizer": disp, "budget": budget,
                        "val_bpb": _meta_val_bpb(run), "flores_all": fl_all, "flores_trained": fl_tr,
                        "blimp": blimp, "code_bpb": code_bpb,
                        "gsm8k": _gsm8k(run), "humaneval": heval, "mbpp": mbpp,
                        "mgsm": _mgsm(run)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug-map", default="configs/extrinsic_slug_map.json",
                    help="report-name -> tokenizer-lm slug map (JSON).")
    ap.add_argument("--report-config", default="configs/report_tokenizers.json")
    ap.add_argument("--paths-config", default=None,
                    help="Cross-repo roots JSON (default: ../configs/paths.json next to this "
                         "script). Env vars TOKENIZER_LM_ROOT etc. override individual keys.")
    ap.add_argument("--output", default="results/report_extrinsic_index.json")
    args = ap.parse_args(argv)
    if args.paths_config:
        _apply_paths(args.paths_config)
    out = {"extrinsic_index": {
        "per_tokenizer": build_records(args.slug_map, args.report_config),
        "stability_10b_20b": build_stability()}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    pt = out["extrinsic_index"]["per_tokenizer"]
    for n, r in pt.items():
        avail = [k for k in ("val_bpb", "flores_all", "code_bpb", "mc_math", "mbpp")
                 if not isinstance(r[k], str)]
        print(f"  {n:32s} std={str(r['declared_std_slug'])[:30]:30s} "
              f"matched(std/mc)={int(r['std_matched'])}/{int(r['mc_matched'])} "
              f"avail_primary={avail}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
