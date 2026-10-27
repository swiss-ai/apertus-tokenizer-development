#!/usr/bin/env python
"""Incremental driver: run the analysis pipeline only for newly-added tokenizers
and merge the per-tokenizer results into the existing on-disk analysis JSONs.

Use this instead of `scripts/run_report.sh` when the only change since the last
full analysis run is adding (not modifying) one or more tokenizer rows in
`configs/report_tokenizers.json`. All metrics in the analysis pipeline are
per-tokenizer-decomposable (no global cross-tokenizer aggregates are consumed
downstream), so overlaying the new tokenizers' rows into the existing
JSONs yields an output identical to a full re-run on those tokenizers, at
~1/(N/k) the cost (k = #new, N = #total).

Report assembly was retired 2026-07-16: this driver produces analysis artifacts
only and writes no .md report. Apertus documentation is hand-maintained in
~/apertus-tokenizer-development, which is the only place Apertus results are
documented; tables are generated into those docs from that repo.

Steps:
  1. Build a temp subset tokenizer-config with only the new names.
  2. Run `tokenizer-analysis` for broad / core / full on the subset, writing to
     a scratch dir. All three get `--code-ast-config` to match `run_report.sh`.
  3. Merge each subset analysis_results_full.json into the corresponding
     existing dir's analysis_results_full.json (per-tokenizer overlay).
  4. Run the secondary scripts (sanity, junk-tokens, FineWeb-Edu compression,
     non-emitting tokens) on the subset and merge their per-tokenizer entries
     into the existing output JSONs.
  5. Skip `superbpe_vs_base.py` (it scores fixed SuperBPE↔base pairs;
     unaffected by new tokenizers unless one of them is in a known pair).
  6. Re-run extrinsic index (full; cheap) and family plots.

Limitations:
  - For modifying (not adding) an existing tokenizer, delete it from the dest
    JSONs first or use the full pipeline. Overlay only adds/overwrites the
    listed names; orphan entries from a prior run are not pruned.
  - The merge does NOT touch global summary aggregates that exist outside
    per-tokenizer dicts. Consumers read per-tokenizer paths, so this is fine in
    practice, but any future code that reads a cross-tokenizer summary would see
    stale (pre-add) numbers until a full re-run.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Set


def _load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _save(path: str, obj: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _known_tokenizers_in(dest: Dict) -> Set[str]:
    """Best-effort: scan dest for a likely per-tokenizer dict and return its keys.
    Used to disambiguate 'this dict is keyed by tokenizer' vs 'this is just
    a generic dict' in the recursive merge."""
    candidates: List[Set[str]] = []
    for top_key, top_val in dest.items():
        if isinstance(top_val, dict):
            pt = top_val.get("per_tokenizer")
            if isinstance(pt, dict) and pt:
                candidates.append(set(pt.keys()))
    if not candidates:
        return set()
    # Intersect: a name that appears as a per-tokenizer key in multiple metrics
    # is almost certainly a tokenizer name.
    return set.intersection(*candidates) if len(candidates) > 1 else candidates[0]


def merge_per_tokenizer(dest: Any, src: Any, new_tokens: Set[str],
                        known: Set[str]) -> None:
    """Recursively overlay any dict entries from *src* whose keys are tokenizer
    names (members of `new_tokens` OR `known`) into *dest*.

    - If src has a key in `new_tokens` at any level, copy src[k] to dest[k].
    - Otherwise, descend in lockstep on shared dict keys; never overwrite a
      leaf that exists in both dest and src (those are non-tokenizer aggregates
      that should be preserved from the dest's full-roster run).
    - If a dict path exists in src but not dest, copy it whole (creates the
      necessary structure for the new tokenizers).
    """
    if not (isinstance(dest, dict) and isinstance(src, dict)):
        return
    for k, src_val in src.items():
        if k in new_tokens:
            dest[k] = src_val
        elif k in dest:
            if isinstance(dest[k], dict) and isinstance(src_val, dict):
                # Special case: if this dict appears to be keyed by tokenizer
                # name and the src has entries that overlap with the dest's
                # known tokenizer set, only overlay the new-token entries.
                merge_per_tokenizer(dest[k], src_val, new_tokens, known)
            # else: leaf in both — preserve dest (non-tokenizer aggregate).
        else:
            # Path doesn't exist in dest at all; copy it. This is the common
            # case for nested per-language dicts that include the new
            # tokenizer entries.
            dest[k] = src_val


def _run(cmd: List[str], description: str) -> None:
    print(f"\n>>> {description}")
    print(f"    {' '.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f"FAILED: {description} (exit {r.returncode})", file=sys.stderr)
        sys.exit(r.returncode)
    print(f"    [{time.perf_counter() - t0:.1f}s]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", required=True,
                    help="Full report tokenizer config (the new tokenizers must "
                         "already be entered here).")
    ap.add_argument("--tokenizers", nargs="+", required=True,
                    help="Names of newly-added tokenizers to compute and merge.")
    ap.add_argument("--lib", default=os.path.join(os.path.expanduser("~"), "tokenizer-intrinsic-evals"),
                    help="tokenizer-intrinsic-evals root (venv, CLI, generic scripts, shared "
                         "language/measurement/AST configs, FLORES parallel data).")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Don't delete the scratch dir after success (for debugging).")
    args = ap.parse_args(argv)

    # Two-repo layout (2026-07-16, mirrors run_report.sh). lib = the generic library;
    # apx = this Apertus repo (Apertus scripts, configs, and the results/ output). The CLI
    # resolves parallel/<lang>.txt relative to cwd, so we chdir into lib.
    lib = os.path.abspath(os.environ.get("LIB_ROOT", args.lib))
    apx = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(lib, ".venv", "bin")):
        raise SystemExit(f"ERROR: library venv not found under {lib}/.venv/bin (set --lib or LIB_ROOT)")
    cfg_path = os.path.abspath(args.tokenizer_config)   # resolve before chdir (relative to invocation dir)
    os.chdir(lib)                                        # so the CLI resolves parallel/<lang>.txt
    full_cfg = _load(cfg_path)

    missing = [t for t in args.tokenizers if t not in full_cfg]
    if missing:
        print(f"unknown tokenizer name(s) in --tokenizers: {missing}",
              file=sys.stderr)
        return 2
    new_tokens = set(args.tokenizers)

    # 1. Subset config
    scratch = tempfile.mkdtemp(prefix="add_tokenizers_")
    subset_cfg_path = os.path.join(scratch, "subset_tokenizers.json")
    _save(subset_cfg_path, {n: full_cfg[n] for n in args.tokenizers})
    print(f"Subset config -> {subset_cfg_path} ({len(args.tokenizers)} tokenizer(s))")
    print(f"Scratch dir   -> {scratch}")

    py = os.path.join(lib, ".venv", "bin", "python")
    tok_analysis = os.path.join(lib, ".venv", "bin", "tokenizer-analysis")
    tok_sanity = os.path.join(lib, ".venv", "bin", "tokenizer-sanity-check")
    mc = os.path.join(lib, "configs", "text_measurement_config_lines.json")
    # AST config is at the LIB repo ROOT, not under configs/. The old code passed
    # "configs/starcoder_ast_config.json", which does not exist and silently substitutes SYNTHETIC
    # code (the same bug fixed in run_report.sh). Use the root path.
    ast_cfg = os.path.join(lib, "starcoder_ast_config.json")

    # 2. Run analysis CLI for broad / core / full on the subset. Outputs land in this Apertus
    #    repo's results/ (apx); language/measurement/AST configs come from the library (lib).
    #    All three panels pass --code-ast-config, matching run_report.sh: the code-AST metrics
    #    (ast_boundary_alignment, operator_isolation_rate) are computed on the code corpus and are
    #    identical across the three panels, but each panel's analysis_results_full.json must carry
    #    them for a merged tokenizer to look like one from a full run. A previous version passed
    #    --no-code-ast for broad/full, which silently omitted those keys for incrementally-added
    #    tokenizers (they would then be dropped from any correlation read off report_flores60).
    analysis_specs = [
        ("broad", "configs/flores60_lang_config.json",
         "results/report_flores60", "flores60",
         ["--code-ast-config", ast_cfg]),
        ("core",  "configs/core_lang_config.json",
         "results/report_core", "core",
         ["--code-ast-config", ast_cfg]),
        ("full",  "configs/flores_devtest_lang_config.json",
         "results/report_flores_devtest", "flores_devtest",
         ["--code-ast-config", ast_cfg]),
    ]
    subset_outputs = {}
    for label, lang_cfg, dest_dir, dataset, extra in analysis_specs:
        out_dir = os.path.join(scratch, f"analysis_{label}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [tok_analysis,
               "--tokenizer-config", subset_cfg_path,
               "--language-config", os.path.join(lib, lang_cfg),
               "--output-dir", out_dir,
               "--measurement-config", mc,
               "--save-full-results",
               "--cer-time-budget", "60",
               "--dataset", dataset,
               *extra]
        # broad + core also get per-language / faceted plots in run_report.sh,
        # but those are full-roster plots that are regenerated by the report
        # build; skipping them here saves time.
        _run(cmd, f"[1/{len(analysis_specs)}] analysis subset: {label}")
        subset_outputs[label] = (os.path.join(out_dir, "analysis_results_full.json"),
                                 os.path.join(apx, dest_dir,
                                              "analysis_results_full.json"))

    # 3. Merge per-tokenizer entries into existing analysis JSONs
    for label, (src_path, dest_path) in subset_outputs.items():
        if not os.path.exists(src_path):
            print(f"WARN: subset analysis missing {src_path}; skipping merge",
                  file=sys.stderr)
            continue
        src = _load(src_path)
        dest = _load(dest_path)
        known = _known_tokenizers_in(dest)
        merge_per_tokenizer(dest, src, new_tokens, known)
        _save(dest_path, dest)
        print(f"   merged {label}: {dest_path}")

    # 4. Secondary per-tokenizer scripts (run on subset, merge into existing outputs)
    secondary = [
        # (description, command (subset-output flag separate), dest_path)
        ("sanity",
         [tok_sanity,
          "--tokenizer-config", subset_cfg_path,
          "--output-dir", os.path.join(scratch, "sanity"),
          "--use-sample-data",
          "--language-config", os.path.join(lib, "configs", "flores60_lang_config.json"),
          "--quiet", "--exit-zero"],
         os.path.join(scratch, "sanity", "sanity_results.json"),
         os.path.join(apx, "results", "report_sanity", "sanity_results.json")),
        ("FineWeb-Edu compression",
         [py, os.path.join(lib, "scripts", "fineweb_edu_compression.py"),
          "--tokenizer-config", subset_cfg_path,
          "--output", os.path.join(scratch, "finewebedu.json"),
          "--max-rows", "1000"],
         os.path.join(scratch, "finewebedu.json"),
         os.path.join(apx, "results", "report_finewebedu_compression.json")),
        ("junk-token counts",
         [py, os.path.join(lib, "scripts", "junk_tokens.py"),
          "--tokenizer-config", subset_cfg_path,
          "--output", os.path.join(scratch, "junk.json")],
         os.path.join(scratch, "junk.json"),
         os.path.join(apx, "results", "report_junk_tokens.json")),
        ("non-emitting tokens",
         [py, os.path.join(lib, "scripts", "nonemitting_tokens.py"),
          "--tokenizer-config", subset_cfg_path,
          "--math-bytes", "12000000", "--code-bytes", "12000000",
          "--output", os.path.join(scratch, "nonemitting.json")],
         os.path.join(scratch, "nonemitting.json"),
         os.path.join(apx, "results", "report_nonemitting_tokens.json")),
    ]
    for desc, cmd, src_path, dest_path in secondary:
        _run(cmd, f"secondary: {desc}")
        if not os.path.exists(src_path):
            print(f"WARN: secondary output missing {src_path}; skipping merge",
                  file=sys.stderr)
            continue
        if not os.path.exists(dest_path):
            shutil.copyfile(src_path, dest_path)
            print(f"   wrote (no prior file): {dest_path}")
            continue
        src = _load(src_path)
        dest = _load(dest_path)
        merge_per_tokenizer(dest, src, new_tokens, _known_tokenizers_in(dest))
        _save(dest_path, dest)
        print(f"   merged: {dest_path}")

    # 5. Extrinsic index, family plots (each cheap; full-roster).
    # Report assembly was retired 2026-07-16: this driver produces analysis artifacts only. Apertus
    # documentation is hand-maintained in ~/apertus-tokenizer-development and tables are generated
    # into it from there. The retired assembler is archived at
    # ~/apertus-tokenizer-development/archive/tooling/build_tokenizer_report.py.
    _run([py, os.path.join(apx, "scripts", "extrinsic_index.py"),
          "--slug-map", os.path.join(apx, "configs", "extrinsic_slug_map.json"),
          "--report-config", cfg_path,
          "--output", os.path.join(apx, "results", "report_extrinsic_index.json")],
         "extrinsic index")

    _run([py, os.path.join(apx, "scripts", "family_plots.py"),
          "--analysis", os.path.join(apx, "results", "report_flores_devtest",
                                     "analysis_results_full.json"),
          "--output-dir", os.path.join(apx, "results", "family_plots")],
         "family plots")

    if not args.keep_temp:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nDONE. Added: {', '.join(args.tokenizers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
