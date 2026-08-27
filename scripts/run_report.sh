#!/usr/bin/env bash
# Reproducible driver for the Apertus tokenizer-selection intrinsic panels.
# Runs the standard pipeline on the broad (60-lang), core (13-lang), and full (205-lang)
# FLORES sets (lines measurement, faceted plots, full results), the sanity check, the
# standalone FineWeb-Edu English compression, and the vocabulary / extrinsic JSON producers.
#
# TWO-REPO LAYOUT (2026-07-16). This Apertus pipeline was moved OUT of the generic library
# tokenizer-intrinsic-evals so the library holds nothing Apertus-specific. It now spans two repos:
#   LIB  (tokenizer-intrinsic-evals): the analysis CLI + its venv, the generic per-tokenizer
#        scripts (fineweb_edu_compression, junk_tokens, nonemitting_tokens), the shared language /
#        measurement configs, the StarCoder AST config, and the FLORES `parallel/` data. The CLI
#        resolves `parallel/<lang>.txt` relative to cwd, so this script runs with cwd=LIB.
#   SELF (this repo, apertus-tokenizer-development): the Apertus-specific scripts
#        (superbpe_vs_base, extrinsic_index, family_plots), the Apertus configs
#        (report_tokenizers.json, extrinsic_slug_map.json), and the output dir results/.
# Override LIB with the LIB_ROOT env var if the library lives elsewhere.
#
# REPORT ASSEMBLY IS RETIRED (2026-07-16). This script produces analysis artifacts only; it no
# longer builds REPORT.md or any other .md report. Apertus documentation is hand-maintained in
# this repo and is the only place Apertus results are documented. Generated tables are injected
# into those docs by this repo's own table generator, not from here. The retired assembler is
# archived at archive/tooling/build_tokenizer_report.py. Do not re-add a step that writes a .md
# report: doing so recreates documents that get confounded with the hand-maintained ones (which
# is what happened on 2026-07-14/15).
# -e so a killed or failing step aborts the whole pipeline at the point of failure, rather than
# marching on with a missing panel (which happened 2026-07-14 when the 205-language step was
# OOM-killed under the old `set -uo pipefail`).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${LIB_ROOT:-${HOME}/tokenizer-intrinsic-evals}"
[ -x "$LIB/.venv/bin/tokenizer-analysis" ] || { echo "ERROR: library CLI not found at $LIB/.venv/bin (set LIB_ROOT)" >&2; exit 1; }
cd "$LIB"                                # so the CLI resolves parallel/<lang>.txt
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

PY="$LIB/.venv/bin"                      # venv with tokenizer_analysis + matplotlib + tokenizers
MC=configs/text_measurement_config_lines.json        # LIB (relative to cwd=LIB)
AST=starcoder_ast_config.json            # LIB repo ROOT, not configs/. A wrong/missing path does
                                         # NOT disable code metrics: it silently substitutes
                                         # SYNTHETIC code (also affects operator isolation's code
                                         # domain), so every step passes it explicitly.
CFG="$SELF/configs/report_tokenizers.json"           # SELF (Apertus candidate roster)
SLUG_MAP="$SELF/configs/extrinsic_slug_map.json"     # SELF (report-name -> tokenizer-lm slug)
OUT="$SELF/results"                      # SELF (Apertus panels; gitignored)
for f in "$CFG" "$SLUG_MAP" "$AST" configs/flores60_lang_config.json configs/core_lang_config.json \
         configs/flores_devtest_lang_config.json "$MC"; do
  [ -e "$f" ] || { echo "ERROR: required input not found: $f (cwd=$LIB)" >&2; exit 1; }
done

echo "[1/12] broad (60-language) pipeline ..."
$PY/tokenizer-analysis --tokenizer-config "$CFG" \
  --language-config configs/flores60_lang_config.json \
  --output-dir "$OUT/report_flores60" --measurement-config $MC \
  --per-language-plots --faceted-plots --code-ast-config $AST \
  --save-full-results --cer-time-budget 60 --update-results-md --dataset flores60

echo "[2/12] core pipeline ..."
$PY/tokenizer-analysis --tokenizer-config "$CFG" \
  --language-config configs/core_lang_config.json \
  --output-dir "$OUT/report_core" --measurement-config $MC \
  --per-language-plots --faceted-plots \
  --code-ast-config $AST \
  --save-full-results --cer-time-budget 60 --update-results-md --dataset core

echo "[3/12] full (205-language FLORES+) pipeline (1012-sentence devtest split; no per-language/faceted plots) ..."
$PY/tokenizer-analysis --tokenizer-config "$CFG" \
  --language-config configs/flores_devtest_lang_config.json \
  --output-dir "$OUT/report_flores_devtest" --measurement-config $MC \
  --code-ast-config $AST \
  --save-full-results --cer-time-budget 60 --update-results-md --dataset flores_devtest

echo "[4/12] sanity check ..."
$PY/tokenizer-sanity-check --tokenizer-config "$CFG" \
  --output-dir "$OUT/report_sanity" \
  --use-sample-data --language-config configs/flores60_lang_config.json \
  --quiet --exit-zero

echo "[5/12] FineWeb-Edu English compression ..."
$PY/python "$LIB/scripts/fineweb_edu_compression.py" --tokenizer-config "$CFG" \
  --output "$OUT/report_finewebedu_compression.json" --max-rows 1000

echo "[6/12] junk-token counts ..."
$PY/python "$LIB/scripts/junk_tokens.py" --tokenizer-config "$CFG" \
  --output "$OUT/report_junk_tokens.json"

echo "[7/12] SuperBPE-vs-base vocab analysis ..."
$PY/python "$SELF/scripts/superbpe_vs_base.py" --output "$OUT/report_superbpe_vs_base.json"

echo "[8/12] vocabulary usage (Active/Rare/Uncommon/Unseen + Scaffold) on FineWeb + math + code ..."
# OOM-prone: it holds the ~73 MB corpus plus a per-tokenizer vocab map. If it is killed, re-run
# the same command with --resume (it keeps what is already in --output) until the count stops
# rising. A full pass over the roster typically needs several resumes.
$PY/python "$LIB/scripts/nonemitting_tokens.py" --tokenizer-config "$CFG" \
  --include-refs --math-bytes 12000000 --code-bytes 12000000 \
  --output "$OUT/report_nonemitting_tokens.json"

echo "[9/12] encode throughput on the English FineWeb-Edu snippet ..."
# Added 2026-07-17. The reported §5 numbers previously came from no artifact at all; this step
# makes them reproducible. Single-core by construction (RAYON_NUM_THREADS=1 inside the script),
# so it is sensitive to load on a shared node: treat cross-run differences below a few percent
# as noise rather than a tokenizer effect.
$PY/python "$SELF/scripts/fineweb_edu_throughput.py" --tokenizer-config "$CFG" \
  --max-rows 1000 --repeats 11 \
  --output "$OUT/report_finewebedu_throughput.json"

echo "[10/12] FineWeb2-proportional multilingual compression ..."
# Added 2026-07-17. Previously produced only in scratch *_focus dirs, one per tokenizer, which is
# why two builds of the same tokenizer disagreed on disk. One roster-wide run replaces them.
$PY/python "$SELF/scripts/fineweb2_proportional_compression.py" --tokenizer-config "$CFG" \
  --target-bytes 5000000 --seed 0 \
  --output "$OUT/report_fineweb2_proportional.json"

echo "[11/12] extrinsic (downstream-LM) index from tokenizer-lm ..."
$PY/python "$SELF/scripts/extrinsic_index.py" --slug-map "$SLUG_MAP" --report-config "$CFG" \
  --output "$OUT/report_extrinsic_index.json"

echo "[12/12] family-faceted plots (PA-BPE training families) ..."
$PY/python "$SELF/scripts/family_plots.py" \
  --analysis "$OUT/report_flores_devtest/analysis_results_full.json" \
  --output-dir "$OUT/family_plots"

echo "DONE -> analysis artifacts under $OUT/ (no .md report is built; see header)"
