#!/usr/bin/env python
"""Faceted plots by PA-BPE training language family.

Reads the PA-BPE tuned parity config to get the `family -> [flores_codes]` mapping
(the linguistic groupings on which the parity-aware candidates were trained), and
the full-FLORES analysis JSON to get per-(tokenizer, language) values for
compression_rate and vocabulary_utilization. Emits one SVG per metric, faceted by
family: one panel per family (only families with >=2 languages present in the
analysis), with one coloured line per tokenizer over the family's languages.

Restricted to the 4 candidates + Apertus baseline + 5 open-source references; the
27 ablation rows are intentionally omitted so the panels stay readable.
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 7 candidates + Apertus baseline + 5 open-source refs. This list was kept in step with
# ROUNDTRIP_SUBSET in build_tokenizer_report.py, which was retired and archived on 2026-07-16
# (~/apertus-tokenizer-development/archive/tooling/). Nothing enforces that pairing any more, so
# this roster now stands on its own.
TOKENIZERS = [
    "PA-Clean-capped", "PA-Clean-plus2-capped",
    "PA-Clean-plus3-cap-hw-cv2", "PA-Clean-plus3-cap-cv2",
    "PA-Apertus-capped",
    "SuperBPE-clean-fw2full-hw", "SuperBPE-apertus-fw2full-hw",
    "Apertus",
    "Gemma 3", "Llama 4", "OLMo 2", "Qwen 3", "K2 Think",
]

DEFAULT_FAMILY_CFG = (
    "/users/cmeister747/pa_tokenizers_branch/configs/"
    "parity_aware_config_grouped_fineweb2full_quota_tuned.json"
)


def parse_family_config(path):
    """Return {family_name: sorted [flores_codes]}; skips empty/non-language families
    (code, english, math have no fineweb2_sample inputs in the config)."""
    cfg = json.load(open(path))
    out = {}
    for g in cfg.get("languages", []):
        name = g.get("name")
        langs = []
        for p in g.get("input", []):
            m = re.search(r"fineweb2_sample/([^/]+)/", p)
            if m:
                langs.append(m.group(1))
        if langs:
            out[name] = sorted(set(langs))
    return out


def load_per_lang(analysis_json, metric):
    """Return {tokenizer: {lang: value}} for the requested metric."""
    R = json.load(open(analysis_json))
    out = {}
    if metric == "compression_rate":
        per_tok = R.get("compression_rate", {}).get("per_tokenizer", {})
        for t, d in per_tok.items():
            pl = d.get("per_language", {}) or {}
            # per_language values may be scalars or dicts with .compression_rate
            out[t] = {lang: (v.get("compression_rate") if isinstance(v, dict) else v)
                      for lang, v in pl.items()}
    elif metric == "vocabulary_utilization":
        per_tok = R.get("vocabulary_utilization", {}).get("per_tokenizer", {})
        for t, d in per_tok.items():
            pl = d.get("per_language", {}) or {}
            out[t] = {lang: (v.get("utilization") if isinstance(v, dict) else v)
                      for lang, v in pl.items()}
    else:
        raise ValueError(f"unknown metric: {metric}")
    return out


def plot_family_facets(values, families, tokenizers, metric_label, out_path,
                       ylabel, log_y=False):
    """Produce one SVG with a grid of panels (one per family that has >=2 langs in
    the analysis), each panel a line plot of metric vs language with one line per
    tokenizer."""
    # Keep only families with >=2 langs present in the analysis for at least one tokenizer
    coverage = {}
    union_langs = set()
    for t in tokenizers:
        if t in values:
            union_langs |= set(values[t])
    panels = []
    for fam, langs in families.items():
        present_langs = [l for l in langs if l in union_langs]
        if len(present_langs) >= 2:
            panels.append((fam, present_langs))
    panels.sort(key=lambda x: (-len(x[1]), x[0]))  # widest first; then alphabetical
    if not panels:
        print(f"no families with >=2 covered langs; nothing to plot for {metric_label}")
        return

    n = len(panels)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(tokenizers))]

    for idx, (fam, langs) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        x = np.arange(len(langs))
        for j, t in enumerate(tokenizers):
            ys = [values.get(t, {}).get(l) for l in langs]
            ys = [np.nan if v is None else v for v in ys]
            ax.plot(x, ys, marker="o", markersize=3, linewidth=1.0,
                    color=colors[j], label=t, alpha=0.85)
        ax.set_title(f"{fam} (n={len(langs)})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=70, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        if log_y:
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=8)

    # blank the unused cells
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    # One shared legend at the bottom of the figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 5), bbox_to_anchor=(0.5, -0.01),
               fontsize=9, frameon=False)
    fig.suptitle(f"{metric_label} per language, faceted by PA-BPE training family",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} family panels)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-config", default=DEFAULT_FAMILY_CFG,
                    help="PA-BPE tuned parity config that defines family -> [flores_codes].")
    ap.add_argument("--analysis", default="results/report_flores_devtest/analysis_results_full.json",
                    help="Analysis JSON (default: full FLORES devtest, 205 langs).")
    ap.add_argument("--output-dir", default="results/family_plots",
                    help="Directory for the output SVGs.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.family_config):
        print(f"family config not found: {args.family_config}", file=sys.stderr)
        return 1
    if not os.path.exists(args.analysis):
        print(f"analysis JSON not found: {args.analysis}", file=sys.stderr)
        return 1

    families = parse_family_config(args.family_config)
    print(f"parsed {len(families)} families with languages: "
          f"{', '.join(sorted(families))}")

    os.makedirs(args.output_dir, exist_ok=True)
    for metric_key, label, ylab, log in [
        ("compression_rate", "Compression rate (FLORES sentences per token, higher = denser)",
         "sent/tok", False),
        ("vocabulary_utilization", "Vocabulary utilization (fraction of vocab used per language, higher = more reuse)",
         "vocab util", False),
    ]:
        vals = load_per_lang(args.analysis, metric_key)
        out_path = os.path.join(args.output_dir, f"{metric_key}_by_family.svg")
        plot_family_facets(vals, families, TOKENIZERS, label, out_path, ylab, log_y=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
