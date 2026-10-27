#!/usr/bin/env python
"""Per-language faceted plots for the focus candidates.

One panel per tokenizer. Bars are per-language values for a hand-picked subset
of languages spanning the European head to the low-resource tail. Two metrics
are supported:

  compression : FLORES sentences per token (n_lines / total_tokens). Parallel
                sentences, so comparable across scripts.
  vocab_util  : raw count of distinct vocabulary ids used to encode the
                language's FLORES corpus (not a percentage).

Data: FLORES parallel txt files (997 sentences per language), encoded with
add_special_tokens=False. Run from the repo root.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tokenizers import Tokenizer

PARALLEL_DIR = "path/to/flores_parallel"  # FLORES parallel text, one file per language
# Apertus v1 vocab == Mistral-Nemo-Base-2407 (verified identical FLORES
# sentences-per-token, e.g. Tibetan 0.0031).
APERTUS_V1 = "path/to/Apertus-70B-2509/tokenizer.json"
O200K = "path/to/gpt_4o_hf.json"  # o200k (GPT-4o) tokenizer.json

# (panel title, tokenizer.json path). Order matches per_language_compression.
TOKENIZERS = [
    ("preliminary_mul", "preliminary_mul/tokenizer.json"),
    ("preliminary_enh", "preliminary_enh/tokenizer.json"),
    ("preliminary_euh", "preliminary_euh/tokenizer.json"),
    ("preliminary_mul_200k", "preliminary_mul_200k/tokenizer.json"),
    ("Apertus v1", APERTUS_V1),
    ("o200k (GPT-4o)", O200K),
]

# Family colors. EU/Semitic/CJK/Indic/Other match per_language_compression.svg;
# OtherLatin (#a6761d) is added for Turkish and Swahili.
FAMILY_COLOR = {
    "EU": "#2c7fb8",
    "Semitic": "#7fcdbb",
    "CJK": "#d95f0e",
    "Indic": "#31a354",
    "Other": "#756bb1",
    "OtherLatin": "#a6761d",
}

# (FLORES code, short label, family). Order = head to tail, grouped by family.
LANGS = [
    ("eng_Latn", "English", "EU"),
    ("deu_Latn", "German", "EU"),
    ("fra_Latn", "French", "EU"),
    ("rus_Cyrl", "Russian", "EU"),
    ("arb_Arab", "Arabic", "Semitic"),
    ("cmn_Hani", "Mandarin", "CJK"),
    ("jpn_Jpan", "Japanese", "CJK"),
    ("kor_Hang", "Korean", "CJK"),
    ("hin_Deva", "Hindi", "Indic"),
    ("ben_Beng", "Bengali", "Indic"),
    ("tam_Taml", "Tamil", "Indic"),
    ("tha_Thai", "Thai", "Other"),
    ("bod_Tibt", "Tibetan", "Other"),
    ("tur_Latn", "Turkish", "OtherLatin"),
    ("swh_Latn", "Swahili", "OtherLatin"),
]

LEGEND_ORDER = ["EU", "Semitic", "CJK", "Indic", "Other", "OtherLatin"]
LEGEND_LABEL = {
    "EU": "European",
    "Semitic": "Semitic (Arabic)",
    "CJK": "CJK",
    "Indic": "Indic",
    "Other": "Thai / Tibetan",
    "OtherLatin": "Turkish / Swahili",
}


def load_lines(code):
    path = os.path.join(PARALLEL_DIR, f"{code}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing FLORES parallel file: {path}")
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def per_language_value(tok, lines, metric):
    total_tokens = 0
    used = set()
    for enc in tok.encode_batch(lines, add_special_tokens=False):
        total_tokens += len(enc.ids)
        if metric == "vocab_util":
            used.update(enc.ids)
    if metric == "compression":
        if total_tokens == 0:
            raise ValueError("zero tokens encoded")
        return len(lines) / total_tokens
    return len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["compression", "vocab_util"], required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    corpora = {code: load_lines(code) for code, _, _ in LANGS}

    # values[panel_title][code] = number
    values = {}
    for title, path in TOKENIZERS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing tokenizer: {path}")
        tok = Tokenizer.from_file(path)
        values[title] = {
            code: per_language_value(tok, corpora[code], args.metric)
            for code, _, _ in LANGS
        }
        print(title, {LANGS[i][1]: round(values[title][LANGS[i][0]], 4) for i in range(len(LANGS))})

    labels = [lab for _, lab, _ in LANGS]
    colors = [FAMILY_COLOR[fam] for _, _, fam in LANGS]
    xs = list(range(len(LANGS)))
    ymax = max(v for panel in values.values() for v in panel.values())

    if args.metric == "compression":
        ylabel = "FLORES sentences per token"
        suptitle = "Per-language compression (FLORES sentences per token)"
        fmt = lambda v: f"{v:.3f}"
    else:
        ylabel = "distinct vocabulary ids used"
        suptitle = "Per-language vocabulary utilization (distinct ids used on FLORES, 997 sentences)"
        fmt = lambda v: f"{int(round(v))}"

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.1), sharey=True)
    axes = axes.ravel()
    for ax, (title, _) in zip(axes, TOKENIZERS):
        heights = [values[title][code] for code, _, _ in LANGS]
        ax.bar(xs, heights, color=colors, width=0.78)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, ymax * 1.12)
        ax.grid(axis="y", color="#b0b0b0", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        for x, h in zip(xs, heights):
            ax.text(x, h + ymax * 0.012, fmt(h), ha="center", va="bottom",
                    fontsize=6.0, rotation=90)
    for ax in axes[::3]:
        ax.set_ylabel(ylabel, fontsize=9)

    handles = [Patch(facecolor=FAMILY_COLOR[f], label=LEGEND_LABEL[f]) for f in LEGEND_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(LEGEND_ORDER),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(f"{args.out}.svg")
    fig.savefig(f"{args.out}.png", dpi=150)
    print("wrote", args.out + ".svg", args.out + ".png")


if __name__ == "__main__":
    main()
