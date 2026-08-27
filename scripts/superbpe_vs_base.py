#!/usr/bin/env python
"""SuperBPE-vs-base vocabulary analysis (aggregate across pairs).

For each (SuperBPE, PA-BPE base) pair this computes what the SuperBPE stage-2
changed relative to the subword base it was grown from:
  - shared / sacrificed (base-only) / added (SuperBPE-only) token counts,
  - the fraction of added tokens that are true superwords (contain an internal
    space, i.e. span >=2 pretokenized words),
  - per-language usage of the added tokens on FLORES devtest (is the gain
    English-centric or broad?),
  - the script make-up of the sacrificed tokens (which languages lose subwords),
  - code-related added superwords, characterized per pretokenization family.

Writes results/report_superbpe_vs_base.json; the report builder renders it.
Tokenizer byte-level space marker is 'Ġ' (U+0120); an internal occurrence marks
a superword.  Nothing here is hardcoded into the report — every number traces
to this artifact.
"""
import argparse
import collections
import json
import os
import re
import unicodedata

from tokenizers import Tokenizer

SP = chr(0x120)  # 'Ġ' byte-level space marker
FLORES_DEVTEST = ("/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/"
                  "flores_parallel_data_devtest")
TLM = "/users/cmeister747/pa_tokenizers_branch/experiments/tokenizer-lm-toks"
SB = "/users/cmeister747/superbpe/trained_tokenizers"

# (display name, base display name, family, mode, superbpe dir, base dir)
PAIRS = [
    ("SuperBPE-clean-fw2full-hw",   "PA-Clean-capped",      "clean",   "hw",
     f"{SB}/clean_multi_capped_fw2full_hw",  f"{TLM}/nfc_clean_multi_capped_hybrid_window_tuned"),
    ("SuperBPE-apertus-fw2full-hw", "PA-Apertus-capped",    "apertus", "hw",
     f"{SB}/apertus_capped_fw2full_hw",      f"{TLM}/nfc_apertus_capped_hybrid_window_tuned"),
    ("SuperBPE-gpt4-fw2full-hw",    "PA-gpt4-fineweb2full", "gpt4",    "hw",
     f"{SB}/gpt4_fineweb2full_hw",           f"{TLM}/nfc_gpt4_fineweb2full_hybrid_window"),
    ("SuperBPE-clean-fw2full",      "PA-Clean-capped-base", "clean",   "base",
     f"{SB}/clean_multi_capped_fw2full",     f"{TLM}/nfc_clean_multi_capped_tuned"),
    ("SuperBPE-apertus-fw2full",    "PA-Apertus-base",      "apertus", "base",
     f"{SB}/apertus_capped_fw2full",         f"{TLM}/nfc_apertus_capped_tuned"),
    ("SuperBPE-gpt4-fw2full",       "PA-gpt4-fw2full-base", "gpt4",    "base",
     f"{SB}/gpt4_fineweb2full",              f"{TLM}/nfc_gpt4_fineweb2full"),
]

# Code-rich sample: numbers (digit-grouping differs by pretok family), operators,
# indentation, identifiers, comments, strings, across a couple of languages.
CODE_SAMPLE = '''def running_mean(values, window=100):
    # compute the rolling average over a fixed window
    total = 0
    out = []
    for i in range(len(values)):
        total += values[i]
        if i >= window:
            total -= values[i - window]
        if i >= window - 1:
            out.append(total / window)
    return out

class Buffer(BaseBuffer):
    MAX_SIZE = 1048576
    def __init__(self, capacity=None):
        self.capacity = capacity or 4096
        self.items = {}
    def put(self, key, value):
        if key not in self.items:
            self.items[key] = []
        self.items[key].append(value)

for idx in range(0, 1000, 25):
    result = compute(idx) * 3.14159 + 0xFF
    print(f"idx={idx} -> {result:.2f}")
'''


def nwords(tok_str):
    return tok_str.lstrip(SP).count(SP) + 1


def first_script(text):
    for ch in text:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for s in ("LATIN", "CYRILLIC", "ARABIC", "HEBREW", "CJK", "HIRAGANA",
                  "KATAKANA", "HANGUL", "DEVANAGARI", "BENGALI", "TAMIL", "THAI",
                  "GREEK", "GEORGIAN", "ARMENIAN", "ETHIOPIC"):
            if s in name:
                return s
        return "OTHER"
    return "EMPTY"


def analyze_pair(name, base_name, family, mode, sb_dir, base_dir, langs):
    sbpe = Tokenizer.from_file(f"{sb_dir}/tokenizer.json")
    base = Tokenizer.from_file(f"{base_dir}/tokenizer.json")
    vs, vb = set(sbpe.get_vocab()), set(base.get_vocab())
    added, sacr, shared = vs - vb, vb - vs, vb & vs

    def dec(tk, t):
        i = tk.token_to_id(t)
        return tk.decode([i]) if i is not None else ""

    superw = [t for t in added if nwords(t) >= 2]
    # sacrificed script make-up
    sac_scripts = collections.Counter(first_script(dec(base, t)) for t in sacr)
    nonlatin = sum(c for s, c in sac_scripts.items() if s not in ("LATIN", "EMPTY", "OTHER"))

    # per-language usage of added tokens (FLORES devtest)
    added_ids = {i for i in (sbpe.token_to_id(t) for t in added) if i is not None}
    lang_share = {}
    for lg in langs:
        lines = open(f"{FLORES_DEVTEST}/{lg}.txt").read().splitlines()
        ids = [i for e in sbpe.encode_batch(lines) for i in e.ids]
        n = len(ids)
        lang_share[lg] = (sum(1 for i in ids if i in added_ids) / n) if n else 0.0
    ranked = sorted(lang_share.items(), key=lambda kv: -kv[1])

    # code: how much do added tokens / added superwords fire, and which superwords
    e = sbpe.encode(CODE_SAMPLE)
    ctoks = [sbpe.id_to_token(i) for i in e.ids]
    cadded = [t for t in ctoks if t in added]
    csuper = [t for t in cadded if nwords(t) >= 2]
    csuper_dec = sorted({dec(sbpe, t) for t in csuper})
    # added superwords that look code-relevant (syntax chars or code keywords)
    kw = re.compile(r"[=(){}\[\];:.<>/_+\-*]|self|def |return |import |class |for |if |elif |while |print")
    code_sw = [dec(sbpe, t) for t in superw if kw.search(dec(sbpe, t))]

    import random
    random.seed(0)
    return {
        "name": name, "base": base_name, "family": family, "mode": mode,
        "shared": len(shared), "sacrificed": len(sacr), "added": len(added),
        "superword_frac": round(len(superw) / len(added), 4) if added else None,
        "sacrificed_nonlatin_frac": round(nonlatin / len(sacr), 4) if sacr else None,
        "sacrificed_examples": [dec(base, t) for t in random.sample(sorted(sacr), min(10, len(sacr)))],
        "added_superword_examples": [dec(sbpe, t) for t in random.sample(superw, min(12, len(superw)))],
        "eng_added_share": round(lang_share.get("eng_Latn", 0.0), 3),
        "mean_added_share": round(sum(lang_share.values()) / len(lang_share), 3) if lang_share else None,
        "top_langs": [[l, round(s, 3)] for l, s in ranked[:8]],
        "zero_langs": [l for l, s in ranked if s < 0.005][:12],
        "code_added_frac": round(len(cadded) / len(ctoks), 4) if ctoks else None,
        "code_superword_frac": round(len(csuper) / len(ctoks), 4) if ctoks else None,
        "code_superwords_in_sample": csuper_dec,
        "n_code_relevant_superwords": len(code_sw),
        "code_relevant_superword_examples": sorted(set(code_sw))[:20],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    langs = sorted(os.path.basename(f)[:-4]
                   for f in os.scandir(FLORES_DEVTEST) if f.name.endswith(".txt"))
    pairs = []
    for spec in PAIRS:
        name = spec[0]
        if not (os.path.exists(f"{spec[4]}/tokenizer.json") and os.path.exists(f"{spec[5]}/tokenizer.json")):
            print(f"  SKIP {name}: tokenizer or base missing")
            continue
        print(f"  analyzing {name} vs {spec[1]} ...")
        pairs.append(analyze_pair(*spec, langs=langs))
    out = {"superbpe_vs_base": {
        "method": ("Each SuperBPE compared to the PA-BPE subword base it was grown from. 'added' = "
                   "tokens only in SuperBPE; 'sacrificed' = tokens only in the base; superword = "
                   "token with an internal space (spans >=2 pretokenized words). Per-language added "
                   "usage and code firing measured on FLORES devtest and a fixed code sample."),
        "pairs": pairs}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(out, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {args.output} ({len(pairs)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
