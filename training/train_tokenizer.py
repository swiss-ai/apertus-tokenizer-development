#!/usr/bin/env python3
"""Train one of 6 tokenizer variants for the 30-language PA-BPE experiment.

Variants:
  bpe_baseline     - standard BpeTrainer
  pa_bpe_dev       - ParityBpeTrainer base variant, dev-driven
  pa_bpe_hybrid    - ParityBpeTrainer base, global_merges=64000 (hybrid)
  pa_bpe_window    - ParityBpeTrainer window variant (W=100, alpha=2)
  pa_bpe_ratios    - ParityBpeTrainer base, FLORES bytes/line ratios (no dev)
  pa_bpe_dev_gpt4o - ParityBpeTrainer base + GPT-4o regex pretokenizer

All variants share the same 5 GiB training corpus, 150 MiB/lang floor,
mT5-proportional sampling, and a fixed special-tokens list.

Usage:
  train_tokenizer.py --variant <name>          # full run
  train_tokenizer.py --variant <name> --smoke-test   # tiny corpus + vocab
"""
import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyarrow

from tokenizers import Tokenizer, Regex, decoders, normalizers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer, ParityBpeTrainer

from pa_bpe_iterators import (
    LanguageTrainCorpus,
    LanguageDevCorpus,
    ListedFileCorpus,
    compute_quota_bytes,
)
from pretokenizer_regexes import (
    REGEX_GPT4O,
    REGEX_CLEAN_MULTI,
    REGEX_CLEAN_MULTI_CAPPED,
    REGEX_CLEAN_MULTI_PLUS,
    REGEX_CLEAN_MULTI_PLUS_CAPPED,
    REGEX_CLEAN_MULTI_PLUS2_CAPPED,
    REGEX_CLEAN_MULTI_PLUS3_CAPPED,
    REGEX_CLEAN_MULTI_PLUS2_REPCAP8_CAPPED,
    REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED,
    REGEX_APERTUS_CAPPED,
)

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "configs" / "lang_manifest_30.json"
MC4_WEIGHTS_PATH = REPO_ROOT / "configs" / "mc4_weights.json"
FLORES_RATIOS_PATH = REPO_ROOT / "configs" / "flores_ratios.json"

# Hyperparameters shared across all variants (paper-faithful).
VOCAB_SIZE = 128_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]

# Apertus 99-token special spec (IDs 0-98). Sourced from
# configs/apertus_special_tokens.json which mirrors the team Google sheet.
# When a variant sets `special_tokens_override` to this list, the trainer
# pre-adds these 99 tokens so they occupy IDs 0-98 in the final vocab.
def _load_apertus_specials():
    import json as _json
    p = REPO_ROOT / "configs" / "apertus_special_tokens.json"
    if not p.exists():
        return None
    d = _json.loads(p.read_text(encoding="utf-8"))
    return d["tokens"]
APERTUS_SPECIAL_TOKENS = _load_apertus_specials()
TOTAL_BYTES = 5 * 1024 * 1024 * 1024       # 5 GiB
FLOOR_BYTES = 150 * 1024 * 1024            # 150 MiB
GLOBAL_SEED = 20260409

# Smoke test: tiny quotas + small vocab for local end-to-end validation.
SMOKE_TOTAL_BYTES = 20 * 1024 * 1024       # 20 MiB
SMOKE_FLOOR_BYTES = 2 * 1024 * 1024        # 2 MiB
SMOKE_VOCAB_SIZE = 8000

# Quick test: FULL corpus + small vocab. Used to verify memory/environment on
# a real compute node before committing to a 12 h training run.
QUICK_VOCAB_SIZE = 8000

# -------- variant config -----------------------------------------------------

VARIANTS = {
    "bpe_baseline": {
        "trainer": "bpe",
        "pretok": "whitespace",
    },
    "pa_bpe_dev": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 0,
        "signal": "dev",
    },
    "pa_bpe_hybrid": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 64_000,
        "signal": "dev",
    },
    "pa_bpe_window": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "window",
        "global_merges": 0,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "dev",
    },
    "pa_bpe_ratios": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
    },
    "pa_bpe_dev_gpt4o": {
        "trainer": "parity-bpe",
        "pretok": "gpt4o",
        "variant": "base",
        "global_merges": 0,
        "signal": "dev",
    },
    "pa_bpe_reverse_hybrid": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 64_000,
        "global_merges_at_end": True,
        "signal": "dev",
    },
    "pa_bpe_hybrid_window": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "dev",
    },
    # Dev-set-size ablations. Same hyperparams as pa_bpe_dev / pa_bpe_hybrid
    # but truncate each language's FLORES+ dev file to the first N non-empty
    # lines (parallel, so "size 100" = 100 parallel sentences × 30 langs).
    "pa_bpe_dev_devsize100": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 0,
        "signal": "dev",
        "dev_max_lines": 100,
    },
    "pa_bpe_dev_devsize300": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 0,
        "signal": "dev",
        "dev_max_lines": 300,
    },
    "pa_bpe_hybrid_devsize100": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 64_000,
        "signal": "dev",
        "dev_max_lines": 100,
    },
    "pa_bpe_hybrid_devsize300": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "base",
        "global_merges": 64_000,
        "signal": "dev",
        "dev_max_lines": 300,
    },
    # Window-size ablations. Same hyperparams as pa_bpe_window but vary the
    # moving-window size (W=100 is the baseline in the paper setting).
    "pa_bpe_window_ws50": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "window",
        "global_merges": 0,
        "window_size": 50,
        "alpha": 2.0,
        "signal": "dev",
    },
    "pa_bpe_window_ws150": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "window",
        "global_merges": 0,
        "window_size": 150,
        "alpha": 2.0,
        "signal": "dev",
    },
    "pa_bpe_window_ws200": {
        "trainer": "parity-bpe",
        "pretok": "whitespace",
        "variant": "window",
        "global_merges": 0,
        "window_size": 200,
        "alpha": 2.0,
        "signal": "dev",
    },

    # Replication of the old pabpe-128k-nfc-gpt4-reg_moddata.json tokenizer.
    # (Historical name — what we now call `_fineweb2full` was originally
    # `_moddata` / `_downstream`; retired 2026-05-18 alongside the
    # silent-exclusion fix in pa_bpe_iterators.py.) Uses the grouped
    # FineWeb-2 full-sample config (not the 30-lang manifest), NFC
    # normalization, GPT-4o regex pretok, asymmetric ByteLevel decoder
    # (add_prefix_space=True on the decoder side, matching defaults),
    # no special tokens.
    "pa_bpe_nfc_gpt4_fineweb2full": {
        "trainer": "parity-bpe",
        "pretok": "gpt4o_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        # Uses the _quota.json sibling (0.70x per-family shrink). The
        # no-quota config materialized 33 GB on disk into ~460 GB resident
        # and OOM'd in base mode too (SLURM 2256970/2257135), not just
        # hybrid+window. See TOKENIZER_TRAINING.md sec 6.5 / sec 6.9.
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_gpt4_fineweb2full",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # Same trainer settings as pa_bpe_nfc_gpt4_fineweb2full, but data spec
    # is derived from /users/cmeister747/tokenizer-lm/configs/data/balanced.json
    # (34 groups; ~9.5 GB text budget) and ratios are computed from FLORES+
    # dev bytes-per-line. Used to retry the FineWeb-2 full-sample variant
    # after the original no-quota run OOM'd at 450 GB on the full 33 GB
    # corpus.
    "pa_bpe_nfc_gpt4_balanced": {
        "trainer": "parity-bpe",
        "pretok": "gpt4o_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_balanced.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_gpt4_balanced",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # Hybrid+window variants of the two grouped-config runs above, intended
    # as drop-in replacements consumed by the LM pipeline at ~/tokenizer-lm/.
    # - variant=window + global_merges=64000 mirrors the pa_bpe_hybrid_window
    #   algorithm: 64k language-blind merges first, then 64k parity-window merges.
    # - output_group="tokenizer-lm-toks" redirects the final artifact directly
    #   into experiments/tokenizer-lm-toks/<output_dir_name>/ (smoke/quick tests
    #   still go to archive/).
    # - post_training_special_tokens adds <s>/</s>/<unk>/<pad> via
    #   Tokenizer.add_special_tokens after training, matching the pattern of
    #   the existing tokenizer-lm-toks/nfc_gpt4_{fineweb2full,balanced}/ files.
    "pa_bpe_nfc_gpt4_fineweb2full_hybrid_window": {
        "trainer": "parity-bpe",
        "pretok": "gpt4o_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        # Uses *_quota.json sibling: same 25 family groups and per-group `ratio`
        # as the original grouped FineWeb-2 full-sample config, plus per-family
        # quota_bytes (uniform 0.70x shrink of on-disk parquet sums; see
        # TOKENIZER_TRAINING.md sec 6.3) so the hybrid+window run fits in
        # 450 GB. Quota was added because the original config (no quotas)
        # OOM'd at 463 GB resident in SLURM 1900564.
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_gpt4_fineweb2full_hybrid_window",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    "pa_bpe_nfc_gpt4_balanced_hybrid_window": {
        "trainer": "parity-bpe",
        "pretok": "gpt4o_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_balanced.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_gpt4_balanced_hybrid_window",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # Sibling variants of the four pa_bpe_nfc_gpt4_* tokenizer-lm-toks/ entries
    # above, identical in every respect except the pretokenizer regex: this set
    # uses REGEX_CLEAN_MULTI (a Mistral-Nemo-style clean multilingual regex —
    # no English contractions, single-digit groups, narrower [ ]? leading-char
    # than GPT-4o/Mistral-Nemo). NFC normalization is still applied via the
    # _nfc suffix on the pretok name. Output dirs sit alongside the gpt4 set,
    # prefixed nfc_clean_multi_ instead of nfc_gpt4_, so the two pretokenizer
    # families are directly comparable on otherwise-identical data + trainer.
    "pa_bpe_nfc_clean_multi_fineweb2full": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        # See pa_bpe_nfc_gpt4_fineweb2full: uses the _quota.json sibling
        # because the no-quota config OOM'd at ~460 GB in base mode.
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_fineweb2full",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_balanced": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_balanced.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_balanced",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_fineweb2full_hybrid_window": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        # Same quota config used by pa_bpe_nfc_gpt4_fineweb2full_hybrid_window
        # — the 0.70x per-family shrink keeps this hybrid+window run under
        # the 450 GB SLURM cap (see TOKENIZER_TRAINING.md sec 6.3).
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_fineweb2full_hybrid_window",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_balanced_hybrid_window": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_balanced.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_balanced_hybrid_window",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # ---- standard-BPE baselines (trainer="bpe", no parity signal) ----------
    # Plain BpeTrainer over the same grouped corpora, for comparison against
    # the parity-aware tokenizer-lm-toks/ variants. NFC + the named pretok
    # regex; full 256-byte alphabet + 4 post-training specials, same as the
    # PA set. fineweb2full uses the _quota.json config (no-quota OOMs
    # at ~460 GB regardless of trainer — see TOKENIZER_TRAINING.md §6.7).
    # Output dirs are prefixed `bpe_` to keep them distinct from the PA dirs.
    "bpe_nfc_clean_multi_balanced": {
        "trainer": "bpe",
        "pretok": "clean_multi_nfc",
        "grouped_config": "configs/parity_aware_config_balanced.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_clean_multi_balanced",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "bpe_nfc_clean_multi_fineweb2full": {
        "trainer": "bpe",
        "pretok": "clean_multi_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_clean_multi_fineweb2full",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "bpe_nfc_gpt4_fineweb2full": {
        "trainer": "bpe",
        "pretok": "gpt4o_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_gpt4_fineweb2full",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # ===== _tuned experiment batch (2026-05-23) ============================
    # Data config `_tuned`: European family ratios x1.2 (subtle favoring,
    # English unchanged); removed kas_Deva + lij_Latn (poor data quality);
    # regrouped ydd_Hebr/kas_Arab/knc_Arab/uzs_Arab -> semitic (script-aware).
    # Pretok schemes: clean-multi-capped and apertus-capped (run-capped).
    # 6 runs on the tuned config + 2 on the old config (apertus-capped only)
    # to isolate the apertus-capped pretokenizer against the old data.
    # ----- tuned config: {BPE, PA base, PA hybrid+window} x {clean_multi, apertus} capped
    "bpe_nfc_clean_multi_capped_tuned": {
        "trainer": "bpe",
        "pretok": "clean_multi_capped_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_clean_multi_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "bpe_nfc_apertus_capped_tuned": {
        "trainer": "bpe",
        "pretok": "apertus_capped_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_apertus_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    # Plain upstream BPE with the plus3 pretokenizer (companion to the
    # parity-aware plus3 variants — same data config, same vocab budget).
    "bpe_nfc_clean_multi_plus3_capped_tuned": {
        "trainer": "bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_clean_multi_plus3_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_capped_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_apertus_capped_tuned": {
        "trainer": "parity-bpe",
        "pretok": "apertus_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_apertus_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_capped_hybrid_window_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_capped_hybrid_window_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_apertus_capped_hybrid_window_tuned": {
        "trainer": "parity-bpe",
        "pretok": "apertus_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_apertus_capped_hybrid_window_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    # ----- old (untuned) config: apertus-capped only, BPE + PA hybrid+window
    "bpe_nfc_apertus_capped_oldcfg": {
        "trainer": "bpe",
        "pretok": "apertus_capped_nfc",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_apertus_capped_oldcfg",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_apertus_capped_hybrid_window_oldcfg": {
        "trainer": "parity-bpe",
        "pretok": "apertus_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_apertus_capped_hybrid_window_oldcfg",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # ===== _tuned ablations (2026-05-24), hybrid+window + apertus-capped only =====
    # Single-variable changes from the _tuned config (RESULTS.md §12 follow-up):
    #  - _noregroup: keep European x1.2 + quality removals, DROP the regrouping
    #    (ydd_Hebr/kas_Arab/knc_Arab/uzs_Arab stay in their original family groups).
    #  - _x1p1:      keep removals + regrouping, soften European bump x1.2 -> x1.1.
    "pa_bpe_nfc_apertus_capped_hybrid_window_tuned_noregroup": {
        "trainer": "parity-bpe",
        "pretok": "apertus_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_noregroup.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_apertus_capped_hybrid_window_tuned_noregroup",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_apertus_capped_hybrid_window_tuned_x1p1": {
        "trainer": "parity-bpe",
        "pretok": "apertus_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_x1p1.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_apertus_capped_hybrid_window_tuned_x1p1",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # ===== clean_multi_plus comparison run (2026-05-24) =====
    # clean_multi_plus_capped = clean_multi_capped + (a) Tibetan tsek attaches to
    # the next syllable (fixes the bod_Tibt/dzo_Tibt fragmentation that clean_multi
    # had vs apertus) and (b) the seven canonical English contractions as a
    # standalone first branch (so don't -> [don,'t]). Tuned data config, hyb+win is
    # the production target; PA base is the comparison point.
    "pa_bpe_nfc_clean_multi_plus_capped_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus_capped_hybrid_window_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus_capped_hybrid_window_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # plus2 extends plus by also attaching ASCII apostrophe (U+0027) and right
    # curly quote (U+2019) to the next word. Targets the residual fra/ita/cat/mlt
    # gap vs Apertus (l'eau, c'est, l'arte). Single-variable change vs plus.
    "pa_bpe_nfc_clean_multi_plus2_capped_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus2_capped_hybrid_window_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_hybrid_window_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # plus3 = plus2 + guarded trailing apostrophe attachment. Closes Maltese
    # morpheme gap (ta', gh', m'), dialect English (talkin'), math primes
    # (f', A'). Multilingual base preserved by the (?!\p{L}) guard.
    "pa_bpe_nfc_clean_multi_plus3_capped_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # targA / targAplus variants: plus3 pretok + targeted ratio dampening on
    # the chrome-producing families. See VOCAB_FILTERING_PLAN.md v5 and
    # gen_ratio_variants.py for the data-config rationale.
    #   targA     : taikadai 3.160 -> 1.500 (only)
    #   targAplus : taikadai 3.160 -> 1.500  AND  dravidian 2.966 -> 2.000
    "pa_bpe_nfc_clean_multi_plus3_capped_tuned_targA": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_targA.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_tuned_targA",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_targA": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_targA.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_targA",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_tuned_targAplus": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_targAplus.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_tuned_targAplus",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_targAplus": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_targAplus.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_targAplus",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # v6 principled reweighting variants. See VOCAB_FILTERING_PLAN.md §8 and
    # gen_ratio_variants.py. Combines encoding cost × max(f_data, f_speakers)
    # penalty, with an empirical taikadai cap. No global RATIO_CAP.
    #   consv2: D_REF=10 GB,  S_REF=50 M,  taikadai_cap=2.00
    #   modv2:  D_REF=50 GB,  S_REF=200 M, taikadai_cap=1.75
    "pa_bpe_nfc_clean_multi_plus3_capped_tuned_consv2": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_tuned_consv2",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_tuned_modv2": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "base",
        "global_merges": 0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_modv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_tuned_modv2",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_modv2": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_modv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_modv2",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # plus2 hyb+win with the consv2 data config — companion to the plus3 consv2.
    # Lets us isolate the trailing-apostrophe arm (plus3 vs plus2) from the
    # ratio-dampening effect.
    "pa_bpe_nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # plus3 hyb+win consv2 with raised global_merges (default 64 k -> 70/75/80 k).
    # The global phase picks merges by raw corpus frequency, so a larger
    # global budget gives English (33 % of corpus by bytes) more merges in
    # exchange for fewer parity merges to the tail families.
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 70_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm75k": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 75_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm75k",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # gm70k / gm80k with the vocab budget bumped to 130 900 (Apertus has
    # 131 072 BPE + 1 000 added = 132 072 effective; 130 900 leaves headroom
    # for our 4 post-training specials so the final tokenizer.json matches
    # Apertus closely). Lets us isolate "vocab size advantage" from
    # "training corpus mix" when comparing to Apertus on English compression.
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k_v130900": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 70_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_900,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k_v130900",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k_v130900": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_900,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": [],
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k_v130900",
        "post_training_special_tokens": ["<s>", "</s>", "<unk>", "<pad>"],
    },

    # 8-run ablation matrix exploring English-compression levers.
    # All use plus3 pretok + PA-BPE hyb+win + 99 Apertus specials (pre-added
    # at IDs 0-98) + vocab_size=130_999 (= 130_900 merges + 99 specials).
    # Suffix `_sp` marks the 99-special-token regime.
    # Tail cuts: mande/baltic/celtic/uralic/nigercongo_{other,voltaniger}
    # pinned to ratio=1.0 (see configs/..._tailcuts*.json).
    # Eng+5G: 18 extra FineWeb CC-MAIN dumps + quota 7.72 -> 12.72 GB
    # (see configs/..._eng5g.json).
    #
    # A1: baseline-shape + 99 specials at gm90k  (fills baseline gm=90 cell)
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm90k_v130_sp": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm90k_v130_sp",
    },
    # A2-A4: data-only effect at 3 gm levels (English +5 GB, no tail cuts)
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 70_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm70k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm80k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm90k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_gm90k_v130_sp_eng5g",
    },
    # A5: tail-cuts-only anchor at gm80k
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp",
    },
    # A6-A8: combined (tail cuts + English +5 GB) at 3 gm levels
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm70k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 70_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm70k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g",
    },

    # plus2 mirrors of A6/A7/A8 (consv2 + tailcuts + eng5g, 99 specials,
    # vocab=130_999). Same data + ratio config as the plus3 ablations, only
    # the pretokenizer differs (plus2 = trailing-apostrophe arm OFF).
    "pa_bpe_nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm70k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "window",
        "global_merges": 70_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm70k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g",
    },

    # plus3_repcap8: plus3 + same-character repeat-run cap at 8 (both word
    # and punct arms). Implementation via a guard arm at the top of the
    # regex; see pretokenizer_regexes.py REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED.
    # Three jobs: standard BPE + repcap8 + English-boosted data, and PA-BPE
    # A7/A8 (gm80k/gm90k) mirrors with the new pretokenizer.
    "bpe_nfc_clean_multi_plus3_repcap8_capped_tuned_consv2_tailcuts_v130_sp_eng5g": {
        "trainer": "bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "bpe_nfc_clean_multi_plus3_repcap8_capped_tuned_consv2_tailcuts_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 80_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm80k_v130_sp_eng5g",
    },
    "pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 130_999,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v130_sp_eng5g",
    },
    # Retrained A8 v131k sp120 with the FIXED repcap8 regex (lookaheads added
    # to word + punct arms; see pretokenizer_regexes.py). Same data/ratios
    # as the previous A8 v131k sp120; the `_fr` suffix marks "fixed regex" to
    # keep both artifacts on disk for comparison.
    "pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp124_eng5g_fr": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 131_072,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp124_eng5g_fr",
    },
    # PA-Clean-plus3-cap-hw-consv2 with repcap8 (FIXED regex) + v131k + sp122
    # + eng5g. Mirrors the existing nfc_clean_multi_plus3_capped_hybrid_window_tuned_consv2
    # baseline (consv2 ratios, default hw gm=64k, NO tailcuts) but with
    # the repcap8 pretokenizer, 122 specials, vocab=131_072, and the +5GB
    # English data quota.
    "pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_v131k_sp124_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 64_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 131_072,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_v131k_sp124_eng5g",
    },
    # Previous A8 v131k sp120 (buggy regex, retained for comparison; do NOT delete the
    # existing trained artifact at nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp120_eng5g/).
    "pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp120_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus3_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 131_072,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_tailcuts_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp120_eng5g",
    },
    # ---- SHORTCOMINGS ratio-rebalance candidates (2026-06-16) ----------------
    # 7 European-prioritized rebalances of the consv2 (mul) base, designed to
    # close the FLORES head-language compression gap vs Apertus (French/Arabic/
    # Spanish/Portuguese/Russian/English). Each clones the
    # consv2_v131k_sp124_eng5g variant (vocab 131072, sp124, eng5g) and differ
    # only in grouped_config (per-family ratios), global_merges, and
    # output_dir_name. Run at BOTH global-merge settings: gm=64k (no gm tag) and
    # gm=90k (_gm90k tag) -> 14 variants total.
    # The semitic (Arabic) ratio is floored at the consv2 value 1.5133 in every
    # config -- never decreased (per user constraint).
    **{
        f"pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_{slug}{gmtag}_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus3_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{slug}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_{slug}{gmtag}_v131k_sp124_eng5g",
        }
        for slug in ("eurofloor", "tailcap", "sinofund", "flatten", "reparam", "balanced", "gapprop")
        for (gm, gmtag) in ((64_000, ""), (90_000, "_gm90k"))
    },
    # ---- plus2 sibling of the rebalance ablation (2026-06-17) -----------------
    # Exact same 14 jobs as the plus3 block above, but with the
    # clean_multi_plus2_repcap8_capped pretok (plus2 base + the same repcap8
    # guard arm + capped). Everything else identical: same consv2 rebalance
    # configs (ratios), gm 64k/90k, vocab 131072, sp124, eng5g. For the
    # plus2-vs-plus3 pretokenizer comparison.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{slug}{gmtag}_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{slug}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{slug}{gmtag}_v131k_sp124_eng5g",
        }
        for slug in ("eurofloor", "tailcap", "sinofund", "flatten", "reparam", "balanced", "gapprop")
        for (gm, gmtag) in ((64_000, ""), (90_000, "_gm90k"))
    },
    # ---- eurosino European-push sweep (2026-06-17, plus2/gm80k) --------------
    # Funding recouped from SE/East-Asian families (sinotibetan=Mandarin,
    # taikadai=Thai/Lao, austroasiatic=Vietnamese/Khmer); Hindi/Bengali/Tamil
    # (indoaryan/dravidian) and isolates preserved at consv2. Escalating
    # romance/germanic to push European toward Apertus v1. plus2 pretok, gm=80k.
    # Arabic (semitic) floored at the consv2 value. (The plus3/gm90k version,
    # jobs 2553237-2553239, was cancelled and superseded by this.)
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{slug}_gm80k_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": 80_000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{slug}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{slug}_gm80k_v131k_sp124_eng5g",
        }
        for slug in ("eurosinomod", "eurosinostr", "eurosinomax")
    },
    # ---- European-DATA boost sweep (2026-06-17) ------------------------------
    # European families re-sourced from fineweb-2 quality_10-filterrobots at
    # 6GB/family (romance/germanic/slavic/baltic/celtic/uralic), raising European
    # weight in the data-weighted (ratio-blind) global phase -- the lever the
    # diagnosis identified (per-family ratios are near-inert; data + gm are not).
    # gm sweep {90k,100k,110k} sets how much the now-European-heavy frequency
    # phase captures. plus2 pretok, consv2 ratios. Heavy European tilt (~82% of
    # pool) -- expect large European gain AND a large long-tail cost.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_gm{gmk}k_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_gm{gmk}k_v131k_sp124_eng5g",
        }
        for (gm, gmk) in ((90_000, 90), (100_000, 100), (110_000, 110))
    },
    # ---- Moderate European-DATA boost sweep (2026-06-17) ---------------------
    # Same as the 6GB eudata sweep but at 3GB and 4GB per European family
    # (~5x and ~7x baseline French data, vs ~11x at 6GB) -- brackets the
    # European-data magnitude x gm surface. Configs ..._consv2_eudata{3,4}_eng5g.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata{tag}_gm{gmk}k_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata{tag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata{tag}_gm{gmk}k_v131k_sp124_eng5g",
        }
        for tag in ("3", "4")
        for (gm, gmk) in ((90_000, 90), (100_000, 100), (110_000, 110))
    },
    # ---- eudata6_gm90k RETRAIN (2026-06-18): fixed repcap8 regex (the guard now
    # excludes digits, \p{N}, so long number runs aren't force-split at 8) + a
    # BOS/EOS template post-processor (adds <s> ... </s> only when
    # add_special_tokens=True). Sibling dir; the pre-fix run is left in place.
    "pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_gm90k_v131k_sp124_eng5g_bospost": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 90_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 131_072,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "add_bos_eos_post_processor": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_gm90k_v131k_sp124_eng5g_bospost",
    },
    # ---- 200k-vocab eudata rebalancing ablation (2026-06-18) -----------------
    # vocab 200k on the eudata data source (EU 6GB from bulk fineweb-2), gm
    # {100k,120k}. 4 configs spanning least->most EU/high-resource-rebalanced:
    #   eudata_mling : non-EU+semitic ratios x1.3 (parity counterweight), full data
    #   eudata       : reference (consv2 ratios, non-EU at natural size)
    #   eudata_eucut : non-EU-cuttable data x0.5
    #   eudata_eumax : non-EU data x0.25 + EU ratios x1.4 (semitic never cut)
    # No post-processor (ablation; added to the chosen winner later).
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": cfgpath,
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (tag, cfgpath) in (
            ("eudata_mling", "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_mling_eng5g.json"),
            ("eudata",       "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eng5g.json"),
            ("eudata_eucut", "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eucut_eng5g.json"),
            ("eudata_eumax", "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eumax_eng5g.json"),
        )
        for (gm, gmk) in ((100_000, 100), (120_000, 120))
    },
    # ---- 131k EU-via-Sinotibetan ablation (2026-06-18) -----------------------
    # Improve EU without sacrificing English, cost on Sinotibetan. European data
    # kept at 6GB (eudata); Sinotibetan data kept at 0.75GB. Ablate three knobs:
    #   sinotibetan ratio {1.2, 1.5, 1.8}  (cut from the consv2 value 2.224)
    #   tailcuts {off, on}  (on = baltic/celtic/uralic/mande/nigercongo_other+
    #                        voltaniger ratio -> 1.0; tag suffix "tc")
    #   gm {90k, 110k}
    # = 6 configs x 2 gm = 12 runs. English at eng5g throughout.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v131k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 131_072,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{tag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v131k_sp124_eng5g",
        }
        for tag in ("eudata_sino12", "eudata_sino15", "eudata_sino18",
                    "eudata_sino12tc", "eudata_sino15tc", "eudata_sino18tc")
        for (gm, gmk) in ((90_000, 90), (110_000, 110))
    },
    # ---- 200k EU + Sinotibetan-as-sole-funder (2026-06-18) -------------------
    # Like eudata_eumax (EU ratios x1.4) but the non-EU cut is concentrated ONLY
    # on Sinotibetan (ratio 1.2, data x0.25); all other non-EU kept at consv2 so
    # large-presence families (Hindi/Indic/...) capture the freed merges, and
    # semitic/Arabic ratio x1.3. Optimizes EU/English, deprioritizes Sinotibetan.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_eusino_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eudata_eusino_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_eusino_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (gm, gmk) in ((100_000, 100), (120_000, 120))
    },
    # ---- English-data-boost recipes (2026-06-18) ----------------------------
    # English boosted to ~21 GB (full FineWeb-1 sample) so English/EU5 stays high
    # in the data-weighted global phase (the lever that actually moves English).
    #   engfull_eu3   : 131k, EU5 ~15GB (eudata3) -> English-PRESERVING
    #   engfull_eu6   : 131k, EU5 ~30GB (eudata6) -> EU-MAXIMIZING
    #   engfull_eusino: 200k, eusino (EU x1.4 + sino 1.2 + Hindi kept) + English boost
    # All: Sinotibetan ratio cut, semitic ratio 2.0 + Arabic data 2GB.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": vocab,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{tag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g",
        }
        for (tag, gm, gmk, vocab, vk) in (
            ("engfull_eu3", 110_000, 110, 131_072, "131k"),
            ("engfull_eu6", 110_000, 110, 131_072, "131k"),
            ("engfull_eusino", 120_000, 120, 200_000, "200k"),
        )
    },
    # ---- follow-ups (2026-06-19) --------------------------------------------
    # engfull_eu6_hiar: 131k, indoaryan 3.2 + semitic 2.5 (parity-only Hindi/Arabic
    #   recovery; English/EU untouched).
    # eusino_v2{a,b,c}: 200k engfull_eusino routes to improve English+Zh+Arabic at
    #   a bit of EU's expense -- restore sino data to 0.75 (Zh), semitic 2.5 (Arabic),
    #   trim EU data (English). a: sino r1.2/EU4GB; b: sino r1.8/EU4GB; c: sino r1.5/EU3GB.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": vocab,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{tag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g",
        }
        for (tag, gm, gmk, vocab, vk) in (
            ("engfull_eu6_hiar", 110_000, 110, 131_072, "131k"),
            ("eusino_v2a", 120_000, 120, 200_000, "200k"),
            ("eusino_v2b", 120_000, 120, 200_000, "200k"),
            ("eusino_v2c", 120_000, 120, 200_000, "200k"),
        )
    },
    # ---- (2026-06-19) data-share test + tail-recoup --------------------------
    # engfull_eu6_hindata: 131k, indoaryan data 0.8->4GB (test Hindi moves with data).
    # eusino_v2c_tail: 200k eusino_v2c + tail ratios floored 2.5, gm120->100 (more
    #   parity budget) -- recoup merges for the FLORES tail (improve Gini).
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gm,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": vocab,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{tag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{tag}_gm{gmk}k_v{vk}_sp124_eng5g",
        }
        for (tag, gm, gmk, vocab, vk) in (
            ("engfull_eu6_hindata", 110_000, 110, 131_072, "131k"),
            ("eusino_v2c_tail", 100_000, 100, 200_000, "200k"),
            ("engfull_eu6_hindata2", 110_000, 110, 131_072, "131k"),  # indoaryan 2GB (vs 4GB)
            # tail recoup at higher gm (keep EU dense); small Hindi boost on eu3
            ("eusino_v2c_tail", 110_000, 110, 200_000, "200k"),
            ("eusino_v2c_tail", 120_000, 120, 200_000, "200k"),
            ("engfull_eu3_hindata", 110_000, 110, 131_072, "131k"),
            # Fr/De-targeted data boost (fra_Latn/deu_Latn share up) + moderate sino cut
            ("frde1", 110_000, 110, 131_072, "131k"),
            ("frde2", 110_000, 110, 131_072, "131k"),
        )
    },
    # ---- (2026-06-21) eusino_v2c reallocation ablation ----------------------
    # Relative to eusino_v2c (200k, gm120): deprioritize Arabic (semitic data
    # and/or ratio) and/or boost EU data (romance/germanic/slavic), graduated
    # mild -> extreme (ar 1GB/2.0 + EU 5GB = ar_d10r20_eu5). Plus gm130
    # (head-favoring), swiss (EU gain concentrated on Fr/De/It), and kor (add
    # bulk Korean+Greek to the singletons family to help Korean and relatively
    # deprioritize Japanese). Mandarin (sinotibetan) untouched. No post-processor.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (vtag, gmk, ctag) in (
            ("eusino_v2c_ar_d15",        120, "eusino_v2c_ar_d15"),
            ("eusino_v2c_ar_r22",        120, "eusino_v2c_ar_r22"),
            ("eusino_v2c_ar_d15r22",     120, "eusino_v2c_ar_d15r22"),
            ("eusino_v2c_ar_d10r20",     120, "eusino_v2c_ar_d10r20"),
            ("eusino_v2c_eu4",           120, "eusino_v2c_eu4"),
            ("eusino_v2c_eu5",           120, "eusino_v2c_eu5"),
            ("eusino_v2c_ar_d15_eu4",    120, "eusino_v2c_ar_d15_eu4"),
            ("eusino_v2c_ar_d15r22_eu4", 120, "eusino_v2c_ar_d15r22_eu4"),
            ("eusino_v2c_ar_d10r20_eu5", 120, "eusino_v2c_ar_d10r20_eu5"),
            ("eusino_v2c",               130, "eusino_v2c"),
            ("eusino_v2c_swiss",         120, "eusino_v2c_swiss"),
            ("eusino_v2c_kor",           120, "eusino_v2c_kor"),
        )
    },
    # ---- (2026-06-22) eusino_v2c reallocation FOLLOW-UPS --------------------
    # First round showed: Arabic cuts are a cross-script dead end for EU (cost
    # Arabic, no EU gain); EU data is the EU lever; gm130 helps English+Mandarin;
    # kor fixed Korean but cratered Japanese; EU boosts regress the tail. So:
    # Arabic kept intact throughout. euw = EU on rom/ger only (Swiss core). eu6sw
    # pushes rom/ger to 6GB. kor_mild = gentler Korean boost. euw_kormild and
    # euw_tailfloor (floor 6 tiny families to 2.0 to protect/grow the tail) are
    # integrated candidates, each also at gm130.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (vtag, gmk, ctag) in (
            ("eusino_v2c_euw",            120, "eusino_v2c_euw"),
            ("eusino_v2c_euw",            130, "eusino_v2c_euw"),
            ("eusino_v2c_eu6sw",          120, "eusino_v2c_eu6sw"),
            ("eusino_v2c_eu6sw",          130, "eusino_v2c_eu6sw"),
            ("eusino_v2c_kor_mild",       120, "eusino_v2c_kor_mild"),
            ("eusino_v2c_euw_kormild",    130, "eusino_v2c_euw_kormild"),
            ("eusino_v2c_euw_tailfloor",  120, "eusino_v2c_euw_tailfloor"),
            ("eusino_v2c_euw_tailfloor",  130, "eusino_v2c_euw_tailfloor"),
        )
    },
    # ---- (2026-06-22) reallocation ROUND 3 ---------------------------------
    # Round-2 showed: EU data scales (rom/ger 6GB = eu6sw gets De/It/Es below
    # Apertus, Fr +3.1); gm130 holds English+Mandarin; kor_mild fixes Korean
    # (Japanese ~parity); the aggressive tail floor cratered Hindi/Indic (+18/+21%)
    # because data-rich austronesian + a 2.0 floor starved the parity phase.
    # Round 3 lands the integrated candidate eu6sw+kor_mild at gm120/gm130, tests a
    # GENTLE tail floor (only the 5 tiniest families to 1.6, austronesian excluded),
    # and eu6sw at gm110 (tail relief via a larger parity phase, no floor).
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (vtag, gmk, ctag) in (
            ("eusino_v2c_eu6sw_kormild",            120, "eusino_v2c_eu6sw_kormild"),
            ("eusino_v2c_eu6sw_kormild",            130, "eusino_v2c_eu6sw_kormild"),
            ("eusino_v2c_eu6sw_kormild_tailgentle", 120, "eusino_v2c_eu6sw_kormild_tailgentle"),
            ("eusino_v2c_eu6sw_kormild_tailgentle", 130, "eusino_v2c_eu6sw_kormild_tailgentle"),
            ("eusino_v2c_eu6sw",                    110, "eusino_v2c_eu6sw"),
            # (round 3b) tail-favoring end of the integrated candidate: EU + Korean
            # fix at gm110 (larger parity phase grows Indic/Hindi/tail; cost = Mandarin/English).
            ("eusino_v2c_eu6sw_kormild",            110, "eusino_v2c_eu6sw_kormild"),
        )
    },
    # ---- (2026-06-22) round 4: French-ward EU shift on eu6sw_kormild ---------
    # Conservative refinement of eu6sw_kormild: raise French file share
    # (fra 50->80, one variant 90), trim the over-compressed EU (Italian 35,
    # Romanian 14, Danish/Swedish/Dutch/Polish 35) to fund French/German, and
    # split Mandarin into its own group cut to 0.11GB while the Tibetan/Burmese/
    # Wu/Cantonese sino tail keeps its full per-file share. tail12 = the 5 tiniest
    # families x1.2 (slight). gm120 included for the tail variant (more parity room)
    # and as the front-runner for the no-tail shift.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (vtag, gmk, ctag) in (
            ("eusino_v2c_frde",        130, "eusino_v2c_frde"),
            ("eusino_v2c_frde",        120, "eusino_v2c_frde"),
            ("eusino_v2c_frde_tail12", 130, "eusino_v2c_frde_tail12"),
            ("eusino_v2c_frde_tail12", 120, "eusino_v2c_frde_tail12"),
            ("eusino_v2c_frde_fr90",   130, "eusino_v2c_frde_fr90"),
        )
    },
    # ---- (2026-06-22) round 5: dial Korean back, restore Japanese+Chinese ----
    # Round-4 (still running) over-boosted Korean (kor_mild ~457MB) and cut
    # Mandarin, flipping Korean denser than Mandarin. Here Korean is dialed to
    # ~120MB (original 13 singletons + a 23MB Korean add), Japanese and Chinese
    # are back at original amounts (no kor_mild, no cmn truncation; sinotibetan
    # untouched). French shift (frde) and eu6sw EU kept. Distinct names so the
    # round-4 jobs are unaffected.
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": 200_000,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_v200k_sp124_eng5g",
        }
        for (vtag, gmk, ctag) in (
            ("eusino_v2c_frde_kr120",        130, "eusino_v2c_frde_kr120"),
            ("eusino_v2c_frde_kr120",        120, "eusino_v2c_frde_kr120"),
            ("eusino_v2c_frde_kr120_tail12", 130, "eusino_v2c_frde_kr120_tail12"),
            ("eusino_v2c_frde_kr120_tail12", 120, "eusino_v2c_frde_kr120_tail12"),
            ("eusino_v2c_frde_kr120_fr90",   130, "eusino_v2c_frde_kr120_fr90"),
        )
    },
    # ---- (2026-06-22) r5b: eu6sw_gm130 + French ADDED (no EU trims) + Korean ~120MB
    # French raised by adding romance quota (7GB, fra 80) so the other EU languages
    # keep their eu6sw amounts (vs frde which funded French by trimming them).
    # Korean ~120MB; Japanese/Chinese/sinotibetan at original. gm130 only.
    "pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_fradd_kr120_gm130k_v200k_sp124_eng5g": {
        "trainer": "parity-bpe",
        "pretok": "clean_multi_plus2_repcap8_capped_nfc",
        "variant": "window",
        "global_merges": 130_000,
        "window_size": 100,
        "alpha": 2.0,
        "signal": "ratios",
        "vocab_size": 200_000,
        "grouped_config": "configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eusino_v2c_fradd_kr120_eng5g.json",
        "special_tokens_override": APERTUS_SPECIAL_TOKENS,
        "decoder_add_prefix_space": True,
        "output_group": "tokenizer-lm-toks",
        "output_dir_name": "nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_fradd_kr120_gm130k_v200k_sp124_eng5g",
    },
    # ---- (2026-06-22) production-recipe variants: vocab 200064 + code boosts ----
    # Exact frde_kr120 recipe (gm130). v200064: same config, vocab 200064 (128-aligned).
    # code2g: code quota raised to 2.10GB (full-ish starcoder_sample). code3gpy: new 3GB
    # code sample, ~80% python (40 python + 10 other-lang files from full starcoderdata).
    **{
        f"pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_{vk}_sp124_eng5g": {
            "trainer": "parity-bpe",
            "pretok": "clean_multi_plus2_repcap8_capped_nfc",
            "variant": "window",
            "global_merges": gmk * 1000,
            "window_size": 100,
            "alpha": 2.0,
            "signal": "ratios",
            "vocab_size": vocab,
            "grouped_config": f"configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_{ctag}_eng5g.json",
            "special_tokens_override": APERTUS_SPECIAL_TOKENS,
            "decoder_add_prefix_space": True,
            "output_group": "tokenizer-lm-toks",
            "output_dir_name": f"nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_{vtag}_gm{gmk}k_{vk}_sp124_eng5g",
        }
        for (vtag, gmk, vocab, vk, ctag) in (
            ("eusino_v2c_frde_kr120",         130, 200_064, "v200064", "eusino_v2c_frde_kr120"),
            ("eusino_v2c_frde_kr120_code2g",  130, 200_000, "v200k",   "eusino_v2c_frde_kr120_code2g"),
            ("eusino_v2c_frde_kr120_code3gpy",130, 200_000, "v200k",   "eusino_v2c_frde_kr120_code3gpy"),
        )
    },
}

# -------- tokenizer assembly -------------------------------------------------


def build_pretokenizer(pretok: str) -> pre_tokenizers.PreTokenizer:
    # The "_nfc" suffix on any pretok name is a hint to also attach an NFC
    # normalizer (handled in build_tokenizer) — it does not change the
    # pretokenizer itself.
    if pretok == "whitespace":
        return pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("gpt4o", "gpt4o_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_GPT4O),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi", "clean_multi_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_capped", "clean_multi_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus", "clean_multi_plus_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus_capped", "clean_multi_plus_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus2_capped", "clean_multi_plus2_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS2_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus3_capped", "clean_multi_plus3_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS3_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus3_repcap8_capped", "clean_multi_plus3_repcap8_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("clean_multi_plus2_repcap8_capped", "clean_multi_plus2_repcap8_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_CLEAN_MULTI_PLUS2_REPCAP8_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    if pretok in ("apertus_capped", "apertus_capped_nfc"):
        return pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(REGEX_APERTUS_CAPPED),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    raise ValueError(f"unknown pretok: {pretok}")


def build_tokenizer(cfg: dict) -> Tokenizer:
    tok = Tokenizer(BPE())
    if cfg.get("pretok", "").endswith("_nfc") or cfg.get("normalizer") == "nfc":
        tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = build_pretokenizer(cfg["pretok"])
    tok.decoder = decoders.ByteLevel(
        add_prefix_space=cfg.get("decoder_add_prefix_space", False),
        trim_offsets=True,
        use_regex=True,
    )
    return tok


# -------- reproducibility metadata -------------------------------------------


def get_git_info() -> dict:
    info = {"commit": None, "branch": None, "dirty": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        pass
    try:
        info["branch"] = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
            capture_output=True, text=True,
        )
        info["dirty"] = (r.returncode != 0)
    except Exception:
        pass
    return info


def get_wheel_info() -> dict:
    import importlib.metadata as md
    info = {"version": None, "wheel_path": None, "wheel_sha256": None}
    try:
        info["version"] = md.version("tokenizers")
    except Exception:
        return info
    try:
        dist = md.distribution("tokenizers")
        so_files = [f for f in (dist.files or []) if str(f).endswith((".so", ".pyd"))]
        if so_files:
            abs_path = dist.locate_file(so_files[0])
            info["wheel_path"] = str(abs_path)
            h = hashlib.sha256()
            with open(abs_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            info["wheel_sha256"] = h.hexdigest()
    except Exception:
        pass
    return info


# -------- grouped-config training path --------------------------------------


def train_grouped_variant(args, cfg, out_dir, smoke, quick,
                          tokenizer_out, manifest_out):
    """Train from a grouped config file (per-language-group explicit parquet
    lists). Dispatches on cfg["trainer"]: "parity-bpe" uses ParityBpeTrainer
    with per-group ratios; "bpe" uses a standard BpeTrainer over the
    concatenated groups (no ratios). Independent of the 30-lang manifest
    pipeline. Used by replication runs like ``pa_bpe_nfc_gpt4_fineweb2full`` and
    the standard-BPE baselines ``bpe_nfc_*``.
    """
    group_config_path = REPO_ROOT / cfg["grouped_config"]
    group_cfg = json.loads(group_config_path.read_text(encoding="utf-8"))

    # Hyperparameters: honor smoke/quick overrides if set, otherwise full.
    # A variant can override the default VOCAB_SIZE by setting "vocab_size"
    # in its cfg dict (e.g. 130_900 for an Apertus-comparable budget).
    if smoke:
        vocab_size = SMOKE_VOCAB_SIZE
        per_lang_cap = SMOKE_FLOOR_BYTES     # tiny per-group in smoke
    elif quick:
        vocab_size = QUICK_VOCAB_SIZE
        per_lang_cap = None                  # real corpus, smaller vocab
    else:
        vocab_size = cfg.get("vocab_size", VOCAB_SIZE)
        per_lang_cap = None                  # full corpus, full vocab

    special_tokens = cfg.get("special_tokens_override", SPECIAL_TOKENS)

    tok = build_tokenizer(cfg)

    # Build one ListedFileCorpus per language group, in config order.
    # Per-group quota resolution: smoke/quick override; otherwise honor the
    # config's per-group quota_bytes (if present); otherwise read everything.
    train_corpora = []
    ratios = []
    ordered_names = []
    for lang in group_cfg["languages"]:
        name = lang["name"]
        files = lang["input"]
        ratio = lang.get("ratio", 1.0)
        if per_lang_cap is not None:
            this_quota = per_lang_cap
        else:
            this_quota = lang.get("quota_bytes")  # None if absent
        corpus = ListedFileCorpus(
            file_paths=files,
            group_name=name,
            quota_bytes=this_quota,
        )
        train_corpora.append(corpus)
        ratios.append(ratio)
        ordered_names.append(name)

    total_files = sum(len(c.file_paths) for c in train_corpora)
    total_bytes_on_disk = sum(
        sum(Path(p).stat().st_size for p in c.file_paths)
        for c in train_corpora
    )
    print(f"[{args.variant}] grouped-config: {len(train_corpora)} groups, "
          f"{total_files} parquet files, {total_bytes_on_disk/2**30:.2f} GB on disk",
          flush=True)
    print(f"[{args.variant}] vocab={vocab_size} special_tokens={special_tokens}",
          flush=True)
    print(f"[{args.variant}] per-group sizes:")
    for lang, corpus in zip(group_cfg["languages"], train_corpora):
        size = sum(Path(p).stat().st_size for p in corpus.file_paths)
        print(f"    {lang['name']:28s} ratio={lang.get('ratio', 1.0):.4f}  "
              f"n_files={len(corpus.file_paths):4d}  size={size/2**20:8.1f} MiB",
              flush=True)

    start_time = datetime.now()

    if cfg["trainer"] == "bpe":
        # Standard BPE baseline over the same grouped corpora. No parity
        # signal / ratios — all groups are concatenated into one stream and
        # vocab is built by plain frequency. initial_alphabet seeds the full
        # 256-byte ByteLevel alphabet (same fix as the parity path).
        #
        # BpeTrainer fills vocab_size *exactly*, unlike ParityBpeTrainer which
        # under-produces from num_merges. So reserve room for the
        # post_training_special_tokens that get appended after save —
        # otherwise final vocab = vocab_size + n_post overshoots the
        # `got_vs > vocab_size` guard below (the bug that failed SLURM
        # 2282607-2282609). Net effect: final tokenizer == vocab_size exactly.
        n_post = len(cfg.get("post_training_special_tokens") or [])
        bpe_target = vocab_size - n_post
        bpe_kwargs = dict(
            vocab_size=bpe_target,
            min_frequency=MIN_FREQUENCY,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        if special_tokens:
            bpe_kwargs["special_tokens"] = special_tokens
        bpe_trainer = BpeTrainer(**bpe_kwargs)
        combined = itertools.chain.from_iterable(train_corpora)
        tok.train_from_iterator(combined, trainer=bpe_trainer)
        print(f"[{args.variant}] BPE vocab_size target={bpe_target} "
              f"(= {vocab_size} budget − {n_post} post-training specials)",
              flush=True)

    elif cfg["trainer"] == "parity-bpe":
        trainer_kwargs = dict(
            num_merges=vocab_size,
            variant=cfg["variant"],
            min_frequency=MIN_FREQUENCY,
            global_merges=cfg["global_merges"],
            total_symbols=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        if special_tokens:
            trainer_kwargs["special_tokens"] = special_tokens
        if cfg["variant"] == "window":
            trainer_kwargs["window_size"] = cfg.get("window_size", 100)
            trainer_kwargs["alpha"] = cfg.get("alpha", 2.0)
        pa_trainer = ParityBpeTrainer(**trainer_kwargs)

        if cfg["signal"] != "ratios":
            raise ValueError(
                f"grouped-config path only supports signal='ratios', got {cfg['signal']}"
            )

        pa_trainer.train_from_iterator(
            tok,
            train_iterators=list(train_corpora),
            ratio=ratios,
        )

    else:
        raise ValueError(f"unknown trainer: {cfg['trainer']}")

    end_time = datetime.now()
    tok.save(str(tokenizer_out))
    print(f"[{args.variant}] saved {tokenizer_out}", flush=True)

    post_specials = cfg.get("post_training_special_tokens")
    if post_specials:
        added = tok.add_special_tokens(post_specials)
        tok.save(str(tokenizer_out))
        print(f"[{args.variant}] appended {added} post-training special tokens "
              f"{post_specials} and re-saved {tokenizer_out}", flush=True)

    # Collect per-group actuals.
    per_group = []
    for lang, corpus in zip(group_cfg["languages"], train_corpora):
        per_group.append({
            "name": lang["name"],
            "ratio": lang.get("ratio", 1.0),
            "n_files": len(corpus.file_paths),
            "actual_bytes": corpus.actual_bytes,
            "actual_lines": corpus.actual_lines,
            "n_shards_touched": len(corpus.shard_log),
            "shard_log": [s.to_dict() for s in corpus.shard_log],
        })

    run_manifest = {
        "variant": args.variant,
        "variant_config": cfg,
        "smoke_test": smoke,
        "quick_test": quick,
        "grouped_config_path": str(group_config_path),
        "vocab_size": vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": special_tokens,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cli_args": sys.argv,
        "git": get_git_info(),
        "tokenizers_wheel": get_wheel_info(),
        "pyarrow_version": pyarrow.__version__,
        "python_version": sys.version,
        "env_snapshot": {
            k: os.environ.get(k)
            for k in ("LANG", "LC_ALL", "RAYON_NUM_THREADS", "SLURM_JOB_ID",
                      "SLURM_JOB_NODELIST", "SLURMD_NODENAME")
        },
        "per_group": per_group,
    }
    manifest_out.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{args.variant}] wrote {manifest_out}", flush=True)

    reloaded = Tokenizer.from_file(str(tokenizer_out))
    got_vs = reloaded.get_vocab_size()
    if got_vs > vocab_size:
        raise SystemExit(
            f"[{args.variant}] vocab size overshoot: got {got_vs}, expected ≤ {vocab_size}"
        )
    if got_vs < vocab_size:
        deficit_pct = (vocab_size - got_vs) / vocab_size * 100
        print(f"[{args.variant}] NOTE: vocab_size={got_vs} "
              f"(requested {vocab_size}, deficit {deficit_pct:.1f}%)", flush=True)
    print(f"[{args.variant}] done in {run_manifest['elapsed_seconds']:.1f}s",
          flush=True)


# -------- main ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    parser.add_argument("--output-dir", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true",
                      help="tiny quotas + vocab=8000 for local smoke testing")
    mode.add_argument("--quick", action="store_true",
                      help="full corpus, vocab=8000 — sanity-check memory + "
                           "environment on a real compute node before a 12 h run")
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]
    smoke = args.smoke_test
    quick = args.quick

    if smoke:
        total_bytes = SMOKE_TOTAL_BYTES
        floor_bytes = SMOKE_FLOOR_BYTES
        vocab_size = SMOKE_VOCAB_SIZE
    elif quick:
        total_bytes = TOTAL_BYTES
        floor_bytes = FLOOR_BYTES
        vocab_size = QUICK_VOCAB_SIZE
    else:
        total_bytes = TOTAL_BYTES
        floor_bytes = FLOOR_BYTES
        vocab_size = VOCAB_SIZE

    # In smoke/quick mode, clamp global_merges so the hybrid variant still makes
    # sense (global_merges=64000 > vocab=8000 would be nonsensical).
    if (smoke or quick) and "global_merges" in cfg and cfg["global_merges"] >= vocab_size:
        cfg = dict(cfg)
        cfg["global_merges"] = vocab_size // 2

    # Output layout:
    #   experiments/paper/fineweb2_training/<short>/            main 30-lang variants
    #   experiments/paper/fineweb2_training/ablations/<short>/  devsize / window-size ablations
    #   experiments/archive/<short>/                            grouped-config replications
    # <short> drops the redundant "pa_bpe_" prefix carried on most variant keys.
    ABLATION_VARIANTS = {
        "pa_bpe_dev_devsize100", "pa_bpe_dev_devsize300",
        "pa_bpe_hybrid_devsize100", "pa_bpe_hybrid_devsize300",
        "pa_bpe_window_ws50", "pa_bpe_window_ws150", "pa_bpe_window_ws200",
    }

    def _short(name: str) -> str:
        return name[len("pa_bpe_"):] if name.startswith("pa_bpe_") else name

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = cfg.get("output_dir_name") or _short(args.variant)
        if smoke:
            suffix = f"{base}_smoke"
        elif quick:
            suffix = f"{base}_quick"
        else:
            suffix = base

        # A variant can override the default output group. Honored only for
        # full runs — smoke/quick fall through to the normal routing so they
        # don't pollute the consumed-tokenizer dir.
        output_group = cfg.get("output_group") if not (smoke or quick) else None

        if output_group == "tokenizer-lm-toks":
            out_dir = REPO_ROOT / "experiments" / "tokenizer-lm-toks" / suffix
        elif "grouped_config" in cfg:
            out_dir = REPO_ROOT / "experiments" / "archive" / suffix
        elif args.variant in ABLATION_VARIANTS:
            out_dir = REPO_ROOT / "experiments" / "paper" / "fineweb2_training" / "ablations" / suffix
        else:
            out_dir = REPO_ROOT / "experiments" / "paper" / "fineweb2_training" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_out = out_dir / "tokenizer.json"
    manifest_out = out_dir / "run_manifest.json"

    # Grouped-config variants (e.g. pa_bpe_nfc_gpt4_fineweb2full) use a completely
    # different data pipeline: ListedFileCorpus over parquets hand-picked in
    # a config JSON, no mC4 sampling, no 30-lang manifest.
    if "grouped_config" in cfg:
        return train_grouped_variant(args, cfg, out_dir, smoke, quick,
                                     tokenizer_out, manifest_out)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mc4 = json.loads(MC4_WEIGHTS_PATH.read_text(encoding="utf-8"))
    flores = json.loads(FLORES_RATIOS_PATH.read_text(encoding="utf-8"))

    ordered = manifest["languages"]  # authoritative order, defines lang idx

    tok = build_tokenizer(cfg)

    # Per-language training corpora (positional order = language index).
    train_corpora = []
    for lang in ordered:
        code = lang["code"]
        share = mc4["languages"][code]["share_by_tokens"]
        quota = compute_quota_bytes(share, total_bytes, floor_bytes)
        train_corpora.append(
            LanguageTrainCorpus(
                train_root=Path(lang["train_root"]),
                lang_code=code,
                quota_bytes=quota,
                seed=GLOBAL_SEED,
            )
        )

    total_quota = sum(c.quota_bytes for c in train_corpora)
    print(f"[{args.variant}] variant config: {cfg}", flush=True)
    print(f"[{args.variant}] vocab={vocab_size} total_quota={total_quota/2**30:.2f} GiB "
          f"(budget={total_bytes/2**30:.2f} GiB, floor={floor_bytes/2**20:.1f} MiB/lang)",
          flush=True)
    print(f"[{args.variant}] per-lang quotas:")
    for lang, corpus in zip(ordered, train_corpora):
        print(f"    {lang['code']:10s} share={mc4['languages'][lang['code']]['share_by_tokens']*100:6.3f}%  "
              f"quota={corpus.quota_bytes/2**20:7.1f} MiB", flush=True)

    start_time = datetime.now()

    # The full 256-byte ByteLevel alphabet is seeded as initial_alphabet so
    # bytes that never appear in training (e.g. byte 0x0A / `Ċ`) still have a
    # base token in vocab; otherwise encoding text with such bytes produces UNK.
    bytelevel_alphabet = pre_tokenizers.ByteLevel.alphabet()

    # ---- dispatch ---------------------------------------------------------
    if cfg["trainer"] == "bpe":
        bpe_trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=MIN_FREQUENCY,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=bytelevel_alphabet,
            show_progress=True,
        )
        combined = itertools.chain.from_iterable(train_corpora)
        tok.train_from_iterator(combined, trainer=bpe_trainer)

    elif cfg["trainer"] == "parity-bpe":
        trainer_kwargs = dict(
            num_merges=vocab_size,
            variant=cfg["variant"],
            min_frequency=MIN_FREQUENCY,
            global_merges=cfg["global_merges"],
            global_merges_at_end=cfg.get("global_merges_at_end", False),
            total_symbols=True,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=bytelevel_alphabet,
            show_progress=True,
        )
        if cfg["variant"] == "window":
            trainer_kwargs["window_size"] = cfg["window_size"]
            trainer_kwargs["alpha"] = cfg["alpha"]
        pa_trainer = ParityBpeTrainer(**trainer_kwargs)

        if cfg["signal"] == "dev":
            dev_max_lines = cfg.get("dev_max_lines")
            dev_corpora = [
                LanguageDevCorpus(Path(lang["flores_dev_path"]),
                                  max_lines=dev_max_lines)
                for lang in ordered
            ]
            pa_trainer.train_from_iterator(
                tok,
                train_iterators=list(train_corpora),
                dev_iterators=dev_corpora,
            )
        elif cfg["signal"] == "ratios":
            ratios = [flores["languages"][lang["code"]]["ratio"] for lang in ordered]
            pa_trainer.train_from_iterator(
                tok,
                train_iterators=list(train_corpora),
                ratio=ratios,
            )
        else:
            raise ValueError(f"unknown signal: {cfg['signal']}")

    else:
        raise ValueError(f"unknown trainer: {cfg['trainer']}")

    end_time = datetime.now()

    if cfg.get("add_bos_eos_post_processor"):
        bos, eos = "<s>", "</s>"
        bid, eid = tok.token_to_id(bos), tok.token_to_id(eos)
        if bid is None or eid is None:
            raise RuntimeError(
                f"add_bos_eos_post_processor: {bos!r}/{eos!r} missing from vocab "
                f"(bos={bid}, eos={eid})"
            )
        # Fires only when add_special_tokens=True: BOS ... EOS around the
        # sequence (single); for pairs, segment B uses type_id 1 (the standard
        # segment convention, matching Apertus's own pair template).
        tok.post_processor = TemplateProcessing(
            single=f"{bos} $A {eos}",
            pair=f"{bos} $A {eos} {bos}:1 $B:1 {eos}:1",
            special_tokens=[(bos, bid), (eos, eid)],
        )
        print(f"[{args.variant}] post_processor: '{bos} $A {eos}' "
              f"(BOS={bid}, EOS={eid})", flush=True)

    tok.save(str(tokenizer_out))
    print(f"[{args.variant}] saved {tokenizer_out}", flush=True)

    # ---- run manifest -----------------------------------------------------
    per_lang = []
    for lang, corpus in zip(ordered, train_corpora):
        per_lang.append({
            "code": lang["code"],
            "train_root": str(corpus.train_root),
            "quota_bytes": corpus.quota_bytes,
            "actual_bytes": corpus.actual_bytes,
            "actual_lines": corpus.actual_lines,
            "n_shards_touched": len(corpus.shard_log),
            "shard_log": [s.to_dict() for s in corpus.shard_log],
        })

    run_manifest = {
        "variant": args.variant,
        "variant_config": cfg,
        "smoke_test": smoke,
        "quick_test": quick,
        "vocab_size": vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "total_bytes_budget": total_bytes,
        "floor_bytes_per_lang": floor_bytes,
        "global_seed": GLOBAL_SEED,
        "mc4_source": mc4["source"],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cli_args": sys.argv,
        "git": get_git_info(),
        "tokenizers_wheel": get_wheel_info(),
        "pyarrow_version": pyarrow.__version__,
        "python_version": sys.version,
        "env_snapshot": {
            k: os.environ.get(k)
            for k in ("LANG", "LC_ALL", "RAYON_NUM_THREADS", "SLURM_JOB_ID",
                      "SLURM_JOB_NODELIST", "SLURMD_NODENAME")
        },
        "per_language": per_lang,
    }
    manifest_out.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{args.variant}] wrote {manifest_out}", flush=True)

    # ---- sanity encode ----------------------------------------------------
    reloaded = Tokenizer.from_file(str(tokenizer_out))
    got_vs = reloaded.get_vocab_size()
    # Strict upper bound: the trainer should never produce more merges than
    # asked for.  Undershooting is allowed (min_frequency can filter out
    # merges, especially on tiny smoke-test corpora).
    if got_vs > vocab_size:
        raise SystemExit(
            f"[{args.variant}] vocab size overshoot: got {got_vs}, "
            f"expected ≤ {vocab_size}"
        )
    if got_vs < vocab_size:
        deficit_pct = (vocab_size - got_vs) / vocab_size * 100
        print(
            f"[{args.variant}] NOTE: vocab_size={got_vs} "
            f"(requested {vocab_size}, deficit {deficit_pct:.1f}%). "
            "Expected for smoke tests; investigate if this happens at full scale.",
            flush=True,
        )
    sanity = {}
    for lang in ordered:
        with open(lang["flores_dev_path"], encoding="utf-8") as f:
            first_line = next((l.strip() for l in f if l.strip()), "")
        if first_line:
            enc = reloaded.encode(first_line)
            sanity[lang["code"]] = {
                "n_chars": len(first_line),
                "n_tokens": len(enc.tokens),
                "compression_rate": len(first_line) / max(1, len(enc.tokens)),
            }
    (out_dir / "sanity_encode.json").write_text(
        json.dumps(sanity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[{args.variant}] sanity-encoded 30 FLORES lines, "
          f"elapsed={run_manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
