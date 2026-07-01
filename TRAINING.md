# Training and reproduction

How the four `preliminary_*` tokenizers in this repository were trained, with enough detail to reproduce them and verify the result against the deployed files. For what the tokenizers are and how to use them, see [README.md](README.md). For the design rationale (pretokenization, algorithm, special tokens), see [apertus_tokenizer_design.md](apertus_tokenizer_design.md).

The exact recipe artifacts are bundled under [`training/`](training/): the four config files, the training driver, the data loader, the pretokenizer regexes, and the special-token spec. The Rust trainer lives in a separate repository (see Trainer). The training data is our cluster's copies of public datasets, so reproducing off-cluster requires repointing the dataset paths (see Data).

## 1. Reproduce

1. Build the trainer. The parity-aware BPE trainer is implemented in Hugging Face `tokenizers` PR #1974 (https://github.com/huggingface/tokenizers/pull/1974). Check out that PR's branch and build the Python extension in release mode:
   ```bash
   cd bindings/python
   maturin develop --release
   ```
   Use `--release`. `pip install -e .` and a plain `maturin develop` build the debug profile, which is unoptimized and about 10x slower at training, with no error or warning. A debug build of the extension is about 115 MB, a release build about 10 MB. See `bindings/python/README.md` in the tokenizers repository.
2. Set up Python. Use a virtual environment with the release `tokenizers` extension installed (our environment uses `pa_venv`). Put the files from this repository's `training/` directory on that machine.
3. Point the configs at the datasets. The config paths use a `${DATA_ROOT}` placeholder (see Data); set it to where these datasets live in your environment.
4. Train. From the directory holding `train_tokenizer.py`:
   ```bash
   python train_tokenizer.py --variant <KEY>
   ```
   The variant keys are in Per-tokenizer recipes. Each run reads its config, builds the corpus, trains, and writes a `tokenizer.json` with no post-processor.
5. Add the post-processor. The trained file has no BOS/EOS post-processor; add it for deployment (see Special tokens and post-processor).
6. Verify. Compare against the deployed tokenizer (see Verification).

We ran each variant as one SLURM job: `sbatch --wrap "python train_tokenizer.py --variant <KEY>"`, account `infra01`, `--mem=800G`, `--cpus-per-task=64`, and `--time=04:00:00` for the 131k tokenizers or `03:00:00` for the 200k (about 1.7 h on a release build), using the `pa_venv` Python.

## 2. Trainer

The trainer is `ParityBpeTrainer` (in `src/models/bpe/`), implemented in Hugging Face `tokenizers` PR #1974 (https://github.com/huggingface/tokenizers/pull/1974). It is a byte-level BPE trainer with a hybrid global-then-parity merge schedule:

- `window_size=100`, `alpha=2.0`, `total_symbols=True`.
- Global phase: the first `gm` (`global_merges`) merges are chosen by data-weighted pooled frequency across all languages.
- Parity phase: the remaining merges are chosen by ratio-adjusted compression with a sliding window. The per-language window cap is `alpha / number_of_languages`.
- `total_symbols=True` sets the merge target to `vocab_size` minus the current symbol count, so the vocabulary lands at the target size exactly.

The driver `training/train_tokenizer.py` builds the normalizer, pretokenizer, and trainer, then calls `train_from_iterator` over the corpora produced by `training/pa_bpe_iterators.py`.

## 3. Pretokenizer and normalizer

NFC normalization, then a byte-level pretokenizer regex. `preliminary_mul` uses `clean_multi_plus3_repcap8`; the other three use `clean_multi_plus2_repcap8`. The `repcap8` guard caps a standalone run of 8 or more identical characters at 8 and excludes digit runs. The definitions are in `training/pretokenizer_regexes.py`. The rationale is in [apertus_tokenizer_design.md](apertus_tokenizer_design.md).

## 4. Special tokens and post-processor

124 special tokens (the `sp124` set) occupy ids 0 to 123. The full list is `training/apertus_special_tokens.json`: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat, tool, multimodal, reflection, and think tokens; the PII tokens `<iban-pii>`(24), `<email-pii>`(25), `<ip-pii>`(26); and the remaining reserve slots.

Training writes a `tokenizer.json` with no post-processor. For deployment the tokenizers add a `TemplateProcessing` post-processor:

- single: `<s> $A </s>`
- pair: `<s>:0 $A:0 </s>:0 <s>:1 $B:1 </s>:1`

So `add_special_tokens=True` wraps a single sequence as `<s> ... </s>`.

## 5. Data

Each config groups languages into families and reads `quota_bytes` of text per family. The loader `training/pa_bpe_iterators.py` (`ListedFileCorpus`) fills a family's quota round-robin with an equal per-file byte cap, so within a family a language's file count is roughly its share of the data.

The inputs are public datasets, with the multilingual data drawn from a quality-filtered variant of FineWeb-2 (`HuggingFaceFW/fineweb-2`):

- English: a FineWeb English web sample.
- European and other boosted families: a quality-filtered version of FineWeb-2 (`HuggingFaceFW/fineweb-2`), using our own quality and robots filtering. This filtered variant is our preprocessing, not a standard public release; it is not distributed here, so these families cannot be reproduced byte-for-byte without it.
- Baseline per-language families: a per-language sample of FineWeb-2 (`HuggingFaceFW/fineweb-2`).
- Code: StarCoder.
- Evaluation only (not training): FLORES-200 devtest.

The config files under `training/configs/` reference paths under a `${DATA_ROOT}` placeholder; set it to where these datasets live before training. `preliminary_mul_200k`'s singletons family also lists a roughly 23 MB Korean subsample of the FineWeb-2 data.

## 6. Per-tokenizer recipes

Config files are `training/configs/parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_<tag>_eng5g.json`. Pass the whole variant key to `--variant`.

| folder | variant key | tag | gm | vocab | pretok |
|---|---|---|---|---|---|
| `preliminary_enh` | `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_engfull_eu3_gm110k_v131k_sp124_eng5g` | engfull_eu3 | 110000 | 131072 | plus2 |
| `preliminary_euh` | `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_frde2_gm110k_v131k_sp124_eng5g` | frde2 | 110000 | 131072 | plus2 |
| `preliminary_mul` | `pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_reparam_v131k_sp124_eng5g` | reparam | 64000 | 131072 | plus3 |
| `preliminary_mul_200k` | `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_frde_kr120_gm130k_v200064_sp124_eng5g` | eusino_v2c_frde_kr120 | 130000 | 200064 | plus2 |

The exact per-family `quota_bytes`, `ratio`, and file lists are in the bundled config. The data character of each:

- `preliminary_enh` (English-preserving). English 22 GB (the full FineWeb-1 sample); the European families (romance, germanic, slavic, baltic, celtic, uralic) about 3 GB each; plus an Arabic data and ratio fix.
- `preliminary_euh` (EU-dense). English 22 GB; romance and germanic raised to 10 GB each, with French and German given a larger file share (`fra_Latn`, `deu_Latn`); baltic, celtic, and slavic 6 GB; Sinotibetan data cut.
- `preliminary_mul` (most balanced, 131k). English 12.72 GB; slavic 2.77 GB, romance 2.39 GB, germanic 2.03 GB; code 1.47 GB; singletons 1.26 GB. Uses the `consv2` baseline with the `reparam` ratio adjustment and the `plus3` pretokenizer.
- `preliminary_mul_200k` (Fr/De-strong, 200k). English 22 GB; romance and germanic 6 GB, with the French file share raised and Italian, Danish, Swedish, Dutch, and Polish trimmed; baltic, celtic, slavic 3 GB; Korean about 120 MB; Mandarin at its original about 135 MB; Arabic unchanged; code 1.47 GB StarCoder. The vocabulary is 200064 (128-aligned); ids 0 to 199999 are identical to the earlier 200000 build, with 64 tokens appended.

## 7. Verification

Retrain a variant, add the post-processor, then compare the model against the deployed file. Load both `tokenizer.json` files and check that `model.vocab` and `model.merges` match. For `preliminary_mul_200k`, ids 0 to 199999 match the deployed 200064 file and the extra 64 are appended. A match shows the bundled config and driver reproduce the deployed tokenizer. The vocabulary lands at the target size exactly because of `total_symbols=True`.

## 8. Provenance

The full training record (every job, config, and result, including the search that produced these recipes) is in the training repository's `EXPERIMENTS_CHRONOLOGICAL.md`, `EXPERIMENTS_RESULTS.md`, `EXPERIMENTS_PLAN.md`, and `EXPERIMENTAL_SETUP.md`. Those files are not part of this repository.
