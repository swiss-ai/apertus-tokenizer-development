# training/

Recipe artifacts for reproducing the `preliminary_*` tokenizers. See [../TRAINING.md](../TRAINING.md) for the full procedure.

- `configs/` holds the four production config files, one per tokenizer. Each lists the language families, the per-family `quota_bytes` and `ratio`, and the input file lists. The paths are our cluster's copies of public datasets; repoint them as described in ../TRAINING.md.
  - `parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_engfull_eu3_eng5g.json` for `preliminary_enh`
  - `parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_frde2_eng5g.json` for `preliminary_euh`
  - `parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_reparam_eng5g.json` for `preliminary_mul`
  - `parity_aware_config_grouped_fineweb2full_quota_tuned_consv2_eusino_v2c_frde_kr120_eng5g.json` for `preliminary_mul_200k`
- `train_tokenizer.py` is the training driver. Its `VARIANTS` dict maps each variant key to its config, `global_merges`, `vocab_size`, pretokenizer, and special tokens. Run `python train_tokenizer.py --variant <KEY>`.
- `pa_bpe_iterators.py` is the data loader (`ListedFileCorpus`): it fills each family's `quota_bytes` round-robin with an equal per-file cap.
- `pretokenizer_regexes.py` holds the `clean_multi_plus2_repcap8` and `clean_multi_plus3_repcap8` pretokenizer definitions.
- `apertus_special_tokens.json` is the 124 special tokens (`sp124`), ids 0 to 123.

The Rust `ParityBpeTrainer` is not bundled here; it lives in the tokenizers fork (see ../TRAINING.md, Trainer).
