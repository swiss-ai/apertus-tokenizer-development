# Romansh tokenization on RCP and future Clariden reruns

The four `romansh-*.cfg` configurations consume independent prepared populations
of `swiss-ai/apertus-pretrain-romansh` at
`c2f3a672246aedfac6733e49bfba18851eab8b3a`. Preparation lives in
`data-pipeline-pretrain/pipelines/romansh`, based on the current LC reserve stack
(PR #52, including PRs #38 and #37). Tokenization uses this
repository's existing Parquet provenance, grouped dump and strict validation
workflow, based on PR #21. Use `source_key` rather than the upstream `id`, which
can be missing or repeated. Each configuration preserves its complete documents.
Use that pipeline stack at runtime: PR #21 passes the sequence length guard
introduced in PR #38, which the standalone PR #37 writer does not accept.

## Production on RCP

Run inside an allocated CPU pod with 16 CPUs and sufficient memory for the
largest individual record (64 GiB is the initial production allocation).
Mount `mlo-scratch` at `/mloscratch` and at both canonical roots:

```text
/capstor/store/cscs/swissai/infra01/datasets
/capstor/store/cscs/swissai/infra01/datasets_tokenized
```

On the PVC, preparation is under `apertus-pretrain-romansh-v1`, and tokens are
under `apertus-pretrain-romansh-v1_apertus_v2`. The canonical mounts let RCP and
Clariden resolve identical map source roots without editing maps after transfer.
Stage clean committed pipeline/tokenizer checkouts and the exact Git-LFS
`preliminary_mul_200k/tokenizer.json`; retain their full commits, tokenizer SHA-256
and dependency freeze. Put the pipeline's `src` on `PYTHONPATH` and the selected
Python environment on `PATH`. Each prepared population must have its real
`_SUCCESS.json` and exact examples inventory before tokenization.

For each of `monolingual`, `poly-aligned`, `poly-non-aligned`, and `synthetic`,
substitute its configuration and category below (`poly_aligned` and
`poly_non_aligned` use underscores for category names):

```bash
config=tokenization_scripts/configs_apertus_v2_rcp/romansh-monolingual.cfg
bash tokenization_scripts/tokenize_script.sh "$config" --prepare-only
source "$config"
category=monolingual
SLURM_JOB_ID=romansh-monolingual-rcp bash tokenization_scripts/tokenize.sh \
  "$config" "$DATASET_OUTPUT_FOLDER_NAME/$category/dump-0" \
  "$PATH_TO_OUTPUT_FOLDER/logs/$DATASET_NAME/dump-0" \
  "$PATH_TO_PREPROCESSING_METADATA/dumps/$category/paths_file_0.txt" \
  "$PATH_TO_PREPROCESSING_METADATA/completed-dumps/$category"
bash tokenization_scripts/validate_tokenization.sh "$config"
```

There is one dump per configuration, with four file tasks/workers and four
tokenizer threads per worker. Do not overlap workers on the same dump. Validation
compares all expected Parquet row coordinates, token/index/map hashes, tokenizer
identity and BOS/EOS, then publishes `_SUCCESS.json` last. The sequence metadata guard is disabled because preparation does not
precompute `sequence_tokens`. Documents remain complete; reserve selection
runs afterward. Use the existing LC reserve recipe separately before choosing
the dense-mixture kept population; report source, reserved and kept tokens.

Transfer the sealed prepared and token releases, reserve outputs and their
complete source-map dependencies to Clariden using `rcp-clariden-transfer`.
Verify destination hashes, publish markers last, and apply the `infra01` shared
permissions convention. Transfer does not change provenance roots. Admission,
reserve completion, sampler acceptance and trainer consumption are separate.

## Future Clariden reruns

The paired files under `configs_apertus_v2/` use the same prepared population,
tokenizer, ID column, manifests, groups and limits. They use the standard Slurm
backend. From the repository root, run the normal workflow documented in
[README.md](README.md), with
`configs_apertus_v2/romansh-<configuration>.cfg`. A rerun must use a fresh output
identity or verified task-scoped retry; never run the worker against an accepted
release since its ordinary retry behavior recreates its dump output.
