# Tokenization scripts

These scripts split Parquet inputs into dumps, tokenize each dump into Megatron
`.bin/.idx` pairs with `.map` source maps, and validate and seal a manifest-backed
grouped output. Clariden runs them through Slurm. RCP runs the same scripts as
explicit `runai submit` jobs.

## Requirements

- A `data-pipeline-pretrain` checkout on branch `token-source-maps` (PR #37). On
  Clariden, `env.toml` mounts it from `$HOME/data-pipeline-pretrain`; see the top-level
  README for the clone command. Elsewhere, put its `src` directory on `PYTHONPATH` in
  every preparation, worker and validation job.
- The tokenizer payload. The `tokenizer.json` files are stored in Git LFS, so fetch the
  one the config names before running:

  ```bash
  git lfs install
  git lfs pull --include preliminary_mul_200k/tokenizer.json
  ```

  A relative `TOKENIZER` path resolves from `tokenization_scripts/`. Validation compares
  the tokenizer's SHA-256 with the digest in every source map, so validate with the same
  file that tokenized the data.

## Clariden

Run from the repository root:

```bash
config_path=tokenization_scripts/configs_apertus_v2/stackv31-repo-context-4k-v1.cfg
./tokenization_scripts/tokenize_script.sh "$config_path"                 # prepare and submit
./tokenization_scripts/tokenize_script.sh "$config_path" --prepare-only  # prepare only
./tokenization_scripts/tokenize_script.sh "$config_path" --dont_compute_dumps  # resubmit unfinished dumps
```

Preparation runs `prepare_dumps.py` through `srun --partition=debug
--account="$ACCOUNT"`. The `debug` partition is hard-coded in `tokenize_script.sh`. Each
dump worker is then submitted with `sbatch`, using `ACCOUNT`, `PARTITION`, `NODES`,
`TIME`, `CPUS_PER_TASK`, `GPUS`, `NO_REQUEUE` and `RESERVATION` from the config.

After every worker has succeeded, validate and seal the output from a compute
allocation with at least `TOKENIZATION_VALIDATION_WORKERS` CPUs:

```bash
srun -A <account> --cpus-per-task=<cpus> --environment=tokenization_scripts/env.toml \
  ./tokenization_scripts/validate_tokenization.sh "$config_path"
```

## Entry points

### `tokenize_script.sh`

```text
tokenize_script.sh <config-file> [--dont_compute_dumps] [--prepare-only]
```

| argument | meaning |
|---|---|
| `<config-file>` | Config to source. |
| `--prepare-only` | Create the dump manifests and directory links, then exit without submitting workers. |
| `--dont_compute_dumps` | Skip preparation and submit only the path manifests still under `PATH_TO_PREPROCESSING_METADATA/dumps`. |

### `tokenize.sh`

```text
tokenize.sh <config-file> <output-folder> <logging-dir> <paths-file> \
  [completed-folder] [batch-size] [batch-bytes] [workers] [threads]
```

This is the per-dump worker. It deletes and recreates `<output-folder>` and
`<logging-dir>`, tokenizes the files listed in `<paths-file>`, and moves `<paths-file>`
to `<completed-folder>` after success. Never run two workers for the same dump at once.

| position | argument | overrides config key | default when omitted or empty |
|---|---|---|---|
| 1 | `config-file` | | required |
| 2 | `output-folder` | | required; passed as `--output-folder` |
| 3 | `logging-dir` | | required; passed as `--logging-dir` |
| 4 | `paths-file` | | required; passed as `--paths-file` |
| 5 | `completed-folder` | | `completed-dumps/` in the parent of the directory that holds `paths-file` |
| 6 | `batch-size` | `TOKENIZER_BATCH_SIZE` | config value, else `10000` |
| 7 | `batch-bytes` | `TOKENIZER_BATCH_BYTES` | config value, else `33554432` |
| 8 | `workers` | `TOKENIZER_WORKERS` | config value, else `NUMBER_OF_DATATROVE_TASKS` |
| 9 | `threads` | `TOKENIZER_THREADS` | config value, else `SLURM_CPUS_PER_TASK` (or `CPUS_PER_TASK`) divided by the workers, capped at 144 |

`threads` sets `RAYON_NUM_THREADS`. `TOKENIZATION_LAUNCH_BACKEND` selects how the Python
step runs: `slurm` (the default) uses `srun --environment=env.toml`, and `rcp` runs
Python directly in the current container.

### `validate_tokenization.sh`

```text
validate_tokenization.sh <config-file> [implementation-commit]
```

| argument | meaning |
|---|---|
| `<config-file>` | Config used for tokenization. |
| `[implementation-commit]` | Full 40-character commit of the tokenizer code that produced the output. Defaults to `HEAD` of this checkout. |

The script runs under `set -u` and calls `validate_megatron.py`. Besides the keys every
config sets, the config must define these keys, which `tokenize_script.sh` treats as
optional:

| key | requirement |
|---|---|
| `DATASET_MANIFEST` | Existing JSONL examples manifest. |
| `REQUIRED_DATASET_MARKER` | Existing completion marker with `complete: true`, `smoke: false` and `pins.examples_manifest_sha256` equal to the manifest's SHA-256. |
| `EXPECTED_GROUP_HEADS` | Non-empty comma-separated categories. |
| `MAX_SEQUENCE_TOKENS` | Prepared-metadata guard; `0` disables it. |
| `TOKENIZATION_VALIDATION_MAX_SEQUENCE_TOKENS` | Positive payload length bound; defaults to `MAX_SEQUENCE_TOKENS`. Set separately when the metadata guard is off. |

The output must be grouped as `<category>/[...]/dump-<id>/<rank>_tokens.{bin,idx,map}`. On
success the script writes `TOKENIZATION_MANIFEST.jsonl`, `CATEGORY_COUNTS.json`,
`TOKENIZATION_RUN.json` and, last, a byte-identical `_SUCCESS.json` under
`DATASET_OUTPUT_FOLDER_NAME`. It refuses to run if `_SUCCESS.json` already exists.

### `prepare_dumps.py`

`tokenize_script.sh` calls it with these arguments.

| argument | default | set from | meaning |
|---|---|---|---|
| `--dataset-folder` | required | `PATH_TO_RAW_DATASET` | Root scanned recursively for `.parquet` files, or the root that manifest paths are relative to. |
| `--preprocessing-metadata-folder` | required | `PATH_TO_PREPROCESSING_METADATA` | Receives `dumps/` and the `raw-dataset-link` symlink. |
| `--n-dumps` | automatic | `DUMPS_NUMBER` | Dumps per group, capped at the group's file count. |
| `--max-dump-bytes` | `150000000000` | `MAX_DUMP_BYTES` | Without `--n-dumps`, each group gets ceil(group bytes / this value) dumps. |
| `--manifest` | none | `DATASET_MANIFEST` | JSONL inventory; only listed files are admitted. |
| `--manifest-path-key` | `relative_path` | `MANIFEST_PATH_KEY` | Manifest field holding the path relative to `--dataset-folder`. |
| `--group-fields` | empty | `DUMP_GROUP_FIELDS` | Comma-separated fields that become nested output directories. Requires `--manifest`. |
| `--group-metadata` | empty | `DUMP_GROUP_METADATA` | JSON object or array that supplies group fields missing from the manifest. |
| `--group-metadata-root` | empty | `DUMP_GROUP_METADATA_ROOT` | Top-level key that holds the metadata entries. |
| `--group-metadata-lookup-field` | empty | `DUMP_GROUP_METADATA_LOOKUP_FIELD` | Manifest field whose value selects a metadata entry. Required with `--group-metadata`. |
| `--group-metadata-id-field` | `id` | `DUMP_GROUP_METADATA_ID_FIELD` | Identifier field when the metadata is an array. |
| `--expected-groups` | `0` (off) | `EXPECTED_GROUP_COUNT` | Fail unless grouping yields exactly this many groups. |
| `--expected-group-heads` | empty | `EXPECTED_GROUP_HEADS` | Fail unless the first group components equal this comma-separated set. |
| `--filter-in` | none | not passed | Keep only paths containing one of these substrings. |
| `--filter-out` | none | not passed | Drop paths containing any of these substrings. |

### `preprocess_megatron.py`

`tokenize.sh` calls it with these arguments.

| argument | default | set from | meaning |
|---|---|---|---|
| `--tokenizer-name-or-path` | required | `TOKENIZER` | Tokenizer file or Hugging Face model id. |
| `--eos-token` | none | not passed | EOS token appended after each document. |
| `--batch-size` | `10000` | position 6 | Maximum documents per tokenizer call. |
| `--batch-bytes` | `33554432` | position 7 | Maximum UTF-8 input bytes per tokenizer call. |
| `--output-folder` | required | position 2 | Output directory for the token pairs and maps. |
| `--logging-dir` | none | position 3 | Datatrove logging directory. |
| `--n-tasks` | `8` | `NUMBER_OF_DATATROVE_TASKS` | Datatrove tasks, capped at the number of files in the paths file. One output rank per task. |
| `--n-workers` | `-1` (equal to `--n-tasks`) | position 8 | Concurrently running tasks. |
| `--dataset` | required | `PATH_TO_PREPROCESSING_METADATA/raw-dataset-link` | Root the paths file is relative to. |
| `--paths-file` | required | position 4 | One input path per line. |
| `--column` | `text` | `COLUMN_KEY` | Text column. |
| `--id-column` | `id` | `ID_COLUMN` | Document identifier column. |
| `--token-map-source-root` | empty (the read root) | `TOKEN_MAP_SOURCE_ROOT` | Absolute dataset root recorded in the source maps. |
| `--rehydrate` | `False` | `REHYDRATE_FLAG` | `true`, `1` or `yes` applies the rehydrater. |
| `--extension` | `.parquet` | `EXTENSION` | Input extension. A value containing `jsonl` uses the JSONL reader and writes no source maps. |
| `--include-boolean-column` | empty | `INCLUDE_BOOLEAN_COLUMN` | Boolean column that selects rows. Parquet only, and requires `--include-reason-column`. |
| `--include-reason-column` | empty | `INCLUDE_REASON_COLUMN` | Reason column paired with the boolean column. |
| `--included-reason` | `included` | `INCLUDED_REASON` | Reason value of included rows; may be empty. |
| `--max-sequence-tokens` | `0` (off) | `MAX_SEQUENCE_TOKENS` | Fail when a tokenized sequence exceeds this length. |

### `validate_megatron.py`

`validate_tokenization.sh` calls it with these arguments.

| argument | default | set from | meaning |
|---|---|---|---|
| `--dataset` | required | `PATH_TO_RAW_DATASET` | Prepared dataset root. Every map must record a root that resolves to the same path, otherwise it fails with "wrong prepared root". |
| `--manifest` | required | `DATASET_MANIFEST` | Prepared examples manifest. |
| `--dataset-marker` | required | `REQUIRED_DATASET_MARKER` | Prepared-dataset completion marker. |
| `--output-folder` | required | `DATASET_OUTPUT_FOLDER_NAME` | Token output root to validate and seal. |
| `--tokenizer` | required | `TOKENIZER` | `tokenizer.json` used for tokenization. |
| `--config` | required | the config file | Config whose SHA-256 is recorded in the seal. |
| `--implementation-commit` | required | position 2, else `HEAD` | Commit of the tokenizer code that produced the output. |
| `--validator-commit` | required | `HEAD` of this checkout | Commit of the validator. |
| `--dataset-name` | required | `DATASET_NAME` | Dataset name recorded in the seal. |
| `--tokenizer-name` | required | `TOKENIZER_NAME` | Tokenizer name recorded in the seal. |
| `--text-column` | `text` | `COLUMN_KEY` | Text column every map must record. |
| `--id-column` | `id` | `ID_COLUMN` | Identifier column every map must record. |
| `--expected-categories` | required | `EXPECTED_GROUP_HEADS` | Categories the manifest and the output's top level must equal. |
| `--max-sequence-tokens` | required | `TOKENIZATION_VALIDATION_MAX_SEQUENCE_TOKENS`, falling back to `MAX_SEQUENCE_TOKENS` | Positive maximum tokens per sequence in validation. |
| `--workers` | `8` | `TOKENIZATION_VALIDATION_WORKERS` | Parallel validation processes. |
| `--validation-mode` | `strict` | `TOKENIZATION_VALIDATION_MODE` | `strict` or `lightweight_infrastructure`, see below. |

## RCP

Configs under `configs_apertus_v2_rcp/` use RCP `/mloscratch` paths. Every job uses the
same template; all paths after `--command --` must be absolute and visible in the
container:

```bash
runai submit -p <project> --name <job> --image <image> \
  --gpu 0 --cpu <cpus> --memory <memory> --node-pools <cpu-pool> --backoff-limit 1 \
  --existing-pvc claimname=<scratch-claim>,path=/mloscratch \
  --run-as-uid <uid> --run-as-gid <gid> \
  --environment USER=<user> --environment LOGNAME=<user> \
  --environment PATH=<venv>/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --environment PYTHONPATH=<data-pipeline-pretrain-checkout>/src \
  --environment TOKENIZATION_LAUNCH_BACKEND=rcp \
  --environment SLURM_JOB_ID=rcp-<job> \
  --command -- /bin/bash <tokenizer-checkout>/tokenization_scripts/<script> <arguments>
```

Run the steps in order, and start each one only after the previous one has succeeded:

| step | `<script> <arguments>` |
|---|---|
| prepare | `tokenize_script.sh <config> --prepare-only` |
| tokenize, one job per `PATH_TO_PREPROCESSING_METADATA/dumps/**/paths_file_<id>.txt` | `tokenize.sh <config> <output> <logging> <paths-file> <completed> <batch-size> <batch-bytes> <workers> <threads>` |
| validate | `validate_tokenization.sh <config> <implementation-commit>` |

Derive the worker paths as `tokenize_script.sh` does, where `<group>` is the path of
the paths file's directory relative to `dumps/` (empty for ungrouped dumps):

| argument | path |
|---|---|
| `<output>` | `DATASET_OUTPUT_FOLDER_NAME/<group>/dump-<id>` |
| `<logging>` | `PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs/TOKENIZER_NAME/DATASET_NAME/<group>/dump-<id>` |
| `<completed>` | `PATH_TO_PREPROCESSING_METADATA/completed-dumps/<group>` |

Keep `<workers> * <threads>` at or below `<cpus>`. `SLURM_JOB_ID` only labels the
worker's row in the statistics CSV.

## Configuration reference

The config is sourced as shell, so do not put untrusted content in it.

### Dataset and output identity

- `TOKENIZER`: tokenizer JSON path. Relative paths resolve from `tokenization_scripts/`.
- `TOKENIZER_NAME`: tokenizer name used in output paths and metadata.
- `DATASET_NAME`: dataset name used in output paths, job names and metadata.
- `COLUMN_KEY`: source text column.
- `ID_COLUMN`: source identifier column; defaults to `id`.
- `TOKEN_MAP_SOURCE_ROOT`: optional absolute dataset root recorded in the source maps
  instead of the physical read root. It needs a data-pipeline-pretrain checkout whose
  `ProvenanceParquetReader` accepts `provenance_dataset_root` (PR #37).
  `validate_megatron.py` requires every recorded root to resolve to the same path as
  `PATH_TO_RAW_DATASET`, so if this key resolves to a different path, validation
  rejects the maps with "token map has the wrong prepared root".
- `PATH_TO_RAW_DATASET`: input root.
- `PATH_TO_OUTPUT_FOLDER`: output and log root.
- `PATH_TO_PREPROCESSING_METADATA`: dump manifests, completion state and symlinks.
- `DATASET_OUTPUT_FOLDER_NAME`: token output root; defaults to
  `PATH_TO_OUTPUT_FOLDER/TOKENIZER_NAME/DATASET_NAME`.

### Input admission and dump layout

- `REQUIRED_DATASET_MARKER`: optional file that must exist before preparation.
- `DATASET_MANIFEST`: optional JSONL source inventory. When set, only listed files are
  admitted.
- `MANIFEST_PATH_KEY`: manifest field containing a path relative to the dataset root;
  defaults to `relative_path`.
- `DUMPS_NUMBER`: optional fixed dump count per group.
- `MAX_DUMP_BYTES`: target maximum bytes per automatically sized dump; defaults to
  `150000000000`.
- `DUMP_GROUP_FIELDS`: comma-separated manifest or metadata fields mirrored into output
  directories.
- `DUMP_GROUP_METADATA`, `DUMP_GROUP_METADATA_ROOT`,
  `DUMP_GROUP_METADATA_LOOKUP_FIELD` and `DUMP_GROUP_METADATA_ID_FIELD`: optional JSON
  lookup for group fields that are not in the manifest.
- `EXPECTED_GROUP_COUNT` and `EXPECTED_GROUP_HEADS`: grouping assertions; preparation
  fails when they do not match.

### Row selection and tokenization limits

- `INCLUDE_BOOLEAN_COLUMN`: optional boolean column that selects rows.
- `INCLUDE_REASON_COLUMN`: reason column paired with the boolean column.
- `INCLUDED_REASON`: reason value of included rows; defaults to `included` and may be
  empty.
- `REHYDRATE_FLAG`: whether to apply the data-pipeline rehydrater.
- `EXTENSION`: source extension; defaults to `.parquet`.
- `MAX_SEQUENCE_TOKENS`: per-sequence token ceiling; `0` (the default) disables it.
- `NUMBER_OF_DATATROVE_TASKS`: Datatrove tasks, and therefore output ranks, per dump.
- `TOKENIZER_BATCH_SIZE`: maximum documents per tokenizer call; defaults to `10000`.
- `TOKENIZER_BATCH_BYTES`: maximum UTF-8 input bytes per tokenizer call; defaults to
  `33554432`. A single larger document is processed alone.
- `TOKENIZER_WORKERS`: concurrent Datatrove workers; defaults to the task count.
- `TOKENIZER_THREADS`: Rayon threads per worker; defaults to the allocated CPUs divided
  by the workers, capped at 144. Keep `TOKENIZER_WORKERS * TOKENIZER_THREADS` at or
  below `CPUS_PER_TASK`.

### Validation and Clariden submission

- `TOKENIZATION_VALIDATION_WORKERS`: validation processes; defaults to `8`.
- `TOKENIZATION_VALIDATION_MODE`: `strict` (the default) or
  `lightweight_infrastructure`. Strict mode hashes the prepared Parquet files and the
  `.bin` and `.idx` files and checks every index offset. Lightweight mode skips those
  hashes and offset checks but still checks inventory, sizes, Parquet footers,
  rank completeness, `.bin/.idx/.map` structure, source coverage and sequence lengths.
- `ACCOUNT`, `PARTITION`, `NODES`, `TIME`, `CPUS_PER_TASK`, `GPUS`, `NO_REQUEUE` (a raw
  `sbatch` flag such as `--no-requeue`) and optional `RESERVATION`: `sbatch` options for
  the workers. Preparation uses only `ACCOUNT`.

## Output and recovery

Preparation creates these paths under `PATH_TO_PREPROCESSING_METADATA`:

```text
dumps/                    # path manifests not yet completed
completed-dumps/          # path manifests moved here after successful tokenization
raw-dataset-link          # symlink to PATH_TO_RAW_DATASET
tokenized-dir-link        # symlink to DATASET_OUTPUT_FOLDER_NAME
```

Every submission, including `--dont_compute_dumps`, rewrites
`tokenize-<TOKENIZER_NAME>-<DATASET_NAME>.csv` with a header row, and each worker
appends one row of statistics.

Token outputs live under `DATASET_OUTPUT_FOLDER_NAME`, optionally in group directories,
then in `dump-<id>` directories. Logs live under `PATH_TO_OUTPUT_FOLDER/logs`.

To recover, resubmit with `--dont_compute_dumps`: only the path manifests still under
`dumps/` are submitted. Do not regenerate dumps while workers from the previous
inventory are running. Run validation only after every path manifest has moved to
`completed-dumps/`.
