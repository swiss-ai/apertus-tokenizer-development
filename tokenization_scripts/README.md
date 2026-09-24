# Tokenization pipeline runbook

These scripts prepare Parquet inputs, tokenize them into Megatron `.bin/.idx` pairs,
write source maps, and optionally validate and seal a manifest-backed grouped dataset.
Clariden is the primary execution environment and uses the built-in Slurm submission
path. RCP is a secondary environment that uses explicit `runai submit` jobs; this
repository intentionally does not provide an RCP launcher.

Run commands from a pinned repository checkout. The Python environment must contain
`data-pipeline-pretrain` and its Datatrove, PyArrow, NumPy, and tokenizer dependencies.
When `data-pipeline-pretrain` is not installed as a package, add its `src` directory to
`PYTHONPATH` in every preparation, worker, and validation job.

## Entry points

### `tokenize_script.sh`

```text
tokenize_script.sh <config-file> [--dont_compute_dumps] [--prepare-only]
```

This is the normal Clariden entry point. It loads the config, checks any required
input marker, prepares path manifests, and submits one Slurm job per dump.

- `--prepare-only` creates the dump manifests and directory links, then exits without
  submitting tokenization workers. This is also the preparation phase used on RCP.
- `--dont_compute_dumps` skips preparation and submits only the path manifests still
  present under `PATH_TO_PREPROCESSING_METADATA/dumps`. A successful worker moves its
  path manifest to `completed-dumps`, so this flag is suitable for bounded retries.

### `tokenize.sh`

```text
tokenize.sh <config-file> <output-folder> <logging-dir> <paths-file> \
  [completed-folder] [batch-size] [batch-bytes] [workers] [threads]
```

This is the worker entry point submitted by `tokenize_script.sh` on Clariden and
submitted directly with `runai submit` on RCP. It deletes only the dump-scoped output
and logging directories passed to it, tokenizes the exact files in `paths-file`, and
moves that path manifest to `completed-folder` only after success. Never run two
workers for the same dump concurrently.

`TOKENIZATION_LAUNCH_BACKEND` controls how commands execute inside the current job:

- unset or `slurm`: use `srun` with `env.toml`;
- `rcp`: invoke Python directly in the prepared container environment.

It is not a cluster selector or launcher. There is deliberately no `EXECUTION_SITE`
setting; the selected config paths and the submission command determine the site.

### `validate_tokenization.sh`

```text
validate_tokenization.sh <config-file> [implementation-commit]
```

This validates a completed manifest-backed grouped tokenization and publishes its seal.
If `implementation-commit` is omitted, the current tokenizer-repository commit is used.
The wrapper derives all other arguments from the config and calls
`validate_megatron.py`.

On success it writes the following files under `DATASET_OUTPUT_FOLDER_NAME`:

- `TOKENIZATION_MANIFEST.jsonl`;
- `CATEGORY_COUNTS.json`;
- `TOKENIZATION_RUN.json`;
- byte-identical `_SUCCESS.json`, written last.

Do not create or copy `_SUCCESS.json` manually. Validation must cover every prepared
source file exactly once before the marker is published.

### Lower-level Python commands

`prepare_dumps.py`, `preprocess_megatron.py`, and `validate_megatron.py` expose their
complete argument lists through `--help`. Prefer the shell entry points for production
runs because they consistently resolve configs, tokenizer paths, output layout, and
validation metadata.

## Primary environment: Clariden

Configs under `configs_apertus_v2/` use Clariden-visible paths and Slurm resources.
From the repository root:

```bash
config_path=tokenization_scripts/configs_apertus_v2/FineMath-CommonCrawl-subset.cfg
./tokenization_scripts/tokenize_script.sh "$config_path"
```

For the sealed Stack v3.1 repository-context dataset:

```bash
config_path=tokenization_scripts/configs_apertus_v2/stackv31-repo-context-4k-v1.cfg
./tokenization_scripts/tokenize_script.sh "$config_path"
```

The entry point prepares dumps with `srun` and submits workers with `sbatch`. Slurm
account, partition, node, CPU, GPU, time, reservation, and requeue settings come from
the config. After every worker succeeds, validate and seal the output from a Clariden
compute allocation, not a login node:

```bash
srun --environment=tokenization_scripts/env.toml \
  ./tokenization_scripts/validate_tokenization.sh "$config_path"
```

Request at least `TOKENIZATION_VALIDATION_WORKERS` CPUs and enough memory for the
selected validation mode when creating the allocation.

To prepare without submission, or to retry only unfinished dumps:

```bash
./tokenization_scripts/tokenize_script.sh "$config_path" --prepare-only
./tokenization_scripts/tokenize_script.sh "$config_path" --dont_compute_dumps
```

## Secondary environment

<details>
<summary>RCP workflow with direct <code>runai submit</code> jobs</summary>

### Overview

Configs under `configs_apertus_v2_rcp/` contain RCP `/mloscratch` paths. All paths
passed after `--command --` must be absolute paths visible inside the container and
mounted PVC. Pin the container image, tokenizer checkout, data-pipeline-pretrain
checkout, config, tokenizer payload, and implementation commits before launching.

The templates below intentionally use placeholders because RunAI project names,
images, PVC claims, and checkout locations are operator-specific.

### 1. Prepare dump manifests

Run preparation in a bounded RunAI job. `--prepare-only` is essential: the shell
orchestrator's submission phase is Slurm-only.

```bash
runai submit -p <project> \
  --name <prepare-job> \
  --image <pinned-image> \
  --gpu 0 --cpu 4 --memory 32G \
  --node-pools <cpu-capable-pool> --backoff-limit 1 \
  --existing-pvc claimname=<scratch-claim>,path=/mloscratch \
  --run-as-uid <uid> --run-as-gid <gid> \
  --environment USER=<user> --environment LOGNAME=<user> \
  --environment PATH=<runtime-venv>/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --environment TOKENIZATION_LAUNCH_BACKEND=rcp \
  --environment PYTHONPATH=<data-pipeline-pretrain-checkout>/src \
  --command -- /bin/bash \
  <tokenizer-checkout>/tokenization_scripts/tokenize_script.sh \
  <tokenizer-checkout>/tokenization_scripts/configs_apertus_v2_rcp/<dataset>.cfg \
  --prepare-only
```

Require the preparation job to succeed before enumerating
`PATH_TO_PREPROCESSING_METADATA/dumps/**/paths_file_*.txt`. Treat those files as the
exact worker inventory.

### 2. Submit one worker per dump

Derive the dump-scoped output, logging, and completion paths exactly as
`tokenize_script.sh` does. For a grouped dump they are:

```text
output     = DATASET_OUTPUT_FOLDER_NAME/<group>/dump-<dump-id>
logging    = PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs/TOKENIZER_NAME/DATASET_NAME/<group>/dump-<dump-id>
completed  = PATH_TO_PREPROCESSING_METADATA/completed-dumps/<group>
```

Submit each distinct path manifest once:

```bash
runai submit -p <project> \
  --name <unique-dump-job> \
  --image <pinned-image> \
  --gpu 0 --cpu <worker-cpus> --memory <worker-memory> \
  --node-pools <cpu-capable-pool> --backoff-limit 1 \
  --existing-pvc claimname=<scratch-claim>,path=/mloscratch \
  --run-as-uid <uid> --run-as-gid <gid> \
  --environment USER=<user> --environment LOGNAME=<user> \
  --environment PATH=<runtime-venv>/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --environment TOKENIZATION_LAUNCH_BACKEND=rcp \
  --environment PYTHONPATH=<data-pipeline-pretrain-checkout>/src \
  --environment SLURM_JOB_ID=rcp-<unique-dump-job> \
  --command -- /bin/bash \
  <tokenizer-checkout>/tokenization_scripts/tokenize.sh \
  <absolute-config> <absolute-output> <absolute-logging> <absolute-paths-file> \
  <absolute-completed-folder> <batch-size> <batch-bytes> <workers> <threads>
```

`SLURM_JOB_ID` is compatibility metadata consumed by the shared statistics output; it
does not make the job a Slurm job. Ensure `workers * threads` does not exceed the CPUs
allocated to the RunAI job. Use the same config and path manifest for a bounded retry;
the worker safely replaces only its own dump-scoped output.

### 3. Validate and seal

After every expected path manifest has moved to `completed-dumps`, submit a separate
validation job using the same pinned environment:

```bash
runai submit -p <project> \
  --name <validation-job> \
  --image <pinned-image> \
  --gpu 0 --cpu <validation-cpus> --memory <validation-memory> \
  --node-pools <cpu-capable-pool> --backoff-limit 1 \
  --existing-pvc claimname=<scratch-claim>,path=/mloscratch \
  --run-as-uid <uid> --run-as-gid <gid> \
  --environment USER=<user> --environment LOGNAME=<user> \
  --environment PATH=<runtime-venv>/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --environment PYTHONPATH=<data-pipeline-pretrain-checkout>/src \
  --command -- /bin/bash \
  <tokenizer-checkout>/tokenization_scripts/validate_tokenization.sh \
  <absolute-config> <tokenization-implementation-commit>
```

Require the job to succeed, require `TOKENIZATION_RUN.json` and `_SUCCESS.json` to be
byte-identical, and verify the recorded config, tokenizer, implementation, source
marker, manifest, category, rank, world-size, and byte totals before admitting or
transferring the artifact.

</details>

## Configuration reference

The config is sourced as shell syntax. Do not put untrusted content in a config.

### Dataset and output identity

- `TOKENIZER`: tokenizer JSON path. Relative paths resolve from `tokenization_scripts/`.
- `TOKENIZER_NAME`: stable output and metadata name for the tokenizer.
- `DATASET_NAME`: stable dataset/job name.
- `COLUMN_KEY`: source text column.
- `ID_COLUMN`: source identifier column; defaults to `id`.
- `PATH_TO_RAW_DATASET`: input root visible on the selected cluster.
- `PATH_TO_OUTPUT_FOLDER`: output/log root.
- `PATH_TO_PREPROCESSING_METADATA`: dump manifests, completion state, and symlinks.
- `DATASET_OUTPUT_FOLDER_NAME`: optional exact token payload root; otherwise defaults to
  `PATH_TO_OUTPUT_FOLDER/TOKENIZER_NAME/DATASET_NAME`.

### Input admission and dump layout

- `REQUIRED_DATASET_MARKER`: optional marker that must exist before preparation.
- `DATASET_MANIFEST`: optional JSONL source inventory. When set, only listed files are
  admitted.
- `MANIFEST_PATH_KEY`: manifest field containing a path relative to the dataset root;
  defaults to `relative_path`.
- `DUMPS_NUMBER`: optional fixed dump count per group.
- `MAX_DUMP_BYTES`: target upper bytes per automatically sized dump.
- `DUMP_GROUP_FIELDS`: comma-separated manifest/lookup fields mirrored into output
  directories.
- `DUMP_GROUP_METADATA`, `DUMP_GROUP_METADATA_ROOT`,
  `DUMP_GROUP_METADATA_LOOKUP_FIELD`, and `DUMP_GROUP_METADATA_ID_FIELD`: optional JSON
  lookup used when grouping values are outside the source manifest.
- `EXPECTED_GROUP_COUNT` and `EXPECTED_GROUP_HEADS`: fail-closed grouping assertions.

### Row selection and tokenization limits

- `INCLUDE_BOOLEAN_COLUMN`: optional boolean column controlling row admission.
- `INCLUDE_REASON_COLUMN`: optional exclusion/admission reason column paired with the
  boolean column.
- `INCLUDED_REASON`: exact reason value for included rows; it may intentionally be
  empty.
- `REHYDRATE_FLAG`: whether to apply the data-pipeline rehydrater.
- `EXTENSION`: source extension, normally `.parquet`.
- `MAX_SEQUENCE_TOKENS`: exact per-example token ceiling; `0` disables the guard.
- `NUMBER_OF_DATATROVE_TASKS`: output task/rank count.
- `TOKENIZER_BATCH_SIZE`: maximum documents per tokenizer call.
- `TOKENIZER_BATCH_BYTES`: maximum UTF-8 input bytes per tokenizer call.
- `TOKENIZER_WORKERS`: concurrent Datatrove workers; defaults to the task count.
- `TOKENIZER_THREADS`: Rayon threads per worker. The default is bounded by the allocated
  CPU count and 144 threads.

### Validation and Clariden submission

- `TOKENIZATION_VALIDATION_WORKERS`: validation process count; defaults to `8`.
- `TOKENIZATION_VALIDATION_MODE`: `strict` or `lightweight_infrastructure`. Strict mode
  hashes prepared Parquet and token/index payloads. Lightweight mode still validates
  exact inventory, sizes, Parquet footer identity, rank/world-size completeness,
  `.bin/.idx/.map` structure, source coverage, and sequence lengths, but avoids a second
  full payload hash pass.
- `ACCOUNT`, `NODES`, `PARTITION`, `TIME`, `CPUS_PER_TASK`, `GPUS`, `NO_REQUEUE`, and
  optional `RESERVATION` configure the Clariden Slurm submission path. RCP resources
  are selected explicitly on each `runai submit` command.

## Output and recovery contract

Preparation creates these paths under `PATH_TO_PREPROCESSING_METADATA`:

```text
dumps/                    # path manifests not yet completed
completed-dumps/          # manifests moved here after successful tokenization
raw-dataset-link          # symlink to PATH_TO_RAW_DATASET
tokenized-dir-link        # symlink to DATASET_OUTPUT_FOLDER_NAME
```

Token outputs live under `DATASET_OUTPUT_FOLDER_NAME`, optionally grouped into nested
directories and then `dump-<id>` directories. Logs live under
`PATH_TO_OUTPUT_FOLDER/logs`.

For recovery, inspect the scheduler state and logs first. Resubmit only failed dumps
whose path manifests remain under `dumps`; never regenerate dumps while workers from
the previous inventory are active. Run validation only after the expected worker set is
complete, and treat the final `_SUCCESS.json` as the admission marker.
