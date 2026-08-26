#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Parse args: accept config file and optional --dont_compute_dumps flag
DONT_COMPUTE_DUMPS=0
CONFIG_FILE=""
for arg in "$@"; do
  if [ "$arg" = "--dont_compute_dumps" ]; then
    DONT_COMPUTE_DUMPS=1
  elif [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$arg"
  fi
done

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config-file> [--dont_compute_dumps]"
  exit 1
fi

# Check if the file exists, then load it
if [ -f "$CONFIG_FILE" ]; then
  CONFIG_FILE=$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")
  source "$CONFIG_FILE"
else
  echo "Error: Config file $CONFIG_FILE not found."
  exit 1
fi

RES_OPT=""
if [ -n "${RESERVATION:-}" ] || [ -n "${reservation:-}" ]; then
  VAL="${RESERVATION:-${reservation:-}}"
  RES_OPT="--reservation=${VAL}"
fi

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv # Used later by tokenize.sh
PATH_TO_DATATROVE_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs                    # Where datatrove logs are stored
PATH_TO_SLURM_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/slurm_logs                            # Where slurm logs are stored
DATASET_OUTPUT_FOLDER_NAME=${DATASET_OUTPUT_FOLDER_NAME:-$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME}
ID_COLUMN=${ID_COLUMN:-id}
INCLUDE_BOOLEAN_COLUMN=${INCLUDE_BOOLEAN_COLUMN:-}
EXCLUSION_REASON_COLUMN=${EXCLUSION_REASON_COLUMN:-exclusion_reason}
PROVENANCE_PIPELINE_JSON=${PROVENANCE_PIPELINE_JSON:-}
PROVENANCE_GROUP_KEYS=${PROVENANCE_GROUP_KEYS:-}
PROVENANCE_DIGEST_FILES=${PROVENANCE_DIGEST_FILES:-}
DUMP_GROUP_FIELDS=${DUMP_GROUP_FIELDS:-}
DUMP_GROUP_METADATA=${DUMP_GROUP_METADATA:-}
DUMP_GROUP_METADATA_ROOT=${DUMP_GROUP_METADATA_ROOT:-}
DUMP_GROUP_METADATA_LOOKUP_FIELD=${DUMP_GROUP_METADATA_LOOKUP_FIELD:-}
DUMP_GROUP_METADATA_ID_FIELD=${DUMP_GROUP_METADATA_ID_FIELD:-id}
EXPECTED_GROUP_COUNT=${EXPECTED_GROUP_COUNT:-0}
EXPECTED_GROUP_HEADS=${EXPECTED_GROUP_HEADS:-}
MAX_DUMP_BYTES=${MAX_DUMP_BYTES:-150000000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-10000}

if [ "$DONT_COMPUTE_DUMPS" -eq 0 ]; then
  mkdir -p "$PATH_TO_PREPROCESSING_METADATA/completed-dumps" #used later by tokenize.sh
  mkdir -p "$PATH_TO_SLURM_LOGGING_DIR"
  mkdir -p "$DATASET_OUTPUT_FOLDER_NAME"
  ln -sfn "$DATASET_OUTPUT_FOLDER_NAME" "$PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link"

  prepare_args=(
    --dataset-folder "$PATH_TO_RAW_DATASET"
    --preprocessing-metadata-folder "$PATH_TO_PREPROCESSING_METADATA"
    --group-fields "$DUMP_GROUP_FIELDS"
    --expected-groups "$EXPECTED_GROUP_COUNT"
    --expected-group-heads "$EXPECTED_GROUP_HEADS"
    --max-dump-bytes "$MAX_DUMP_BYTES"
  )
  if [ -n "${DUMPS_NUMBER:-}" ]; then
    prepare_args+=(--n-dumps "$DUMPS_NUMBER")
  fi
  if [ -n "${DATASET_MANIFEST:-}" ]; then
    prepare_args+=(
      --manifest "$DATASET_MANIFEST"
      --manifest-path-key "${MANIFEST_PATH_KEY:-relative_path}"
    )
  fi
  if [ -n "$DUMP_GROUP_METADATA" ]; then
    prepare_args+=(
      --group-metadata "$DUMP_GROUP_METADATA"
      --group-metadata-root "$DUMP_GROUP_METADATA_ROOT"
      --group-metadata-lookup-field "$DUMP_GROUP_METADATA_LOOKUP_FIELD"
      --group-metadata-id-field "$DUMP_GROUP_METADATA_ID_FIELD"
    )
  fi
  srun --environment="$SCRIPT_DIR/env.toml" --partition=debug --account="$ACCOUNT" \
    --job-name=dumps_prep --export=ALL \
    python3 "$SCRIPT_DIR/prepare_dumps.py" "${prepare_args[@]}"
else
  echo "Skipping dump creation and directory setup due to --dont_compute_dumps flag"
fi

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" >"$CSV_RESULTS_FILE"
dumps_root=$PATH_TO_PREPROCESSING_METADATA/dumps
found_dumps=0
while IFS= read -r paths_file; do
  found_dumps=$((found_dumps + 1))
  relative_path=${paths_file#"$dumps_root"/}
  group_path=$(dirname "$relative_path")
  if [ "$group_path" = . ]; then
    group_path=""
  fi
  dump_name=$(basename "$paths_file")
  dump=${dump_name#paths_file_}
  dump=${dump%.txt}
  output_folder=$DATASET_OUTPUT_FOLDER_NAME
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME
  completed_folder=$PATH_TO_PREPROCESSING_METADATA/completed-dumps
  job_group=""
  if [ -n "$group_path" ]; then
    output_folder=$output_folder/$group_path
    logging_dir=$logging_dir/$group_path
    completed_folder=$completed_folder/$group_path
    job_group=-${group_path//\//-}
  fi
  output_folder=$output_folder/dump-$dump
  logging_dir=$logging_dir/dump-$dump
  job_name=tokenize-$DATASET_NAME$job_group-dump-$dump
  submit_args=(
    --partition="$PARTITION"
    --account="$ACCOUNT"
    --nodes="$NODES"
    --time="$TIME"
    --cpus-per-task="$CPUS_PER_TASK"
    --job-name="$job_name"
    --output="$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out"
    --error="$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err"
  )
  if [ -n "$RES_OPT" ]; then
    submit_args+=("$RES_OPT")
  fi
  if [ "${GPUS:-0}" -gt 0 ]; then
    submit_args+=(--gres="gpu:$GPUS")
  fi
  if [ -n "${NO_REQUEUE:-}" ]; then
    submit_args+=("$NO_REQUEUE")
  fi
  sbatch "${submit_args[@]}" "$SCRIPT_DIR/tokenize.sh" \
    "$CONFIG_FILE" "$output_folder" "$logging_dir" "$paths_file" \
    "$group_path" "$completed_folder"
done < <(find "$dumps_root" -type f -name 'paths_file_*.txt' -print | sort)

if [ "$found_dumps" -eq 0 ]; then
  echo "No prepared dumps found under $dumps_root" >&2
  exit 1
fi
