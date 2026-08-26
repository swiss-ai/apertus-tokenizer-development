#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

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
  source "$CONFIG_FILE"
else
  echo "Error: Config file $CONFIG_FILE not found."
  exit 1
fi

RES_OPT=""
if [ -n "${RESERVATION:-}" ] || [ -n "${reservation:-}" ]; then
  VAL="${RESERVATION:-$reservation}"
  RES_OPT="--reservation=${VAL}"
fi

submit_stackv31() (
  set -euo pipefail

  : "${PATH_TO_RAW_DATASET:?}"
  : "${GROUP_MANIFEST:?}"
  : "${CATEGORY_MAP:?}"
  : "${PATH_TO_OUTPUT_FOLDER:?}"
  : "${PATH_TO_PREPROCESSING_METADATA:?}"
  : "${TOKENIZER:?}"
  : "${TOKENIZER_NAME:?}"

  output_root=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME
  work_root=$PATH_TO_PREPROCESSING_METADATA/$TOKENIZER_NAME
  tokenizer_path=$TOKENIZER
  if [[ "$tokenizer_path" != /* ]]; then
    tokenizer_path=$SCRIPT_DIR/$tokenizer_path
  fi

  if [ "$DONT_COMPUTE_DUMPS" -eq 0 ]; then
    python3 "$SCRIPT_DIR/stackv31.py" prepare \
      --input-root "$PATH_TO_RAW_DATASET" \
      --group-manifest "$GROUP_MANIFEST" \
      --category-map "$CATEGORY_MAP" \
      --output-root "$output_root" \
      --work-root "$work_root" \
      --tokenizer-path "$tokenizer_path" \
      --column "$COLUMN_KEY" \
      --id-column "$ID_COLUMN" \
      --include-boolean-column "$INCLUDE_BOOLEAN_COLUMN" \
      --exclusion-reason-column "$EXCLUSION_REASON_COLUMN" \
      --output-layout "$OUTPUT_LAYOUT" \
      --expected-languages "$EXPECTED_LANGUAGE_COUNT" \
      --expected-categories "$EXPECTED_CATEGORIES" \
      --expected-policy-tag "$EXPECTED_POLICY_TAG" \
      --expected-signals-revision "$EXPECTED_SIGNALS_REVISION" \
      --expected-source-revision "$EXPECTED_SOURCE_REVISION" \
      --expected-tokenizer-sha256 "$EXPECTED_TOKENIZER_SHA256" \
      --target-jobs "$TARGET_JOBS" \
      --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
  fi

  assignment_count=$(
    python3 "$SCRIPT_DIR/stackv31.py" assignment-count --work-root "$work_root"
  )
  if [ "$assignment_count" -lt 1 ]; then
    echo "Prepared Stack v3.1 run has no assignments"
    exit 1
  fi
  last_assignment=$((assignment_count - 1))
  mkdir -p "$work_root/slurm"

  token_args=(
    --parsable
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --nodes="${NODES:-1}"
    --time="$TIME"
    --cpus-per-task="$CPUS_PER_TASK"
    --array="0-$last_assignment"
    --job-name=tokenize-stackv31
    --output="$work_root/slurm/%x-%A_%a.out"
    --error="$work_root/slurm/%x-%A_%a.err"
  )
  if [ -n "$RES_OPT" ]; then
    token_args+=("$RES_OPT")
  fi
  if [ "${GPUS:-0}" -gt 0 ]; then
    token_args+=(--gres="gpu:$GPUS")
  fi
  if [ -n "${NO_REQUEUE:-}" ]; then
    token_args+=("$NO_REQUEUE")
  fi

  token_job=$(sbatch "${token_args[@]}" "$SCRIPT_DIR/stackv31_tokenize.sh" \
    "$work_root" "$NUMBER_OF_DATATROVE_WORKERS")
  validation_args=(
    --parsable
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --time=04:00:00
    --cpus-per-task=16
    --dependency="afterok:$token_job"
    --job-name=validate-stackv31
    --output="$work_root/slurm/%x-%j.out"
    --error="$work_root/slurm/%x-%j.err"
  )
  if [ -n "$RES_OPT" ]; then
    validation_args+=("$RES_OPT")
  fi
  validation_job=$(sbatch "${validation_args[@]}" \
    "$SCRIPT_DIR/stackv31_validate.sh" \
    "$work_root" "$output_root" "$MAX_MAP_OVERHEAD" "$MIXTURE_SAMPLER_PYTHON")

  echo "Tokenization array: $token_job"
  echo "Validation: $validation_job"
)

if [ -n "${GROUP_MANIFEST:-}" ]; then
  submit_stackv31
  exit
fi

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv # Used later by tokenize.sh
PATH_TO_DATATROVE_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs                    # Where datatrove logs are stored
PATH_TO_SLURM_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/slurm_logs                            # Where slurm logs are stored
DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME             # Where tokenized data is stored

if [ "$DONT_COMPUTE_DUMPS" -eq 0 ]; then
  mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed-dumps #used later by tokenize.sh
  mkdir -p $PATH_TO_SLURM_LOGGING_DIR
  mkdir -p $DATASET_OUTPUT_FOLDER_NAME
  ln -sfn $DATASET_OUTPUT_FOLDER_NAME $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

  # Create dumps
  srun --environment="$SCRIPT_DIR/env.toml" --partition=debug --account=$ACCOUNT --job-name=dumps_prep --export=ALL bash -lc "python3 '$SCRIPT_DIR/prepare_dumps.py' --dataset-folder '${PATH_TO_RAW_DATASET}' --preprocessing-metadata-folder '${PATH_TO_PREPROCESSING_METADATA}' --n-dumps '${DUMPS_NUMBER}'"
else
  echo "Skipping dump creation and directory setup due to --dont_compute_dumps flag"
fi

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" >$CSV_RESULTS_FILE
# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<<$paths_file)
  output_folder=$DATASET_OUTPUT_FOLDER_NAME/dump-$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME/dump-$dump
  sbatch $RES_OPT --partition=$PARTITION --account=$ACCOUNT --nodes=$NODES --gres=gpu:$GPUS --time=$TIME --cpus-per-task=$CPUS_PER_TASK $NO_REQUEUE --job-name=tokenize-$DATASET_NAME-dump-$dump --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err "$SCRIPT_DIR/tokenize.sh" $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER $logging_dir $CSV_RESULTS_FILE $paths_file $NUMBER_OF_DATATROVE_TASKS $COLUMN_KEY $REHYDRATE_FLAG $EXTENSION
done
