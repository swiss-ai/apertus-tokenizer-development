#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${1:-}
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
  echo "Usage: $0 <stackv31-config>"
  exit 1
fi

source "$CONFIG_FILE"
: "${PATH_TO_RAW_DATASET:?}"
: "${GROUP_MANIFEST:?}"
: "${CATEGORY_MAP:?}"
: "${PATH_TO_OUTPUT_FOLDER:?}"
: "${PATH_TO_PREPROCESSING_METADATA:?}"
: "${TOKENIZER:?}"
: "${TOKENIZER_NAME:?}"

OUTPUT_ROOT=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME
WORK_ROOT=$PATH_TO_PREPROCESSING_METADATA/$TOKENIZER_NAME
TOKENIZER_PATH=$TOKENIZER
if [[ "$TOKENIZER_PATH" != /* ]]; then
  TOKENIZER_PATH=$SCRIPT_DIR/$TOKENIZER_PATH
fi

python3 "$SCRIPT_DIR/stackv31.py" prepare \
  --input-root "$PATH_TO_RAW_DATASET" \
  --group-manifest "$GROUP_MANIFEST" \
  --category-map "$CATEGORY_MAP" \
  --output-root "$OUTPUT_ROOT" \
  --work-root "$WORK_ROOT" \
  --tokenizer-path "$TOKENIZER_PATH" \
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

ASSIGNMENT_COUNT=$(python3 "$SCRIPT_DIR/stackv31.py" assignment-count --work-root "$WORK_ROOT")
if [ "$ASSIGNMENT_COUNT" -lt 1 ]; then
  echo "Prepared run has no assignments"
  exit 1
fi
LAST_ASSIGNMENT=$((ASSIGNMENT_COUNT - 1))
mkdir -p "$WORK_ROOT/slurm"

SBATCH_ARGS=(
  --parsable
  --account="$ACCOUNT"
  --partition="$PARTITION"
  --time="$TIME"
  --cpus-per-task="$CPUS_PER_TASK"
  --array="0-$LAST_ASSIGNMENT"
  --job-name=tokenize-stackv31
  --output="$WORK_ROOT/slurm/%x-%A_%a.out"
  --error="$WORK_ROOT/slurm/%x-%A_%a.err"
)
if [ "${GPUS:-0}" -gt 0 ]; then
  SBATCH_ARGS+=(--gres="gpu:$GPUS")
fi
if [ -n "${NO_REQUEUE:-}" ]; then
  SBATCH_ARGS+=("$NO_REQUEUE")
fi

TOKEN_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/stackv31_tokenize.sh" \
  "$WORK_ROOT" "$NUMBER_OF_DATATROVE_WORKERS")
VALIDATION_JOB=$(sbatch --parsable \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time=04:00:00 \
  --cpus-per-task=16 \
  --dependency="afterok:$TOKEN_JOB" \
  --job-name=validate-stackv31 \
  --output="$WORK_ROOT/slurm/%x-%j.out" \
  --error="$WORK_ROOT/slurm/%x-%j.err" \
  "$SCRIPT_DIR/stackv31_validate.sh" \
  "$WORK_ROOT" "$OUTPUT_ROOT" "$MAX_MAP_OVERHEAD" "$MIXTURE_SAMPLER_PYTHON")

echo "Tokenization array: $TOKEN_JOB"
echo "Validation: $VALIDATION_JOB"
