#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${1:-}
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config-file> [implementation-commit]" >&2
  exit 1
fi
CONFIG_FILE=$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")
source "$CONFIG_FILE"
if [[ "$TOKENIZER" != /* ]]; then
  TOKENIZER=$(cd "$SCRIPT_DIR" && realpath "$TOKENIZER")
fi

implementation_commit=${2:-$(git -C "$SCRIPT_DIR/.." rev-parse HEAD)}
output_folder=${DATASET_OUTPUT_FOLDER_NAME:-$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME}
validation_workers=${TOKENIZATION_VALIDATION_WORKERS:-8}

python3 "$SCRIPT_DIR/validate_megatron.py" \
  --dataset "$PATH_TO_RAW_DATASET" \
  --manifest "$DATASET_MANIFEST" \
  --dataset-marker "$REQUIRED_DATASET_MARKER" \
  --output-folder "$output_folder" \
  --tokenizer "$TOKENIZER" \
  --config "$CONFIG_FILE" \
  --implementation-commit "$implementation_commit" \
  --dataset-name "$DATASET_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --text-column "$COLUMN_KEY" \
  --id-column "${ID_COLUMN:-id}" \
  --expected-categories "$EXPECTED_GROUP_HEADS" \
  --max-sequence-tokens "$MAX_SEQUENCE_TOKENS" \
  --workers "$validation_workers"
