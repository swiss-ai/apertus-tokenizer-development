#!/bin/bash

#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --no-requeue
#SBATCH --time=00:10:00

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE="$SCRIPT_DIR/env.toml"

set -eo pipefail

CONFIG_FILE=$1
output_folder=$2
logging_dir=$3
paths_file=$4
COMPLETED_DUMPS_FOLDER=${5:-$(dirname "$(dirname "$paths_file")")/completed-dumps}

source "$CONFIG_FILE"

input_folder=$PATH_TO_PREPROCESSING_METADATA/raw-dataset-link
CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv
ID_COLUMN=${ID_COLUMN:-id}
INCLUDE_BOOLEAN_COLUMN=${INCLUDE_BOOLEAN_COLUMN:-}
INCLUDE_REASON_COLUMN=${INCLUDE_REASON_COLUMN:-}
INCLUDED_REASON=${INCLUDED_REASON-included}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-10000}
MAX_SEQUENCE_TOKENS=${MAX_SEQUENCE_TOKENS:-0}
TOKENIZATION_LAUNCH_BACKEND=${TOKENIZATION_LAUNCH_BACKEND:-slurm}

# Setup ENV
export HF_HUB_ENABLE_HF_TRANSFER=0
# Setup directories
rm -rf "$output_folder"
rm -rf "$logging_dir"
mkdir -p "$output_folder"

echo "START TIME: $(date) | Preprocessing $paths_file with $NUMBER_OF_DATATROVE_TASKS tasks per node with the $TOKENIZER tokenizer. Storing tokenized dataset in $output_folder"
start_s=$(date)
start=$(date +%s)

preprocess_command=(
  python3 "$SCRIPT_DIR/preprocess_megatron.py"
  --tokenizer-name-or-path "$TOKENIZER" \
  --output-folder "$output_folder" \
  --logging-dir "$logging_dir" \
  --n-tasks "$NUMBER_OF_DATATROVE_TASKS" \
  --dataset "$input_folder" \
  --paths-file "$paths_file" \
  --column "$COLUMN_KEY" \
  --id-column "$ID_COLUMN" \
  --extension "${EXTENSION:-.parquet}" \
  --rehydrate "$REHYDRATE_FLAG" \
  --include-boolean-column "$INCLUDE_BOOLEAN_COLUMN" \
  --include-reason-column "$INCLUDE_REASON_COLUMN" \
  --included-reason "$INCLUDED_REASON" \
  --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"
  --max-sequence-tokens "$MAX_SEQUENCE_TOKENS"
)
if [ "$TOKENIZATION_LAUNCH_BACKEND" = rcp ]; then
  "${preprocess_command[@]}"
elif [ "$TOKENIZATION_LAUNCH_BACKEND" = slurm ]; then
  srun --environment="$ENV_FILE" numactl --membind=0-3 "${preprocess_command[@]}"
else
  echo "Unsupported tokenization launch backend: $TOKENIZATION_LAUNCH_BACKEND" >&2
  exit 1
fi

end=$(date +%s)
end_s=$(date)
echo "FINISH TIME: $(date) | Preprocessed $paths_file ! Stored in $output_folder"

# Stats
wc=$((end - start))

if [ "$TOKENIZATION_LAUNCH_BACKEND" = rcp ]; then
  dataset_total_size=$(python3 "$SCRIPT_DIR/compute_dump_size.py" "$paths_file")
else
  dataset_total_size=$(srun --environment="$ENV_FILE" python3 "$SCRIPT_DIR/compute_dump_size.py" "$paths_file")
fi

processed_total_size=$(du -shLb "$output_folder" | cut -f1)

bw=$(awk "BEGIN {print $dataset_total_size/$wc}")
total_tokens_processed=$(($(du -shLcb "$output_folder"/*.bin | tail -n1 | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/') / 4))
throughput=$(awk "BEGIN {print $total_tokens_processed/$wc}")

echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$NUMBER_OF_DATATROVE_TASKS,$wc,$bw,$total_tokens_processed,$throughput"
echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$NUMBER_OF_DATATROVE_TASKS,$wc,$bw,$total_tokens_processed,$throughput" >>"$CSV_RESULTS_FILE"

sleep 10
ls -lS "$output_folder"
mkdir -p "$COMPLETED_DUMPS_FOLDER"
mv "$paths_file" "$COMPLETED_DUMPS_FOLDER/"
