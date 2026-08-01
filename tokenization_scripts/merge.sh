#!/bin/bash

#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --time=4:00:00

ENV_FILE="./env.toml"

# Parse arguments
# First argument is output prefix
# Second argument is comma-separated dump IDs
# Third argument is completed dumps folder path
# Fourth argument is merged completed dumps folder path
# Remaining arguments are input directories
output_prefix=$1
dump_ids_str=$2
completed_dumps_folder=$3
merged_completed_dumps_folder=$4
shift 4
input_dirs=("$@")

if [ -z "$output_prefix" ] || [ -z "$dump_ids_str" ] || [ -z "$completed_dumps_folder" ] || [ -z "$merged_completed_dumps_folder" ] || [ ${#input_dirs[@]} -eq 0 ]; then
  echo "Error: Not enough arguments"
  echo "Usage: $0 <output-prefix> <dump-ids-csv> <completed-dumps-folder> <merged-completed-dumps-folder> <input-dir1> [input-dir2 ...]"
  exit 1
fi

set -eo pipefail

# Setup ENV
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "START TIME: $(date) | Merging ${#input_dirs[@]} dumps into $output_prefix"
echo "Input directories: ${input_dirs[@]}"
start_s=$(date)
start=$(date +%s)

# Run merge with srun --environment to execute inside the container
srun --environment=$ENV_FILE \
  python3 merge_tokenized_datasets.py \
  --input-dirs "${input_dirs[@]}" \
  --output-prefix "$output_prefix"

end=$(date +%s)
end_s=$(date)
echo "FINISH TIME: $(date) | Merged into $output_prefix"

# Stats
wc=$((end - start))

# Calculate merged output size
merged_size=$(du -shLb ${output_prefix}* 2>/dev/null | awk '{sum+=$1} END {print sum}')

# Count total tokens in merged files (.bin files, each token is 4 bytes)
total_tokens_merged=$(($(du -shLcb ${output_prefix}*.bin 2>/dev/null | tail -n1 | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/' || echo 0) / 4))

echo "=== Merge Statistics ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $start_s"
echo "End: $end_s"
echo "Duration: $wc seconds"
echo "Input directories: ${#input_dirs[@]}"
echo "Output prefix: $output_prefix"
echo "Merged size: $merged_size bytes"
echo "Total tokens: $total_tokens_merged"
echo "======================="

sleep 10
ls -lh ${output_prefix}*

# Now copy completed dumps to merged completed dumps folder
# Extract bucket number from output prefix
bucket_num=$(basename "$output_prefix" | grep -oP '(?<=bucket_)\d+')

if [ -n "$bucket_num" ]; then
  BUCKET_COMPLETED_DUMPS="$merged_completed_dumps_folder/bucket_$bucket_num.txt"
  
  echo "=== Copying completed dumps metadata ==="
  echo "Creating: $BUCKET_COMPLETED_DUMPS"
  
  # Clear/create file
  > "$BUCKET_COMPLETED_DUMPS"
  
  # Convert comma-separated dump IDs to array
  IFS=',' read -ra dump_ids <<< "$dump_ids_str"
  
  # Collect all parquet files that compose this bucket (from completed-dumps)
  for dump_id in "${dump_ids[@]}"; do
    # The completed-dumps folder contains paths_file_<dump_id>.txt files
    completed_dump_file="$completed_dumps_folder/paths_file_$dump_id.txt"
    if [ -f "$completed_dump_file" ]; then
      cat "$completed_dump_file" >> "$BUCKET_COMPLETED_DUMPS"
      echo "  Added dump $dump_id ($(wc -l < "$completed_dump_file") files)"
    else
      echo "  Warning: Completed dumps file not found: $completed_dump_file"
    fi
  done
  
  total_files=$(wc -l < "$BUCKET_COMPLETED_DUMPS")
  echo "Total parquet files in bucket $bucket_num: $total_files"
  echo "========================================"
else
  echo "Warning: Could not extract bucket number from output prefix: $output_prefix"
fi
