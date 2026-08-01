#!/bin/bash

#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --time=4:00:00

ENV_FILE="./env.toml"

# Parse arguments
# First argument is output prefix
# Remaining arguments are input directories
output_prefix=$1
shift
input_dirs=("$@")

if [ -z "$output_prefix" ] || [ ${#input_dirs[@]} -eq 0 ]; then
  echo "Error: Not enough arguments"
  echo "Usage: $0 <output-prefix> <input-dir1> [input-dir2 ...]"
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
