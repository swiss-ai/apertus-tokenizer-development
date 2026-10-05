#!/bin/bash
# Thin CPU-only Clariden wrapper around the standard tokenization worker.
set -euo pipefail
config=$1
paths_file=$2
runtime=$3
edf=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$config"
if [ "$(git -C "$runtime" rev-parse HEAD)" != fc1e5a86526e38c5f31aa6a099473f5a95b511e7 ]; then
  echo "The pinned provenance/reserve runtime is required" >&2
  exit 1
fi
relative=${paths_file#"$PATH_TO_PREPROCESSING_METADATA/dumps/"}
group=$(dirname "$relative")
dump=$(basename "$paths_file" .txt)
dump=${dump#paths_file_}
export PYTHONPATH="$runtime/src"
# The direct-process backend runs inside this already allocated EDF container.
export TOKENIZATION_LAUNCH_BACKEND=rcp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$script_dir/.."
srun --ntasks=1 --gpus=0 --gres=none --environment="$edf" \
  env PYTHONPATH="$runtime/src" TOKENIZATION_LAUNCH_BACKEND=rcp \
  bash "$script_dir/tokenize.sh" "$config" \
  "$DATASET_OUTPUT_FOLDER_NAME/$group/dump-$dump" \
  "$PATH_TO_OUTPUT_FOLDER/logs/$DATASET_NAME/$group/dump-$dump" \
  "$paths_file" "$PATH_TO_PREPROCESSING_METADATA/completed-dumps/$group"
