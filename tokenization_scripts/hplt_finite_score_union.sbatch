#!/bin/bash
# Qualify original supply early; seal after the recovered LC split completes.
set -euo pipefail
config=$1
runtime=$2
edf=$3
control=$4
original=$5
variant=$6
stage=${7:-seal}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$config"
output="$PATH_TO_OUTPUT_FOLDER/kept-unions/$variant"
tokenizer=$(realpath "$script_dir/$TOKENIZER")
srun --ntasks=1 --gpus=0 --gres=none --environment="$edf" \
  env PYTHONPATH="$runtime/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python "$script_dir/hplt_finite_score_union.py" "$stage" \
  --variant "$variant" --original "$original" \
  --recovered "$DATASET_OUTPUT_FOLDER_NAME.lc-reserve-v4" --output "$output" \
  --control "$control/union/$variant" --tokenizer "$tokenizer" --workers 24
