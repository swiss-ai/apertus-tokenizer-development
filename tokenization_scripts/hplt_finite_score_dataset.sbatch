#!/bin/bash
# One whole-node allocation per variant, reusing the standard dump workers.
set -euo pipefail
config=$1
runtime=$2
edf=$3
control=$4
stage=${5:-all}
case "$stage" in
  all|prepare|finalize) ;;
  *) echo "Stage must be all, prepare or finalize" >&2; exit 2 ;;
esac
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Slurm executes a spool copy; the config remains in the repository.
if [ ! -f "$script_dir/tokenize_script.sh" ]; then
  script_dir=$(cd "$(dirname "$config")/.." && pwd)
fi
if [ "${HPLT_INSIDE_EDF:-0}" != 1 ]; then
  srun --ntasks=1 --gpus=0 --gres=none --environment="$edf" \
    env HPLT_INSIDE_EDF=1 PYTHONPATH="$runtime/src" TOKENIZATION_LAUNCH_BACKEND=rcp \
    bash "$script_dir/hplt_finite_score_dataset.sbatch" "$config" "$runtime" "$edf" "$control" "$stage"
  exit
fi
source "$config"
if [ "$(git -C "$runtime" rev-parse HEAD)" != fc1e5a86526e38c5f31aa6a099473f5a95b511e7 ]; then
  echo "Pinned token-map/runtime checkout is required" >&2
  exit 1
fi
cd "$script_dir/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
tokenizer=$(realpath "$script_dir/$TOKENIZER")
resume_args=(--metadata "$PATH_TO_PREPROCESSING_METADATA" --config "$config"
  --marker "$REQUIRED_DATASET_MARKER" --manifest "$DATASET_MANIFEST"
  --tokenizer "$tokenizer" --runtime-commit "$(git -C "$runtime" rev-parse HEAD)"
  --implementation-commit "$(git -C "$script_dir/.." rev-parse HEAD)")
if [ ! -f "$PATH_TO_PREPROCESSING_METADATA/HPLT_DUMPS.json" ]; then
  bash "$script_dir/tokenize_script.sh" "$config" --prepare-only
  python "$script_dir/hplt_finite_score_resume.py" freeze "${resume_args[@]}"
fi
python "$script_dir/hplt_finite_score_resume.py" check "${resume_args[@]}"
if [ "$stage" = prepare ]; then
  exit
fi
if [ "$stage" = all ]; then
while IFS= read -r paths_file; do
  relative=${paths_file#"$PATH_TO_PREPROCESSING_METADATA/dumps/"}
  group=$(dirname "$relative")
  dump=$(basename "$paths_file" .txt)
  dump=${dump#paths_file_}
  bash "$script_dir/tokenize.sh" "$config" \
    "$DATASET_OUTPUT_FOLDER_NAME/$group/dump-$dump" \
    "$PATH_TO_OUTPUT_FOLDER/logs/$DATASET_NAME/$group/dump-$dump" \
    "$paths_file" "$PATH_TO_PREPROCESSING_METADATA/completed-dumps/$group"
done < <(find "$PATH_TO_PREPROCESSING_METADATA/dumps" -type f -name 'paths_file_*.txt' -print | sort)
fi
python "$script_dir/hplt_finite_score_resume.py" check "${resume_args[@]}"
if [ -n "$(find "$PATH_TO_PREPROCESSING_METADATA/dumps" -type f -name 'paths_file_*.txt' -print -quit)" ]; then
  echo "Pending dump manifests prevent finalization" >&2
  exit 1
fi
if [ ! -f "$DATASET_OUTPUT_FOLDER_NAME/_SUCCESS.json" ]; then
  bash "$script_dir/validate_tokenization.sh" "$config"
fi
reserve="$runtime/pipelines/lc-reserve/main_1_reserve.py"
root=$DATASET_OUTPUT_FOLDER_NAME
reserve_control="$control/reserve/$DATASET_NAME"
if [ ! -f "$root.lc-reserve-v4/_SPLIT_SUCCESS.json" ]; then
  python "$reserve" histogram --root "$root" --control "$reserve_control" --workers 24
  python "$reserve" plan --root "$root" --control "$reserve_control" --fraction 0.10 --seed 20260907 --dataset-id "$DATASET_NAME"
  python "$reserve" split --control "$reserve_control" --output-root "$root.lc-reserve-v4" --workers 24
  python "$reserve" verify --control "$reserve_control" --output-root "$root.lc-reserve-v4" --workers 24
  python "$reserve" finalize --control "$reserve_control" --output-root "$root.lc-reserve-v4"
fi
