#!/usr/bin/env bash
#SBATCH --job-name=finetranslations-source-token-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
if [[ $# != 8 ]]; then
  printf 'usage: %s REPO PIPELINE TOKENIZER SOURCE PREPARED OUTPUT COMMIT PIPELINE_COMMIT\n' "$0" >&2
  exit 2
fi
repo="$1" pipeline="$2" tokenizer="$3" source="$4" prepared="$5" output="$6" commit="$7" pipeline_commit="$8"
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --porcelain)"
test -f "$prepared/_SUCCESS.json"
test ! -e "$output"
# PYTHONPATH supplies the pinned pipeline and runtime dependencies; the repository
# itself is prepended so the module and regression tests use this exact checkout.
export PYTHONPATH="$repo:$pipeline/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python="${TOKEN_REPAIR_PYTHON:-python}"
cd "$repo"
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --exact --gpus=0 --gres=none "$python" -m pytest -q tests/test_finetranslations_token_repair.py
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --exact --gpus=0 --gres=none "$python" -u -m tokenization_scripts.repair_finetranslations_tokens \
  --source-root "$source" --prepared-root "$prepared" --tokenizer "$tokenizer" \
  --output-root "$output" --dataset-id finetranslations-edu-span-bounded-ip-v1 \
  --producer-commit "$commit" --pipeline-commit "$pipeline_commit" --workers 64
