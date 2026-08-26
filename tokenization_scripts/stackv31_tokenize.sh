#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_ROOT=$1
WORKERS=$2
: "${SLURM_ARRAY_TASK_ID:?}"

srun --environment="$SCRIPT_DIR/env.toml" \
  numactl --membind=0-3 \
  python3 "$SCRIPT_DIR/stackv31.py" run-assignment \
  --work-root "$WORK_ROOT" \
  --assignment "$SLURM_ARRAY_TASK_ID" \
  --workers "$WORKERS"
