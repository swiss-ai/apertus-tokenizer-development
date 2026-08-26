#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_ROOT=$1
MAP_ROOT=$2
MAX_MAP_OVERHEAD=$3
MIXTURE_SAMPLER_PYTHON=$4

srun --environment="$SCRIPT_DIR/env.toml" \
  env PYTHONPATH="$MIXTURE_SAMPLER_PYTHON:${PYTHONPATH:-}" \
  python3 "$SCRIPT_DIR/stackv31.py" validate \
  --work-root "$WORK_ROOT" \
  --map-root "$MAP_ROOT" \
  --max-map-overhead "$MAX_MAP_OVERHEAD" \
  --require-assignment-markers
