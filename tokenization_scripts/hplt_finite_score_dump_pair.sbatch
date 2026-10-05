#!/bin/bash
# Two independent standard dump workers use disjoint 128-core steps on one node.
set -euo pipefail
config=$1
runtime=$2
edf=$3
shift 3
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] || { [ "$#" -eq 2 ] && [ "$1" = "$2" ]; }; then
  echo "Pass one or two distinct prepared paths-file manifests" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$config"
if [ "${SLURM_CPUS_PER_TASK:-0}" -lt "$(( $# * CPUS_PER_TASK ))" ]; then
  echo "Allocation cannot hold the requested dump CPU steps" >&2
  exit 2
fi
pids=()
for paths_file in "$@"; do
  bash "$script_dir/hplt_finite_score_worker.sbatch" "$config" "$paths_file" "$runtime" "$edf" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
