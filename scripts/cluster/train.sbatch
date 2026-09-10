#!/usr/bin/env bash
#SBATCH --job-name=tmqmg_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
ROOT="${TMQMG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
PYTHON="${TMQMG_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${1:-painn_3d}"
SEEDS=(2737188456 1409281222 2696188251 3663425383 451851728)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -gt 4 ]; then
    printf '%s\n' 'Array index must be between 0 and 4.' >&2
    exit 1
fi
exec "$PYTHON" tools/retrain.py "$MODEL" --seed "${SEEDS[$INDEX]}"
