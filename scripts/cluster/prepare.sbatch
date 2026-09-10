#!/usr/bin/env bash
#SBATCH --job-name=tmqmg_prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
ROOT="${TMQMG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"
PYTHON="${TMQMG_PYTHON:-$ROOT/.venv/bin/python}"
SOURCE="${TMQMG_SOURCE_ROOT:-$ROOT/source_inputs}"
exec "$PYTHON" -m tmqmg_es.cli prepare --cache-dir "$ROOT/reproduction/cache" \
    --natqg-root "$SOURCE/natqg" --natqg-type "${1:-baseline}"
