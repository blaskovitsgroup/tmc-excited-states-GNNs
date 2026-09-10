#!/usr/bin/env bash
set -euo pipefail
APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$APP/.." && pwd)"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    printf '%s\n' "The package environment is missing. Run: bash \"$APP/setup_painn.sh\"" >&2
    exit 1
fi
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -u predictor_webapp/app.py "$@"
