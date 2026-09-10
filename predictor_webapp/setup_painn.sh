#!/usr/bin/env bash
set -euo pipefail
APP="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$APP/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf '%s\n' 'Python 3.11 was not found. Install Python 3.11, then rerun: bash setup_painn.sh' >&2
    exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for this environment"'
if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Existing .venv must use Python 3.11"'
.venv/bin/python -m pip install -e '.[painn]'
.venv/bin/python -m pip check
printf '%s\n' "Installation complete. Start PaiNN with: bash \"$APP/start_painn.sh\""
