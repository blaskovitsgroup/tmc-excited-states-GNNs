"""Launch PaiNN with the package environment, without shell activation."""
import os
from pathlib import Path
import subprocess
import sys


def environment_python(root, windows=None):
    windows = os.name == "nt" if windows is None else windows
    return Path(root) / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def main():
    root = Path(__file__).resolve().parents[1]
    python = environment_python(root)
    if not python.is_file():
        print("Package environment missing. Run setup_painn.py with Python 3.11 first.", file=sys.stderr)
        return 1
    try:
        return subprocess.call([str(python), "-u", str(root / "predictor_webapp/app.py"), *sys.argv[1:]], cwd=root)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
