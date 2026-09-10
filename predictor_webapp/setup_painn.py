"""Install the package into a local environment on Windows, macOS or Linux."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

from run_painn import environment_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="also install tests and plotting tools")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Run setup with Python 3.11.")
    root = Path(__file__).resolve().parents[1]
    python = environment_python(root)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(root / ".venv")
    try:
        subprocess.run([str(python), "-c", 'import sys; assert sys.version_info[:2] == (3,11), "Existing environment must use Python 3.11"'], check=True)
        extra = ".[painn,dev]" if args.dev else ".[painn]"
        subprocess.run([str(python), "-m", "pip", "install", "-e", extra], cwd=root, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
    except subprocess.CalledProcessError as exc:
        print("Installation did not complete. Resolve the error above before launching PaiNN.", file=sys.stderr)
        return exc.returncode
    print("Installation complete. Run run_painn.py with Python.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
