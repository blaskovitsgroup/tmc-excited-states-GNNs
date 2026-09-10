"""Check handoff contents against SHA-256 hashes without changing files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "publication/BUNDLE_MANIFEST.json").read_text())
    failed = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected["sha256"]:
            failed.append(relative)
    if failed:
        raise SystemExit("Missing or changed files:\n" + "\n".join(failed))
    print(f"PASS: {len(manifest['files'])} files match the handoff manifest.")


if __name__ == "__main__":
    main()
