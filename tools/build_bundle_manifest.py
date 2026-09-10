"""Record SHA-256 hashes for a reviewed local handoff; run only when releasing."""
import json
from pathlib import Path
from verify_bundle import ROOT, sha256

EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv",
            "reproduction", "release-assets"}


def main():
    destination = ROOT / "publication/BUNDLE_MANIFEST.json"
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file() or path == destination or path.name == ".DS_Store":
            continue
        files[str(relative)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    destination.write_text(json.dumps({"schema_version": 1,
        "description": "collaborator handoff; paths relative to GitHub folder",
        "files": files}, indent=2) + "\n")
    print(f"Recorded {len(files)} files in {destination}")


if __name__ == "__main__":
    main()
