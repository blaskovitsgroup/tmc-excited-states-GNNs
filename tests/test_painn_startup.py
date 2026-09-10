from pathlib import Path
import socket
import subprocess
import sys

import pytest

from predictor_webapp import app


def test_missing_assets_are_explained(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(app, "ENSEMBLE_MANIFEST", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["app.py", "--selftest"])
    assert app.main() == 1
    assert "Restore the trained-model assets" in capsys.readouterr().err


def test_occupied_port_is_explained(monkeypatch, capsys):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        monkeypatch.setattr(sys, "argv", ["app.py", "--port", str(sock.getsockname()[1])])
        assert app.main() == 1
    assert "choose another port" in capsys.readouterr().err


def test_unavailable_cuda_is_explained(monkeypatch):
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="--device cpu"):
        app.Predictor("cuda")


def test_launcher_resolves_environment_from_other_directory(tmp_path):
    root = tmp_path / "package with spaces"
    (root / ".venv/bin").mkdir(parents=True)
    (root / "predictor_webapp").mkdir()
    launcher = root / "predictor_webapp/start_painn.sh"
    launcher.write_text((Path(__file__).parents[1] / "predictor_webapp/start_painn.sh").read_text())
    python = root / ".venv/bin/python"
    python.write_text('#!/bin/sh\npwd\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(["bash", str(launcher), "--port", "8766"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert str(root) in result.stdout
    assert "predictor_webapp/app.py\n--port\n8766" in result.stdout


def test_launcher_explains_missing_environment(tmp_path):
    launcher = tmp_path / "start_painn.sh"
    launcher.write_text((Path(__file__).parents[1] / "predictor_webapp/start_painn.sh").read_text())
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "setup_painn.sh" in result.stderr
