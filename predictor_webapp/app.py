#!/usr/bin/env python3
"""Local prediction web app for the finalized PaiNN ensemble.

Drop a .xyz of a transition-metal complex (+ net charge) into a local webpage
and get every property the surrogate predicts: 30 excitation energies and
oscillator strengths (gas phase + acetone), simulated absorption spectra,
UV/vis/nIR band maxima and broadness, visible charge-transfer class, NTO metal
fractions, solvatochromic shifts, and calibrated uncertainty + OOD flags.

Model: PaiNN coordinate-based 3D model, five-seed ensemble. This is the
finalized model configuration that runs from an XYZ geometry and formal
molecular charge without precomputed descriptors or NBO-derived graphs.

Usage:
    export KMP_DUPLICATE_LIB_OK=TRUE
    python app.py [--port 8765] [--device cpu|mps|cuda] [--no-browser]

Stdlib server only — no flask/gradio dependency. Heavy deps (torch, PyG) come
from the existing `tmqmg_es` environment.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # noqa: E402 (before torch)

import argparse
import hashlib
import json
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PACKAGE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PACKAGE_ROOT))  # import the finalized implementation

try:
    import numpy as np
    import torch
    from torch_geometric.data import Batch, Data
    from tmqmg_es import config as C
    from tmqmg_es.data.geometry import radius_graph
    from tmqmg_es.train.trainer import TrainConfig, build_model
except (ImportError, OSError) as exc:
    raise SystemExit(
        f"PaiNN dependencies could not be loaded: {exc}\n"
        f'Run: bash "{APP_DIR / "setup_painn.sh"}"\n'
        f'Then: bash "{APP_DIR / "start_painn.sh"}"'
    ) from exc

ENSEMBLE_MANIFEST = PACKAGE_ROOT / "manifests" / "final_ensemble_manifest.json"
PREDS_NPZ = PACKAGE_ROOT / "outputs" / "locked_predictions" / "painn_3d.npz"

# Elements present in the tmQMg training structures (ligands + 30 TMs).
# Anything outside this set has an untrained embedding -> hard warning.
TRAINING_ELEMENTS = set(
    "H B C N O F Si P S Cl As Se Br I".split()
) | set(C.TRANSITION_METALS)

MODEL_CARD = {
    "name": "PaiNN five-seed ensemble",
    "backbone": "PaiNN (equivariant coordinate-based 3D GNN)",
    "inputs": "XYZ atomic identities and coordinates + formal molecular charge",
    "n_seeds": 5,
    "joint_test_mae_eV": 0.126725296362022,
    "gasphase_test_mae_eV": 0.12980012360118187,
    "acetone_test_mae_eV": 0.12365046912286218,
    "gasphase_hit_0p2_eV": 0.806320987654321,
    "acetone_hit_0p2_eV": 0.8249831649831649,
    "reference": "tmQMg* TD-DFT labels, 74,273 mononuclear TMCs",
    "note": ("Uses the finalized identity-aware split. PaiNN is the neural "
             "configuration that needs only an XYZ geometry and formal charge; "
             "graph-fusion models require additional precomputed graph data."),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# XYZ parsing (tolerant text version of tmqmg_es.data.dataset.parse_xyz)
# --------------------------------------------------------------------------
def parse_xyz_text(text: str):
    """Return (symbols, Z[n], pos[n,3], comment). Raises ValueError with a
    user-readable message on malformed input."""
    lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # strip leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise ValueError("File is empty.")
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError("Not a valid XYZ file: first line must be the atom count.")
    comment = lines[1].strip() if len(lines) > 1 else ""
    atom_lines = [ln for ln in lines[2:] if ln.strip()]
    if len(atom_lines) < n:
        raise ValueError(f"XYZ declares {n} atoms but only {len(atom_lines)} atom lines found.")
    symbols, pos = [], np.empty((n, 3), dtype=np.float32)
    for i, ln in enumerate(atom_lines[:n]):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Atom line {i + 1} malformed: '{ln.strip()}'")
        sym = parts[0]
        if sym.isdigit():  # atomic number instead of symbol
            zi = int(sym)
            rev = {v: k for k, v in C.SYMBOL_TO_Z.items()}
            if zi not in rev:
                raise ValueError(f"Unknown atomic number {zi} on line {i + 1}.")
            sym = rev[zi]
        else:
            sym = sym[0].upper() + sym[1:].lower()
        if sym not in C.SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol '{parts[0]}' on atom line {i + 1}.")
        symbols.append(sym)
        try:
            pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            raise ValueError(f"Non-numeric coordinates on atom line {i + 1}.")
    z = np.array([C.SYMBOL_TO_Z[s] for s in symbols], dtype=np.int64)
    if n < 2:
        raise ValueError("Need at least 2 atoms.")
    if not np.isfinite(pos).all():
        raise ValueError("Coordinates contain NaN/inf.")
    return symbols, z, pos, comment


def detect_charge(comment: str):
    """Try to read a net charge from the XYZ comment line."""
    if not comment:
        return None
    m = re.search(r"(?:charge|chrg|q)\s*[=:]\s*(-?\d+)", comment, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # bare "charge multiplicity" pair, e.g. "-1 1"
    m = re.fullmatch(r"\s*(-?\d+)\s+(\d+)\s*", comment)
    if m:
        return int(m.group(1))
    return None


def formula_of(symbols):
    from collections import Counter
    cnt = Counter(symbols)
    order = sorted(cnt, key=lambda s: (s != "C", s != "H", s))
    return "".join(f"{s}{cnt[s] if cnt[s] > 1 else ''}" for s in order)


# --------------------------------------------------------------------------
# Model loading + inference
# --------------------------------------------------------------------------
class Predictor:
    def __init__(self, device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Start with --device cpu instead.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available. Start with --device cpu instead.")
        self.device = torch.device(device)
        self.lock = threading.Lock()
        manifest = json.loads(ENSEMBLE_MANIFEST.read_text())
        model_manifest = manifest["models"]["painn_3d"]
        members = [m for m in model_manifest["members"] if m["status"] == "valid"]
        if len(members) != MODEL_CARD["n_seeds"]:
            raise RuntimeError(
                f"Expected {MODEL_CARD['n_seeds']} valid PaiNN members; found {len(members)}."
            )
        self.models, self.cutoff = [], 5.0
        self.member_seeds = []
        for member in members:
            checkpoint_path = PACKAGE_ROOT / member["artifact"]
            if sha256_file(checkpoint_path) != member["artifact_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device,
                                  weights_only=False)
            except TypeError:  # older torch without weights_only kwarg
                ckpt = torch.load(checkpoint_path, map_location=self.device)
            cfg = TrainConfig(**ckpt["cfg"])
            if cfg.model_id != "painn_3d" or cfg.backbone != "painn" or cfg.regime != "A":
                raise RuntimeError(f"Unexpected model configuration in {checkpoint_path}")
            if int(cfg.seed) != int(member["seed"]):
                raise RuntimeError(f"Seed mismatch in {checkpoint_path}")
            self.cutoff = cfg.cutoff
            m = build_model(cfg).to(self.device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            self.models.append(m)
            self.member_seeds.append(int(member["seed"]))
        self.var_scale = float(
            model_manifest["validation_calibration"]["epistemic_variance_scale"]
        )
        # Reference distribution for the same molecule-level ensemble-
        # disagreement statistic. It is diagnostic, not a validated OOD test.
        self.ood_ref = None
        if PREDS_NPZ.exists():
            locked = np.load(PREDS_NPZ, allow_pickle=False)
            member_predictions = locked["energy_prediction"]  # [K,N,2,30]
            self.ood_ref = np.sort(
                member_predictions.std(axis=0, ddof=0).mean(axis=(1, 2))
            )
        print(f"[app] loaded {len(self.models)} PaiNN seeds on {self.device}, "
              f"cutoff={self.cutoff}, var_scale={self.var_scale:.3f}")

    # ---- graph construction (mirrors tmqmg_es.external.build_external_data)
    def build_data(self, z, pos, charge: float) -> tuple[Data, dict]:
        ei = radius_graph(pos, self.cutoff)
        metal_mask = np.isin(z, [C.SYMBOL_TO_Z[m] for m in C.TRANSITION_METALS])
        n_metals = int(metal_mask.sum())
        metal_idx = int(np.argmax(metal_mask)) if n_metals else 0
        d = Data(z=torch.tensor(z, dtype=torch.long),
                 pos=torch.tensor(pos, dtype=torch.float32),
                 edge_index=torch.tensor(ei, dtype=torch.long))
        d.num_nodes = len(z)
        d.metal_idx = torch.tensor([metal_idx], dtype=torch.long)
        d.metal_z = torch.tensor([int(z[metal_idx])], dtype=torch.long)
        d.charge = torch.tensor([float(charge)], dtype=torch.float32)
        d.n_atoms = torch.tensor([len(z)], dtype=torch.long)
        meta = {"n_metals": n_metals,
                "metal_symbol": None if not n_metals else
                [s for s, zz in C.SYMBOL_TO_Z.items() if zz == z[metal_idx]][0]}
        return d, meta

    @torch.no_grad()
    def predict(self, z, pos, charge: float) -> dict:
        data, meta = self.build_data(z, pos, charge)
        batch = Batch.from_data_list([data]).to(self.device)
        outs = []
        with self.lock:
            for m in self.models:
                o = m(batch)
                outs.append(o)

        res = {"meta": meta}
        for s in C.SOLVENTS:
            E = np.stack([o[s]["E"][0].cpu().numpy() for o in outs])          # [K,30]
            logf = np.stack([o[s]["logf"][0].cpu().numpy() for o in outs])    # [K,30]
            band = np.stack([torch.sigmoid(o[s]["band_logits"][0]).cpu().numpy()
                             for o in outs])
            peak_E = np.stack([o[s]["peak_E"][0].cpu().numpy() for o in outs])  # [K,3]
            peak_logf = np.stack([o[s]["peak_f"][0].cpu().numpy() for o in outs])
            peak_logwidth = np.stack([o[s]["sigma"][0].cpu().numpy() for o in outs])
            ct = np.stack([torch.softmax(o[s]["ct_logits"][0], -1).cpu().numpy()
                           for o in outs])                                    # [K,4]
            mfrac = np.stack([o[s]["mfrac"][0].cpu().numpy() for o in outs])  # [K,2]

            E_mean = E.mean(0)
            E_std_raw = E.std(0)                                   # epistemic
            E_std = np.sqrt(E_std_raw ** 2 * self.var_scale)       # recalibrated
            # Match locked_eval.py: aggregate ensemble members in transformed
            # space, then invert log1p for user-facing oscillator strengths.
            f_mean = np.expm1(np.clip(logf.mean(0), None, 20.0))
            pk_E = peak_E.mean(0)
            pk_f = np.expm1(np.clip(peak_logf.mean(0), None, 20.0))
            pk_sig = np.expm1(np.clip(peak_logwidth.mean(0), None, 20.0))
            band_prob = band.mean(0)

            peaks = {}
            for ri, r in enumerate(C.REGIONS):
                exists = bool(band_prob[ri] >= 0.5)
                peaks[r] = {
                    "exists": exists,
                    "presence_probability": float(band_prob[ri]),
                    "E_max_eV": float(pk_E[ri]),
                    "lambda_max_nm": float(C.HC_EV_NM / pk_E[ri]) if pk_E[ri] > 1e-3 else None,
                    "f_max": float(pk_f[ri]),
                    "broadness_sigma": float(pk_sig[ri]),
                }
            ctp = ct.mean(0)
            grid = np.linspace(0.75, 8.5, 500)
            inten = (f_mean[None, :] * np.exp(
                -(grid[:, None] - E_mean[None, :]) ** 2 / (2 * 0.2 ** 2))).sum(1)

            res[s] = {
                "E_eV": E_mean.tolist(),
                "E_std_eV": E_std.tolist(),
                "lambda_nm": (C.HC_EV_NM / np.clip(E_mean, 1e-3, None)).tolist(),
                "f": f_mean.tolist(),
                "peaks": peaks,
                "ct": {"probs": {c: float(p) for c, p in zip(C.CT_CLASSES, ctp)},
                       "top": C.CT_CLASSES[int(ctp.argmax())],
                       "valid": peaks["vis"]["exists"]},
                "nto": {"M_frac_occ": float(mfrac.mean(0)[0]),
                        "M_frac_virt": float(mfrac.mean(0)[1]),
                        "valid": peaks["vis"]["exists"]},
                "spectrum": {"E_grid_eV": grid.tolist(),
                             "intensity": inten.tolist()},
                "_ood_raw": float(np.mean(E_std_raw)),
            }

        gas_vis = res["gasphase"]["peaks"]["vis"]
        ace_vis = res["acetone"]["peaks"]["vis"]
        if gas_vis["exists"] and ace_vis["exists"]:
            sh_l = float(ace_vis["lambda_max_nm"] - gas_vis["lambda_max_nm"])
            sh_f = float(ace_vis["f_max"] - gas_vis["f_max"])
            direction = ("bathochromic (red shift)" if sh_l > 5 else
                         "hypsochromic (blue shift)" if sh_l < -5 else "small shift")
        else:
            sh_l, sh_f, direction = None, None, "not defined"
        res["shift"] = {"lambda_delta_nm": sh_l, "f_delta": sh_f,
                        "direction": direction,
                        "note": ("acetone minus gas phase, visible-region lambda_max; "
                                 "defined only when both visible bands are predicted present")}

        # Percentile of joint gas/acetone disagreement against the test set.
        ood_raw = float(np.mean([res[s].pop("_ood_raw") for s in C.SOLVENTS]))
        pct = None
        if self.ood_ref is not None and len(self.ood_ref):
            pct = float(np.searchsorted(self.ood_ref, ood_raw) / len(self.ood_ref) * 100)
        res["ood"] = {"mean_ensemble_std_eV": ood_raw, "percentile_vs_test": pct}
        return res


PREDICTOR: Predictor | None = None


def run_prediction(payload: dict) -> dict:
    xyz_text = payload.get("xyz", "")
    filename = payload.get("filename", "molecule.xyz")
    symbols, z, pos, comment = parse_xyz_text(xyz_text)

    charge = payload.get("charge", None)
    detected = detect_charge(comment)
    if charge is None or charge == "":
        charge = detected if detected is not None else 0
    charge = int(charge)

    warnings = []
    elems = sorted(set(symbols))
    unknown = [e for e in elems if e not in TRAINING_ELEMENTS]
    if unknown:
        warnings.append({"level": "error",
                         "text": f"Element(s) {', '.join(unknown)} never seen in training "
                                 "— predictions for this molecule are unreliable."})
    n_tm = sum(1 for s in symbols if s in C.TRANSITION_METALS)
    if n_tm == 0:
        warnings.append({"level": "error",
                         "text": "No transition metal found. The model was trained only on "
                                 "mononuclear transition-metal complexes."})
    elif n_tm > 1:
        warnings.append({"level": "warn",
                         "text": f"{n_tm} transition-metal atoms found; training data is "
                                 "mononuclear. Readout is centered on the first metal."})
    if len(symbols) > 120:
        warnings.append({"level": "warn",
                         "text": f"{len(symbols)} atoms — larger than typical training "
                                 "molecules; treat with caution."})
    if charge not in (-1, 0, 1):
        warnings.append({"level": "warn", "text": f"Charge {charge:+d} is outside the "
                        "training range (-1, 0, +1); this prediction is extrapolative."})

    res = PREDICTOR.predict(z, pos, charge)

    pct = res["ood"]["percentile_vs_test"]
    if pct is not None:
        if pct >= 99:
            warnings.append({"level": "warn",
                             "text": "Ensemble disagreement is above the 99th percentile "
                                     "of the test reference. This indicates unusually "
                                     "high model disagreement, but is not proof of OOD status."})
        elif pct >= 90:
            warnings.append({"level": "warn",
                             "text": "Ensemble disagreement above the 90th percentile of "
                                     "the in-domain test set — elevated uncertainty."})

    res.update({
        "ok": True,
        "molecule": {
            "filename": filename,
            "formula": formula_of(symbols),
            "n_atoms": len(symbols),
            "charge": charge,
            "charge_autodetected": detected is not None and payload.get("charge") in (None, ""),
            "metal": res["meta"]["metal_symbol"],
            "elements": elems,
        },
        "model": MODEL_CARD,
        "warnings": warnings,
    })
    res.pop("meta", None)
    return res


# --------------------------------------------------------------------------
# HTTP server (stdlib)
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # allow the page to call the API even if it was opened as a file://
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS preflight (file:// -> http://127.0.0.1)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (APP_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path == "/example.xyz" and (APP_DIR / "example.xyz").exists():
            self._send(200, (APP_DIR / "example.xyz").read_bytes(), "text/plain")
        elif self.path == "/api/info":
            self._json(200, {"model": MODEL_CARD})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self._json(404, {"ok": False, "error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 10_000_000:
                return self._json(413, {"ok": False, "error": "File too large."})
            payload = json.loads(self.rfile.read(length))
            return self._json(200, run_prediction(payload))
        except ValueError as e:
            return self._json(400, {"ok": False, "error": str(e)})
        except Exception:
            traceback.print_exc()
            return self._json(500, {"ok": False,
                                    "error": "Internal error — see terminal for details."})

    def log_message(self, fmt, *args):  # quiet
        pass


def selftest() -> int:
    """Reproduce the stored ensemble predictions for bundled test ID ABAJOF."""
    global PREDICTOR
    PREDICTOR = Predictor("cpu")
    mid = "ABAJOF"
    xyz = (PROJECT_ROOT / "examples" / f"{mid}.xyz").read_text()
    res = run_prediction({"xyz": xyz, "filename": f"{mid}.xyz", "charge": 0})
    d = np.load(PREDS_NPZ, allow_pickle=False)
    i = int(np.where(d["molecule_ids"].astype(str) == mid)[0][0])
    ok = True
    for s in C.SOLVENTS:
        solvent_index = C.SOLVENTS.index(s)
        stored = d["energy_prediction"][:, i, solvent_index, :].mean(axis=0)
        dE = float(np.abs(np.array(res[s]["E_eV"]) - stored).max())
        print(f"[selftest] {s}: max |E_app - E_stored| = {dE:.2e} eV")
        ok &= dE < 1e-4
        stored_f = np.expm1(d["logf_prediction"][:, i, solvent_index, :].mean(axis=0))
        df = float(np.abs(np.array(res[s]["f"]) - stored_f).max())
        print(f"[selftest] {s}: max |f_app - f_stored| = {df:.2e}")
        ok &= df < 1e-5
        for region_index, region in enumerate(C.REGIONS):
            stored_band = d["band_probability"][:, i, solvent_index, region_index].mean()
            db = abs(res[s]["peaks"][region]["presence_probability"] - stored_band)
            ok &= db < 1e-5
    print("[selftest]", "PASS — app reproduces the trained ensemble exactly."
          if ok else "FAIL — predictions deviate from stored ensemble output!")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="PaiNN prediction webpage")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the app reproduces stored ensemble predictions, then exit")
    args = ap.parse_args()

    if not 1 <= args.port <= 65535:
        ap.error("Port must be between 1 and 65535.")

    global PREDICTOR
    server = None
    try:
        if args.selftest:
            return selftest()
        try:
            server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        except OSError as exc:
            raise ValueError(
                f"Cannot open port {args.port}: {exc}. Stop the existing server "
                "or choose another port, for example --port 8766."
            ) from exc
        print("[app] loading ensemble (5 x PaiNN) ...", flush=True)
        PREDICTOR = Predictor(args.device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if server is not None:
            server.server_close()
        print(f"[app] Cannot start: {exc}", file=sys.stderr)
        if isinstance(exc, FileNotFoundError):
            print("Restore the trained-model assets described in README.md. "
                  "A source-code-only download cannot run predictions.", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}"
    print(f"[app] ready  ->  {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
