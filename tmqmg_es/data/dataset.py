"""PyG dataset: XYZ geometry -> graph with excited-state targets and masks.

One ``Data`` per molecule. Geometry is solvent-independent; solvent-specific
targets are stored along a leading axis (index 0 = gasphase, 1 = acetone) so the
backbone runs once and the decoder is conditioned on the solvent token at
read-out (workflow sections 5, 8, 10).

Radius-graph edges are precomputed here (pure torch, no torch_cluster) so the
same code runs on Mac MPS and cluster CUDA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .. import config as C
from .geometry import parse_xyz, radius_graph, symbol_to_z


_CT_INDEX = {c: i for i, c in enumerate(C.CT_CLASSES)}
PREPROCESSING_VERSION = "tmqmg-preprocess-2-undirected-graphs"
CACHE_SCHEMA_VERSION = 3


def _solvent_targets(row: pd.Series, solvent: str) -> dict:
    E = [row[f"E_{i}_{solvent}_eV"] for i in range(1, C.N_STATES + 1)]
    logf = [row[f"logf_{i}_{solvent}"] for i in range(1, C.N_STATES + 1)]
    peak_E = [row.get(f"E_max_{r}_{solvent}_eV", np.nan) for r in C.REGIONS]
    peak_f = [np.log1p(row.get(f"f_max_{r}_{solvent}", np.nan)) for r in C.REGIONS]
    peak_mask = [bool(row[f"mask_peak_{r}_{solvent}"]) for r in C.REGIONS]
    sigma = [row.get(f"sigma_{r}_{solvent}", np.nan) for r in C.REGIONS]
    ct = row[f"transition_nature_vis_{solvent}"]
    ct_idx = _CT_INDEX.get(ct, -1)
    mfrac = [
        row.get(f"M_frac_occ_{solvent}", np.nan),
        row.get(f"M_frac_virt_{solvent}", np.nan),
    ]
    return {
        "E": E,
        "logf": logf,
        "peak_E": peak_E,
        "peak_f": peak_f,
        "peak_mask": peak_mask,
        "band": peak_mask,
        "sigma": sigma,
        "ct": ct_idx,
        "ct_mask": bool(row[f"mask_ct_{solvent}"]),
        "mfrac": mfrac,
        "nto_mask": bool(row[f"mask_nto_{solvent}"]),
    }


def build_data(
    row: pd.Series, xyz_dir: Path, cutoff: float, natqg_dir: Path | None = None
) -> Data | None:
    path = xyz_dir / f"{row['id']}.xyz"
    if not path.exists():
        return None
    z, pos = parse_xyz(path)
    ei = radius_graph(pos, cutoff)

    nq = None
    if natqg_dir is not None:
        from .natqg import parse_gml

        # graph type from the directory name, e.g. ".../d-NatQ_graphs" -> "d-NatQ"
        gtype = Path(natqg_dir).name.replace("_graphs", "")
        gml = Path(natqg_dir) / f"{row['id']}.gml"
        if not gml.exists() or (nq := parse_gml(gml, gtype))["n_nodes"] != len(z):
            return None  # require an aligned NatQG graph for Regime C

    gas = _solvent_targets(row, "gasphase")
    ace = _solvent_targets(row, "acetone")

    def stack(key):
        # leading batch dim of 1 so PyG collates graph-level targets to [B, 2, ...]
        return torch.tensor([gas[key], ace[key]], dtype=torch.float32).unsqueeze(0)

    metal_is = np.isin(z, [symbol_to_z(m) for m in C.TRANSITION_METALS])
    metal_idx = int(np.argmax(metal_is)) if metal_is.any() else 0

    data = Data(
        z=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(ei, dtype=torch.long),
    )
    data.num_nodes = len(z)
    data.metal_idx = torch.tensor([metal_idx], dtype=torch.long)
    data.metal_z = torch.tensor([int(z[metal_idx])], dtype=torch.long)
    data.charge = torch.tensor([float(row["charge"])], dtype=torch.float32)
    data.n_atoms = torch.tensor([len(z)], dtype=torch.long)
    data.mol_id = row["id"]

    data.E = stack("E")  # [1, 2, 30]
    data.logf = stack("logf")  # [1, 2, 30]
    data.peak_E = stack("peak_E")  # [1, 2, 3]
    data.peak_f = stack("peak_f")  # [1, 2, 3]
    data.peak_mask = torch.tensor(
        [[gas["peak_mask"], ace["peak_mask"]]], dtype=torch.bool
    )
    data.band = stack("band")  # [1, 2, 3], all molecules supervised
    data.sigma = stack("sigma")  # [1, 2, 3]
    data.ct = torch.tensor([[gas["ct"], ace["ct"]]], dtype=torch.long)  # [1, 2]
    data.ct_mask = torch.tensor([[gas["ct_mask"], ace["ct_mask"]]], dtype=torch.bool)
    data.mfrac = stack("mfrac")  # [1, 2, 2]
    data.nto_mask = torch.tensor([[gas["nto_mask"], ace["nto_mask"]]], dtype=torch.bool)
    if nq is not None:
        # NatQG nodes align 1:1 with atoms, so the 3D `batch` vector also pools
        # the NatQG branch; nq_edge_index auto-increments by num_nodes.
        data.nq_x = torch.tensor(nq["node_feat"], dtype=torch.float32)
        data.nq_edge_index = torch.tensor(nq["edge_index"], dtype=torch.long)
        data.nq_edge_attr = torch.tensor(nq["edge_feat"], dtype=torch.float32)
    return data


class TMQMGDataset(InMemoryDataset):
    """In-memory dataset built from the cleaned parquet + XYZ directory."""

    def __init__(
        self,
        root: str | Path,
        cleaned: pd.DataFrame | None = None,
        ids: list[str] | None = None,
        cutoff: float = 5.0,
        tag: str = "full",
        natqg_dir: Path | None = None,
        source_fingerprint: str | None = None,
    ):
        self._cleaned = cleaned
        self._ids = ids
        self._cutoff = cutoff
        self._tag = tag
        self._natqg_dir = natqg_dir
        self._source_fingerprint = source_fingerprint or self._fallback_fingerprint()
        super().__init__(str(root))
        self._validate_metadata()
        self.load(self.processed_paths[0])

    def _fallback_fingerprint(self) -> str:
        """Stable test/smoke fallback; production passes a source manifest hash."""
        digest = hashlib.sha256()
        digest.update(PREPROCESSING_VERSION.encode("ascii"))
        digest.update(str(self._cutoff).encode("ascii"))
        digest.update(str(self._natqg_dir).encode("utf-8"))
        if self._ids is not None:
            for molecule_id in sorted(self._ids):
                digest.update(str(molecule_id).encode("utf-8"))
                digest.update(b"\n")
        if self._cleaned is not None:
            digest.update(
                pd.util.hash_pandas_object(
                    self._cleaned.sort_values("id", kind="stable"), index=False
                ).values.tobytes()
            )
        return digest.hexdigest()

    @property
    def processed_file_names(self):
        short = self._source_fingerprint[:16]
        return [f"tmqmg_{self._tag}_{short}_c{self._cutoff}.pt"]

    @property
    def metadata_path(self) -> Path:
        return Path(self.processed_paths[0]).with_suffix(".metadata.json")

    def _expected_metadata(self) -> dict:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "preprocessing_version": PREPROCESSING_VERSION,
            "source_fingerprint": self._source_fingerprint,
            "tag": self._tag,
            "cutoff_angstrom": self._cutoff,
            "natqg_dir": None if self._natqg_dir is None else str(self._natqg_dir),
        }

    def _validate_metadata(self) -> None:
        if not Path(self.processed_paths[0]).exists():
            return
        if not self.metadata_path.exists():
            raise RuntimeError(
                f"cache exists without provenance metadata: {self.processed_paths[0]}"
            )
        actual = json.loads(self.metadata_path.read_text())
        expected = self._expected_metadata()
        if actual != expected:
            raise RuntimeError(
                "cache provenance mismatch; refusing stale cache\n"
                f"expected={expected}\nactual={actual}"
            )

    def process(self):
        cleaned = self._cleaned
        if cleaned is None:
            cleaned = pd.read_parquet(C.PHASE1_DIR / "cleaned_tmQMg_star.parquet")
        if self._ids is not None:
            cleaned = cleaned[cleaned["id"].isin(set(self._ids))]
        data_list, order = [], []
        for _, row in cleaned.iterrows():
            d = build_data(row, C.XYZ_DIR, self._cutoff, natqg_dir=self._natqg_dir)
            if d is not None:
                data_list.append(d)
                order.append(row["id"])
        Path(self.processed_paths[0]).with_suffix(".ids.json").write_text(
            json.dumps(order)
        )
        self.save(data_list, self.processed_paths[0])
        self.metadata_path.write_text(
            json.dumps(self._expected_metadata(), indent=2) + "\n"
        )

    def id_to_index(self) -> dict[str, int]:
        """Map molecule id -> position in this dataset (from the sidecar)."""
        order = json.loads(
            Path(self.processed_paths[0]).with_suffix(".ids.json").read_text()
        )
        return {mid: i for i, mid in enumerate(order)}
