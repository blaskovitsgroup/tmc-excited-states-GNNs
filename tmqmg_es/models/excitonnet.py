"""Solvent-conditioned multitask excited-state model.

3D backbone -> metal-centered attention readout -> per-solvent conditioning ->
multitask heads (states, band existence, peaks, CT, and NTO). Solvent shifts
are derived from paired predictions rather than learned as a direct target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import softmax

from .backbone import (
    PaiNNBackbone,
    SchNetBackbone,
)
from .heads import BandHead, CTHead, NTOHead, PeakHead, StateDecoder
from .natqg_encoder import NatQGEncoder

BACKBONES = {
    "schnet": SchNetBackbone,
    "painn": PaiNNBackbone,
}


class MetalAttentionReadout(nn.Module):
    """Mean pool + metal-atom embedding + metal-query attention pool."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out_dim = 3 * hidden

    def forward(self, h, batch, ptr, metal_idx):
        # global index of the metal atom in each graph
        metal_global = ptr[:-1] + metal_idx
        metal_emb = h[metal_global]  # [B, H]

        # mean pool per graph
        n_graphs = ptr.numel() - 1
        counts = (ptr[1:] - ptr[:-1]).clamp(min=1).float()[:, None]
        mean = torch.zeros(n_graphs, h.size(1), device=h.device)
        mean.index_add_(0, batch, h)
        mean = mean / counts

        # metal-query attention over all atoms
        q = self.q(metal_emb)[batch]  # broadcast query to nodes
        score = (q * self.k(h)).sum(-1) / (h.size(1) ** 0.5)
        alpha = softmax(score, batch)  # softmax within graph
        weighted = self.v(h) * alpha[:, None]
        attn = torch.zeros(n_graphs, h.size(1), device=h.device)
        attn.index_add_(0, batch, weighted)

        return torch.cat([mean, metal_emb, attn], dim=-1)


class TMCExcitonNet(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        n_interactions: int = 4,
        cutoff: float = 5.0,
        n_rbf: int = 50,
        cond_dim: int = 256,
        n_states: int = 30,
        dropout: float = 0.1,
        backbone: str = "schnet",
        backbone_initialization_seed: int = 0,
        regime: str = "A",
        natqg_node_dim: int = 21,
        natqg_edge_dim: int = 16,
        natqg_layers: int = 3,
        use_ct: bool = True,
        use_nto: bool = True,
        use_peak: bool = True,
        use_band: bool = True,
    ):
        super().__init__()
        self.regime = regime  # A=3D | C=3D+NBO fusion | G=graph-only
        self.solvent_emb = nn.Embedding(2, hidden)  # 0=gas, 1=acetone

        self.natqg = None
        if regime in ("C", "G"):
            self.natqg = NatQGEncoder(
                natqg_node_dim, natqg_edge_dim, hidden, natqg_layers, dropout
            )

        if regime == "G":
            # graph-only MPNN: no 3D backbone / metal-attention read-out
            self.backbone = None
            self.readout = None
            cin = self.natqg.out_dim + hidden + 1  # graph + solvent + charge
        else:
            self.backbone = BACKBONES[backbone](
                hidden, n_interactions, n_rbf, cutoff
            )
            self.readout = MetalAttentionReadout(hidden)
            cin = self.readout.out_dim + hidden + 1  # + solvent + charge
            if regime == "C":
                cin += self.natqg.out_dim  # fuse NBO graph embedding
        self.cond = nn.Sequential(
            nn.Linear(cin, cond_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        self.state = StateDecoder(cond_dim, n_states)
        self.band = BandHead(cond_dim) if use_band else None
        self.peak = PeakHead(cond_dim) if use_peak else None
        self.ct = CTHead(cond_dim) if use_ct else None
        self.nto = NTOHead(cond_dim) if use_nto else None

    def _context(self, g, charge, solvent_id: int):
        s = self.solvent_emb(
            torch.full((g.size(0),), solvent_id, dtype=torch.long, device=g.device)
        )
        return self.cond(torch.cat([g, s, charge[:, None]], dim=-1))

    def embed(self, batch) -> torch.Tensor:
        """Return the molecular embedding before solvent conditioning."""
        n_graphs = batch.ptr.numel() - 1
        if self.regime == "G":  # graph-only: NBO/2D graph alone
            return self.natqg(
                batch.nq_x,
                batch.nq_edge_index,
                batch.nq_edge_attr,
                batch.batch,
                n_graphs,
            )
        h = self.backbone(batch.z, batch.pos, batch.edge_index, batch.batch)
        g = self.readout(h, batch.batch, batch.ptr, batch.metal_idx)
        if self.natqg is not None:  # regime C: fuse NBO graph
            g_nq = self.natqg(
                batch.nq_x,
                batch.nq_edge_index,
                batch.nq_edge_attr,
                batch.batch,
                n_graphs,
            )
            g = torch.cat([g, g_nq], dim=-1)
        return g

    def forward(self, batch) -> dict:
        charge = batch.charge
        g = self.embed(batch)

        out: dict = {"gasphase": {}, "acetone": {}, "context": {}}
        for sid, sname in ((0, "gasphase"), (1, "acetone")):
            c = self._context(g, charge, sid)
            out["context"][sname] = c
            pred = dict(self.state(c))
            if self.band is not None:
                pred["band_logits"] = self.band(c)
            if self.peak is not None:
                pred.update(self.peak(c))
            if self.ct is not None:
                pred["ct_logits"] = self.ct(c)
            if self.nto is not None:
                pred["mfrac"] = self.nto(c)
            out[sname] = pred

        return out
