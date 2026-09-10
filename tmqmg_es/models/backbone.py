"""SchNet and PaiNN coordinate-based message-passing backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ..provenance import sha256_file


class RBFExpansion(nn.Module):
    def __init__(self, cutoff: float, n_rbf: int = 50):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2 / 4.0

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (d[:, None] - self.centers[None, :]) ** 2)


class Interaction(nn.Module):
    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.lin_in = nn.Linear(hidden, hidden)
        self.lin_out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h, edge_index, edge_feat, edge_weight):
        row, col = edge_index
        w = self.filter_net(edge_feat) * edge_weight[:, None]
        msg = self.lin_in(h)[col] * w
        agg = scatter(msg, row, dim=0, dim_size=h.size(0), reduce="sum")
        return self.lin_out(agg)


def cosine_cutoff(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    return 0.5 * (torch.cos(torch.pi * d / cutoff) + 1.0) * (d < cutoff)


class SchNetBackbone(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        n_interactions: int = 4,
        n_rbf: int = 50,
        cutoff: float = 5.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(max_z, hidden)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.interactions = nn.ModuleList(
            [Interaction(hidden, n_rbf) for _ in range(n_interactions)]
        )
        self.out_dim = hidden

    def forward(self, z, pos, edge_index, batch=None):
        h = self.embed(z)
        row, col = edge_index
        d = (pos[row] - pos[col]).norm(dim=-1)
        edge_feat = self.rbf(d)
        edge_weight = cosine_cutoff(d, self.cutoff)
        for inter in self.interactions:
            h = h + inter(h, edge_index, edge_feat, edge_weight)
        return h


class PaiNNMessage(nn.Module):
    """Equivariant message: updates scalar s and vector v features."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )
        self.filter = nn.Linear(n_rbf, 3 * hidden)

    def forward(self, s, v, edge_index, rbf, unit, fcut):
        row, col = edge_index
        x = self.phi(s)[col] * (self.filter(rbf) * fcut)  # [E, 3H]
        ds, gv1, gv2 = x.split(self.hidden, dim=-1)
        dv = gv1[:, None, :] * v[col] + gv2[:, None, :] * unit[:, :, None]
        s_agg = scatter(ds, row, dim=0, dim_size=s.size(0), reduce="sum")
        v_agg = scatter(dv, row, dim=0, dim_size=s.size(0), reduce="sum")
        return s + s_agg, v + v_agg


class PaiNNUpdate(nn.Module):
    """Equivariant gated update mixing scalar and vector channels."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * hidden)
        )

    def forward(self, s, v):
        Uv, Vv = self.U(v), self.V(v)  # [N, 3, H]
        a = self.mlp(torch.cat([s, Vv.norm(dim=1)], dim=-1))  # [N, 3H]
        a_ss, a_sv, a_vv = a.split(self.hidden, dim=-1)
        ds = a_ss + a_sv * (Uv * Vv).sum(dim=1)
        dv = a_vv[:, None, :] * Uv
        return s + ds, v + dv


class PaiNNBackbone(nn.Module):
    """PaiNN-style equivariant message passing (Schütt et al. 2021).

    Same (z, pos, edge_index) -> scalar node embeddings interface as SchNet, so
    it drops into TMCExcitonNet behind a config flag. Vector features capture
    directional/angular information the distance-only SchNet cannot.
    """

    def __init__(
        self,
        hidden: int = 128,
        n_interactions: int = 4,
        n_rbf: int = 50,
        cutoff: float = 5.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(max_z, hidden)
        self.rbf = RBFExpansion(cutoff, n_rbf)
        self.message = nn.ModuleList(
            [PaiNNMessage(hidden, n_rbf) for _ in range(n_interactions)]
        )
        self.update = nn.ModuleList(
            [PaiNNUpdate(hidden) for _ in range(n_interactions)]
        )
        self.out_dim = hidden

    def forward(self, z, pos, edge_index, batch=None):
        s = self.embed(z)
        v = torch.zeros(z.size(0), 3, s.size(1), device=z.device, dtype=s.dtype)
        row, col = edge_index
        rij = pos[row] - pos[col]
        d = rij.norm(dim=-1)
        unit = rij / (d[:, None] + 1e-8)
        rbf = self.rbf(d)
        fcut = cosine_cutoff(d, self.cutoff)[:, None]
        for msg, upd in zip(self.message, self.update):
            s, v = msg(s, v, edge_index, rbf, unit, fcut)
            s, v = upd(s, v)
        return s
