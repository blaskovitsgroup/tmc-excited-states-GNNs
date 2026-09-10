"""NatQG branch: a GINE-style MPNN over the NBO quantum graph (Regime C).

Messages flow from each stored source node to its target node. The data parser
materializes both directions for the undirected baseline and u-NatQ graphs,
whereas d-NatQ retains its native donor-to-acceptor directions. The pooled
graph embedding is fused with the 3D backbone embedding in TMCExcitonNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class NatQGConv(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h, edge_index, edge_emb):
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(h[src] + edge_emb)
        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        return h + self.upd(agg)


class NatQGEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # input BatchNorm standardizes the heterogeneously-scaled NBO features
        self.node_bn = nn.BatchNorm1d(node_dim)
        self.edge_bn = nn.BatchNorm1d(edge_dim)
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.layers = nn.ModuleList([NatQGConv(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * hidden  # mean + sum pool

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = self.node_in(self.node_bn(x))
        e = (
            self.edge_in(self.edge_bn(edge_attr))
            if edge_attr.numel()
            else torch.zeros(
                edge_index.size(1), self.node_in.out_features, device=x.device
            )
        )
        for layer in self.layers:
            h = layer(h, edge_index, e)
        h = self.drop(h)
        mean = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="mean")
        summ = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="sum")
        return torch.cat([mean, summ], dim=-1)
