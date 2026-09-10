import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import pytest

from tmqmg_es.data.natqg import SCHEMAS, feature_dims, parse_gml
from tmqmg_es.models.excitonnet import TMCExcitonNet
from tmqmg_es.models.natqg_encoder import NatQGEncoder

NF, EF = feature_dims("d-NatQ")  # (21, 16)


def test_per_type_schemas():
    # baseline is a genuine 2D graph: no NBO node features, no 3D bond_distance
    assert feature_dims("baseline") == (4, 1)
    assert feature_dims("u-NatQ") == (21, 16)
    assert feature_dims("d-NatQ") == (21, 16)
    assert "feature_bond_distance" not in SCHEMAS["baseline"]["edge"]
    assert "feature_node_position" not in SCHEMAS["baseline"]["node"]
    # d-NatQ carries the directed donor->acceptor edge chemistry
    assert "feature_donor_nbo_energy" in SCHEMAS["d-NatQ"]["edge"]
    assert "feature_stabilisation_energy_max" in SCHEMAS["d-NatQ"]["edge"]


def _write_graph(path, *, directed: bool, source: int = 0, target: int = 1):
    directed_line = "  directed 1\n" if directed else ""
    path.write_text(
        "graph [\n"
        f"{directed_line}"
        "  node [\n"
        "    id 0\n"
        "    feature_atomic_number 6\n"
        "  ]\n"
        "  node [\n"
        "    id 1\n"
        "    feature_atomic_number 8\n"
        "  ]\n"
        "  edge [\n"
        f"    source {source}\n"
        f"    target {target}\n"
        "    feature_wiberg_bond_order_int 2\n"
        "    feature_wiberg_bond_order 1.5\n"
        "  ]\n"
        "]\n"
    )


@pytest.mark.parametrize("graph_type", ["baseline", "u-NatQ"])
def test_undirected_graphs_materialize_reverse_edges(tmp_path, graph_type):
    path = tmp_path / "undirected.gml"
    _write_graph(path, directed=False)
    graph = parse_gml(path, graph_type)
    assert graph["edge_index"].tolist() == [[0, 1], [1, 0]]
    assert graph["edge_feat"].shape == (2, feature_dims(graph_type)[1])
    np.testing.assert_array_equal(graph["edge_feat"][0], graph["edge_feat"][1])


def test_undirected_self_loop_is_not_duplicated(tmp_path):
    path = tmp_path / "self_loop.gml"
    _write_graph(path, directed=False, source=0, target=0)
    graph = parse_gml(path, "baseline")
    assert graph["edge_index"].tolist() == [[0], [0]]
    assert graph["edge_feat"].shape == (1, 1)


def test_dnatq_retains_native_direction(tmp_path):
    path = tmp_path / "directed.gml"
    _write_graph(path, directed=True)
    graph = parse_gml(path, "d-NatQ")
    assert graph["edge_index"].tolist() == [[0], [1]]
    assert graph["edge_feat"].shape == (1, 16)


def test_parser_rejects_directionality_schema_mismatch(tmp_path):
    path = tmp_path / "wrong_direction.gml"
    _write_graph(path, directed=True)
    with pytest.raises(ValueError, match="unexpectedly declared directed"):
        parse_gml(path, "baseline")


def test_natqg_encoder_pools_per_graph():
    enc = NatQGEncoder(NF, EF, hidden=16, n_layers=2)
    n, e = 10, 24
    out = enc(
        torch.randn(n, NF),
        torch.randint(0, n, (2, e)),
        torch.randn(e, EF),
        torch.zeros(n, dtype=torch.long),
        n_graphs=1,
    )
    assert out.shape == (1, enc.out_dim)


def _mol(n_atoms, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n_atoms, 3, generator=g) * 3
    d = (pos[:, None] - pos[None]).norm(dim=-1)
    d.fill_diagonal_(1e9)
    ei = torch.nonzero(d <= 6.0).T
    ne = 3 * n_atoms
    data = Data(z=torch.randint(1, 30, (n_atoms,), generator=g), pos=pos, edge_index=ei)
    data.num_nodes = n_atoms
    data.metal_idx = torch.tensor([0])
    data.charge = torch.tensor([0.0])
    data.nq_x = torch.randn(n_atoms, NF, generator=g)
    data.nq_edge_index = torch.randint(0, n_atoms, (2, ne), generator=g)
    data.nq_edge_attr = torch.randn(ne, EF, generator=g)
    return data


def test_regimeC_batches_and_forwards():
    batch = next(iter(DataLoader([_mol(12, 1), _mol(9, 2), _mol(15, 3)], batch_size=3)))
    assert int(batch.nq_edge_index.max()) < batch.num_nodes
    model = TMCExcitonNet(
        hidden=24, n_interactions=2, cond_dim=48, backbone="painn", regime="C"
    ).train()
    out = model(batch)
    assert out["gasphase"]["E"].shape == (3, 30)
    out["gasphase"]["E"].sum().backward()
    gn = sum(
        p.grad.abs().sum().item()
        for p in model.natqg.parameters()
        if p.grad is not None
    )
    assert gn > 0


def test_graph_only_regime_G():
    """Regime G has no 3D backbone; the NBO/2D graph alone drives the heads."""
    batch = next(iter(DataLoader([_mol(12, 1), _mol(9, 2)], batch_size=2)))
    model = TMCExcitonNet(hidden=24, n_interactions=2, cond_dim=48, regime="G").train()
    assert model.backbone is None and model.natqg is not None
    out = model(batch)
    assert out["gasphase"]["E"].shape == (2, 30)
    out["gasphase"]["E"].sum().backward()
    gn = sum(
        p.grad.abs().sum().item()
        for p in model.natqg.parameters()
        if p.grad is not None
    )
    assert gn > 0


def test_regime_branches():
    assert TMCExcitonNet(regime="A").natqg is None
    assert TMCExcitonNet(regime="C").natqg is not None
    assert TMCExcitonNet(regime="G").backbone is None
