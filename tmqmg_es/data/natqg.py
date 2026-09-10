"""NatQG graph parsing (Regime C): NBO-derived quantum graphs from tmQMg.

Parses the GML graphs hosted on the Sigma2 NIRD archive (baseline / u-NatQ /
d-NatQ; Kneiding et al., D2DD00129B). Nodes are atoms (same count as the XYZ).

The THREE graph types carry DIFFERENT attribute keys, so each has its own
feature schema (an earlier single-schema parser silently zero-padded the
mismatched keys, discarding the baseline's 2D descriptors and d-NatQ's directed
donor->acceptor edge chemistry). Schemas below were read directly from the GML
files:

* baseline : a 2D connectivity graph. Node = atomic number, covalent radius,
  electronegativity, node degree (pure periodic-table / topology). Edge =
  integer Wiberg bond order. ``bond_distance`` and ``node_position`` (both
  3D-derived) are deliberately EXCLUDED so it is a genuine geometry-free 2D
  baseline.
* u-NatQ   : undirected NBO graph. 21 NBO node features + bond/antibond edge
  features.
* d-NatQ   : directed donor->acceptor NBO graph. Same 21 node features + the
  directed donor/acceptor/stabilisation-energy edge features.

Categorical ``*_type`` string fields and ``node_position`` are never read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NBO_NODE = (
    "feature_atomic_number",
    "feature_natural_atomic_charge",
    "feature_natural_electron_population_valence",
    "feature_natural_electron_configuration_s_occupation",
    "feature_natural_electron_configuration_p_occupation",
    "feature_natural_electron_configuration_d_occupation",
    "feature_n_lone_pairs",
    "feature_lone_pair_energy_min_max_difference",
    "feature_lone_pair_max_energy",
    "feature_lone_pair_max_occupation",
    "feature_lone_pair_max_s_occupation",
    "feature_lone_pair_max_p_occupation",
    "feature_lone_pair_max_d_occupation",
    "feature_n_lone_vacancies",
    "feature_lone_vacancy_energy_min_max_difference",
    "feature_lone_vacancy_min_energy",
    "feature_lone_vacancy_min_occupation",
    "feature_lone_vacancy_min_s_occupation",
    "feature_lone_vacancy_min_p_occupation",
    "feature_lone_vacancy_min_d_occupation",
    "feature_hydrogen_count",
)

SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "baseline": {
        "node": (
            "feature_atomic_number",
            "feature_covalent_radius",
            "feature_electronegativity",
            "feature_node_degree",
        ),
        "edge": ("feature_wiberg_bond_order_int",),  # 2D only; no bond_distance
    },
    "u-NatQ": {
        "node": _NBO_NODE,
        "edge": (
            "feature_wiberg_bond_order",
            "feature_bond_distance",
            "feature_n_bn",
            "feature_n_nbn",
            "feature_bond_energy_min_max_difference",
            "feature_bond_max_energy",
            "feature_bond_max_occupation",
            "feature_bond_max_s_occupation",
            "feature_bond_max_p_occupation",
            "feature_bond_max_d_occupation",
            "feature_antibond_energy_min_max_difference",
            "feature_antibond_min_energy",
            "feature_antibond_min_occupation",
            "feature_antibond_min_s_occupation",
            "feature_antibond_min_p_occupation",
            "feature_antibond_min_d_occupation",
        ),
    },
    "d-NatQ": {
        "node": _NBO_NODE,
        "edge": (
            "feature_wiberg_bond_order",
            "feature_bond_distance",
            "feature_donor_nbo_energy",
            "feature_donor_nbo_occupation",
            "feature_donor_nbo_s_occupation",
            "feature_donor_nbo_p_occupation",
            "feature_donor_nbo_d_occupation",
            "feature_donor_nbo_min_max_energy_gap",
            "feature_acceptor_nbo_energy",
            "feature_acceptor_nbo_occupation",
            "feature_acceptor_nbo_s_occupation",
            "feature_acceptor_nbo_p_occupation",
            "feature_acceptor_nbo_d_occupation",
            "feature_acceptor_nbo_min_max_energy_gap",
            "feature_stabilisation_energy_average",
            "feature_stabilisation_energy_max",
        ),
    },
}

UNDIRECTED_GRAPH_TYPES = frozenset(("baseline", "u-NatQ"))


def feature_dims(graph_type: str) -> tuple[int, int]:
    s = SCHEMAS[graph_type]
    return len(s["node"]), len(s["edge"])


def parse_gml(path: str | Path, graph_type: str = "d-NatQ") -> dict:
    """Return {node_feat [N,Fn], edge_index [2,E], edge_feat [E,Fe], n_nodes}.

    Features are selected per ``graph_type`` (see SCHEMAS). A lightweight manual
    parser (much faster than networkx over 74k files); tolerant of repeated
    ``node_position`` keys and the multigraph/directed flags.
    """
    schema = SCHEMAS[graph_type]
    node_idx = {k: i for i, k in enumerate(schema["node"])}
    edge_idx = {k: i for i, k in enumerate(schema["edge"])}
    n_nf, n_ef = len(schema["node"]), len(schema["edge"])

    nodes: list[np.ndarray] = []
    src: list[int] = []
    dst: list[int] = []
    edges: list[np.ndarray] = []
    mode = None
    cur = None
    cur_src = cur_dst = -1
    declared_directed = False

    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if s == "node [":
                mode, cur = "node", np.zeros(n_nf, dtype=np.float32)
            elif s == "edge [":
                mode, cur = "edge", np.zeros(n_ef, dtype=np.float32)
                cur_src = cur_dst = -1
            elif mode is None and s.startswith("directed "):
                declared_directed = bool(int(s.partition(" ")[2]))
            elif s == "]" and mode == "node":
                nodes.append(cur)
                mode = None
            elif s == "]" and mode == "edge":
                if cur_src < 0 or cur_dst < 0:
                    raise ValueError(f"edge without valid endpoints in {path}")
                src.append(cur_src)
                dst.append(cur_dst)
                edges.append(cur)
                mode = None
            elif mode == "node":
                key, _, val = s.partition(" ")
                j = node_idx.get(key)
                if j is not None:
                    cur[j] = float(val)
            elif mode == "edge":
                key, _, val = s.partition(" ")
                if key == "source":
                    cur_src = int(val)
                elif key == "target":
                    cur_dst = int(val)
                else:
                    j = edge_idx.get(key)
                    if j is not None:
                        cur[j] = float(val)

    if graph_type in UNDIRECTED_GRAPH_TYPES and declared_directed:
        raise ValueError(f"{graph_type} graph is unexpectedly declared directed: {path}")
    if graph_type == "d-NatQ" and not declared_directed:
        raise ValueError(f"d-NatQ graph is not declared directed: {path}")
    if src and (max(src + dst) >= len(nodes)):
        raise ValueError(f"edge endpoint exceeds node count in {path}")

    # GML stores each undirected baseline/u-NatQ edge once. Message passing is
    # directional, so materialize both directions while preserving every edge
    # feature vector. A self-loop already supplies its own reverse direction.
    if graph_type in UNDIRECTED_GRAPH_TYPES:
        original = list(zip(src, dst, edges, strict=True))
        for edge_src, edge_dst, edge_features in original:
            if edge_src != edge_dst:
                src.append(edge_dst)
                dst.append(edge_src)
                edges.append(edge_features.copy())

    node_feat = np.stack(nodes) if nodes else np.zeros((0, n_nf), np.float32)
    if src:
        edge_index = np.vstack([src, dst]).astype(np.int64)
        edge_feat = np.stack(edges)
    else:
        edge_index = np.zeros((2, 0), np.int64)
        edge_feat = np.zeros((0, n_ef), np.float32)
    return {
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_feat": edge_feat,
        "n_nodes": len(nodes),
    }


def natqg_dir(root: Path, graph_type: str) -> Path:
    return Path(root) / f"{graph_type}_graphs"
