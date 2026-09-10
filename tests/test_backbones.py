import pytest
import torch

from tmqmg_es.models.backbone import PaiNNBackbone, SchNetBackbone


def _radius_graph(pos, cutoff=5.0):
    d = (pos[:, None] - pos[None]).norm(dim=-1)
    d.fill_diagonal_(1e9)
    return torch.nonzero(d <= cutoff).T


def _random_mol(n=18, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(1, 30, (n,), generator=g)
    pos = torch.randn(n, 3, generator=g) * 3
    return z, pos


def _rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    return q * torch.det(q).sign()


def test_painn_scalar_output_is_rotation_invariant():
    bb = PaiNNBackbone(hidden=32, n_interactions=3, n_rbf=20, cutoff=5.0).eval()
    z, pos = _random_mol()
    R = _rotation()
    with torch.no_grad():
        h1 = bb(z, pos, _radius_graph(pos))
        pos2 = pos @ R.T
        h2 = bb(z, pos2, _radius_graph(pos2))
    assert torch.allclose(h1, h2, atol=1e-4)


def test_schnet_output_is_rotation_invariant():
    bb = SchNetBackbone(hidden=32, n_interactions=3, n_rbf=20, cutoff=5.0).eval()
    z, pos = _random_mol()
    R = _rotation()
    with torch.no_grad():
        h1 = bb(z, pos, _radius_graph(pos))
        h2 = bb(z, pos @ R.T, _radius_graph(pos @ R.T))
    assert torch.allclose(h1, h2, atol=1e-4)


def test_both_backbones_shapes():
    z, pos = _random_mol()
    ei = _radius_graph(pos)
    for bb in (
        SchNetBackbone(hidden=16, n_interactions=2),
        PaiNNBackbone(hidden=16, n_interactions=2),
    ):
        h = bb(z, pos, ei)
        assert h.shape == (z.numel(), 16)


