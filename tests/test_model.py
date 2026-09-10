import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch_geometric.loader import DataLoader  # noqa: E402

from tmqmg_es import config as C  # noqa: E402
from tmqmg_es.data.dataset import build_data  # noqa: E402
from tmqmg_es.models.excitonnet import TMCExcitonNet  # noqa: E402
from tmqmg_es.train.losses import MultiTaskLoss  # noqa: E402
from .conftest import requires_data  # noqa: E402


@pytest.fixture(scope="module")
def small_batch(phase1):
    cleaned = phase1["cleaned"]
    data = [build_data(r, C.XYZ_DIR, 5.0) for _, r in cleaned.head(8).iterrows()]
    data = [d for d in data if d is not None]
    return next(iter(DataLoader(data, batch_size=8)))


@requires_data
def test_forward_shapes(small_batch):
    model = TMCExcitonNet(hidden=32, n_interactions=2, cond_dim=64)
    out = model(small_batch)
    B = small_batch.num_graphs
    assert out["gasphase"]["E"].shape == (B, C.N_STATES)
    assert out["gasphase"]["logf"].shape == (B, C.N_STATES)
    assert out["gasphase"]["band_logits"].shape == (B, len(C.REGIONS))
    assert out["gasphase"]["ct_logits"].shape == (B, len(C.CT_CLASSES))
    assert out["gasphase"]["mfrac"].shape == (B, 2)
    assert "shift" not in out


@requires_data
def test_energies_strictly_ordered(small_batch):
    model = TMCExcitonNet(hidden=32, n_interactions=2, cond_dim=64)
    E = model(small_batch)["gasphase"]["E"]
    assert bool((E[:, 1:] >= E[:, :-1]).all())  # cumulative softplus -> increasing


@requires_data
def test_loss_finite_and_backprops(small_batch):
    model = TMCExcitonNet(hidden=32, n_interactions=2, cond_dim=64)
    loss_fn = MultiTaskLoss()
    loss, logs = loss_fn(model(small_batch), small_batch)
    assert np.isfinite(loss.detach().item())
    assert all(np.isfinite(v) for v in logs.values())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_uncertainty_metrics_masked_alignment():
    """nll/coverage must mask variance with the SAME mask as mean/true — a NaN
    early in `true` must not shift the variance row against the survivors."""
    from tmqmg_es.train.metrics import coverage, nll_gaussian

    true = np.array([np.nan, 1.0, 2.0, 3.0])
    mean = np.array([0.0, 1.0, 2.0, 3.0])  # exact on all valid entries
    std = np.array([1e6, 0.1, 0.1, 0.1])  # huge std sits on the NaN slot
    assert coverage(mean, std, true, z=1.0) == 1.0  # all valid within 1 std
    var = std**2
    nll = nll_gaussian(mean, var, true)
    # perfectly predicted with std 0.1 -> NLL ~ 0.5*log(2*pi*0.01) < 0; a prefix
    # slice would pair the 1e12 variance with a valid point and blow this up
    assert nll < 0
