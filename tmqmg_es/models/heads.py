"""Prediction heads with physical constraints.

State energies are emitted as a strictly increasing sequence via a softplus
cumulative sum, mirroring the dataset's ordering (state 1 = lowest energy).
Oscillator strengths are predicted as log1p(f) >= 0. Regional peak energies
are constrained to remain inside their UV, visible, or near-IR windows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HC_EV_NM

VIS_LO_EV = HC_EV_NM / 825.0
VIS_HI_EV = HC_EV_NM / 350.0


def _mlp(din: int, dhidden: int, dout: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dhidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(dhidden, dout),
    )


class StateDecoder(nn.Module):
    """Thirty ordered excitation energies (eV) and log1p(f) values."""

    def __init__(self, d: int, n_states: int = 30, hidden: int = 256):
        super().__init__()
        self.n_states = n_states
        self.delta = _mlp(d, hidden, n_states)  # softplus -> positive gaps
        self.logf = _mlp(d, hidden, n_states)
        # bias initial gaps to ~0.2 eV so E_30 starts near a physical ~6 eV
        # (softplus(-1.5) ~= 0.20) instead of ~21 eV
        nn.init.constant_(self.delta[-1].bias, -1.5)

    def forward(self, c: torch.Tensor) -> dict:
        gaps = F.softplus(self.delta(c))  # [B, S] strictly positive
        energies = torch.cumsum(gaps, dim=-1)  # increasing sequence
        logf = F.softplus(self.logf(c))  # log1p(f) >= 0
        return {"E": energies, "logf": logf}


class BandHead(nn.Module):
    """Logits for whether UV, visible, and near-IR bands are defined."""

    def __init__(self, d: int, n_regions: int = 3, hidden: int = 128):
        super().__init__()
        self.net = _mlp(d, hidden, n_regions)

    def forward(self, c):
        return self.net(c)


class PeakHead(nn.Module):
    """UV/Vis/nIR peak energy (eV), peak log1p(f), and band broadness sigma."""

    def __init__(self, d: int, n_regions: int = 3, hidden: int = 128):
        super().__init__()
        self.energy = _mlp(d, hidden, n_regions)
        self.logf = _mlp(d, hidden, n_regions)
        self.sigma = _mlp(d, hidden, n_regions)

    def forward(self, c):
        raw_energy = self.energy(c)
        uv = VIS_HI_EV + F.softplus(raw_energy[:, 0])
        visible = VIS_LO_EV + (VIS_HI_EV - VIS_LO_EV) * torch.sigmoid(raw_energy[:, 1])
        nir = VIS_LO_EV * torch.sigmoid(raw_energy[:, 2])
        return {
            "peak_E": torch.stack([uv, visible, nir], dim=1),
            "peak_f": F.softplus(self.logf(c)),
            "sigma": F.softplus(self.sigma(c)),
        }


class CTHead(nn.Module):
    """Visible charge-transfer class logits (ddT/LLCT/MLCT/LMCT)."""

    def __init__(self, d: int, n_classes: int = 4, hidden: int = 128):
        super().__init__()
        self.net = _mlp(d, hidden, n_classes)

    def forward(self, c):
        return self.net(c)


class NTOHead(nn.Module):
    """Metal fraction of occupied / virtual NTOs in (0, 1)."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(d, hidden, 2)

    def forward(self, c):
        return torch.sigmoid(self.net(c))
