"""Masked and normalized multitask objective."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from ..config import HC_EV_NM, SOLVENTS

DEFAULT_WEIGHTS = {
    "E": 1.0,
    "f": 0.2,
    "spectrum": 0.2,
    "band": 0.2,
    "peak": 0.2,
    "ct": 0.1,
    "nto": 0.1,
    "consistency": 0.05,
}

VIS_LO_EV = HC_EV_NM / 825.0
VIS_HI_EV = HC_EV_NM / 350.0


def _masked_huber(pred, true, mask=None, delta=1.0):
    ok = torch.isfinite(true)
    if mask is not None:
        ok = ok & mask
    if ok.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[ok], true[ok], delta=delta)


def _spectrum(E, logf, grid, sigma=0.2):
    f = torch.expm1(logf.clamp(max=20)).clamp(min=0)
    valid = torch.isfinite(E) & torch.isfinite(f)
    E = torch.where(valid, E, torch.zeros_like(E))
    f = torch.where(valid, f, torch.zeros_like(f))
    diff = grid[None, None, :] - E[:, :, None]
    return (f[:, :, None] * torch.exp(-(diff**2) / (2 * sigma**2))).sum(1)


def _mean_terms(terms: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(terms).mean() if terms else zero


class MultiTaskLoss(torch.nn.Module):
    def __init__(
        self,
        weights: dict | None = None,
        ct_class_weights: torch.Tensor | None = None,
        band_pos_weights: torch.Tensor | None = None,
        spectrum_sigma: float = 0.2,
        n_grid: int = 128,
    ):
        super().__init__()
        unknown = set(weights or {}) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown loss weights: {sorted(unknown)}")
        self.w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.register_buffer("grid", torch.linspace(0.5, 8.0, n_grid))
        self.register_buffer(
            "band_pos_weights",
            torch.ones(2, 3) if band_pos_weights is None else band_pos_weights,
        )
        self.ct_class_weights = ct_class_weights
        self.sigma = spectrum_sigma

    def forward(self, out: dict, batch) -> tuple[torch.Tensor, dict]:
        terms: dict[str, list[torch.Tensor]] = defaultdict(list)
        zero = self.grid.sum() * 0.0

        for solvent_index, solvent_name in enumerate(SOLVENTS):
            pred = out[solvent_name]
            energy_true = batch.E[:, solvent_index, :]
            logf_true = batch.logf[:, solvent_index, :]

            terms["E"].append(_masked_huber(pred["E"], energy_true))
            terms["f"].append(_masked_huber(pred["logf"], logf_true))

            spectrum_pred = _spectrum(pred["E"], pred["logf"], self.grid, self.sigma)
            spectrum_true = _spectrum(energy_true, logf_true, self.grid, self.sigma)
            spectrum_pred_norm = spectrum_pred / (
                spectrum_pred.amax(1, keepdim=True) + 1e-6
            )
            spectrum_true_norm = spectrum_true / (
                spectrum_true.amax(1, keepdim=True) + 1e-6
            )
            mse = F.mse_loss(spectrum_pred_norm, spectrum_true_norm)
            cosine_loss = (
                1 - F.cosine_similarity(spectrum_pred_norm, spectrum_true_norm, dim=1)
            ).mean()
            terms["spectrum"].append(0.5 * (mse + cosine_loss))

            if "band_logits" in pred:
                terms["band"].append(
                    F.binary_cross_entropy_with_logits(
                        pred["band_logits"],
                        batch.band[:, solvent_index, :],
                        pos_weight=self.band_pos_weights[solvent_index],
                    )
                )

            if "peak_E" in pred:
                peak_mask = batch.peak_mask[:, solvent_index, :]
                peak_parts = [
                    _masked_huber(
                        pred["peak_E"],
                        batch.peak_E[:, solvent_index, :],
                        peak_mask,
                    ),
                    _masked_huber(
                        pred["peak_f"],
                        batch.peak_f[:, solvent_index, :],
                        peak_mask,
                    ),
                    _masked_huber(
                        pred["sigma"],
                        torch.log1p(batch.sigma[:, solvent_index, :].clamp(min=0)),
                        peak_mask,
                    ),
                ]
                terms["peak"].append(torch.stack(peak_parts).mean())
                terms["consistency"].append(self._consistency(pred, peak_mask[:, 1]))

            if "ct_logits" in pred:
                ct_true = batch.ct[:, solvent_index].clone()
                ct_true[~batch.ct_mask[:, solvent_index]] = -1
                if (ct_true >= 0).any():
                    terms["ct"].append(
                        F.cross_entropy(
                            pred["ct_logits"],
                            ct_true,
                            weight=self.ct_class_weights,
                            ignore_index=-1,
                        )
                    )

            if "mfrac" in pred:
                nto_mask = batch.nto_mask[:, solvent_index][:, None].expand_as(
                    pred["mfrac"]
                )
                terms["nto"].append(
                    _masked_huber(
                        pred["mfrac"],
                        batch.mfrac[:, solvent_index, :],
                        nto_mask,
                    )
                )

        components = {name: _mean_terms(terms[name], zero) for name in DEFAULT_WEIGHTS}
        total = sum(self.w[name] * components[name] for name in components)
        logs = {name: float(value.detach()) for name, value in components.items()}
        logs["total"] = float(total.detach())
        return total, logs

    def _consistency(self, pred, visible_mask, beta=5.0, k=10.0):
        if visible_mask.sum() == 0:
            return pred["E"].sum() * 0.0
        energy, logf = pred["E"], pred["logf"]
        gate = torch.sigmoid(k * (energy - VIS_LO_EV)) * torch.sigmoid(
            k * (VIS_HI_EV - energy)
        )
        weights = torch.softmax(beta * logf + torch.log(gate + 1e-6), dim=1)
        state_peak_energy = (weights * energy).sum(1)
        return F.huber_loss(
            state_peak_energy[visible_mask],
            pred["peak_E"][:, 1][visible_mask],
        )
