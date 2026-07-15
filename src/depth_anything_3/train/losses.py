"""
Losses for monocular depth fine-tuning. SiLogLoss is ported directly from Depth Anything V2's
metric_depth training code and is the only loss used by default (train.py), matching that
reference. GradientLoss is an optional multi-scale gradient-matching regularizer (MiDaS/DA2
style edge-awareness term); it is available but weighted 0 by default so the default training
recipe stays faithful to the SiLogLoss-only reference.

GlobalLocalLoss approximates the "global-local loss" (L_gl) term from the DA3 paper's Teacher
training objective (L_T = alpha*L_grad + L_gl + L_N + L_sky + L_obj). The paper attributes the
exact alignment formula to an external reference ("ROE alignment", cited as [95]) rather than
spelling it out, so this is the standard closed-form least-squares scale-and-shift-invariant loss
(MiDaS-style), applied once over the whole image (global) and once per patch on a grid (local) --
not a verified reproduction of that reference's exact method.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask].clamp(min=1e-4)) - torch.log(pred[valid_mask].clamp(min=1e-4))
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
        return loss


class GradientLoss(nn.Module):
    """Multi-scale gradient-matching loss on log-depth (opt-in, weight 0 by default)."""

    def __init__(self, scales: int = 4):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        log_pred = torch.zeros_like(pred)
        log_target = torch.zeros_like(target)
        log_pred[valid_mask] = torch.log(pred[valid_mask])
        log_target[valid_mask] = torch.log(target[valid_mask])
        diff = (log_pred - log_target) * valid_mask
        mask = valid_mask.float()

        total = pred.new_tensor(0.0)
        for scale in range(self.scales):
            step = 2**scale
            d = diff[:, ::step, ::step]
            m = mask[:, ::step, ::step]

            grad_x = torch.abs(d[:, :, :-1] - d[:, :, 1:])
            mask_x = m[:, :, :-1] * m[:, :, 1:]
            grad_y = torch.abs(d[:, :-1, :] - d[:, 1:, :])
            mask_y = m[:, :-1, :] * m[:, 1:, :]

            denom = mask_x.sum() + mask_y.sum()
            if denom > 0:
                total = total + ((grad_x * mask_x).sum() + (grad_y * mask_y).sum()) / denom
        return total / self.scales


def _solve_scale_shift(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form least-squares scale `a` and shift `b` minimizing
    sum_i mask_i * (a*pred_i + b - target_i)^2, solved independently per leading-dim element
    (pred/target/mask: [N, H, W] -> a, b: [N]). Standard 2x2 normal-equations solve.
    """
    m = mask.float()
    sum_pp = (pred * pred * m).flatten(1).sum(-1)
    sum_p = (pred * m).flatten(1).sum(-1)
    sum_pt = (pred * target * m).flatten(1).sum(-1)
    sum_t = (target * m).flatten(1).sum(-1)
    sum_m = m.flatten(1).sum(-1)

    det = (sum_pp * sum_m - sum_p * sum_p).clamp(min=eps)
    a = (sum_m * sum_pt - sum_p * sum_t) / det
    b = (sum_pp * sum_t - sum_p * sum_pt) / det
    return a, b


def _scale_shift_align_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, min_valid: int
) -> torch.Tensor:
    """Mean masked L1 between `target` and `pred` after per-element scale/shift alignment."""
    valid = mask.flatten(1).float().sum(-1) >= min_valid
    if not bool(valid.any()):
        return pred.new_tensor(0.0)

    pred, target, mask = pred[valid], target[valid], mask[valid]
    a, b = _solve_scale_shift(pred, target, mask)
    aligned = a.view(-1, 1, 1) * pred + b.view(-1, 1, 1)
    m = mask.float()
    l1 = (torch.abs(aligned - target) * m).flatten(1).sum(-1) / m.flatten(1).sum(-1).clamp(min=1.0)
    return l1.mean()


class GlobalLocalLoss(nn.Module):
    """
    Approximate scale-and-shift-invariant "global-local" loss (see module docstring): a global
    term aligning the whole image, plus a local term aligning each cell of a `grid_size x
    grid_size` patch grid independently, both via closed-form least-squares scale/shift.
    """

    def __init__(self, grid_size: int = 4, min_valid_global: int = 20, min_valid_local: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.min_valid_global = min_valid_global
        self.min_valid_local = min_valid_local

    def _to_patches(self, t: torch.Tensor, h_trunc: int, w_trunc: int, ph: int, pw: int) -> torch.Tensor:
        b = t.shape[0]
        t = t[:, :h_trunc, :w_trunc]
        t = t.reshape(b, self.grid_size, ph, self.grid_size, pw)
        return t.permute(0, 1, 3, 2, 4).reshape(b * self.grid_size * self.grid_size, ph, pw)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.detach()
        global_loss = _scale_shift_align_l1(pred, target, mask, self.min_valid_global)

        _, h, w = pred.shape
        ph, pw = h // self.grid_size, w // self.grid_size
        if ph == 0 or pw == 0:
            return global_loss

        h_trunc, w_trunc = ph * self.grid_size, pw * self.grid_size
        pred_p = self._to_patches(pred, h_trunc, w_trunc, ph, pw)
        target_p = self._to_patches(target, h_trunc, w_trunc, ph, pw)
        mask_p = self._to_patches(mask, h_trunc, w_trunc, ph, pw)
        local_loss = _scale_shift_align_l1(pred_p, target_p, mask_p, self.min_valid_local)

        return global_loss + local_loss


class DepthLoss(nn.Module):
    """
    Combined loss: SiLogLoss (always on) + optional GradientLoss + optional GlobalLocalLoss
    (both weight 0, i.e. off, by default).
    """

    def __init__(
        self,
        silog_lambd: float = 0.5,
        grad_weight: float = 0.0,
        grad_scales: int = 4,
        gl_weight: float = 0.0,
        gl_grid_size: int = 4,
    ):
        super().__init__()
        self.silog = SiLogLoss(lambd=silog_lambd)
        self.grad_weight = grad_weight
        self.gradient = GradientLoss(scales=grad_scales) if grad_weight > 0 else None
        self.gl_weight = gl_weight
        self.global_local = GlobalLocalLoss(grid_size=gl_grid_size) if gl_weight > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (total_loss, components) so callers can log each term separately."""
        components = {"silog": self.silog(pred, target, valid_mask)}
        loss = components["silog"]
        if self.gradient is not None:
            components["grad"] = self.gradient(pred, target, valid_mask)
            loss = loss + self.grad_weight * components["grad"]
        if self.global_local is not None:
            components["gl"] = self.global_local(pred, target, valid_mask)
            loss = loss + self.gl_weight * components["gl"]
        return loss, components
