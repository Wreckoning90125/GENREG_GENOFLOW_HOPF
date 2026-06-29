# ================================================================
# D_4 group-equivariant CNN whose group axis carries either the
# ORDINARY regular representation (standard G-CNN) or the alpha'_1
# TWISTED (projective) regular representation -- the non-trivial class
# of H^2(D_4, U(1)) = Z_2 (the paper's N=1 / T-gate case).  D_4
# (90-deg rotations + flips) acts EXACTLY on a square pixel grid.
#
# Design (input-transformation convention, so the director CHANNEL
# action is handled exactly by rho_input):
#   * LiftConv: out[:,g] = conv(rho(g^{-1}) x, W)  -- builds the group
#     axis; the group acts on it by PURE PERMUTATION  f[g] -> f[g0^{-1}g]
#     (= ordinary left-regular rep L on the slot axis).
#   * SlotMix: mixes the 8-slot axis with sum_g c_g R(g) [linear] or
#     sum_g c_g R^alpha(g) [cocycle], the verified operators from
#     projective_equivariant.py.  R commutes with L (the permutation),
#     so the LINEAR net is exactly D_4-equivariant; the COCYCLE net is
#     PROJECTIVELY equivariant (R^alpha commutes with L^alpha) -- the
#     physically correct action for spinor / half-integer-winding
#     features (a +-1/2 defect picks up the cocycle sign under r^4).
#   * shared spatial convs (same filter on every slot) commute with the
#     permutation, so they preserve equivariance.
#
# Equivariance is verified numerically before training (test_*).
# ================================================================

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dihedral_cohomology import Dihedral
from projective_equivariant import CocycleRepresentation

G = Dihedral(1)                      # D_4, order 8
ELS = G.elements()
IDX = {g: i for i, g in enumerate(ELS)}

# 8x8 slot-mixing operator banks (real), reused from the verified substrate
_REP = CocycleRepresentation(1, cocycle=True, use_raw=True)
R_LINEAR = np.real(_REP.conv_basis(twisted=False)).astype(np.float32)   # (8,8,8)
R_TWISTED = np.real(_REP.conv_basis(twisted=True)).astype(np.float32)   # (8,8,8)


# ---- spatial action of D_4 on the last two axes -----------------
def spatial(g, t):
    a, j = g
    if j:
        t = torch.flip(t, dims=(-1,))           # F: horizontal flip
    if a:
        t = torch.rot90(t, k=a, dims=(-2, -1))  # R^a (90-deg)
    return t


# ---- director channel action (cos2th, sin2th) ------------------
def director_channel_matrix(g):
    a, j = g
    Rr = np.array([[-1.0, 0.0], [0.0, -1.0]])   # 2theta += 180deg
    Cs = np.array([[1.0, 0.0], [0.0, -1.0]])    # flip negates sin
    M = np.linalg.matrix_power(Rr, a) @ (Cs if j else np.eye(2))
    return torch.tensor(M, dtype=torch.float32)


def rho_input(g, x):
    """Full D_4 action on a director image x: (B,2,H,W)."""
    x = spatial(g, x)
    M = director_channel_matrix(g).to(x.device)
    return torch.einsum("co,bohw->bchw", M, x)


# ---- lifting conv: director image -> (B,C,8,H,W) ---------------
class LiftConv(nn.Module):
    def __init__(self, c_in=2, c_out=8, k=5):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=True)

    def forward(self, x):
        outs = [self.conv(rho_input(G.inverse(g), x)) for g in ELS]
        return torch.stack(outs, dim=2)          # (B,C,8,H,W)


# ---- slot mix on the 8 group axis (linear or cocycle) ----------
class SlotMix(nn.Module):
    """K learned group-conv filters on the slot axis: out block k is
    sum_g c[k,g] R(g) applied to the 8 slots (per spatial location)."""

    def __init__(self, twisted, K):
        super().__init__()
        ops = R_TWISTED if twisted else R_LINEAR
        self.register_buffer("ops", torch.from_numpy(ops))  # (8,8,8)
        self.coeff = nn.Parameter(torch.randn(K, 8) * (1.0 / np.sqrt(8)))
        self.K = K

    def forward(self, x):
        # x: (B,C,8,H,W) -> (B,C*K,8,H,W)
        Wmix = torch.einsum("kg,goi->koi", self.coeff, self.ops)  # (K,8,8)
        out = torch.einsum("koi,bcihw->bckohw", Wmix, x)
        B, C, K, S, H, Wd = out.shape
        return out.reshape(B, C * K, S, H, Wd)


class SharedSpatial(nn.Module):
    """A spatial Conv2d applied identically to every group slot."""

    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2)

    def forward(self, x):
        B, C, S, H, Wd = x.shape
        y = self.conv(x.permute(0, 2, 1, 3, 4).reshape(B * S, C, H, Wd))
        return y.reshape(B, S, -1, H, Wd).permute(0, 2, 1, 3, 4)


class D4DefectNet(nn.Module):
    """kind in {'plain','linear','cocycle'}."""

    def __init__(self, kind="cocycle", width=8, n_classes=4):
        super().__init__()
        self.kind = kind
        if kind == "plain":
            self.net = nn.Sequential(
                nn.Conv2d(2, width, 5, padding=2), nn.ReLU(),
                nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(2 * width, n_classes)
            return
        tw = (kind == "cocycle")
        self.lift = LiftConv(2, width, k=5)
        self.mix1 = SlotMix(tw, K=2)
        self.sp1 = SharedSpatial(2 * width, 2 * width, k=3)
        self.mix2 = SlotMix(tw, K=1)
        self.sp2 = SharedSpatial(2 * width, 2 * width, k=3)
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, x):
        if self.kind == "plain":
            return self.head(self.net(x).flatten(1))
        h = F.relu(self.lift(x))            # (B,W,8,H,W)
        h = F.relu(self.sp1(self.mix1(h)))  # (B,2W,8,H,W)
        # spatial downsample (per slot)
        B, C, S, H, Wd = h.shape
        h = F.max_pool2d(h.reshape(B * C * S, 1, H, Wd), 2)
        h = h.reshape(B, C, S, H // 2, Wd // 2)
        h = F.relu(self.sp2(self.mix2(h)))  # (B,2W,8,.,.)
        h = h.mean(dim=2)                   # group pool
        h = h.mean(dim=(-2, -1))            # spatial pool
        return self.head(h)


# ---- equivariance verification (canonical gate) ----------------
def _permute(g0, f):
    """Pure-permutation group action on a lifted feature: f[g]->f[g0^{-1}g]."""
    out = torch.empty_like(f)
    for g in ELS:
        out[:, :, IDX[g]] = f[:, :, IDX[G.mul(G.inverse(g0), g)]]
    return out


def test_equivariance():
    torch.manual_seed(0)
    out = {}
    x = torch.randn(2, 2, 16, 16)
    # rho is a group action
    e = 0.0
    for g1 in ELS:
        for g2 in ELS:
            e = max(e, (rho_input(g1, rho_input(g2, x))
                        - rho_input(G.mul(g1, g2), x)).abs().max().item())
    out["rho is a group action"] = e

    lift = LiftConv(2, 4, k=5).eval()
    with torch.no_grad():
        Lx = lift(x)
        e = 0.0
        for g0 in ELS:
            lhs = lift(rho_input(g0, x))         # pure permutation convention
            e = max(e, (lhs - _permute(g0, Lx)).abs().max().item())
        out["lift equivariance (permutation)"] = e

        # LINEAR slot mix commutes with the permutation action
        sm = SlotMix(twisted=False, K=2).eval()
        f = torch.randn(2, 4, 8, 12, 12)
        e = 0.0
        sf = sm(f)
        for g0 in ELS:
            # the permutation acts on the K-replicated output blockwise
            lhs = sm(_permute(g0, f))
            rhs = _permute(g0, sf)
            e = max(e, (lhs - rhs).abs().max().item())
        out["linear slot-mix equivariance"] = e

        # SharedSpatial commutes with the permutation
        sp = SharedSpatial(4, 4, k=3).eval()
        e = 0.0
        sf = sp(f)
        for g0 in ELS:
            e = max(e, (sp(_permute(g0, f)) - _permute(g0, sf)).abs().max().item())
        out["shared-spatial equivariance"] = e

        # FULL linear net: output invariant under input rotation rho(g0)
        net = D4DefectNet(kind="linear", width=6).eval()
        e = 0.0
        base = net(x)
        for g0 in ELS:
            e = max(e, (net(rho_input(g0, x)) - base).abs().max().item())
        out["full linear net D4-invariance"] = e
    return out


if __name__ == "__main__":
    for k, v in test_equivariance().items():
        print(f"{k:38s} max err = {v:.2e}")
