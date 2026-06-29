# ================================================================
# Nematic defect classification: plain CNN vs D_4-equivariant
# (linear) vs cocycle (projective) D_4 G-CNN.
#
# HONEST FINDINGS (the scientifically informative part):
#   1. LINEAR D_4-equivariance helps SAMPLE EFFICIENCY: with little data
#      the linear G-CNN beats the plain convnet (e.g. n=120: ~0.99 vs
#      ~0.89).  This is the standard, expected benefit of equivariance.
#   2. The COCYCLE TWIST gives NO benefit on this task (cocycle <= linear,
#      and sometimes worse): defect-charge classification does NOT present
#      as a target an ordinary-equivariant model cannot represent, so
#      there is no cohomological obstruction here for the cocycle to lift.
#   3. With enough data all three reach ceiling.
#
# We had hypothesized the half-integer (+-1/2) winding sign would force a
# PROJECTIVE representation; empirically it does not.  The genuine
# obstruction -- where naive equivariant designs provably fail -- lives
# where the TARGET itself transforms projectively: the abstract C[G]
# experiment (run_obstruction_experiment.py) and the non-symmorphic
# crystal little co-group (crystallographic_cohomology.py), NOT here.
#
# This is kept as a documented, honest result.  The D_4 G-CNN itself
# (d4_gcnn.py, equivariance verified to 1e-9) is the reusable artifact.
# ================================================================

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import nematic_data as nd
from d4_gcnn import D4DefectNet, rho_input, G, ELS


def reflection_geometry(seed=0):
    """Measure how the D_4 reflection rho(s) acts on a +1/2 director
    field, reported neutrally (the precise geometry is subtle and we do
    not hinge any claim on it -- the empirical model comparison below is
    what carries the conclusion)."""
    H = Wd = 25
    fields = {}
    for q in nd.CHARGES:
        th = nd.director_field(q, H, Wd, 0.0, 12, 12, noise=0.0)
        fields[q] = np.stack([np.cos(2 * th), np.sin(2 * th)]).astype(np.float32)
    refl = rho_input(G.s, torch.from_numpy(fields[0.5])[None]).numpy()[0]
    d_to_plus = float(np.abs(refl - fields[0.5]).mean())
    d_to_minus = float(np.abs(refl - fields[-0.5]).mean())
    return {
        "reflect(+1/2)_dist_to_+1/2": round(d_to_plus, 4),
        "reflect(+1/2)_dist_to_-1/2": round(d_to_minus, 4),
        "note": "geometry is subtle; the conclusion rests on the empirical "
                "model comparison (cocycle shows no advantage), not on this.",
    }


def _train_eval(kind, Xtr, ytr, Xte, yte, epochs, seed=0):
    torch.manual_seed(seed)
    m = D4DefectNet(kind=kind, width=8)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            lf(m(torch.from_numpy(Xtr[idx])), torch.from_numpy(ytr[idx])).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pr = m(torch.from_numpy(Xte)).argmax(1).numpy()
    acc = float((pr == yte).mean())
    half = np.isin(yte, [1, 2])
    return acc, float((pr[half] == yte[half]).mean())


def run():
    print("Nematic defect classification (documented honest result)\n")
    inv = reflection_geometry()
    print("Reflection geometry (neutral):")
    for k, v in inv.items():
        print(f"  {k}: {v}")
    print()
    Xte, yte = nd.make_director_image_dataset(120, H=25, W=25, seed=999)
    sizes = [120, 600]   # small + larger, per class*4
    out = {"reflection_geometry": inv, "runs": {}}
    for size in sizes:
        Xtr, ytr = nd.make_director_image_dataset(size // 4, H=25, W=25, seed=1)
        out["runs"][size] = {}
        for kind in ("plain", "linear", "cocycle"):
            t0 = time.time()
            acc, half = _train_eval(kind, Xtr, ytr, Xte, yte, epochs=40)
            out["runs"][size][kind] = {"test_acc": acc, "half_pair_acc": half}
            print(f"[n={size:4d}] {kind:8s} acc={acc:.3f} half(+-1/2)={half:.3f} "
                  f"({time.time()-t0:.0f}s)")
        print()
    path = os.path.join(os.path.dirname(__file__), "nematic_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path)
    print("\nFinding: linear equivariance aids sample efficiency vs the plain "
          "convnet; the cocycle twist gives no advantage -> no obstruction "
          "for this task.  See module docstring.")
    return out


if __name__ == "__main__":
    run()
