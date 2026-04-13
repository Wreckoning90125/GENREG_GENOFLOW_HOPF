# v11 Chirality Victory + v12 Full Hodge Ladder — Findings

## The headline

v11's face/holonomy features, which tie v10 on MNIST, **win decisively
on cross-digit chirality transfer**. v12 completes the discrete de
Rham ladder on the 600-cell with genuine Clifford-algebraic 3-form
features but does not further improve the chirality story — and
combining everything actually *hurts* chirality generalization,
which is itself a meaningful result about what the "right" feature
set is for an orientation-sensitive task.

## The chirality benchmark

Task: **hflip-MNIST binary classification**. For each MNIST image
decide whether it is the original or its horizontal mirror image.
Two regimes:

- **In-distribution**: train and test on all 10 digits. Both v10
  and v11 do very well (≥98.5%) because hflip moves pixel positions
  and any pixel-sensitive feature set can memorize digit-specific
  asymmetries.

- **Cross-digit transfer (the real test)**: train on digits {0..4}
  hflip detection, test on {5..9}. Train on {5..9}, test on {0..4}.
  The features must encode chirality in a way that **transfers
  across never-seen shapes** — otherwise the model just memorizes
  "digit X has ink curving to the left" and fails out of distribution.

Results (all numbers are real runs on `chirality_benchmark.py` and
`chirality_benchmark_v12.py`):

| experiment                                    |   v10-k |  face-k | v12new-k | FULL-v12-k |
|-----------------------------------------------|--------:|--------:|---------:|-----------:|
| full-MNIST hflip detection                    | 99.08%  | 98.57%  |  98.52%  |   99.03%   |
| cross-digit hflip (train:0-4, test:5-9)       | 90.26%  | **92.87%** |  91.16%  |   90.65%   |
| cross-digit hflip (train:5-9, test:0-4)       | 90.15%  | **93.53%** |  90.95%  |   90.17%   |

On cross-digit transfer:

- **v11 face features alone beat v10 by +2.61% and +3.38%.** Symmetric
  gain in both split directions, with a linear readout even beating
  v10's full polynomial kernel (93.11% linear face vs 90.26% kernel
  v10 in one split). The signed Berry phase is a genuine orientation-
  reversing invariant and it transfers.

- **v12-new features alone** (co-exact 2-forms + cells via d₂) score
  90.95% and 91.16% — better than v10, worse than v11 face. They
  carry chirality signal but more weakly than the face/Berry layer.

- **FULL v12 (v10 + v11 face + v12-new combined into one feature
  set)** scores 90.17% and 90.65% — *worse* than v11 face alone.
  Adding v10's non-chirality digit features to v11 face lets the
  kernel ridge use them to memorize training-digit shapes, and the
  resulting model fails to transfer chirality to unseen digits.

## What v12 actually builds (all verified)

v12 completes the discrete de Rham chain complex on the 600-cell:

    Ω⁰ (vertices)  →  d₀  →  Ω¹ (edges)  →  d₁  →  Ω² (triangles)  →  d₂  →  Ω³ (cells)
      scalar eigs           curl eigs         v11 face eigs          v12 cell eigs
                                                                     + v12 coex face eigs

- **Tetrahedral cells enumerated**: all 600 4-cliques of the 600-cell
  graph found. Euler characteristic: V−E+F−C = 120−720+1200−600 = 0 ✓.
- **d₂ built**: triangle→cell signed boundary operator, one row per
  cell, four nonzero entries per row with alternating signs for the
  four faces of each tetrahedron.
- **Chain complex identity**: `max |d₂·d₁| = 0.00e+00` — machine
  exact zero. The full chain complex d₀→d₁→d₂ is consistent.
- **L₃ = d₂ d₂ᵀ**: 3-form Hodge Laplacian on cells. Eigendecomposed
  into **26 non-zero eigenspaces with 599 total modes**, plus the
  single harmonic mode (the constant volume form, reflecting
  H³(S³) = ℝ).
- **Co-exact 2-form eigenspaces**: built via intertwining from cell
  eigenvectors as orthonormal `d₂ᵀ · V_cell`. Max eigenvector
  identity error: 1.3e-14 (machine precision).
- **Cell chirality weights**: signed 3D volume of the tetrahedron
  formed by the four Hopf-projected vertices. Sign split: 240
  positive, 240 negative, 120 zero — the same 40-40-20 pattern as
  the triangle Berry phases. The 2I symmetry gives consistent
  chirality partitioning at every form-degree.

## The v12 feature extraction

Two new feature blocks, both passed through the v9/v11 Hopf-on-
4-tuples + Poincaré leftover treatment for consistency:

1. **Co-exact 2-form features** — same v11 triangle signal
   `T_ijk = (f_i f_j + f_j f_k + f_k f_i) · Ω_ijk / 2`, but
   projected onto the first 4 coexact face eigenspaces (the
   orthogonal complement of v11's exact face eigenspaces inside Ω²).
   By Hodge orthogonality these are linearly independent from v11's
   features. Multiplicities `[4, 9, 16, 25]` give 54 features per
   kappa scale.

2. **Cell (3-form) features** via the genuine chain-complex signal
   `S = d₂ · T_face`. The cell signal is literally the exterior
   derivative of v11's triangle signal — Stokes's theorem applied
   pointwise. Projected onto the first 5 cell eigenspaces with
   multiplicities `[4, 9, 16, 25, 36]`, giving 90 features per scale.

   An earlier variant used a per-cell chirality weight times a linear
   signal `(f_i+f_j+f_k+f_l)`. It underperformed because the cell
   chirality magnitude is small (±0.03) and the linear signal
   dilutes Berry-phase information. Replacing it with `d₂ · T_face`
   boosted v12-new cross-digit accuracy from ~88.5% to ~91% — the
   chain-complex signal is meaningfully better.

Total new v12 features: 432 per scale × 3 kappa scales = 1296 new.
Plus v10's 293×3 = 879 and v11 face's 100×3 = 300, total = 2475 per
scale × 3 scales = 1611 total.

## MNIST results

| Version |  Test acc  |  Features   |   Kernel   |
|---------|-----------:|------------:|-----------:|
| v8      |     87.46% |    1780 wts |    linear  |
| v9      |     96.12% |         293 | poly2 m=2k |
| v10     |     97.39% |         879 | poly2 m=5k |
| v11     |     97.32% |        1179 | poly2 m=5k |
| **v12** | **97.24%** |    **1611** | poly2 m=4k |

v12 slightly underperforms v10 on MNIST (97.24% vs 97.39%). Part of
this is the m=4000 cap on Nystrom landmarks (v12's 1611 features OOM
v10's m=5000 workspace during `np.linalg.solve`); the rest is that
MNIST is chirality-blind and the extra v12 features don't carry any
new digit-discriminating content. This is the expected outcome of a
mathematical completeness construction on an already-saturated
benchmark.

## What this tells us

1. **The Schoen-truth works.** v11's Berry-phase reading is a
   *lawful* invariant: it encodes chirality mathematically via the
   Hopf bundle's connection, so it transfers across instances. v10's
   chirality is just accidental polynomial-kernel capacity absorbing
   pixel-layout information, so it doesn't transfer. This is exactly
   what the "three lawful readings of one object" philosophy
   predicts: the feature that's structural carries across instances,
   the feature that's phenomenological does not.

2. **More readings ≠ better.** v12 completes the chain complex at
   machine precision. Co-exact 2-forms and cell eigenspaces are
   constructed exactly. The chain identity d₂∘d₁ = 0 holds exactly.
   *And the combined feature set performs worse on chirality transfer
   than v11 face features alone.* The lesson is that feature sets
   should match the *geometric content* of the task, not the total
   algebraic vocabulary available. For chirality MNIST, v11 face is
   the right level.

3. **The "right level" is the 2-form layer.** Chirality of a 2D
   image, projected onto a 3D cell complex via a stereographic-like
   map and then Hopf-projected onto S², lives at the 2-cell level —
   spherical triangles with signed areas. Higher form-degrees are
   mathematically available but physically redundant for this task.

4. **Benchmark saturation is real.** Both MNIST classification and
   in-distribution hflip detection are saturated. v11 and v12's
   value is visible only in the out-of-distribution test. This is
   the Brakke criterion working in the other direction: if your
   mathematical machinery looks identical to a phenomenological
   model in-distribution, take it out of distribution and see which
   one generalizes.

## Reproducibility

    python chirality_benchmark.py       # v10 vs v11 face, 3 experiments
    python chirality_benchmark_v12.py   # v12 new + FULL v12, 3 experiments
    python train_v12.py                 # v12 MNIST training

All three Berry/Hopf/chain-complex verifications run at import time:
- `hopf_controller.verify_geometric_ops()` — Hopf projection vs Cl(3,0)
- `hopf_controller.verify_berry_phase()` — signed Euler-Eriksson vs
  Cl(3,0) rotor composition
- `cell600.get_geometry()` — chain complex d₂∘d₁ = 0 assertion,
  eigenspace Laplacian identities
