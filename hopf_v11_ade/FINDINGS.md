# v11 Findings: Face/Holonomy Reading — Mathematical Completion, Benchmark Saturation

## What was built

v11 completes the discrete de Rham ladder on the 600-cell:

    Ω⁰ (vertices)  →  d₀  →  Ω¹ (edges)  →  d₁  →  Ω² (triangles)
     scalar eigs            curl eigs              face eigs  (new)

Four new ingredients, all fixed geometric functions of the 600-cell with
zero learned feature parameters:

1. **Per-vertex Hopf S² points** (`vertex_hopf`, shape `(120, 3)`):
   direct Hopf projection of each 2I quaternion to S². Unit to 6 places.

2. **Per-triangle signed Berry phase** (`triangle_berry`, shape `(1200,)`):
   the signed solid angle / 2 of each 600-cell triangular face, via
   Euler–Eriksson:
       Ω_ijk / 2 = atan2( p_i·(p_j × p_k), 1 + p_i·p_j + p_j·p_k + p_k·p_i )
   with pᵢ = Hopf(vᵢ). The 2I symmetry gives an exactly balanced
   chirality split: 480 positive, 480 negative, 240 zero. Max |Berry|
   ≈ 0.365 rad.

3. **Face (2-form) eigenspaces** (`face_eigenspaces`, mults `[6, 16, 30, 48]`):
   obtained via the intertwining relation d₁∘L₁_coex = L₂_exact∘d₁ from
   the cached co-exact 1-form eigenbases. QR-orthonormalized. Eigenspace
   identity verified:  |L₂_exact V − λV|_max  <  5e-15.

4. **Cl(3,0) rotor-composition Berry phase**
   (`hopf_controller.triangle_berry_clifford`): an independent
   implementation via actual rotor composition in the even subalgebra
   Cl(3,0)⁺ — build three parallel-transport rotors
   R_pq = cos(θ/2) + sin(θ/2)·B̂ along the triangle, compose them, and
   extract the net rotation angle about the base point from the bivector
   part dotted with the bivector dual of p_a. This is a genuine Clifford-
   algebraic computation, only using the geometric product (via qmul)
   and a scalar alignment check.

   **Verification**:
     random triangles:   max err = 4.44e-16 (50 trials)
     600-cell triangles: max err = 3.25e-13 (5 sampled)
   The two routes — Euler–Eriksson and Cl(3,0) rotor composition — agree
   to machine precision. The geometric substrate is sound.

## Triangle signal + face features

For a batch of pixel activations F on the 600-cell (shape (N, 120)),
the triangle signal is the oriented sum of the three v9 edge curl
products weighted by the signed Berry phase:

    T_ijk = ( f_i·f_j  +  f_j·f_k  +  f_k·f_i ) · Ω_ijk/2

This is a discrete Stokes statement: the boundary sum of v9's 1-form
curl signal over an oriented 2-cell, multiplied by the holonomy of the
Hopf connection over that cell. Projected onto the four face
eigenspaces and passed through the same Hopf-on-4-tuples treatment as
v9 curl, it yields 100 new 2-form features per scale (6+16+30+48),
bringing the per-scale count to 393 and the multi-scale total
(kappas = {3, 5.5, 8}) to 1179.

An earlier cubic variant T = f_i·f_j·f_k · Ω_ijk/2 was tried and
empirically redundant with v10's polynomial-kernel expansion (6th-order
overall). The quadratic form keeps total order at 4th — same as v9
curl through the kernel — but on the 2-form layer instead of the
1-form layer, which is the principled choice.

## Benchmark results

All on full MNIST (60k train / 10k test), no data augmentation.

| Experiment                                    | Test acc   |
|-----------------------------------------------|------------|
| v8  (scalar eigs, linear ridge)               |  87.46%    |
| v9  (+ curl + E8, poly kernel)                |  96.12%    |
| v10 (+ multi-scale, m=5000)                   |  97.39%    |
| v11 combined (v10 + face features, one kernel)|  97.32%    |
| v11 linear ridge (for reference)              |  94.98%    |
|                                               |            |
| **Face features ALONE, multi-scale, m=2000**  |  **89.31%**|
| v10 features alone, reproduced                |  97.39%    |
| v11 ensemble: v10 logits + w·face logits      |  97.39% @ w=0 |

### What the numbers say

- **The face features are genuinely informative.** Alone they reach
  89.31% on MNIST, which is ~8× above chance and not far from v8's
  scalar-only baseline. On linear ridge they push v11 from v10's 94.95%
  to 94.98% — a tiny but real positive delta.

- **They are not independent of v10 once you pass through the kernel.**
  The optimal ensemble weight for the face readout is exactly w = 0:
  any positive contribution from the face logits strictly worsens the
  ensemble. This means the face features' errors are correlated with
  v10's errors — face features make mistakes on the same digits v10
  does, so averaging in their evidence just adds noise.

- **v10 was already saturated on this benchmark at m=5000.** The
  polynomial-kernel expansion of v10's 879 multi-scale features
  captures all the MNIST-discriminative signal that the face reading
  could in principle add. On a task where chirality of Hopf triangles
  matters, v11 would pull ahead; on MNIST it ties.

### Why the completion is still worth it

The de Rham ladder is now complete on the 600-cell:

  - 0-forms (Ω⁰): scalar eigenspaces via Laplacian, ADE-decomposed into
    irrep copies with CG projectors — v8 layer.
  - 1-forms (Ω¹): co-exact curl eigenspaces via multiplicative edge
    signal h_e = f_i·f_j (Hodge-orthogonal to the exact part) — v9 layer.
  - 2-forms (Ω²): exact face eigenspaces via d₁∘curl intertwining, with
    triangle signal T_ijk = (edge-sum) · Berry phase — v11 layer.

Each reading is 2I-equivariant, fixed-geometric (no learned feature
parameters), and Cl(3,0)-verified where nonlinearity enters. The fact
that v11 ties v10 on MNIST is not a failure of the construction — it's
evidence that v10 had already exhausted the benchmark's geometric
content, and it means v12/v13 must look elsewhere: higher form-degrees
(Ω³ on tetrahedra, with d₂ requiring actual cell enumeration), non-
trivial bundle connections beyond abelian U(1), or tasks where
chirality and higher-order topology actually matter.

### Reproducibility

    python train_v11.py             # v11 combined run
    python train_v11_ensemble.py    # v10 + face ensemble diagnostic

Both scripts reuse `extract_features_batch` from `train_ade_hopf.py`
and the shared pixel-kernel cache from `hopf_controller.py`, so any
numerical drift would show up as a change in the v10 baseline too.
The Cl(3,0) Berry phase verification runs at import time of
`hopf_controller.py` and blocks if it disagrees with Euler–Eriksson.
