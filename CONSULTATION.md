# CONSULTATION — Layer 2 of the Hopf → Stellarator program

This document is a consultation framing. It exists so a second source can
be brought in (textbook, paper, or another model session) to answer
specific questions before the next round of code lands. The point is to
stop guessing where the right answer is one citation away.

## Repo state

- **Main branch**: `wreckoning90125/genreg_genoflow_hopf` @ `10b234d`. The
  research program lives in `RESEARCH_PROGRAM.md` (six-layer stack from
  generalized-Hopf seed through topology-preserving relaxation, surface
  extraction, geometric chart, equilibrium + QS optimization, and SPEC
  for non-nested cases).
- **Active branch**: `claude/layer1-hopf-seed-verification-diagnostics`.
- **Pre-existing 2I infrastructure** (do not modify; production code with
  downstream MNIST / QM9 consumers):
  - `cell600.py` — 120 vertices on S^3 as quaternions, 720 oriented
    edges, 1200 triangles, 600 tetrahedra, discrete coboundaries
    `d0, d1, d2`, scalar eigenspaces with multiplicities
    `[1, 4, 9, 16, 25, 36, 9, 16, 4]`, curl eigenspaces
    `[6, 16, 30, 48]`, face / cell eigenspaces with chirality data,
    McKay correspondence to E8 wired in (`e8_marks`,
    `eigenspace_to_e8`).
  - `hopf_controller.py` — Hopf map S^3 -> S^2 (`hopf_project`), section,
    Pancharatnam phase, signed solid-angle Berry, Cl(3, 0) rotor-
    composition Berry (`triangle_berry_clifford`), all cross-verified
    to <1e-8.
  - `ade_geometry.py` — orbit-method irrep copy decomposition, CG
    projector to V_1, ADE wiring on top of the cell600 spectra.

## Layer 1 — what was just shipped

Eight new files, ~1660 lines, no existing file modified. Reuse via
import only.

| File | Role |
|------|------|
| `hopf_seed.py` | Closed-form `B(x; omega1, omega2)` and `A` via Clebsch pair: `B = Im(grad phi_A x grad phi_B)`, `A = Im(phi_A grad phi_B)`, with `phi_A = u^omega1`, `phi_B = v^omega2` on stereographic-lifted S^3 coords. Gives `analytic_iota = omega1 / omega2`, `analytic_linking_number = omega1 * omega2`, `analytic_helicity = 2 pi^2 * nm * n!m! / (n+m)! * R^4` (empirically determined to machine precision; derivation open — see Q2). |
| `hopf_grid.py` | Cartesian grid sampling, central-difference div / curl, Simpson helicity, boundary-flux check. `div B -> 0` as `O(h^2)`; helicity matches analytic formula to `4 x 10^-4` at bbox=8 res=64. |
| `hopf_600cell_witness.py` | **The compromised module.** Stereographic projection of 120 S^3 vertices to R^3, samples B at vertices and A as edge line-integrals (720 edges), then `np.linalg.lstsq(d0, A_edges)` for a Hodge-style split. Confirms `d1 d0 = 0`, `d2 d1 = 0` structurally. **Ignores the entire irrep structure already in `cell600.get_geometry()`.** This is the file that needs to be rewritten in 2I-irrep coordinates. |
| `hopf_fieldlines.py` | LSODA tracer + Poincaré section + linear-fit iota recovery. iota recovered to 1–5% at `(1, 1)`, `(3, 2)`, `(2, 3)`. |
| `hopf_berry_diagnostic.py` | Pancharatnam + Cl(3, 0) rotor-fan Berry phase, both routes agree to ~1e-14 along traced field lines. `(1, 1)` closed loop gives exactly `2 pi`. |
| `hopf_io.py` | HDF5 + VTK (.vti) + JSON sidecar, round-trip tested. |
| `hopf_layer1_cli.py` | End-to-end runner. |
| `tests/test_layer1.py` | 24 cases, all pass. |

## The thesis driving this consultation

Symmetric computation lives natively in tensor categories indexed by
the irreps of the symmetry group. A physical / geometric object lives
in a representation; symmetry decomposes that representation;
computation should happen inside those pieces, not after flattening
everything into soup.

The naïve framing — "|G|-fold speedup" — is only exact for
fundamental-domain simulation. For spectral block-diagonalization on a
non-regular representation, you get a ragged speedup: the cost is the
sum over irreps `rho` of `(d_rho * m_rho)^3` (or smaller, depending on
the operation), where `d_rho` is the irrep dimension and `m_rho` is its
multiplicity in the representation being decomposed. For 2I (order 120)
acting on 720 edges, the speedup is real but not a clean `120 x`. What's
exact is that the *information* in a G-symmetric problem is no larger
than the dimension of its irrep decomposition, so any computation that
respects the symmetry is bounded below by that dimension. That bound is
typically a small multiple of `|G|^{-1}` of the unconstrained dimension
in expectation, not the worst case.

For Layer 2 specifically: the controller / witness should not merely
*observe* symmetric features; its intermediate states should remain
equivariant objects. That is the difference between "geometry as input
decoration" and "geometry as compute substrate."

## Open questions

### Q1 — 2I-irrep decomposition of the 720 edges  [RESOLVED]

**Answer.** 2I acts on the 120 vertices by left multiplication, and the
action is **free** (left-mult by `g != e` fixes no vertex). An edge
`{v, w}` is fixed by `g` iff `g . v = v` (forces `g = e`) or
`g . v = w and g . w = v` (forces `g = w v^{-1}` with `g^2 = e`, so
`g = -1` and `w = -v`). But `w = -v` puts `w` antipodal to `v` at
angular distance `pi`, which is **not** a 600-cell edge (edges are at
angular distance corresponding to the unit-edge-length regular polychoron,
i.e., neighbours at golden-ratio-derived angles, not antipodes). So 2I
acts freely on the 720 edges as well.

Therefore `chi_perm(e) = 720`, `chi_perm(g != e) = 0`, equivalently
`chi_perm = 6 * chi_reg`. The multiplicities are `m_rho = 6 * dim(rho)`:

| irrep | dim | multiplicity | total contribution |
|-------|-----|--------------|--------------------|
| 1     | 1   | 6            | 6                  |
| 2a    | 2   | 12           | 24                 |
| 2b    | 2   | 12           | 24                 |
| 3a    | 3   | 18           | 54                 |
| 3b    | 3   | 18           | 54                 |
| 4a    | 4   | 24           | 96                 |
| 4b    | 4   | 24           | 96                 |
| 5     | 5   | 30           | 150                |
| 6     | 6   | 36           | 216                |
|       |     |              | **720**            |

```
rho_720 = 6 . 1  (+)  12 . 2a  (+)  12 . 2b  (+)  18 . 3a  (+)  18 . 3b
       (+)  24 . 4a  (+)  24 . 4b  (+)  30 . 5  (+)  36 . 6
```

This is the **2I (left)** action. The full 600-cell symmetry group is
H4 of order 14,400 = `(2I x 2I) / Z_2 |x| Z_2`; the H4-irrep
decomposition is a different and larger calculation. For the irrep-
native Hodge decomposition we use the 2I-grading because the existing
eigenspaces in `cell600.py` are 2I-graded (matching multiplicities
`[1, 4, 9, 16, 25, 36, 9, 16, 4]` on vertices, which is the d^2 pattern
of 2I irreps in the regular representation of order 120).

Verified: `Sigma m_i . dim_i = 720` and `chi_perm == 6 . chi_reg` both
true.

Implication for the cochain complex: with `H^1(S^3) = 0`, the harmonic
part of any 1-form is identically zero, so `1-form-edges = image(d0) (+)
image(d1.T)` exactly. Dimension accounting:

- `image(d0)` has dim `120 - 1 = 119` (kernel = constant functions, the
  trivial isotypic component on vertices). By isotypic component:
  0 (trivial) + 4 + 4 + 9 + 9 + 16 + 16 + 25 + 36 = 119.
- `image(d1.T)` has dim `720 - 119 = 601`. By isotypic component:
  6 + 20 + 20 + 45 + 45 + 80 + 80 + 125 + 180 = 601.

That's the dimension table the irrep-native Hodge decomposition has to
reproduce as a hard test, not a soft tolerance.

### Q2 — Analytic derivation of `H = 2 pi^2 . nm . n!m! / (n+m)! . R^4`  [PARTIAL]

**Status.** Empirical formula verified to 1e-12 across 18 (n, m) pairs.
DLMF-anchored derivation recipe is in hand, full write-out deferred.

**Recipe (from DLMF §5.12 + §14.30 + §34.3).** The closed form factors
as

```
n! m! / (n + m)! = (n + m + 1) . B(n + 1, m + 1)
```

where `B` is the beta function (DLMF §5.12). So

```
H / (2 pi^2 R^4) = nm (n + m + 1) B(n + 1, m + 1)
```

which makes explicit what integral collapses to give the formula. The
volume integral on R^3 reduces, via the inverse stereographic Jacobian
and the (n, m) Hopf rational map separating the two angular factors, to
a single beta integral on `[0, 1]`.

Pieces of DLMF actually used:

- §5.12: `B(a, b) = integral_0^1 t^(a-1) (1 - t)^(b-1) dt
                  = Gamma(a) Gamma(b) / Gamma(a + b)` — produces the
  `n! m! / (n + m + 1)!` ratio.
- §14.30: explicit spherical harmonics — for the angular factors.
- §34.3: 3j symbols / Gaunt's integral — for the cross-product of two
  Y_lm on S^2 that arises in `A . curl A`.
- §1.14 (table 1.14.5): Mellin transforms — alternate route via
  Mellin–Barnes if the radial integral is ill-conditioned.

Sketch. Write the (n, m) Hopfion as
`B = grad phi_n  x  grad phi_m / (1 + |z|^2)^2` with
`phi_n = arg(z_1^n z_2^m)` on S^3. Push to R^3 by inverse
stereographic. Integrate `A . B` in spherical R^3 coords. The angular
part collapses by Y_lm orthogonality, leaving an integer factor of nm.
The radial part is

```
integral_0^infty r^{2(n + m) - 1} (1 + r^2)^{-(n + m + 2)} dr
```

which by `t = r^2 / (1 + r^2)` is, up to factors, a beta integral that
gives the `(n + m + 1) B(n + 1, m + 1)` factor. The `2 pi^2 R^4` overall
is the S^3 volume in this normalization.

Locally derivable in one careful pass — does not require external
lookup once the angular orthogonality is in hand. Open in this repo:
write it out as a docstring or a `helicity_derivation.md` companion to
the closed form. Not load-bearing for Layer 2.

### Q3 — Stellarator pipeline: where do Hopfion seeds actually fit?  [RESOLVED]

**Answer.** Not in the QA/QH dichotomy. In the omnigenity / quasi-
isodynamic sector, if anywhere.

Empirical structure of the Landreman QUASR catalogue (371,701 quasi-
symmetric configurations):

- `helicity in {0, 1}` is the **integer helical-period class M** of
  the quasi-symmetry direction `B = B(psi, M . theta - N . phi)`.
  `M = 0` -> QA (200,946 entries). `M = 1, N = nfp` -> QH (170,755
  entries). **NOT the volumetric Hopf invariant.** It's the topological
  label of the magnetic-axis self-linking type — a sibling-but-not-
  equal invariant.
- iota is a **continuous output**, not a topology selector. The
  catalogue sweeps iota over wide ranges within each (nfp, helicity)
  sector. So "rational iota Hopfion seed" doesn't compete with the
  standard pipeline for topological reasons — the standard pipeline
  doesn't constrain iota to be rational at all.
- The current pipeline IS Garren–Boozer near-axis expansion + VMEC /
  DESC refinement (Landreman & Sengupta; Mapping the space of QS
  stellarators using optimized near-axis expansion; etc.).

iota by sector:

```
QA  (helicity = 0):  iota in [0.10, 0.90]  across all nfp  (mean ~0.40-0.46)
QH  (helicity = 1):  iota in [0.50, 4.50]  shifts up with nfp
                       nfp=2: [0.5, 2.5]   nfp=3: [0.5, 2.6]
                       nfp=4: [1.1, 2.6]   nfp=5: [1.1, 2.6]
                       nfp=6: [2.5, 4.5]   nfp=7: [2.5, 4.5]   nfp=8: [2.5, 4.5]
```

QS quality distribution (qs_error): 304k under 1e-4, 159k under 1e-6,
38k under 1e-8.

**Implication for the program.** The Hopfion contribution is only
interesting if the (n, m) topology

- (a) furnishes a better near-axis ansatz than Garren–Boozer (unlikely
      on its face — GB is locally optimal in its sector), or
- (b) targets the **omnigenity / quasi-isodynamic** sector that lies
      outside the `helicity in {0, 1}` dichotomy, or
- (c) accelerates convergence to `qs_error < 1e-8` over GB+VMEC/DESC,
      with measurably better max_kappa or max_msc at fixed qs_error.

Strongest opening: (b). The omnigenity papers in the live literature
(Direct construction of optimized stellarator shapes III: omnigenity
near the magnetic axis; Weakly Quasisymmetric Near-Axis Solutions to
all Orders) point at the sector. **This re-orients Layer 1/2/3 in this
repo: the Hopfion seed isn't a competitor to QA/QH starts; it's a
candidate seed for the omnigenity sector.** That changes which
equilibrium codes we need to interface with downstream and which
metrics we report in the seed metadata sidecar.

This is the most consequential question that came back from
consultation. The next-phase plan needs to be written against this
reorientation.

### Q4 — `omega1 / omega2` vs stellarator iota convention  [PARTIALLY PINNED]

**Action taken.** `hopf_seed.CONVENTION` constant now pins
`omega1 = poloidal winding`, `omega2 = toroidal winding`,
`iota = omega1 / omega2 = poloidal / toroidal` to match the standard
stellarator convention. Naming clarified in module docstring: this
repo's "helicity" is always the volumetric Hopf invariant
`H = int A.B dV`, NOT the QUASR / Landreman `helicity` field which is
the integer `M in {0, 1}` qs_axis_class.

**Still open.** Verification against a live VMEC / DESC import format.
Required before the seed handoff in any QA/QH-handoff branch (which
per Q3/Q7 we are explicitly NOT taking). For the in-repo omnigenity
lab path, this is a non-issue.

### Q5 — Berry phase per `(n, m)`  [OPEN, low priority]

Still want the analytic prediction for closed-loop Pancharatnam phase
as a function of `(n, m)`. Diagnostic, not load-bearing.

### Q6 — Sketch the irrep-native `hodge_decompose`  [SCHEMA TEMPLATE FOUND]

The crystallography sqlite (SG230 only in current ingest) has a
`kg_irreps` / `kg_selection_rules` / `kg_spinor_factor_systems` schema
that ties (Wyckoff orbit) -> (irrep) -> (allowed/forbidden), which is
exactly the irrep-native projector wiring we want, just on Wyckoff
orbits in 3D crystallography rather than cochain spaces on the 600-cell.
The schema is `(factor_system_json, unitary_rotations_json,
metadata_json)` — directly cloneable.

For the immediate code refactor, we don't need the sqlite. We have
enough from Q1: the multiplicities, the existing `scalar_eigenspaces`
and `face_eigenspaces` in `cell600.py`, and the equivariance of `d0`
and `d1`. The plan:

```
For each scalar (vertex) eigenspace V (isotypic of irrep rho_i, dim d_i^2):
    image(d0|V) is a d_i^2-dim subspace of edges, sitting inside the
    rho_i-isotypic of edges (which has total dim 6 d_i^2).
    Project A_edges onto image(d0|V) -> exact part in rho_i.

For each face eigenspace W (isotypic of irrep rho_i):
    image(d1.T|W) is a subspace of edges, sitting inside the
    rho_i-isotypic of edges, orthogonal to image(d0|V).
    Project (A - exact) onto image(d1.T|W) -> coexact part in rho_i.

Residual: harmonic. By H^1(S^3) = 0, must be 0 to floating-point.
This is the hard test.
```

This avoids the explicit 720-edge group action — we use the existing
eigendecomposition as an oracle for the irrep grading. The longer-term
move is the sqlite ingest; the immediate move is the refactor.

### Q7 — Layer 2 framing: in-repo discrete relaxation, or pure handoff?  [RESOLVED → option (b)]

**Decision: build the in-repo omnigenity-relaxation lab. Skip the
QA/QH-handoff path.** Three reasons:

1. **The QA/QH market is saturated.** QUASR is 371,701 entries; the
   Garren–Boozer near-axis pipeline + VMEC / DESC refinement is a
   working stack with 37,936 configurations already at
   `qs_error < 1e-8`. A Hopfion-flavoured seed for QA/QH is a marginal
   improvement to a mature pipeline, competing on initial-guess
   quality. The differentiating contribution is not there.
2. **Omnigenity has weaker theoretical scaffolding and is the natural
   fit for the irrep machinery** already built. Omnigenity is the
   condition that bounce-averaged radial drift vanishes — equivalently,
   that field-line trajectories close (on average) under the bounce
   action. This is fundamentally a symmetry / group-theoretic
   condition. The 2I-irrep Hodge decomposition (machine-zero harmonic
   leak, exact `[0, 4, 9, 16, 25, 36, 9, 16, 4]` dimension matching)
   is exactly the tool that lights up here: omnigenity's bounce-action
   decomposition lives in invariant subspaces of some discrete
   symmetry, and finding the right one is an open problem. The
   600-cell with 2I action is a controlled lab where "which irreps
   host omnigenous fields and which don't" gets sharp answers.
3. **Layer 1 already paid the precision price.** Going from 3% harmonic
   leak to 6e-16 is a 13-orders-of-magnitude reduction in noise floor.
   That precision is wasted on QA/QH seed generation (downstream
   VMEC / DESC noise floor is ~1e-8 at best). But for a structural
   claim about which bounce-action equivalence classes are achievable
   in the 2I-symmetric sector, machine zero is the right precision: it
   lets you make rigorous "this irrep is in the kernel" claims, not
   just "approximately small."

**Stage plan (concrete, dependency-ordered):**

- **Stage 1 — metric Hodge star on the 600-cell.** Closed-form
  circumcentric weights from the icosian quaternion coordinates
  (option (i): exact under 2I, fast). Switch to Whitney-form Galerkin
  (option (ii)) only if (i) reveals spectral pathologies (negative
  eigenvalues from non-Delaunay cells, kernel larger than 2I
  predicts). 1-2 days.
- **Stage 2 — discrete bounce action on edges.** `J(alpha) = oint p
  parallel dl` along closed edge-paths. 2I orbits give 10
  representatives via the classical Hopf decomposition of the 600-cell
  (10 great circles of 12 vertices each, partitioning the 120
  vertices). 2-3 days.
- **Stage 3 — Kraus–Maj variational integrator.** Discrete Lagrangian
  on edges, structure-preserving symplectic integrator on the 1-skeleton.
  4-7 days. The unknown — Kraus–Maj on a discrete polytope is
  non-standard and may want a separate writeup before code.
- **Stage 4 — equivariant decomposition of the bounce action.** Compute
  `J_irrep[i] = <J, P_i J>` per irrep. Omnigenous configurations are
  precisely those where `J_irrep[i] = 0` for `i != trivial`. 1 day,
  uses the projectors already built in Layer 1.

Total: 1.5-2 weeks for a working omnigenity lab on the 600-cell. Honest
off-ramp: if Stage 1 reveals unfixable spectral issues, switch to
Whitney-form (1-2 weeks added). If THAT fails, the 600-cell is too
coarse and we go to the 120-cell or a tetrahedral subdivision. The
off-ramp adds time, doesn't kill the project.

The decision criterion (made explicit): this work is a **mathematical
physics contribution**, not a plasma engineering contribution. The
Layer 1 work already prioritized irrep-native precision over
engineering throughput; the lab is the consistent next move.

## What would close the remaining loop

In priority order:

1. **Stage 1 of the lab** — metric Hodge star, circumcentric weights.
   Closed-form on the 600-cell because of icosian symmetry. 1-2 days.
   Requires explicit go-ahead before swinging.
2. **Q2** — full DLMF-anchored derivation written out. Locally
   feasible; nice to have; not load-bearing for the lab.
3. **Q4 verification** — only required if any handoff to an external
   equilibrium code is contemplated. Not blocking the lab.
4. **Q5** — Berry-phase formula. Per consultation, this is a derived
   quantity from the bounce-action machinery, not a starting point.
   Defer until after Stage 2.
5. **Q6 sqlite ingest** — clone the SG230 schema for 2I / 600-cell.
   Long-term infrastructure, not blocking.
