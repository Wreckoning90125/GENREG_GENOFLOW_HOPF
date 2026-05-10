# Icosahedral quasicrystal experiments

Self-rolled cut-and-project icosahedral quasicrystal generator + a
two-task benchmark that demonstrates **when** irrep-decomposed
geometric features add signal and **when** they don't.

## Why this exists

The clean Python equivalent of `spglib + spgrep + pyxtal` does not
exist for quasicrystals — icosahedral symmetry is incompatible with
translation, so the 230-space-group toolchain stops at the boundary.
This subdirectory rolls the missing piece (cut-and-project from the
6D simple-cubic lattice with a rhombic-triacontahedron window),
verifies it against canonical published properties, and uses it on
two head-to-head ML tasks that show what irrep-decomposed features
buy you.

## Verification — 17 tests pass

`tests/test_qc_canonical.py`. All assertions are against either
canonical published values or analytic identities, with `mpmath`
used at 30–100 digit precision where rep-theory invariants would
otherwise drift numerically:

- **Golden-ratio identities** (τ² = τ + 1, 1/τ = τ − 1) at 100-digit
  precision via `mpmath`.
- **Icosahedral group I structure**: 60 elements, conjugacy classes
  {1, 12, 12, 20, 15}, all proper-rotations (det = +1, orthogonal).
- **Character table orthogonality** (Cotton, Table A.20):
  ⟨χ_i, χ_j⟩ = δ_ij to residual < 10⁻⁴⁵ at 50-digit `mpmath` precision.
- **Icosahedral invariant counts via the SO(3) → I character formula**
  matching the canonical Molien series
  P(t) = (1 + t¹⁵) / [(1 − t⁶)(1 − t¹⁰)] for all l ≤ 30. Notable
  fingerprints (l=15 has 1 invariant; l=30 has 2) are exercised.
- **Cut-and-project structural**: every accepted vertex's perp
  coord is inside the rhombic triacontahedron window; the 6D→3D+3D'
  split is an exact isometry to machine precision; phys and perp
  3D subspaces are 6D-orthogonal to <10⁻¹³.
- **Icosahedral symmetry of the QC vertex set**: every R ∈ I maps
  the vertex set to itself (median R-survival > 80% at finite cutoff,
  100% for the identity, expected behavior near the cutoff boundary).
- **τ³ self-similarity** (Levine & Steinhardt 1986, Eq. 2.3):
  τ³-inflation maps the QC onto itself; > 65% of inflated vertices
  that fall in the original physical box are present in the original
  vertex set at finite cutoff.
- **Density**: vertex density in 3D is order vol(perp window) =
  4·a³·√(5 + 2√5) for edge length a = 1/√2, matched to within
  finite-cutoff Diophantine-discrepancy oscillation (well-known
  for τ-irrational projections; Beck 1989).
- **`spgrep` sanity check**: the 230-group machinery returns
  sensible irrep structure for cubic m-3m (Oh) — the toolchain works
  on the periodic side, just not on the icosahedral one we had to
  roll ourselves.

## Two-task benchmark — when do irrep-decomposed features help?

`bench_vertex_class.py`. Same QC, same 12 nearest neighbors per
vertex, same train/test split, same ridge classifier — only the
feature pipeline differs.

### Task A: vertex-class classification (rotation-invariant target)

Predict which quantile of ||perp|| the vertex falls in (6 classes).
||perp|| is a scalar, invariant under any rotation of the QC.

| Pipeline | n_feat | test acc | std (5 seeds) |
|---|---:|---:|---:|
| naive (NN distances, rot-invariant) | 12 | 0.639 | 0.008 |
| steinhardt Q_l, l ∈ {2,4,6,8,10,12} | 6 | 0.532 | 0.010 |
| icosahedral Q_l, l ∈ {6,10,12} | 3 | 0.479 | 0.008 |
| cell600 ADE eigenspace norms | 9 | 0.305 | 0.002 |
| cell600 ADE coefficients (direction-aware) | 30 | 0.236 | 0.007 |
| **naive + steinhardt** | **18** | **0.677** | **0.004** |

**Reading**: when the target is rotation-invariant, naive distance
features have direct access to the discriminating signal. The
rotation-invariant Q_l features add a +3.8% increment when stacked
with naive distances. Direction-aware features *hurt* because they
add variance without signal.

### Task B: perp-coordinate regression (direction-equivariant target)

Predict the perp-space x-coordinate of each vertex from its local
3D environment. Rotating the QC rotates the perp space too — the
target is *not* rotation-invariant.

| Pipeline | n_feat | R² test | std (5 seeds) |
|---|---:|---:|---:|
| naive (rotation-invariant) | 12 | −0.003 | 0.001 |
| steinhardt (rotation-invariant) | 6 | −0.001 | 0.001 |
| cell600 ADE norms (rotation-invariant) | 9 | −0.002 | 0.001 |
| **cell600 ADE coefficients (direction-aware)** | **30** | **+0.378** | **0.013** |
| naive + ADE coefficients | 42 | +0.375 | 0.012 |

**Reading**: rotation-invariant features (all three of the first
three rows) get **R² ≈ 0** by construction — they cannot predict a
direction-equivariant target. The 2I-irrep-decomposed coefficient
features extract **R² = 0.378** of the variance because they
preserve the orientation information that the rotation-invariant
features destroy.

### Bottom line

Irrep-decomposed direction-aware features earn their cost specifically
when the target co-transforms with the symmetry group. For
rotation-invariant scalar targets, simpler features already cover
the relevant degrees of freedom. The benchmark answers the "when does
this help?" question concretely on a pair of tasks that share
everything except the symmetry of the target.

## How to reproduce

```
PYTHONPATH=. python3 -m pytest experiments/quasicrystal/tests -v
PYTHONPATH=. python3 -m experiments.quasicrystal.bench_vertex_class
```

Outputs to `checkpoints/vertex_class_bench.json` and
`checkpoints/perp_regression_bench.json`. The bench takes ~30 s end
to end at cutoff=5 (~12k vertices).

## Files

- `cnp.py` — cut-and-project generator (6D Z⁶ → 3D physical + 3D
  perp via the orthogonal Elser-type projection; rhombic
  triacontahedron window via the 30-face zonotope half-space form).
- `icosahedral_group.py` — the icosahedral group I (order 60) as
  3×3 rotations, plus the character table at 50-digit `mpmath`
  precision and a closed-form orthogonality verification.
- `features.py` — 4 + 1 feature pipelines (naive distances,
  Steinhardt Q_l, icosahedral-only Q_l, cell600 ADE eigenspace
  norms, cell600 ADE eigenspace coefficients with direction kept).
  The `cell600_*` features reuse the `ade_geometry.get_ade()`
  machinery already in this repo.
- `bench_vertex_class.py` — both tasks, ridge logistic regression
  for classification, ridge regression for the regression task.
- `tests/test_qc_canonical.py` — the 17 verification tests.

## Coverage gap acknowledged

This subdirectory is the "roll our own with verification" path
flagged in `AGENTS.md`. The quasicrystal / superspace tooling
ecosystem is genuinely sparse — there is no Python equivalent of
`spgrep` for (3+d)D superspace groups, and the rep theory for SSGs
lives behind ISOTROPY/Bilbao web forms or in JANA2020. What's here
is a 3D icosahedral cut-and-project (no SSG yet); extending to a
real (3+d)D Bloch decomposition with `mpmath`-precision irreps
would be a follow-up, with the canonical-tables verification path
named in `AGENTS.md` for the SSG case.
