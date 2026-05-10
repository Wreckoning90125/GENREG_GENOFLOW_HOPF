# Lorentz / Möbius fallback

The conformal / hyperbolic toolchain flagged in `AGENTS.md` as the
"fall back to Lorentz invariance wherever it suits us" piece.
Self-contained; uses `clifford`-compatible Cl(1,3) signatures via
direct SL(2, ℂ) matrices (no dependency on the Cl(3,0) machinery
in the parent repo), and `mpmath` for high-precision verification.

## Why this exists

When the data has natural conformal / hyperbolic structure rather
than just Euclidean rotational symmetry, the right invariance is
*Möbius* (= the full conformal group of S² ≅ ℂP¹), not just SO(3).
The double cover is SL(2, ℂ) → PSL(2, ℂ) ≅ SO(3, 1)⁺/{±1}.

Concrete use cases this stack now covers:

1. **EPINET-style hyperbolic-origin networks.** The parent tilings
   live in H² with isometry group PSL(2, ℝ) ⊂ PSL(2, ℂ); the
   projection to 3D Euclidean networks partially obscures their
   hyperbolic origin. Cross-ratio features recover signal the
   Euclidean toolchain drops.
2. **Anisotropic / conformal symmetry on S²** (Möbius-deformed
   point sets, celestial-sphere data, twistor-style coordinates).
3. **Non-abelian SL(2, ℂ) Berry phases** as an extension of the
   existing U(1) Pancharatnam machinery in
   `hopf_controller.pancharatnam_phase`.

## Verification — 12 tests pass

`tests/test_mobius_canonical.py`:

- Stereographic round-trip S² → ℂ → S² is identity to machine
  precision (excluding the north pole).
- All `sl2c_*` generators return matrices with det = 1.
- SU(2) ⊂ SL(2, ℂ) acts on S² as the ordinary SO(3) rotation it
  double-covers: for 10 random axis-angle pairs and 30 random S²
  points each, |R·p − Möbius(M_SU(2))·p| < 10⁻¹⁰.
- Cross-ratio invariant under **random SL(2, ℂ)** including
  non-trivial Lorentz boosts (the part rotation invariants don't
  cover): residual < 10⁻⁸ over 50 random tests.
- Cross-ratio NOT invariant under generic non-Möbius coordinate
  changes (z ↦ z² as a sanity check).
- The six S₄-permutation values of the cross-ratio sum to 3
  (Klein's canonical identity), verified at 50-digit `mpmath`
  precision with residual < 10⁻⁴⁵.

## Head-to-head — when do Möbius-invariant features win?

`bench_mobius_equivalence.py`. 5-class synthetic task on 4-point
S² configurations, 200 samples per class, 5 reporting seeds.

### Task A: Möbius-equivalence classification

Within-class members differ by random SL(2, ℂ) elements (including
Lorentz boosts). The class label is the orbit under the full Möbius
group; only Möbius-invariant features can identify it.

| Pipeline | n_feat | test acc |
|---|---:|---:|
| pairwise_distances (rotation-invariant) | 6 | **0.329 ± 0.030** |
| cross_ratio (Möbius-invariant) | 2 | 0.915 ± 0.116 |
| all_6_cross_ratios (S₄-symmetric) | 12 | **1.000 ± 0.000** |

The rotation-invariant baseline at 33% is barely above chance
(20% for 5 classes) — because Lorentz boosts on S² distort
pairwise distances arbitrarily, distance features cannot
distinguish boost-equivalent configs.

### Task B: Rotation-only-equivalence (control)

Within-class members differ only by SO(3) rotation (no boost).
Distance features should be sufficient; cross-ratio features lose
information because cross-ratio is strictly coarser than the full
rotation invariant.

| Pipeline | n_feat | test acc |
|---|---:|---:|
| pairwise_distances | 6 | **1.000 ± 0.000** |
| cross_ratio | 2 | 0.915 ± 0.116 |
| all_6_cross_ratios | 12 | 1.000 ± 0.000 |

### Read

The right symmetry for the data is the right symmetry for the
features. Lorentz / Möbius isn't a free upgrade — it gives perfect
resolution on conformally-equivalent data, but loses ground on
purely-rotation-equivalent data where the symmetry is over-strong.

The `all_6_cross_ratios` pipeline (full S₄-permutation-symmetric
12-feature representation) is the canonical "conformal fingerprint"
of a 4-point S² configuration: 100% on both tasks.

## How to reproduce

```
PYTHONPATH=. python3 -m pytest experiments/lorentz/tests/ -v
PYTHONPATH=. python3 -m experiments.lorentz.bench_mobius_equivalence
```

Bench takes < 5 seconds end to end.

## Files

- `mobius.py` — Stereographic projection, SL(2, ℂ) generators
  (rotation, boost, random), Möbius action on ℂ ∪ {∞}, cross-ratio
  invariants. Sign convention chosen so SU(2) ⊂ SL(2, ℂ) acts on
  S² as the SO(3) rotation it double-covers (test verified).
- `bench_mobius_equivalence.py` — the head-to-head bench.
- `tests/test_mobius_canonical.py` — the 12 verification tests.

## What this stack does NOT cover yet

- (3 + d)-dimensional superspace-group rep theory (still gated on
  driving JANA2020 / scraping Bilbao / porting Stokes' SSESG tables
  per `AGENTS.md`).
- Wannier-style local-orbital construction on hyperbolic graphs.
- Discrete representations of PSL(2, ℝ) (principal series,
  complementary series, discrete series). The continuous group
  acts here; subgroup discrete-rep machinery would come next.

The path to those is documented in `AGENTS.md` as user-surfacing
gaps; this commit only ships the immediate-use Lorentz/Möbius
primitives + their verification + one demonstration of usefulness.
