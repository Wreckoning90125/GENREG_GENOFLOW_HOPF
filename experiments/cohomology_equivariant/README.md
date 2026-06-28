# Cohomological Obstructions in Equivariant ML

**An H²(G, U(1))-parameterized equivariant feature pipeline, transferred from
Warman & Schäfer-Nameki, *"Transversal Clifford-Hierarchy Gates via Non-Abelian
Surface Codes"* (arXiv:2512.13777v1).**

This experiment takes the paper's *algebraic* machinery — a group 2-cocycle
`α_N ∈ H²(D_{4N}, U(1))`, its trivialization `β_N` on a cyclic subgroup, and the
phase `U_{α,β}` — and uses it where it has direct ML content: the obstruction to
building a `G`-equivariant network when the data carries a **projective**
symmetry. It is the concrete follow-up to the analysis that motivated this
branch ("equivariant feature pipeline parameterized by H²(G, U(1)) for a small
finite G… the cases where naive equivariant designs hit obstructions").

---

## The one-line bridge

> **H²(G, U(1)) classifies the projective representations of G.**

A standard group-equivariant layer (group convolution) is an *intertwiner of an
ordinary linear representation*. If the symmetry acting on the data is
**projective** with a non-trivial cocycle `α`, the linear-equivariant hypothesis
space **provably does not contain** the target intertwiner. That gap is the
cohomology class. The QEC paper is one exploitation of exactly this fact (the
SPT-stacking automorphism is a projective phase); here we exploit the same fact
on the ML side.

| Paper object | Equation | ML role here |
|---|---|---|
| `α_N ∈ H²(D_{4N},U(1))=Z₂` | A.23, A.27 | the cocycle defining the data's projective symmetry (the obstruction class) |
| `ρ_α(g)\|h⟩ = α(g,h)\|gh⟩` | (twisted reg. rep) | the projective representation the data transforms under |
| `R^α(g)` (α-twisted conv) | commutant of `ρ_α` | the **cocycle-aware equivariant layer** |
| `β_N` trivializes `α_N` on `⟨r⟩` | A.29 | the diagonal **gauge** that restores ordinary equivariance on a subgroup |
| `U_{α,β}(g₁,g₂)` | II.22 | the equivariant projection / realized phase |

---

## Files

- **`dihedral_cohomology.py`** — the paper's machinery, in exact integer-exponent
  arithmetic (every U(1) phase is an integer in `Z/(8N)`, so the cocycle identity,
  the Beigi–Shor–Whalen normalization (A.18) and `α|_⟨r⟩ = δβ` are checked with
  zero floating error). Builds `α_N` from the central extension `D_{8N} → D_{4N}`
  (A.20–A.23) and reproduces the paper's **Table I** logical gate
  `T^{1/N} = diag(1, e^{iπ/(4N)})`.
- **`projective_equivariant.py`** — the ML substrate: ordinary vs α-twisted
  regular reps, ordinary vs twisted group-convolution layers (equal parameter
  count, only the cohomology class differs), the `β`-gauge, and the
  subspace-overlap diagnostics that measure the obstruction.
- **`run_obstruction_experiment.py`** — the supervised experiment + benchmark
  (writes `results.json`).
- **`test_cohomology_equivariant.py`** — 41 tests pinning every claim to the
  paper's formulas.

Run:
```bash
pip install numpy            # only hard dependency (already in repo requirements)
python experiments/cohomology_equivariant/dihedral_cohomology.py      # gate table
python experiments/cohomology_equivariant/run_obstruction_experiment.py
python -m pytest experiments/cohomology_equivariant/test_cohomology_equivariant.py -q
```

---

## Canonical anchor: we reproduce the paper's gate

`dihedral_cohomology.py` recovers Table I exactly — the same "verify against
canonical" discipline used elsewhere in this repo:

```
D_4   N=1: U(rs,s)=0.707107+0.707107j -> T      = P(π/4)   (Clifford level n=3)  ✓
D_8   N=2: U(rs,s)=0.923880+0.382683j -> T^1/2  = P(π/8)   (Clifford level n=4)  ✓
D_16  N=4: U(rs,s)=0.980785+0.195090j -> T^1/4  = P(π/16)  (Clifford level n=5)  ✓
```
with `α_N` confirmed (a) to satisfy the 2-cocycle condition, (b) BSW-normalized,
(c) the **non-trivial** class of `Z₂` (brute-force: not a coboundary), and
(d) trivialized on `⟨r⟩` by `β_N` — all in exact arithmetic.

---

## The experiment & results

Task: learn an operator `T` on `C[D_{4N}]` (orientation-pose feature maps over
the symmetry group of the 4N-gon) that is equivariant under the projective
symmetry with cocycle `α_N`. Two networks, **identical except for the cohomology
class** twisting their convolution layer, each with `|G|` complex parameters.

**[A] Non-trivial target `α_N`** — the naive linear-equivariant net plateaus;
the cocycle-aware net solves it:

| group | naive (linear) test err | cocycle-aware test err | obstruction (shared dims / total) |
|---|---|---|---|
| D₄  | 0.735 | 3.0e-15 | 1 / 8 |
| D₈  | 0.643 | 1.3e-15 | 1 / 16 |
| D₁₆ | 0.843 | 1.5e-15 | 1 / 32 |

The two equal-sized hypothesis spaces share only **one** direction (the
identity); a generic twisted target is otherwise outside the naive span.

**[B] Control — trivial class** — roles flip exactly (naive → 1e-15, aware → ~0.6).
This proves the effect is the **cohomology class, not model capacity**.

**[C] `β_N` recovery on `K = ⟨r⟩`** — the paper's trivialization, transferred:
the ordinary K-equivariant model cannot fit the twisted target (resid ≈ 0.74–0.88),
but after the diagonal `β`-gauge `D_β = diag(β_N(h))` the residual collapses to
~1e-15. The gauge identity `D_β L_α(k) D_β⁻¹ = β(k) L(k)` holds to 1e-15 on the
`C[K]` block. This is necessarily **subgroup-local** — no global trivialization
exists for a non-trivial class, which is the obstruction itself.

---

## Honest scope

- This is a **controlled synthetic task** designed to *isolate* the cohomological
  obstruction — the right instrument for the specific claim being tested. It is
  not a natural-image/graph benchmark, and it does not claim a SOTA number. The
  dataset carries a genuine `D_{4N}` projective symmetry by construction.
- The transferable, reusable artifact is the **machinery**: a verified
  `H²(G,U(1))` cocycle/trivialization toolkit plus projective-equivariant layers.
  When you build equivariant ML over a finite group with non-trivial second
  cohomology, this is the substrate that tells you *whether a linear-equivariant
  design can represent your target* and *how to fix it* (twist by the cocycle, or
  gauge by its subgroup trivialization).
- The QEC paper itself remains a quantum-error-correction result; nothing here
  claims a quantum advantage for ML. What transfers is the algebra — exactly the
  `α / β / U_{α,β}` pattern — now doing measurable work on the ML side.
