# Group Cohomology, Projective Equivariance, and the Gyroid↔Diamond Transition

A self-contained study connecting four papers through one piece of mathematics —
**H²(G, U(1))**, the group that classifies 2-cocycles / projective representations:

1. **Warman & Schäfer-Nameki**, *Transversal Clifford-Hierarchy Gates via
   Non-Abelian Surface Codes*, arXiv:2512.13777 — the cocycle `α_N`, its
   trivialization `β_N`, and the transversal phase `U_{α,β}`.
2. **Shan & Thomas**, *Gradient Transformation of the Double Gyroid to the Double
   Diamond in Soft Matter*, ACS Nano 2024 — the experimental DG→DD structural
   pathway and crystallography.
3. **Dimitriyev, Greenvall, Matthew & Grason**, *Not even metastable: Cubic
   double-diamond in diblock copolymer melts*, arXiv:2507.07361 — the 2-parameter
   (tetragonal-stretch + node-fusion) free-energy pathway; DD an unstable saddle.

The unifying fact:

> **H²(G, U(1)) classifies the projective representations of G**, and it is the
> obstruction both to a transversal Clifford gate (Warman) *and* to building a
> linear-equivariant ML model when the symmetry acts projectively — as it does at
> the Brillouin-zone boundary of a **non-symmorphic** crystal like the gyroid.

---

## What is rigorously established (and tested)

| Result | Module | Status |
|---|---|---|
| Warman cocycle `α_N` ∈ H²(D₄ₙ,U(1))=Z₂ built, normalized (A.18), verified vs the paper's closed forms (A.23–A.29), class order exactly 2, for N=1…16 | `dihedral_cohomology.py` | exact |
| Reproduces the paper's logical gate `T^{1/N}=diag(1,e^{iπ/4N})` (Table I) | `dihedral_cohomology.py` | exact |
| Reproduces the paper's **qubit realization** (Sec. IV): D₄ ops as 3-qubit Cliffords, `M^β=CS†(I⊗T)⊗I` | `qubit_realization.py` | machine-precision |
| Projective (`α`-twisted) regular rep + twisted group-convolution intertwiners | `projective_equivariant.py` | exact |
| **The ML obstruction**: a linear-equivariant layer cannot fit a projectively-equivariant target; the cocycle-aware one can; β-gauge recovery on ⟨r⟩ | `run_obstruction_experiment.py` | demonstrated |
| **Non-symmorphic → projective rep**: gyroid 4₁/4₃ screws realize the nontrivial Z₂ class = Warman's; diamond 2₁ trivial | `crystallographic_cohomology.py` | exact (incl. controls) |
| **Gyroid↔diamond morph**: χ jumps −4→−8, forcing a critical slice / graph rewrite; Shan–Thomas crystallography (35.27°=arccos(2/√6)) | `gyroid_diamond_morph.py` | exact |
| Nematic defect classification (honest negative result) | `nematic_experiment.py`, `d4_gcnn.py` | documented |

`python -m pytest test_cohomology_equivariant.py -q` → **59 passed**.

---

## The chain of ideas

### 1. The Warman cocycle, reproduced exactly
`dihedral_cohomology.py` builds `α_N` from the central extension D₈ₙ→D₄ₙ in **exact
integer-exponent arithmetic** (every U(1) phase is an integer in Z/8N), so the
cocycle identity, normalization, and `α_N|⟨r⟩ = δβ_N` are checked with zero
floating error. `qubit_realization.py` reproduces the paper's qubit construction
(Sec. IV) — the D₄ multiplication/irrep operators as 3-qubit Clifford gates and
the boundary gate `M^β` — at machine precision. This is the "we do *exactly* what
the paper describes" layer.

### 2. The genuine ML obstruction (abstract, provable)
`run_obstruction_experiment.py`: learn a target on C[G] that is equivariant under
the **projective** symmetry `ρ_α`. Two networks identical except for the
cohomology class of their group-convolution layer:
- *naive* (ordinary conv, linear rep) → plateaus (rel-err 0.64–0.84);
- *cocycle-aware* (α-twisted conv) → solves it (≈1e-15).
A **control with the trivial class flips the roles**, proving the effect is the
cohomology, not capacity. The paper's `β_N` then trivializes `α` on ⟨r⟩, and the
diagonal β-gauge recovers an ordinary-equivariant solution on the subgroup —
the ML form of "find the trivialization on a subgroup, get the equivariant
projection."

### 3. Where this touches the gyroid/diamond papers
`crystallographic_cohomology.py`: **non-symmorphic space groups force projective
representations.** At a BZ-boundary wavevector the small reps of the little
co-group carry a factor system
`ω(R₁,R₂)=exp[-ik·(τ_{R₁}+R₁τ_{R₂}-τ_{R₁R₂})]`
from the fractional translations of screw axes. We build the space group by
closing generators **modulo the integer lattice** (so `ω` is a genuine cocycle),
verify the cocycle condition, and test triviality in **H²(·,U(1))** (κ allowed to
be root-of-unity valued — the correct setting; a ±1 cocycle on a cyclic group is
a U(1)-coboundary though not a Z₂ one).

Results, all with passing controls (symmorphic → trivial; pure screw on cyclic
C₄ → trivial since H²(C₄,U(1))=0):

- gyroid **4₁ / 4₃** screw + perpendicular 2-fold → **non-trivial** in H²(D₄)=Z₂
  = the **same class as the Warman α₁ (T-gate) cocycle** (by uniqueness of Z₂);
- diamond **2₁** screw + 2-fold → **trivial** at the same k.

This mirrors *gyroid-chiral vs diamond-achiral* and pinpoints exactly where
equivariant ML over the gyroid crystal must use **projective** representations.

### 4. The transition, as a morph with a singularity taxonomy
`gyroid_diamond_morph.py` realizes the DG↔DD transition as a **smooth
order-parameter field** `F_s` (phase-field morph) interpolating the Schoen-G and
Schwarz-D nodal surfaces with a tetragonal stretch (Dimitriyev–Grason parameter).
It computes the **exact periodic cubical Euler characteristic** along the path and
a Morse-theory degenerate-critical-point detector. The taxonomy (your framing):

- **phase-field morph** — `F_s` is smooth in (x,y,z,s) throughout;
- **critical morph** — isolated singular slice(s) where the *level set* changes
  topology (grad F = 0, F = 0);
- **graph rewrite** — the skeleton's combinatorial type jumps (3-valent srs pair
  ↔ 4-valent dia node), **forced** because χ(gyroid)=−4 ≠ χ(diamond)=−8, so no
  continuous path can avoid it;
- **regular morph** — ruled out for exactly that reason.

The crystallography reproduces Shan–Thomas: `[111]_DD ∥ [110]_DG`, with the
secondary angle **35.27° = arccos(2/√6)** verified exactly. The singularity is
real (in the slice / skeleton) while the 4D field stays smooth — the
"smooth cobordism whose 3D cross-section is critical" picture.

### 5. Honest negative result: nematic defects
`nematic_experiment.py` + `d4_gcnn.py` (a D₄ G-CNN whose group axis carries the
linear or α-twisted regular rep; equivariance verified to 1e-9). We hypothesized
the half-integer (±½) winding sign would force a projective representation. **It
does not.** Empirically: linear D₄-equivariance improves *sample efficiency* over
a plain convnet (n=120: ~0.99 vs ~0.89), but the **cocycle twist gives no
advantage** (≤ linear), and with enough data all three reach ceiling. Defect
classification has no cohomological obstruction for the cocycle to lift — the
genuine obstruction lives where the *target* transforms projectively (§2) or at
the non-symmorphic BZ boundary (§3), not here. Kept as a documented control.

---

## Honest scope

- §1, §2, §3, §4 are **rigorous and exactly verified** (machine precision or
  exact arithmetic, with controls and 59 tests).
- §3's screw-axis construction is a **faithful representative** of the gyroid's
  non-symmorphic little co-group (4-fold screw + 2-fold), grounded in standard
  Bradley–Cracknell space-group rep theory. It demonstrates the mechanism and the
  exact Z₂ class; a full irrep computation of all of Ia3̄d at every k is a larger
  undertaking, not claimed here.
- §4's morph is a **phenomenological phase-field interpolation** that correctly
  exhibits the forced topology change / graph rewrite and the crystallographic
  relations; it is *not* the SCFT free-energy-minimizing path of Dimitriyev–Grason
  (whose saddle-point structure is a separate computation).
- §5 is a **negative result**, reported as such.
- No quantum advantage for ML is claimed. What transfers from the Warman paper is
  the *algebra* — the `α / β / U_{α,β}` pattern — which here does measurable work
  (the obstruction experiment) and identifies the real-materials locus (the
  non-symmorphic gyroid) where projective-equivariant ML is mandatory.

## Files & how to run
```bash
pip install numpy torch            # torch only for the nematic G-CNN
python experiments/cohomology_equivariant/dihedral_cohomology.py        # gate table
python experiments/cohomology_equivariant/qubit_realization.py          # Sec IV
python experiments/cohomology_equivariant/run_obstruction_experiment.py # the obstruction
python experiments/cohomology_equivariant/crystallographic_cohomology.py# gyroid screws
python experiments/cohomology_equivariant/gyroid_diamond_morph.py       # morph + taxonomy
python experiments/cohomology_equivariant/nematic_experiment.py         # negative control
python -m pytest experiments/cohomology_equivariant/test_cohomology_equivariant.py -q
```
