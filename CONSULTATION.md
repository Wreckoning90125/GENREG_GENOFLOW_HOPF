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

### Q1 — 2I-irrep decomposition of the 720 edges

`cell600.py` exposes scalar eigenspaces decomposed by **squared** 2I
irrep dimensions (`1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 3'^2, 4'^2, 2'^2`,
matching multiplicities `[1, 4, 9, 16, 25, 36, 9, 16, 4]` on the 120
vertices) and partial curl eigenspaces (`[6, 16, 30, 48]` on 100 of
the 720 edge modes). These are dimensions, not full irrep labels.

**Wanted**: the multiplicity decomposition of the natural 2I action on
the 720 oriented edges. Equivalently, the character of the edge
permutation rep, `chi_edge(g) = #{edges fixed by g}`, decomposed by
inner product with each 2I irrep character.

This is wanted because the irrep-native Hodge decomposition sketched in
Q6 needs the full irrep block structure on edges, not just the partial
spectrum already exposed.

Sub-question: for S^3 topologically `H^1 = 0`, so the harmonic part of
any 1-form should be **identically zero combinatorially** on this
complex (not merely numerically small). Confirm or correct.

### Q2 — Analytic derivation of `H = 2 pi^2 * nm * n!m! / (n+m)! * R^4`

This is the open derivation. Empirical pattern verified across 18 `(n,
m)` pairs at bbox=10, res=128 to ~5 decimals; fits the rationals `H /
pi^2 = 2 nm / C(n + m, n)` for every tested pair with `n + m <= 8`.
Marked as conjectured-derivation in `hopf_seed.analytic_helicity`'s
docstring.

**Wanted**: the analytic derivation, ideally as Schur-orthogonality of
SU(2) representations on monomials in `(u, v)` integrated against Haar.
Sketch:

- The integrand `A . B` is a polynomial in `(u, u_bar, v, v_bar)` and
  their gradients.
- After grouping, the volume integral over R^3 should reduce, via the
  inverse stereographic Jacobian, to an integral over S^3.
- On S^3, the SU(2) invariant measure factorises into Haar over
  `SU(2)` orbits times a beta-function factor in `(|u|^2, |v|^2)`.
- Schur orthogonality kills cross terms; the surviving diagonal terms
  give the `n! m! / (n + m)!` factor.

A textbook reference (Vilenkin, Folland, or the relevant chapter of
Knapp) plus the explicit chain of reductions would close this. The
factor of `2 pi^2` is the volume of S^3 with the chosen normalization;
that part is standard.

### Q3 — Stellarator relevance of rational-iota Hopfion seeds

`iota(n, m) = n / m` is rational by construction. Modern QS designs
(W7-X, HSX, future stellarators) target **irrational** iota precisely
to avoid island-chain resonances. The plan invokes Smiet's GLEMuR
pipeline as the relaxation: rational seed -> relaxed force-balance
equilibrium with islands or chaotic regions admitted.

**Wanted**: confirmation or correction on whether the Hopfion family is
actually used as a starting point in current QS optimization pipelines,
or whether the canonical seed is something else entirely (e.g., a
Solov'ev equilibrium, a near-axis expansion à la Landreman–Sengupta, a
boundary-driven VMEC seed). If the latter, what's the relationship of
the Hopfion seed to those — analog, complement, or non-comparable?

Sub-questions:

- (a) Which `(n, m)` pairs are physically interesting starting points
  for QS optimization, given that the relaxed equilibrium will *not*
  preserve iota exactly?
- (b) Does Smiet-style relaxation broaden the spectrum so the rational-
  vs-irrational distinction becomes irrelevant, or does it preserve
  enough resonance structure that we have to seed near a target?

### Q4 — `omega1 / omega2` vs stellarator iota convention

`analytic_iota = omega1 / omega2`. Stellarator codes (VMEC, GVEC, DESC)
define iota as poloidal-turns-per-toroidal-turn (or its reciprocal in
some conventions). Empirical iota recovery at `(3, 2)` from a `y = 0`
Poincaré section (poloidal angle around `(R_core, 0, 0)` divided by
toroidal angle around z-axis) gives 1.5.

**Wanted**: confirmation that this is the convention the equilibrium
codes will read on import, or a precise statement of any normalization
mismatch (factor of `2 pi`, sign convention, swap).

### Q5 — Berry phase per `(n, m)`

For `(1, 1)` the Pancharatnam accumulator returns exactly `2 pi` over a
closed Villarceau circle. For `(3, 2)` it returns `~ -16` to `-17` rad
over a partially-closed trace. Two Berry routes agree to ~1e-14 in all
cases.

**Wanted**: the analytic prediction for the Berry phase of a closed
`(n, m)` field line as a function of `n` and `m`. Expected form
`2 pi * k(n, m)` for some integer or simple-rational `k` tied to the
Hopf invariant or its refinement (Călugăreanu twist + writhe). This is
diagnostic, not load-bearing for Layer 2, but knowing the formula
lets us turn the Berry route into a topology check rather than a
descriptive readout.

### Q6 — Sketch the irrep-native `hodge_decompose`

Given the answer to Q1 (the 720-edge 2I-irrep decomposition), the
target rewrite of `hopf_600cell_witness.hodge_decompose` should look
like:

```
For each 2I irrep rho with multiplicity m_rho on edges:
    P_rho := projector onto rho-isotypic component
    A_rho := P_rho @ A_edges                    # block of size d_rho * m_rho
    Within the rho block:
        d0_rho := P_rho @ d0 @ P_rho_vertices   # vertex -> edge restricted to rho
        SVD or pseudoinverse on d0_rho gives the exact-form coefficients
        Coexact = A_rho - exact_part
        Harmonic = whatever lies in ker(d0_rho.T) ∩ ker(d1_rho)
                  ;  on S^3 this should be the zero subspace per Q1
Reassemble: A_edges = sum over rho of (exact_rho + coexact_rho)
```

The blocks are small (the largest 2I irrep on 720 edges is bounded by
`d_rho * m_rho` with `d_rho <= 6`), so per-block SVD is cheap, and the
combined cost beats the global lstsq because no rho-block ever exceeds
the size set by its multiplicity in the edge rep.

**Wanted**: confirmation that this sketch is correct, or specific
identification of any subtle error (e.g., whether the d-dimensional
irrep with multiplicity m_rho gives `d_rho * m_rho` modes or just
`m_rho` modes, depending on whether the rep is the full isotypic
component or one copy thereof).

### Q7 — Layer 2 framing: in-repo discrete relaxation, or pure handoff?

`RESEARCH_PROGRAM.md` says Layer 2 is "topology-preserving relaxation:
FEEC, DEC, GLEMuR." The plan says we don't reimplement those — we feed
our seeds to them.

But: the 600-cell-as-FEEC-test-bed angle in the same document says we
*can* run a discrete relaxation on the 600-cell as a small-scale, fully-
controlled lab for the relaxation operators before handing off to
GLEMuR. That would be a real Layer 2 deliverable in this repo: a
discrete topology-preserving relaxation acting on 1-forms on the 600-
cell, conserving discrete helicity to machine precision, demonstrated
on a Hopf seed.

But it requires a **discrete Hodge star with metric weights** (not just
the combinatorial coboundaries), which `cell600.py` does not currently
provide. The metric Hodge star for a triangulation of S^3 needs edge
lengths, face areas, cell volumes; for the 600-cell these are
calculable in closed form (regular polychoron, all elements congruent),
so the addition is bounded.

**Wanted**: a recommendation. Is the right Layer 2 move

- (a) skip the in-repo relaxation, go straight to surface-extraction
  (Layer 3) on a GLEMuR-relaxed external seed, or
- (b) build the metric-weighted Hodge star and a Kraus–Maj-style
  variational integrator on the 600-cell as a lab demonstration before
  any external handoff?

The trade-off: (a) is shorter and more aligned with the "this repo
contributes seeds, not a relaxation code" framing; (b) is longer but
provides a controlled testbed for the operators and validates the
seeds at the irrep level before they leave the lab.

## What would close the loop

Most useful, in priority order:

1. **Q1 character-table answer** — the irrep multiplicities of the
   720-edge 2I rep. Without this the irrep-native rewrite of the
   600-cell witness can't be written. If it's not in any reference,
   the answer is computable from the character of the edge permutation
   rep (equal to the number of edges fixed by each 2I element) inner-
   producted with each 2I irrep character; a paste-back of the 2I
   character table would let me compute it locally.
2. **Q2 derivation reference** — paper or textbook that does the
   SU(2)-Schur-orthogonality integral over S^3 of `|u|^{2n} |v|^{2m}`.
3. **Q3 stellarator-practice answer** — current state of the art on
   what's actually used as a starting equilibrium in QS optimization,
   and whether the Hopfion family is in or out of that workflow.
4. **Q7 recommendation** — Layer 2 in-repo relaxation lab or pure
   handoff.
5. **Q4 / Q5 conventions** — sign / factor / normalization.
6. **Q6 sketch confirmation** — algorithm-level confirmation of the
   irrep-native Hodge decomposition.

Anything not on this list that I should be asking but am not asking
because I don't know to ask, please add.
