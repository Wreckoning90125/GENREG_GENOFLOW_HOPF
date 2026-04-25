# Research program — from Hopf bundles to stellarator geometry

This document lays out the full stack that connects the Hopf / 600-cell /
de Rham work in this repository to the practical problem of designing
good stellarator equilibria. It is a research plan, not a results
document. Every factual claim below is sourced; every interface between
layers is pinned to a concrete data object so the layers can be swapped
or checked independently.

The starting point is a retraction: the 600-cell discrete de Rham code
in this repo is a **laboratory for Hopf-bundle operators and Hodge
decomposition**, not the final physical discretization of a stellarator.
The final discretization lives on toroidal meshes and is the domain of
FEEC / DEC codes and the community equilibrium solvers (VMEC, GVEC,
DESC, SPEC). The contribution this repo is positioned to make lives at
the **topological seed** layer and at the **operator-verification**
layer, not at the equilibrium-solver layer.

The organizing claim:

> Hopf gives the topological seed. G-Frame / GVEC gives the geometric
> chart. FEEC / DEC gives lawful transport. Equilibrium codes (VMEC,
> GVEC, DESC, SPEC) give force balance. SIMSOPT closes the optimization
> loop around quasi-symmetry.

Each of those is a separate layer with its own inputs, outputs, and
well-defined failure modes. The rest of this document walks through
them in order.

## Layer 1 — Topological seed

**Object produced:** a smooth divergence-free vector field `B₀(x)` on a
toroidal domain, with prescribed rotational transform `ι = ω₁/ω₂` and
exact, known linking number between any pair of field lines.

**Method:** Smiet's generalized Hopf construction. The standard Hopf
map gives a field whose field lines are linked circles on nested tori;
the generalized version picks the winding ratio `(ω₁, ω₂)` freely and
yields an integrable foliation where the rotational transform is set
analytically rather than read off after the fact. Every pair of field
lines on the same flux surface has linking number equal to the product
of the winding numbers. This is the cleanest possible starting
condition for a topology-preserving relaxation: the topology of the
initial field is not a numerical artifact, it is an exact closed-form
property.

**Public code:** GLEMuR (Candelaresi et al.), a GPU Lagrangian
mimetic relaxation code that accepts such seed fields and relaxes
them while preserving field-line topology.

**Why this layer matters for the repo:** the Hopf fibration is
already the geometric object the 600-cell work in this repo is built
around. The bridge from "Hopf bundle as representation-theory scaffold"
to "Hopf field as MHD initial condition" is a coordinate change, not
a change of subject.

## Layer 2 — Topology-preserving relaxation

**Object consumed:** the seed field `B₀` from Layer 1, with known
helicity `H = ∫ A · B dV` and known linking structure.

**Object produced:** a relaxed field `B*(x)` that is (approximately)
in force balance, `J × B = ∇p`, while preserving to machine precision:

- `∇ · B = 0`
- total magnetic helicity `H`
- field-line topology (no spurious reconnection)

**Methods and references:**

- **FEEC for magnetic relaxation** — Gawlik & Gay-Balmaz (2025). Finite
  element exterior calculus gives you discrete differential forms whose
  exterior derivatives commute with the continuous ones exactly, so
  `d ∘ d = 0` holds at the discrete level and `div B = 0` is a
  structural property of the discretization rather than a constraint
  enforced by projection.
- **Discrete Exterior Calculus variational integrator** — Kraus & Maj
  (2017). Builds a symplectic / variational integrator for ideal MHD on
  a simplicial mesh; conserves a discrete helicity and energy to
  machine precision over arbitrarily long integrations.
- **Lagrangian mimetic** — Candelaresi, Pontin & Hornig (2015), GLEMuR.
  Advects the field with the flow rather than solving an Eulerian
  update, so topology is preserved by construction (field lines are
  literally carried by particles).

**Failure mode to watch:** if the relaxation wants to reconnect — e.g.
a current sheet forms and numerical dissipation kicks in — the
topology-preserving guarantee is broken. That is a physical signal, not
a bug: it says the chosen seed has no nearby force-balance state with
its topology, and Layer 6 (SPEC) becomes necessary.

## Layer 3 — Axis and surface extraction

**Object consumed:** the relaxed field `B*(x)` on a 3D grid.

**Object produced:**

- a smooth guiding curve `γ(ζ)` (the magnetic axis), extracted as the
  O-point of Poincaré sections
- a family of nested toroidal flux surfaces `Σ_s` for `s ∈ [0, 1]`,
  extracted by field-line tracing from seed points at varying distance
  from the axis
- a chart that labels each surface by its enclosed toroidal flux `s`

**Method:** trace field lines of `B*` across many toroidal transits,
collect the punctures on a fixed poloidal cross-section, identify the
fixed point as the axis, identify nested closed curves as surfaces,
detect islands and chaotic regions by the presence of breakouts.

**Why this is a distinct layer:** equilibrium codes (VMEC, GVEC, DESC)
assume nested flux surfaces as an input ansatz. They do not discover
them. The output of Layer 3 is what decides whether you feed Layer 4
(nested-surface codes) or jump to Layer 6 (SPEC, which admits islands).

## Layer 4 — Geometric chart

**Object consumed:** the extracted axis `γ(ζ)` and surfaces `Σ_s` from
Layer 3.

**Object produced:** a parameterization of each surface around the
guiding curve, suitable as the state of a variational equilibrium
solver.

**Two concrete choices:**

- **G-Frame / GVEC** — B-spline radial basis × Fourier angular basis.
  Uses a generalized Frenet frame along `γ` to avoid the singularities
  of the standard Frenet frame where curvature vanishes.
- **Fourier–Zernike / DESC** — Zernike polynomials radially × Fourier
  angularly. Zernike polynomials are naturally defined on the disk and
  are smooth through the axis, which removes the axis-regularization
  hacks that VMEC needs. Implemented in JAX, so the Jacobian of the
  equilibrium residual with respect to the surface coefficients is
  available by AD.

**Data object at the interface:** either a set of Fourier coefficients
`(R_{mn}, Z_{mn})` for the boundary plus radial basis functions, or
the equivalent Zernike spectrum. Plus profile functions `p(s)` and
either `ι(s)` or `I(s)`.

## Layer 5 — Equilibrium solve and QS optimization

**Object consumed:** the chart from Layer 4 plus boundary and profile
constraints.

**Object produced:** a force-balance equilibrium `J × B = ∇p` on nested
surfaces, and — for quasi-symmetry optimization — a figure of merit
based on the Boozer spectrum of `|B|`.

**Solvers:**

- **VMEC** — historical workhorse. Fortran, fixed axis treatment.
- **GVEC** — modern successor, G-Frame geometry, B-spline radial.
- **DESC** — JAX, Zernike radial, full AD through the residual.
  Supports Gauss–Newton with second-order information because the
  Jacobian is free from AD; this is a qualitative speedup over
  finite-difference Newton schemes in VMEC-class codes.

**Optimization:**

- **SIMSOPT** — wraps the equilibrium solve in a Python optimization
  loop. The standard QS objective is the sum of squared non-symmetric
  Boozer-spectrum coefficients of `|B|`, summed over flux surfaces and
  weighted.

**Data object at the interface:** Boozer coordinates `(ψ, θ_B, ζ_B)`
and the spectrum `B_{mn}(ψ)`. Quasi-symmetry is the statement that
`B_{mn}` is nonzero only for a fixed direction `(M, N)` in `(m, n)`
space.

## Layer 6 — Beyond nested surfaces

**When this layer is needed:** Layer 3 reveals that the relaxed field
has islands or chaotic regions that are physically meaningful (not
numerical noise), so the nested-surface assumption of Layers 4–5 would
discard real physics.

**Code:** SPEC (Stepped-Pressure Equilibrium Code) — partitions the
volume into subregions separated by ideal-MHD barrier surfaces; inside
each subregion the pressure is constant and the field is a Taylor
state. Islands and chaotic regions are admissible in the interior.

**Interface:** SPEC consumes essentially the same surface
parameterization as Layer 4 on the barrier surfaces, plus the list of
enclosed fluxes and helicities in each subregion.

## Where the 600-cell / de Rham code fits

The discrete exterior derivative, Hodge star, and Laplacian that this
repo builds on the 600-cell are **discrete FEEC / DEC operators on a
symmetric lattice**. They are not a substitute for Layer 2 codes on
toroidal meshes. Their role in the program is:

1. **Operator verification.** On the 600-cell you can compute the
   spectrum of the Hodge Laplacian exactly (up to floating point) and
   check it against representation-theoretic predictions from H₄.
   That gives you a controlled ground truth for "is my discrete Hodge
   decomposition correctly separating exact, coexact, and harmonic
   parts?"
2. **Hopf-bundle operators at machine precision.** The Hopf map
   `S³ → S²` acts on the vertices of the 600-cell with a closed-form
   decomposition, so you can test pullback / pushforward of forms
   against analytic predictions.
3. **A test bed for the seed layer.** Before committing to GLEMuR or a
   FEEC code, you can check that your generalized-Hopf seed field has
   the helicity and linking you think it has, using the 600-cell
   operators as a second witness.

What the 600-cell code is **not**: a stellarator equilibrium solver, a
substitute for VMEC/GVEC/DESC, or the right discretization for Layer 2.

## The Surface Evolver connection

MHD equilibrium and Brakke's Surface Evolver solve structurally
similar problems: both minimize an energy functional over embedded
surfaces subject to constraints, and in both cases the second
variation (the Hessian) encodes second-order geometry that controls
stability.

- In Surface Evolver the Hessian encodes mean and Gaussian curvature
  of the surface; the stability criterion is a condition on its
  spectrum.
- In MHD the Mercier criterion is a condition on the curvature of
  flux surfaces (specifically on a combination of magnetic shear,
  pressure gradient, and the average of field-line curvature). The
  ballooning criterion is a condition on geodesic curvature along
  field lines.

Both are quadratic-form-on-a-constrained-manifold stability tests,
which means the Hessian-level analysis techniques transfer. The
relevance to this program is that the AD-through-equilibrium available
in DESC makes these Hessians directly computable, not estimated by
finite differences.

## Data objects at the interfaces (summary table)

| From layer | To layer | Object |
|------------|----------|--------|
| 1 → 2 | Hopf seed → relaxation | `B₀(x)` on a grid + `(ω₁, ω₂)` |
| 2 → 3 | relaxation → extraction | relaxed `B*(x)` on a grid + `H` |
| 3 → 4 | extraction → chart | Poincaré sections, axis `γ(ζ)`, surfaces `Σ_s` |
| 4 → 5 | chart → solver | `(R_{mn}, Z_{mn})` + profiles `p(s), ι(s)` |
| 5 → QS | solver → optimizer | Boozer spectrum `B_{mn}(ψ)` |
| 3 → 6 | extraction → SPEC | barrier surfaces + fluxes + helicities |

## What this repo is positioned to contribute

Not a new equilibrium code. The existing codes (GVEC, DESC, SPEC,
SIMSOPT) are serious, community-supported, and well ahead of anything
a single repository can rebuild. What this repo can contribute is:

- **Seeds.** Generalized-Hopf initial conditions with exact,
  closed-form topology, parameterized by `(ω₁, ω₂)`, delivered as
  `B₀(x)` on a grid that GLEMuR / a FEEC code can consume directly.
- **Verification.** The 600-cell de Rham operators as a second witness
  for Hodge decomposition and helicity on the seed, before handing off
  to a production relaxation code.
- **Bundle-level diagnostics.** Berry-phase / holonomy machinery on
  Hopf bundles as a diagnostic for field-line topology, separate from
  the Poincaré-section extraction in Layer 3.

Each of those is a concrete deliverable with a consumer (GLEMuR input
format, DESC input format, SIMSOPT objective-function hook) and can be
evaluated independently of whether the rest of the program ever ships.

## Addendum (post-consultation): the Hopfion seed targets the omnigenity sector, not QA/QH

Empirical structure of the Landreman QUASR catalogue (371,701 quasi-
symmetric configurations, the live state of QS-stellarator design)
reframes where the Hopfion-seed contribution actually sits in the
pipeline. Documented in `CONSULTATION.md` Q3.

- The QUASR `helicity` field is the **integer helical-period class
  M ∈ {0, 1}** of the quasi-symmetry direction
  `B = B(ψ, M·θ − N·φ)`. M=0 → QA (200,946 entries), M=1 → QH
  (170,755 entries). **This is not the volumetric Hopf invariant.**
  It is a sibling-but-not-equal topological label of the magnetic
  axis self-linking.
- ι is a continuous output, not a topology selector. The catalogue
  sweeps ι freely within each `(nfp, helicity)` sector. So
  "rational ι Hopfion seed" does not compete with the standard
  pipeline for topological reasons — the standard pipeline does not
  constrain ι to be rational at all.
- The current pipeline is Garren–Boozer near-axis expansion + VMEC
  / DESC refinement (Landreman & Sengupta; *Mapping the space of QS
  stellarators using optimized near-axis expansion*; etc.). The
  Hopfion seed is not a competitor to this for QA/QH targets.

The opening for the Hopfion family is in the **omnigenity / quasi-
isodynamic sector** that lies outside the `helicity ∈ {0, 1}`
dichotomy. The relevant literature: *Direct construction of
optimized stellarator shapes III: omnigenity near the magnetic
axis*, *Weakly Quasisymmetric Near-Axis Solutions to all Orders*.
The (n, m) topology of the Hopfion encodes a choice of
field-line-link / -twist that could parameterize a near-axis ansatz
in the omnigenity sector, where Garren–Boozer's QA/QH structure
does not apply.

This re-orients the layer plan:

- **Layer 1** (seeds): unchanged. The closed-form `(ω₁, ω₂)`
  Hopfion family is the right object.
- **Layer 2** (relaxation): the in-repo 600-cell relaxation lab is
  more interesting than originally framed, because relaxing a
  Hopfion under an omnigenity constraint is research, not
  reproduction of existing codes. See `CONSULTATION.md` Q7.
- **Layer 3-5** (extraction → chart → equilibrium → QS): not
  necessarily VMEC/DESC for the QA/QH-Boozer objective. The
  natural downstream codes are still DESC / SIMSOPT, but the
  objective function targets omnigenity (e.g., Landreman omnigenity
  metric) rather than the QA/QH Boozer-spectrum minimisation.

Layer 1 ships independently of this reframing. Layers 2-5 should
be revisited under the omnigenity lens before the next round of
implementation.

## References

- Smiet et al. — generalized Hopf fields with prescribed rotational
  transform.
- Candelaresi, Pontin & Hornig (2015) — GLEMuR: GPU Lagrangian mimetic
  relaxation.
- Gawlik & Gay-Balmaz (2025) — FEEC for magnetic relaxation.
- Kraus & Maj (2017) — DEC variational integrator for ideal MHD.
- Hirshman & Whitson — VMEC.
- Hindenlang et al. — GVEC with G-Frame geometry.
- Dudt & Kolemen et al. — DESC (JAX, Zernike, AD through equilibrium).
- Hudson et al. — SPEC (stepped-pressure equilibria, admits islands).
- Landreman et al. — SIMSOPT optimization framework.
- Brakke — Surface Evolver.
- Mercier, Greene–Johnson — Mercier stability criterion.
- Connor, Hastie, Taylor — ballooning stability.
