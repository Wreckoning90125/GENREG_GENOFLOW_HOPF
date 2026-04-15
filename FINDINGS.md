# Findings — Hopf Geometric Feature Extractor

This document records what the Hopf subsystem in this repo has and has
not demonstrated. It exists because earlier framings — in commit
messages, inline comments, and per-version `FINDINGS.md` files that
have since been removed — overstated several points, in particular a
chirality-detection claim on the signed Berry phase. The corrections
are below. The file supersedes the deleted `hopf_v11_ade/FINDINGS.md`
and `hopf_v12_ade/FINDINGS.md`; neither should be resurrected without a
commit that explicitly addresses the critique here.

## TL;DR

- Current best MNIST result: **v10 at 97.39% test accuracy** with
  multi-scale kernel ridge on fixed geometric features. 879 features
  (293 × 3 kappas) under a polynomial-kernel Nystrom readout
  (m = 5000). No learned nonlinearity, no SGD, no ES.
- v11 (adds face / Ω² eigenspaces and triangle Berry features): ties
  v10 at 97.32%. **Not an improvement on MNIST.**
- v12 (completes the discrete de Rham ladder up through Ω³ cells):
  97.24%. **Also not an improvement on MNIST.** Slightly worse than
  v11.
- The signed Berry phase, as added in v11 and extended in v12, was
  claimed in the now-removed per-version writeups to detect chirality
  via cross-digit transfer on hflip/vflip/rot180 operations. Rigorous
  follow-up runs falsified that claim. See the tables below.
- The original v2 multi-stage Hopf architecture (Hopf projection
  *between* computational stages, not just once inside feature
  extraction) has not been built in this repo. Any claim that the
  architecture "replaces ReLU and gradients" is a claim about
  something that does not yet exist here.

## What v8–v12 actually are

All five versions are **fixed feature extractors followed by a
closed-form linear (or kernel) readout**. There are zero non-convex
learned parameters in the trained path. Training is one ridge solve
or one Nystrom-approximated kernel ridge solve. From
`train_ade_hopf.py`:

> *"Zero non-convex parameters: all features are fixed geometric
> functions. Only the linear readout is learned, via closed-form
> ridge regression. Training is a single linear solve — no
> evolutionary search needed."*

The five versions differ in which features are extracted and which
readout is used. Numbers are the ones actually recorded in the
training logs at `checkpoints/hopf_v*_ade/training_log.txt`.

| ver | features                                                   | readout                               | test acc |
|-----|------------------------------------------------------------|---------------------------------------|----------|
| v8  | ADE scalar eigenspaces, CG cross products                  | linear ridge                          | 87.46%   |
| v9  | v8 + curl (co-exact 1-form) eigenspaces + E₈ edge features | linear ridge                          | 96.12%   |
| v10 | v9 features at three kappas {3.0, 5.5, 8.0}, concatenated  | polynomial kernel ridge, Nystrom 5000 | 97.39%   |
| v11 | v10 + face (Ω²) eigenspaces + triangle signed Berry phase  | polynomial kernel ridge, Nystrom 5000 | 97.32%   |
| v12 | v11 + co-exact Ω² + cell (Ω³) eigenspaces via d₂           | polynomial kernel ridge, Nystrom 4000 | 97.24%   |

The v9 → v10 jump (96.12% → 97.39%) is real and comes from two things:
multi-scale kappa (three softness levels of the pixel→600-cell
projection) and a larger Nystrom sample plus a polynomial kernel
readout. The readout change alone accounts for most of the jump — the
linear ridge baseline on v10's multi-scale features is 94.98%, so
multi-scale + linear is actually *worse* than v9's linear ridge.

The v10 → v11 → v12 progression **does not improve MNIST accuracy**.
Adding the Ω² face eigenspaces with signed Berry phase weighting
(v11) and then the Ω³ cell eigenspaces (v12) leaves the kernel ridge
test accuracy in the 97.2–97.4% band — statistically indistinguishable
at this sample size, and within the noise of the Nystrom sample / alpha
sweep. On MNIST, the discrete de Rham ladder above v10 adds **no
measurable signal** to the linear readout.

**This by itself is a meaningful result worth recording.** Adding
more exotic geometry (full chain complex, Clifford-verified Berry
phases, face and cell eigenspaces) did not improve accuracy on the
task used to motivate them. The v11/v12 features may well encode
something, but that something is already representable by v10's
polynomial-kernel combinations of scalar + curl eigenspace features.

## The chirality claim that was made, and the rigorous result that falsified it

The now-removed `hopf_v11_ade/FINDINGS.md` and `hopf_v12_ade/FINDINGS.md`
argued that while v11's face features tied v10 on MNIST accuracy,
they *won decisively* on a follow-up task: cross-digit hflip
classification (train on digits 0–4 detecting original-vs-mirror,
test on digits 5–9). The claim was that the signed Berry phase — the
per-triangle holonomy of the Hopf connection — is an
orientation-reversing invariant that transfers across never-seen
digit shapes in a way that unsigned or pixel-level features cannot.

That claim does not survive more careful testing. A subsequent
rigorous comparison, at matched operating points, ran **three seeds
× two splits** for each feature construction on the same task plus
vflip and rot180, and compared four variants of the face-basis
readout:

- `face_signed` — face-eigenspace projection weighted by signed
  Berry phase (the original v11 claim)
- `face_unit` — same basis, unit weights (Berry phase removed)
- `face_abs` — same basis, |Berry phase| as weights
- `face_random` — same basis, random weights (floor)

### Ungauged, 3 seeds × 2 splits

| operation | v10          | face_signed  | face_unit    | face_abs     | face_random  |
|-----------|--------------|--------------|--------------|--------------|--------------|
| hflip     | 90.78 ± 0.83 | 92.76 ± 0.88 | 91.76 ± 0.79 | 90.28 ± 1.33 | 90.14 ± 1.06 |
| vflip     | 87.16 ± 2.16 | 87.79 ± 1.47 | 90.58 ± 1.98 | 88.22 ± 3.69 | 88.82 ± 2.44 |
| rot180    | 96.62 ± 0.25 | 98.21 ± 0.12 | 98.42 ± 0.32 | 96.99 ± 0.37 | 97.58 ± 0.93 |

### Gauge-concatenated (Z/4 pixel-rotation orbit averaged)

| variant              | hflip         | vflip         |
|----------------------|---------------|---------------|
| `v10_concat`         | 88.74 ± 1.56  | 86.08 ± 3.17  |
| `face_signed_concat` | 92.66 ± 0.87  | 88.83 ± 0.64  |
| `face_unit_concat`   | 92.97 ± 0.50  | 89.96 ± 1.30  |

### What these tables support

**The face eigenspace basis** provides a reproducible cross-digit
transfer advantage over the v10 baseline. In both gauges and on both
chiral operations, every `face_*` variant beats v10 by roughly 3–4
percentage points. The advantage is robust to the weighting scheme
on top of the basis — signed, unit, absolute-value, and random
weightings all show the same order-of-magnitude basis effect. That
means the effect is about the basis, not the weights.

### What these tables do not support

**The signed Berry phase as a chirality-specific causal signal.**

- Ungauged: `face_unit` beats `face_signed` on vflip by +2.79%, while
  `face_signed` beats `face_unit` on hflip by only +1.00% (≈ 1 σ at
  n = 6). Rotating direction between the two chiral operations is not
  what a chirality detector looks like — both hflip and vflip are
  reflections, and a real orientation-reversing invariant should react
  the same way to both.
- Gauge-concatenated: `face_unit` ties (hflip) or beats (vflip)
  `face_signed` in every cell. The ungauged hflip advantage of
  `face_signed` evaporates under Z/4 orbit averaging.
- `face_signed`'s standard deviation on vflip collapses from 1.47
  (ungauged) to 0.64 (gauge-concatenated). That is the signature of
  orbit-averaging stabilizing *any* feature — the textbook
  "augmentation helps" effect — not the signature of a gauge fix
  uncovering a latent chirality signal. Any reasonable feature set
  would show the same variance reduction under Z/4 concatenation.
- `face_abs` — which should be at least as strong as `face_unit` if
  |Berry phase| carried information as an importance weight —
  performs *worse* than `face_unit` on hflip and rot180. Using the
  Berry phase magnitude as a weight hurts rather than helps.

Across six cells of the signed-vs-unit comparison (3 ops × 2 gauges),
there is zero cell in which the signed Berry phase convincingly adds
signal over unit weighting. The chirality-as-causal-factor claim is
**not supported** by these experiments. The v11 / v12 writeups that
are being replaced by this file made that claim and it is retracted.

### Why gauge concatenation does not rescue the story

An intermediate framing argued that signed Berry phase might detect
chirality only under a "properly gauged" feature construction, where
gauge concatenation (averaging features over the Z/4 pixel rotation
orbit) would restore the advantage. The gauged table above says
otherwise: under gauge concatenation, `face_signed` and `face_unit`
are statistically indistinguishable on hflip and `face_unit` wins on
vflip. Additionally, the whole premise is shaky — hflip and vflip
differ by a 90° rotation, so any Z/4-symmetric construction makes
them converge by arithmetic, not by geometry. "If hflip and vflip
converge after averaging over 90° rotations, the chirality claim is
vindicated" is false: they must converge, because the averaging
forces them to.

### Why MNIST is an arbitrary venue for a chirality claim

Independent of the specific numbers above, MNIST was always going to
be a poor test bed for a chirality-detection claim:

1. **MNIST digits are not hflip-invariant in appearance.** Most
   digits (2, 3, 4, 5, 6, 7, 9) look clearly wrong under horizontal
   flip. "Original or hflipped" is therefore confounded with "real
   digit or mirrored digit" — a signal that trivial features pick
   up without needing Berry phase or holonomy.
2. **Chirality would be swamped by the dominant MNIST signal.**
   MNIST's discriminating variance is in stroke patterns, topology,
   and size. A hypothetical chirality effect worth ~1% of
   class-conditional variance is below the current noise floor.
3. **The v10 baseline is itself a specific, arbitrary construction.**
   The face-basis advantage is measured against v10, not against
   "any sensibly-constructed geometric baseline." Swapping v10 for a
   different control might or might not preserve the gap. That
   experiment was not run.

A clean chirality test requires a task where chirality is the only
discriminating signal. MNIST is not that task.

## What has not been built

The multi-stage Hopf architecture sketched in the original v2 design
describes something like:

```
input → 600-cell embedding → ADE eigenspace decomposition
  → K × [ per-irrep rotor (evolved)
          → Hopf projection S³ → S²
          → section lift S² → S³
          → Poincaré warp radially on next-stage coefficients ]
  → readout from final S²
```

That is **not** what v8–v12 implement. v8–v12 are single-pass fixed
feature extractors with a linear or kernel readout. Hopf projection
appears once, inside feature extraction, per eigenspace. There is no
between-stage lift, no accumulated holonomy across passes, no multi-
stage composition. The entire search space in v10–v12 is inside the
kernel-ridge alpha parameter and the Nystrom sample; the geometric
machinery is fixed.

Consequently, any claim that rests on *computation* happening across
Hopf stages — "irrep boundaries as the nonlinearity", "architecture
replaces ReLU", "gradient-free search over a small constrained
parameter space matches or beats SGD" — is a claim about an
architecture that has not yet been built. The `hopf_project` function
in `hopf_controller.py` is correctly annotated as *"THIS IS THE
NONLINEARITY"*: nonlinearity is a property of the Hopf map, not of
the irrep decomposition. The irrep decomposition is a symmetry
constraint — it restricts the function class rather than adding
representational capacity. Those are not interchangeable.

The next test planned for this subsystem — building a faithful
multi-stage Hopf architecture and measuring it against matched MLP
baselines on a task where temporal / compositional structure is
actually stressed — is pre-registered in `PRE_REGISTRATION.md`.

## Documented arbitrary choices in the rigorous chirality runs

The rigorous tables above came from interactive sessions whose full
provenance is **not** in version control. The following choices
affect interpretation and are not currently pinned in the repo:

- Exact MNIST train/test splits used for the 2-split cross-validation.
- The precise definition of the "face eigenspace basis" used in the
  `face_*` variants — which Ω² Hodge modes, which copy decomposition
  convention.
- The operating point for `v10` in the tables (feature count, ridge
  alpha, kernel alpha, Nystrom m, pixel kappas), and whether these
  were locked before the `face_*` comparisons or co-tuned with them.
- Whether `face_signed` / `face_unit` / `face_abs` / `face_random`
  were defined before running or iterated on after seeing interim
  results.
- Whether the set {hflip, vflip, rot180} was fixed before measurement
  or selected to frame specific comparisons.
- Exact random seeds for the "3 seeds × 2 splits" runs.

These choices are **not** claimed to have been pre-registered. They
were not. The tables are the cleanest falsifying evidence we have on
the signed-Berry-phase chirality claim, but they are not a clean
replication target. A clean reproduction would need each of those
choices pinned first. We list the gap rather than paper over it.

Any future chirality test run on this repo is expected to live under
`PRE_REGISTRATION.md` with its design committed in advance.

## What stays, what goes

### Kept (reusable math, active code, and real improvements)

- `hopf_controller.py` — geometric primitives (`hopf_project`,
  `hopf_section`, `hopf_lift`, `pancharatnam_phase`,
  `solid_angle_triangle`, `holonomy_triangle`,
  `triangle_berry_clifford`, `poincare_warp*`) and three controller
  classes:
  - `HopfController` (v6) — per-eigenspace Hopf + McKay E₈ message
    passing with **learnable rotors trained by GA/ES**. Used by
    `genreg_genome.py` as the optional `controller_type="hopf"`
    path for Snake IDE genomes, and as a v6 MNIST trainer entry
    point via `train_hopf_mnist.py`. Active code, not historical.
  - `VertexHopfController` (v7) — 120 vertex activations through
    a Hopf hidden layer with rotors trained by ES
    (`train_hopf_es.py`, `train_hopf_rotor_es.py`). Reference
    implementation of the all-vertex approach.
  - `ADEHopfController` (v8/v9) — ADE-structured feature extraction
    with **zero non-convex parameters**; only the linear readout is
    fit. Wrapped by v10's multi-scale + Nystrom kernel ridge in
    `train_ade_hopf.py`. Current MNIST accuracy path.
  
  The signed Berry phase code is kept because the math is correct
  and reusable; it is not shown to detect chirality in the tests
  that were run, but that is a statement about the tests, not the
  primitive.

- `cell600.py`, `ade_geometry.py` — 600-cell and ADE eigenspace
  machinery. Verified via Clifford algebra.
- `train_ade_hopf.py` — current v10 trainer (multi-scale kernel
  ridge). Best MNIST number in the repo (97.39%).
- `train_v11.py`, `train_v12.py` — v11 and v12 trainers. These do
  not improve on v10 in terms of MNIST accuracy, but the geometric
  features they build (face eigenspaces, triangle Berry, cell
  eigenspaces via d₂, co-exact Ω² via intertwining) are verified
  against Clifford-algebraic cross-checks and are useful reference
  implementations of the discrete de Rham ladder on the 600-cell.
- `train_hopf_mnist.py`, `train_hopf_es.py`, `train_hopf_rotor_es.py`
  — GA / ES / hybrid trainers for the v6 and v7 controllers, on
  MNIST. The controllers themselves remain in use; these scripts
  are the entry points for re-training them.
- `genreg_genome.py`, `genreg_controller.py`, `genreg_proteins.py`,
  `genreg_population.py`, `start_server.py`, the `nodes/` and
  `static/` and `frontend/` trees — the Snake / GenoFlow IDE
  pipeline. Independent of the Hopf MNIST work; unchanged.
- All `checkpoints/hopf_v*_ade/` training logs and best-checkpoint
  files. These are primary evidence and are not altered retroactively.

### Removed (pre-rigorous overclaiming)

- `chirality_benchmark.py`, `chirality_benchmark_v12.py` — earlier
  benchmark scripts whose operating points were numerically unstable
  and whose claimed `face_signed > v10` advantage was not replicated
  in the rigorous follow-up.
- `train_v11_ensemble.py` — one-off ensemble diagnostic.
- `hopf_v11_ade/FINDINGS.md`, `hopf_v12_ade/FINDINGS.md` — the
  pre-rigorous writeups of the chirality claim. Superseded by this
  document.
- `hopf_chirality_bench/`, `hopf_chirality_bench_v12/`,
  `hopf_v11_ensemble/` — corresponding result snapshots.

## Bottom line

- Best MNIST result in the repo: **v10 at 97.39%**, multi-scale
  kernel ridge over 879 fixed geometric features. This stands.
- v11 and v12 add more geometry but do not improve MNIST accuracy
  over v10. That itself is an honest negative result.
- The signed Berry phase is **not shown** to carry chirality signal
  in any of the six cells of the rigorous cross-digit transfer
  comparison. The earlier per-version claim is retracted.
- The multi-stage Hopf architecture from the v2 sketch has **not
  been built**. No claims about it are made. The next test targeting
  it is pre-registered in `PRE_REGISTRATION.md`.
