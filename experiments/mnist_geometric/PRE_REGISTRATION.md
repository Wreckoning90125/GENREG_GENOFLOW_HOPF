# Pre-registration — next test for the Hopf subsystem

This document pre-registers the next test to be run on the Hopf
subsystem. It exists because the experiments summarized in `FINDINGS.md`
suffered from post-hoc interpretation and arbitrary venue choice. The
remedy is not a cleverer one-shot experiment; it is a test whose
design, success criteria, and null criteria are fixed in version
control *before* any run happens.

**Status: draft. Not yet runnable.** Several choices are deliberately
left unset below. Each must be pinned in a separate commit to this
file before the first training run. Any deviation from the committed
version, once a run has started, requires a new commit that explicitly
amends this document and notes the reason.

**Audit note.** Two setup bugs in the math substrate that this
pre-registration's baseline number depends on (v10 = 97.39%) were
found later: chirality / spinor double cover in the pixel kernel,
and group-action permutation table corruption in
`ade_geometry.perms`. Both are now fixed at the substrate level;
the corrected baseline on the same trainer is 97.86% kernel ridge,
97.23% linear ridge. Any concrete run of the test below should
re-pin the baseline reference against the corrected substrate
before win/loss claims. See `FINDINGS.md` addendum.

## What is being tested

The question: does a **multi-stage Hopf architecture** — the design
sketched in the original v2 spec — do useful work beyond what a fixed
feature extractor followed by a linear or kernel readout already does?

Concretely: does evolution over the constrained parameters of such an
architecture match or beat gradient descent over the unconstrained
parameters of a parameter-matched MLP, on a task where temporal /
compositional structure is actually stressed?

This is **not** a test of the signed Berry phase chirality claim.
That claim has been retracted (see `FINDINGS.md`). The
`holonomy_triangle` / `triangle_berry_clifford` code remains available
in `hopf_controller.py` as reusable geometry, but no claim about
chirality detection is being made here, and the success / null
criteria below do not depend on one.

## Why this is the right next test

The result that the Hopf subsystem currently carries is:

- v10 achieves 97.39% on MNIST with a fixed feature extractor +
  polynomial kernel ridge readout.
- v11 and v12 add more geometry on top (face Ω² eigenspaces with
  signed Berry phase, cell Ω³ eigenspaces via d₂) and **do not
  improve MNIST accuracy**. They sit at 97.32% and 97.24%.
- Attempts to show that the v11/v12 additions carry orthogonal
  signal on a follow-up task (chirality-adjacent cross-digit
  transfer) do not hold up under rigorous comparison.

The honest conclusion is that, *as feature extractors*, the
geometric additions above v10 are redundant on MNIST. That does not
tell us whether the geometry is useful for *computation* — i.e. when
Hopf projection appears between computational stages, not just once
inside feature extraction, and the architecture can accumulate
something across passes. The v2 spec specifically describes that
kind of architecture, and this repo has never built it. This test
is aimed at that gap.

## The architecture under test

The v2 sketch, implemented faithfully:

```
state vector
  → 600-cell embedding (via the existing pixel kernel or an
                        analogous input kernel for the task)
  → ADE eigenspace decomposition
  → K × [
        per-irrep rotor (searched by evolution)
        → Hopf projection S³ → S² per eigenspace with dim ≥ 4
        → section lift S² → S³
        → Poincaré warp radially on next-stage coefficients
    ]
  → readout from final S² / S³ coordinates
```

**Trainable parameters.** Only the per-stage, per-irrep rotors.
These are points on S³, mutated by small rotations. Evolution is the
only search method over this set.

**Readout.** Fixed and specified before training begins. Three
candidates, one of which must be pinned in this file before the
first run:

- (a) small fixed random linear projection;
- (b) closed-form linear ridge, fit once on a held-out segment of
  the training data *before* any rotor search begins and then
  frozen;
- (c) single linear layer co-trained with the rotors under the same
  ES signal.

**What is explicitly forbidden.**

- Adding layers, heads, skip connections, or auxiliary losses beyond
  what the v2 sketch provides.
- Adding a trainable nonlinearity beyond what the sketch provides
  (Hopf projection + Poincaré warp).
- Switching from evolution to SGD or any gradient method for the
  rotors.
- Any post-hoc averaging over rotation orbits, any data augmentation
  not specified before the run, any gauge concatenation added after
  seeing results.
- Tuning any hyperparameter after seeing a test-set number. All
  hyperparameter search uses a fixed validation split committed
  here before the first run.

## The task

Unset. The candidates under consideration are:

- **Acrobot-v1** — discrete action, continuous state, temporal
  composition required. A small tuned MLP achieves near-optimal
  reward; random search struggles; this is a real discriminator.
- **Pendulum-v1** (swing-up) — continuous torque output, requires
  accumulating energy over multiple time steps. A small MLP with
  SGD solves it; random search does not.

Explicitly **not** candidates:

- **CartPole-v1.** Random search solves it, and a 50-parameter
  policy solves it. A win on CartPole proves the architecture can
  solve the easiest benchmark, not that it does anything non-trivial.
- **MNIST** or any classification variant of MNIST (hflip detection,
  cross-digit transfer, rotation invariance). The venue critique
  from `FINDINGS.md` applies. MNIST is not a task where multi-stage
  *computation* is necessary; a linear readout on a good feature
  basis already gets ~97%.
- Any task where the three baselines below cannot be run with
  matching compute.

One of the two candidate tasks must be pinned in this file before
the first run. If neither is tractable in the runtime environment,
this document must be amended to say so, rather than silently
substituting a different task.

## Baselines (required)

All three must be run on the same task, with the same environment-step
budget, the same wall-clock budget, and the same number of seeds as
the Hopf controller:

1. **MLP-SGD** — small feed-forward MLP with approximately the same
   learnable parameter count as the Hopf controller's rotors +
   readout, trained by SGD with a learning rate tuned only on the
   committed validation split.
2. **MLP-ES** — the same small MLP, trained by the same evolution
   strategy as the Hopf controller. Isolates "evolution vs gradient
   descent" from "Hopf architecture vs MLP architecture."
3. **Random search** — same parameter budget, uniform random
   initialization over the same parameter space used by ES, same
   number of evaluations. A floor.

"Parameter count" is counted over all learnable parameters. For the
Hopf controller this is the rotors plus the readout (if any part of
the readout is trained). For the MLP baselines it is all weights and
biases.

## Success criteria — pre-registered

The Hopf architecture "works on this task" **iff**, on N ≥ 5 random
seeds of the final frozen design, it beats **all three** baselines
on mean held-out reward with:

- Welch's t-test p < 0.05 vs each baseline, **and**
- effect size greater than 1 standard deviation of MLP-SGD's per-seed
  reward distribution.

Held-out seeds are seeds that were not used at any point to select
hyperparameters, choose K, pick the readout variant, or tune ES
parameters.

Any weaker outcome — beats random search but not MLP-SGD, ties
MLP-SGD, beats on mean but not per-seed, beats on some aggregation
cherry-picked after the fact — is a null result for the purposes of
this document.

## Null criteria — pre-registered

If the Hopf controller does not beat MLP-SGD under the success
criteria above, the null result will be written up in `FINDINGS.md`
with the full numbers. The null result will **not** automatically
trigger:

- a different task,
- a different K or rotor layout,
- a different readout variant,
- a different ES algorithm,
- the addition of any extra computation layer,
- a switch to SGD training of the rotors,
- rotation / gauge / data augmentation patches,
- ensembling across runs.

Any of the above is a new experiment, which requires its own
pre-registration commit before running.

The null criterion is a **stopping rule**, not an iteration invitation.

## What a positive result would and would not mean

A positive result would mean: *on this specific task, at this
specific parameter budget, the multi-stage Hopf architecture trained
by evolution is at least as effective as a parameter-matched MLP
trained by SGD.*

That is narrower than several claims that have been made in
interactive sessions about this subsystem. A positive result would
**not** mean:

- The chirality / signed-Berry-phase story is vindicated. It was
  retracted; this test is independent.
- The architecture replaces ReLU or gradients in general. One task,
  one parameter budget, one comparison.
- The irrep boundaries are the nonlinearity. They are not, by
  construction. Hopf projection is. This test does not change that.
- Gradient-free search is generally preferable to SGD. Narrow-domain
  result at best.
- The architecture scales to larger tasks. Not measured.

If the result is positive, the `FINDINGS.md` writeup of it must use
language no stronger than the above.

## Choices that must be pinned before the first run

Each of these is explicitly unset. Each must be committed in an
amendment to this file, with a stated reason, before the first
training run.

- [ ] Specific task (Acrobot-v1 or Pendulum-v1 swing-up)
- [ ] K — number of Hopf stages
- [ ] Which eigenspaces receive rotors (all with dim ≥ 4, or a subset)
- [ ] Readout variant: (a) fixed random / (b) pre-fit ridge / (c)
      co-trained linear
- [ ] Parameter budget (applied to all MLP baselines too)
- [ ] ES algorithm and its hyperparameters (population size, noise
      scale, learning rate, antithetic sampling on/off)
- [ ] N — number of seeds for the reported runs
- [ ] Fixed validation split for hyperparameter tuning
- [ ] Environment-step budget and wall-clock budget per seed
- [ ] Reward metric (mean, median, last-100-episode mean, etc.)

### Pinning amendment — 2026-05-06

All ten slots above are pinned below in a single commit, before any
training code for the multi-stage Hopf controller has been written.
Each slot lists the chosen value and the stated reason for choosing
it. No slot is left ambiguous; "approximately" is replaced by a
concrete numeric or set value everywhere a baseline must match.

- [x] **Specific task: `Acrobot-v1`.**
      Discrete 3-action space (torque in {−1, 0, +1}) lets the Hopf
      architecture reuse the vertex-cardinal-affinity pattern that
      worked on Snake — each action maps to an S² direction, and
      vertex affinities give a structural prior. Pendulum-v1's
      continuous torque output requires either a discretization
      hack (which adds design freedom) or a continuous readout
      (which makes the parameter-matched MLP comparison messier).
      Acrobot also has the property that random search struggles
      while a tuned MLP does not, which is the discriminating
      regime named in the original document.

- [x] **K = 2 Hopf stages.**
      K = 1 is the degenerate case (= a fixed feature extractor +
      readout, which is what v10 already was). K = 2 is the minimum
      nontrivial test of the multi-stage hypothesis. K = 3 or higher
      would be a separate pre-registration if K = 2 fails;
      pre-emptively running it would re-introduce the design
      freedom this document exists to remove.

- [x] **Eigenspaces with rotors: dim-4 only (the two dim-4 ADE
      eigenspaces).**
      Hopf projection S³ → S² is naturally 4D-source. Rotors live
      on S³, which acts cleanly on R⁴ as quaternion multiplication.
      Using dim-5 or dim-6 eigenspaces would require defining an
      additional group action (embed S³ in SO(5) / SO(6) somehow),
      which is design freedom we explicitly do not take. Two dim-4
      eigenspaces × K = 2 stages = 4 rotors total (12 free
      parameters, since each unit quaternion has 3 d.o.f.).

- [x] **Readout: (c) co-trained linear.**
      (a) fixed random readout is a bottleneck that risks a false
      null — a correct architecture could still fail to express
      itself through a poorly aligned random projection. (b) pre-fit
      ridge mismatches parameter accounting against the MLP
      baselines (the readout's params are "spent" but not
      "learnable"). (c) is the cleanest test: same parameter
      accounting on both sides, same training signal (ES), same
      readout family. The architectural difference reduces to "MLP
      hidden layer with tanh" vs "K=2 Hopf stages with vMF
      assignment + Hopf projection."

- [x] **Parameter budget: 80 ± 5 learnable parameters.**
      A 6 → 8 → 3 tanh MLP has 6×8 + 8 + 8×3 + 3 = 83 learnable
      parameters. The Hopf controller is sized to match: 12 rotor
      parameters + a co-trained linear readout from the
      stage-K Hopf-projected coefficients to 3 logits, sized to
      bring the total to 80 ± 5. The exact readout dimension is
      computed deterministically from the eigenspace decomposition
      and is committed in code, not chosen after seeing results.

- [x] **ES: antithetic Gaussian NES, population = 40, σ = 0.10,
      lr = 0.05.**
      Standard recipe (Salimans et al. 2017 / OpenAI ES). Antithetic
      sampling halves variance at no extra evaluations. σ and lr
      are tunable on validation seeds 100–104 *before* any reporting
      seed is run; the values committed here are starting points
      only, and the final pinned values for reporting will be
      written into a follow-up amendment commit naming the chosen
      values and their validation-set rewards. The same ES,
      population, σ, lr is used for the MLP-ES baseline.

- [x] **N = 5 reporting seeds.**
      Pre-reg minimum. Seeds 0, 1, 2, 3, 4.

- [x] **Validation split: seeds 100, 101, 102, 103, 104.**
      Used exclusively for ES hyperparameter tuning (σ, lr) and
      SGD hyperparameter tuning (lr, batch size) before any
      reporting seed runs. No reporting seed is touched until the
      validation tuning is finalized and committed.

- [x] **Environment-step budget: 200 ES generations × 40 population
      × 4 episodes per evaluation × 500 max steps/episode =
      1.6 × 10⁷ env-steps per seed.**
      Same budget for all four arms (Hopf-ES, MLP-SGD, MLP-ES,
      random search). MLP-SGD is given the same env-step budget
      via on-policy rollouts (REINFORCE-style); the SGD step count
      is chosen to consume exactly the same env-step budget,
      computed before training.
      Wall-clock budget: 10 minutes per seed per arm, hard cap.

- [x] **Reward metric: mean undiscounted episode reward over 20
      fresh evaluation episodes at the end of training, with a
      different random seed for env initialization than was used
      in training.**
      Avoids end-of-run cherry-picking and stochastic-final-step
      noise. Acrobot's natural reward structure (−1 per step until
      balanced, episode caps at 500) means this is bounded in
      [−500, ~−60] for non-trivial policies.

### Architectural choices that follow from the above

These are not new design freedom; they are determined by the
pinned slots above plus the deterministic structure of the
600-cell ADE decomposition.

- Input kernel: vMF soft assignment of the Acrobot state
  (cos θ₁, sin θ₁, cos θ₂, sin θ₂, ω₁_clipped, ω₂_clipped) onto
  the 120 vertices of the 600-cell, after embedding the angle pair
  into a unit quaternion as (cos θ₁/2 · cos θ₂/2, sin θ₁/2 ·
  cos θ₂/2, cos θ₁/2 · sin θ₂/2, sin θ₁/2 · sin θ₂/2) and
  re-normalizing. Angular velocities modulate the vMF concentration.
  This is the closest analog of the pixel kernel for a 2-angle
  state and was named explicitly in the document above.
- Eigenspace structure (deterministic, from `ade_geometry.get_ade()`):
  E3 (dim-4 irrep) has multiplicity 4 — 16 coefficients total,
  reshaped as a (4 copies × 4 irrep-coords) array. Same for E7
  (the second dim-4 irrep). A "rotor on a dim-4 eigenspace" means
  one S³ rotor (3 d.o.f.) shared across all 4 copies, acting on
  the irrep-coord axis. Sharing across copies preserves the
  multiplicity-space group action (equivariant); per-copy rotors
  would not.
- Stage transition: Poincaré warp on the irrep-coord 4-vectors
  (per copy, per eigenspace) before the next stage's vMF
  projection back onto the 600-cell, as the v2 sketch prescribes.
- Final readout input: at stage K, each eigenspace contributes
  4 copies × 3 Hopf-projected S² coordinates = 12 numbers. Two
  eigenspaces: 24 numbers total. The readout is a single linear
  layer 24 → 3 logits with no bias: 72 weights.
- **Total learnable parameter count: 12 rotor d.o.f. + 72 readout
  weights = 84.** This is within the 80 ± 5 budget pinned above.
- MLP baseline matched to 84: a 6 → 8 → 3 tanh net is 6×8 + 8 +
  8×3 + 3 = 83 params. The 1-param difference is rounded into the
  matched budget.
- Random search baseline: uniform sampling over the same 84-d
  MLP parameter vector, same number of evaluations as MLP-ES.

This correction supersedes the earlier "8 numbers" claim in the
preceding amendment, which was based on an incorrect reading of
the eigenspace multiplicity.

### What this pinning forecloses

- Switching tasks to anything other than Acrobot-v1.
- Adding K = 3 stages, or stages in dim-5 / dim-6 eigenspaces.
- Switching readout family.
- Re-tuning σ, lr, SGD lr after seeing reporting-seed results.
- Re-defining the reward metric to a more favorable aggregation
  after seeing reporting-seed results.

A null result under the criteria already stated in the document
above is a null result. It does not trigger a re-pinning of any
of the slots above.

## Why this document exists

The experiments summarized in `FINDINGS.md` were designed after a
story and iterated until the story looked supported. Gauge
concatenation was proposed as a fix for a chirality interpretation
that the data then failed to support. The venue (MNIST cross-digit
hflip transfer) was arbitrary. The baseline (v10 at a specific
operating point) was arbitrary. Several feature-construction choices
were made without being pinned. When each choice was eventually
nailed down under rigorous comparison, the headline result evaporated.

The failure mode was not one agent getting excited once. It was the
absence of a fixed target. Any design freedom that was not
pre-specified got consumed in the direction of the prior conclusion.
Pre-registration is the cheapest fix: remove design freedom before
seeing results.

This document is the fix. It is not meant to be comprehensive about
the Hopf subsystem's research program. It is meant to be narrow
enough that one test can decisively pass or fail it, and that a
null result has no room to be reframed as success.
