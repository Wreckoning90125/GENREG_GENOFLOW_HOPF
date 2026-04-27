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
