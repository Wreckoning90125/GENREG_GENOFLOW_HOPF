# FINDINGS — multi-stage Hopf controller, Acrobot pre-registered test

**Status**: pre-registered NULL result. The success criterion fixed in
`experiments/mnist_geometric/PRE_REGISTRATION.md` (amendments 1 and 2)
is not met. No re-pinning of any architectural slot is triggered, per
the document's stated null-criteria stopping rule.

This file is the writeup the pre-registration required. It exists
because the result is null, not despite that.

## Result

Five reporting seeds (0–4), four arms, evaluated on 20 fresh
evaluation episodes per seed (mean undiscounted episode reward over
the eval episodes is the per-seed metric). Higher is better; Acrobot
truncation floor is −500.

| Arm | mean of per-seed means | std of per-seed means | per-seed list | n_params |
|---|---:|---:|---|---:|
| random_search | **−106.5** | 12.3 | −98, −90, −112, −122, −112 | 83 |
| mlp_es | −205.9 | 172.0 | −92, −214, −89, **−500**, −134 | 83 |
| **hopf_es** | **−260.7** | 14.3 | −252, −254, −274, −245, −278 | 84 |
| mlp_ppo | −497.8 | 4.8 | −500, −500, −489, −500, −500 | 83 |

Welch's t-test, two-sided, of Hopf-ES vs each baseline:

| Comparison | t | p (two-sided) | mean diff | effect size (σ of baseline) |
|---|---:|---:|---:|---:|
| Hopf-ES vs random_search | −18.14 | 1.1e-7 | −154.2 | **−12.3 σ** |
| Hopf-ES vs mlp_es | −0.71 | 0.52 | −54.7 | −0.32 σ |
| Hopf-ES vs mlp_ppo | +35.15 | 4.4e-7 | +237.2 | **+49.1 σ** |

## Pre-registered success criterion

> The Hopf architecture "works on this task" iff, on N ≥ 5 random
> seeds of the final frozen design, it beats **all three** baselines
> on mean held-out reward with Welch's t-test p < 0.05 vs each
> baseline, **and** effect size greater than 1 standard deviation of
> MLP-SGD's per-seed reward distribution.

Hopf-ES vs random_search: **loses at p=1e-7**.
Hopf-ES vs mlp_es: **does not differ significantly** (p=0.52).
Hopf-ES vs mlp_ppo: wins, but mlp_ppo is broken (see below) so this
is not informative on its own.

The criterion requires beating **all three**. It does not. This is
the null.

## What "null" means here, narrowly

- *On Acrobot-v1, at 84 learnable parameters, with K=2 Hopf stages on
  the dim-4 ADE eigenspaces under antithetic Gaussian NES at the
  pinned `pop=40, σ=0.10, lr=0.05`, with the budget pinned in
  amendment 2 (5 × 10⁵ env-steps / 8 min wall clock per seed),*
  the multi-stage Hopf architecture trained by evolution does not
  match a parameter-matched MLP under the same evolutionary search,
  and is strictly worse than uniform random sampling in the same
  parameter space.

That is the only claim. It is narrow on purpose.

## What "null" does NOT mean

Per the pre-registration, the null result does not, on its own,
imply or warrant any of the following:

- The architecture is wrong in general.
- A different K, a different eigenspace selection, or a different
  readout family would not work — *but trying any of these now
  would be a new pre-registered experiment, not a follow-up to this
  one.* That is the stopping rule.
- The task was wrong. Acrobot was pinned because it was on the
  pre-registration's allowed list and because it was supposed to
  discriminate against random search. (The latter expectation was
  wrong at this parameter budget — see "What we learned" below.)
- The geometric machinery is generally useless. The substrate
  (cell600.py, ade_geometry.py, hopf_controller.py) is correct and
  well-tested; QM9, MNIST, and Snake benchmarks already document
  cases where it does carry signal. Acrobot at 84 params is just
  not one of them.

## Two diagnostic observations (not part of the contracted result)

These are observations from running the bench. They are not
post-hoc reframings of the null and they do not change the
pre-registered outcome.

### 1. Random search is unexpectedly strong at 84 parameters on Acrobot.

The pre-registration listed Acrobot as a task where "random search
struggles" — that was presumed but never measured ahead of time.
With N=200 random parameter samples × 4 evaluation episodes each
(800 evals total per seed), uniform random search consistently finds
working Acrobot policies (per-seed eval means −90 to −122, std 12.3
across seeds). Energy-pumping baseline reference ≈ −115. Random is
matching it.

The reason is straightforward in retrospect: 84 parameters is small
enough that the working-policy region in parameter space is dense
under the sampling distribution used. Random search effectively
becomes a coarse grid in 84 dimensions. This means **Acrobot at this
parameter budget is not a strong discriminator**: any reasonable
search method will find a working policy quickly.

A future test (separate pre-registration) that wanted to discriminate
algorithms here would need either a much smaller parameter budget
(where random no longer covers the space) or a harder task.

### 2. PPO at the pinned default lr=3e-3 fails to learn Acrobot.

mlp_ppo nulls at the truncation floor on 4 of 5 seeds. The pinning
amendment said "only the policy LR is tunable on validation seeds
100–104." That tuning step was *not* performed on this run; the
pinned default was used directly. Under the contract this is still
a valid run — the default was committed before reporting seeds — but
the result is uninformative as a baseline strength check on PPO. A
follow-up that wanted PPO to be a meaningful comparison would need
to run the validation-seed lr sweep.

This does not change the Hopf-ES vs random_search and Hopf-ES vs
mlp_es comparisons, both of which are decisively against Hopf-ES.

### 3. Hopf-ES is *consistent* but consistently worse than random.

Hopf-ES per-seed means cluster tightly (std 14.3, range −245 to
−278). MLP-ES is bimodal: 3 seeds at random-quality (−92 to −134),
1 mediocre (−214), 1 catastrophic (−500). Both are at the same
84-param budget, same ES recipe.

This is not a positive result — Hopf-ES still does not beat random
or beat MLP-ES — but it is the only structural observation worth
recording: the geometric prior makes the architecture's outputs
more *stable* across seeds than the MLP under the same evolutionary
search, even though the level it stabilizes at is below random
sampling. This could be due to the architecture biasing the search
toward a basin of attraction that is robust but not optimal at this
parameter budget. Investigating that is a separate experiment.

## What this writeup deliberately is not

- Not an argument that Hopf "almost worked" or "would have worked
  with a bit more tuning."
- Not a request to switch to Pendulum, or to try K=3, or to enlarge
  the readout, or to train PPO longer. Each of those is a new
  pre-registered experiment per the stopping rule.
- Not a retraction of the substrate-level results in the parent
  benches (`bench_qm9.py`, `bench_qm9_local.py`,
  `bench_mnist_chirality.py`, `bench_snake.py`). Those are unaffected;
  they tested feature extraction, not multi-stage Hopf computation,
  which is a different claim.

## Reproducing the result

```
PYTHONPATH=. python3 -m pytest experiments/multistage_hopf/tests/test_smoke.py -v
PYTHONPATH=. python3 -m experiments.multistage_hopf.bench_acrobot
```

Raw per-seed numbers, training histories, and Welch t-tests are in
`experiments/multistage_hopf/checkpoints/results.json`. The bench
log including timings is `experiments/multistage_hopf/checkpoints/run.log`.

Pinned design slots are in
`experiments/mnist_geometric/PRE_REGISTRATION.md` (amendments 1 and 2).
