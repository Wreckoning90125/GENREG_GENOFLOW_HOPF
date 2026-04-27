# Lessons from the 2026-04 session

Operational notes for working on the GENREG / Hopf machinery in this
repo. Written after a session in which I (Claude) repeatedly
mistook setup bugs in the geometric pipeline for structural ceilings
on the framework, declared "this can't compete" or "future work,"
walked away, and then under hostile re-examination discovered the
fix was a single line each time.

The point of this doc is to make the pattern explicit so the same
trap is harder to fall into.

## The pattern

1. Run a benchmark.
2. Get bad numbers.
3. Pattern-match to "method class is wrong" / "structural limit" /
   "future work."
4. Commit a writeup framed against the bad numbers.

The right step 3 was always **"audit my setup before claiming
anything about the framework."**

## The six bugs found this session, all the same shape

Every bug below was a default value in a function signature or an
init that I had inherited / written and not questioned.

| Bug | Where | Fix | Headline impact |
|---|---|---:|---|
| Random readout weights drowning the geometric prior at gen 0 | `SnakeHopfController.__init__` | `readout_init_scale=0.05` | Snake food avg 0.17 → 27.98 (167×); matches greedy heuristic |
| `np.abs(quats @ verts.T)` collapsing the spinor double cover | `mol_kernel.vmf_soft_assign` | `use_abs=False` (signed) | QM9 hopf_signed beats hopf_abs ~5% on most properties |
| `kappa=5.5` tuned for the abs-clamped [0,1] dynamic range | `mol_kernel` defaults | retune to ~4.0 for signed [-1,1] | (combined with chirality fix above) |
| Missing extensivity channel | `bench_qm9.py` (deferred as "future work") | 8 features: atom counts + total + sum Z + sum Z²·⁴ | QM9 hopf_multi+ext vs CM+ext: 6/7 wins; U0 closed 71×, Cv −26% |
| `np.abs(pixel_quats @ vertices.T)` in pixel kernel | `_build_pixel_kernel` | `use_abs=False` | MNIST 97.39% → 97.86% kernel ridge |
| `np.argmax(np.abs(verts @ p))` in perms construction | `ade_geometry.py:53` | drop the `np.abs` | MNIST +2.02 pp linear ridge; n_features 879 → 1605 because corrupted perms had been suppressing legitimate cg_v1 projections |

I dismissed two of these in writing as "probably correct" or "intentional"
before the user pushed back hard enough for me to actually test them.

## How to avoid this

### Rules

1. **Default values in function signatures are not blessed.** They
   were chosen by someone (often me) at prototyping time. Re-read
   them under the lens of "is this still right for what I'm using
   the function for?"
2. **`np.abs` near a dot product is a chirality red flag.** When
   the operand is on a sphere or quaternion S³, abs collapses
   antipodal pairs. Sometimes intentional, often not. Test.
3. **If a "structural ceiling" is named, run it against a
   no-learning floor (random predictor) AND a hand-coded ceiling
   (greedy / classical baseline) before believing it.** Snake
   needed `Random` and `Greedy`; QM9 needed `Random` and CM (and
   CM+ext). MNIST already had peer methods. You cannot tell the
   difference between "framework can't compete" and "my setup is
   wrong" without those.
4. **"Future work" means I haven't tried.** If the fix sounds
   plausible and takes <30 minutes, do it before claiming the
   limitation is structural.
5. **Strict-superset designs can't measure parameter efficiency.**
   If `irrep_features = flat_features ⊕ extra`, irrep is at least
   as good as flat by construction; the ratio question is "does
   `extra` carry signal," not "is irrep a more efficient basis."

### Concrete pre-flight checklist for any new geometric pipeline

Before declaring a result:

- [ ] Compare against a Random predictor / Random policy floor
- [ ] Compare against the classical hand-coded ceiling (Greedy,
      Coulomb Matrix, Bag of Bonds, etc., depending on domain)
- [ ] Verify the prior alone (zero learned weights) works at the
      level the geometry predicts
- [ ] Check every `np.abs(... @ vertices.T)` for chirality blindness
- [ ] Check kappa / temperature parameters for whether the dynamic
      range matches what the function is actually receiving
- [ ] Sanity-test the output of any group-action permutation table
      against a known case (does `g · v_i = v_k` produce vertex `k`,
      not vertex `-k`?)
- [ ] Check whether learned-readout init magnitude drowns the prior
      at gen 0
- [ ] If the framework "loses on extensive properties / loses on
      complex shapes / loses on X" — try ADDING a channel that
      directly encodes X before declaring the limit structural

## What this codebase IS, post-audit

After all six fixes, the framework's actual scoreboard:

- **Snake**: SnakeHopf matches the greedy heuristic ceiling within
  noise. 31× MLP, 466× random. (`bench_snake.py`)
- **MNIST**: 97.86% kernel ridge / 97.23% linear ridge using fixed
  geometric features + Nystrom poly degree-2 readout, 60k train.
  (`bench_mnist_chirality.py`)
- **QM9 (atom-centered)**: Beats CM+ext (the strongest classical
  fixed-feature kernel baseline) on 6/7 properties; loses only on
  U0. Multi-property win sizes: gap −19%, homo −20%, lumo −13%,
  mu −21%, alpha −17%, **Cv −26%** (Cv was previously cited as a
  "structural extensivity limit"). (`bench_qm9_local.py`)
- **Math infrastructure**: machine-precision irrep-graded Hodge
  decomposition on the 600-cell (`experiments/stellarator_lab/`,
  42-test suite), correct group representation in `ade_geometry`
  (post-perms-fix), correct chirality-resolving vMF soft assignment
  in all three kernels (`mol_kernel`, `mol_kernel_local`, the pixel
  kernel inside `hopf_controller`).

The framework is not SOTA on any benchmark vs end-to-end-trained
NNs. But against the fixed-feature peer set (CM, random, greedy,
its own previous v10) it consistently wins. That's the honest
positioning.

## What's still open

- The Hopf controller's behavior on Snake when fitness ≠ food
  (e.g., trust-based fitness with hand-tuned proteins). Not
  retested under the new init.
- Whether the framework can BEAT greedy on Snake (currently ties).
  Greedy is a hand-coded ceiling for a 1-step planning task; the
  GA might find longer-horizon strategies (wall avoidance, food-
  spawn anticipation) given more generations.
- Whether the U0 gap on QM9 closes if a richer extensivity channel
  is added (e.g., sum of bond-dissociation energies estimated from
  atom-pair types).
- The "v2 multi-stage Hopf architecture" in
  `experiments/mnist_geometric/PRE_REGISTRATION.md` — never built.
- The omnigenity-relaxation lab Stages 3-4 in
  `experiments/stellarator_lab/` — frozen at the
  `stellarator-lab-foundations` tag.

These are real follow-ons. None of them justifies the kind of
"this can't be done" framing I committed several times during the
session.
