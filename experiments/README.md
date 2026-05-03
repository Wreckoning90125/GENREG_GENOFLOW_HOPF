# Experimental sublibraries

These directories contain exploratory work on top of the GENREG /
GenoFlow runtime in the parent directory. They share the math
substrate (`cell600.py`, `hopf_controller.py`, `ade_geometry.py`,
`hopf_decagon.py`, `snake_hopf_controller.py` at top level) but are
not part of the deployable IDE.

## `mnist_geometric/`

MNIST classification via fixed Hopf-geometric features + kernel ridge.

Pre-audit headline (broken-baseline): **97.39% test accuracy** (v10).
Post-audit (chirality + perms fixes in the math substrate at repo
root): **97.86% kernel ridge, 97.23% linear ridge** on the same
trainer. v11/v12 extensions still don't add measurable signal beyond
the substrate fix. See `FINDINGS.md` for the addendum + the original
chirality-claim retraction. The top-level `bench_mnist_chirality.py`
runs the controlled A/B that establishes the +0.50 pp / +2.02 pp
deltas.

Run from the repo root:
```
PYTHONPATH=. python experiments/mnist_geometric/train_ade_hopf.py
PYTHONPATH=. python bench_mnist_chirality.py            # A/B harness
```

## `stellarator_lab/`

Discrete-exterior-calculus machinery on the 600-cell. Closed-form
icosian Hodge stars; machine-precision irrep-graded Hodge
decomposition; generalized-Hopf seed fields; field-line Berry-phase
diagnostics. Frozen at git tag `stellarator-lab-foundations`.

Tests:
```
PYTHONPATH=. pytest experiments/stellarator_lab/tests/
```

These pieces are math infrastructure — they are not by themselves a
contribution to a current ML benchmark, but the math is correct and
well-tested, and the substrate (`hopf_decagon.py` in particular) IS
used by the Snake-aware geometric controller in the GENREG runtime.
