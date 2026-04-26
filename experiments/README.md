# Experimental sublibraries

These directories contain exploratory work on top of the GENREG /
GenoFlow runtime in the parent directory. They share the math
substrate (`cell600.py`, `hopf_controller.py`, `ade_geometry.py`,
`hopf_decagon.py`, `snake_hopf_controller.py` at top level) but are
not part of the deployable IDE.

## `mnist_geometric/`

MNIST classification via fixed Hopf-geometric features + kernel ridge.
Headline: **97.39% test accuracy** (v10) with no learned
nonlinearities. v11/v12 extensions did not improve on v10.
See `FINDINGS.md` for the honest record (including a retracted
chirality claim from earlier iterations).

Run from the repo root:
```
PYTHONPATH=. python experiments/mnist_geometric/train_ade_hopf.py
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
