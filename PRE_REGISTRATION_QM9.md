# Pre-registration — QM9 molecular property prediction

This document pre-registers the first non-MNIST test of the 600-cell
geometric feature extractor. It exists because the Hopf subsystem's
MNIST work (documented in `FINDINGS.md`) saturated at v10, the chirality
claim was retracted, and MNIST is not a venue where the geometric
inductive bias is tested meaningfully. QM9 molecular property prediction
is the first real test of whether the icosahedral eigenspace decomposition
works outside digit classification.

**Status: pipeline built, awaiting QM9 data download.** The pipeline
(`train_qm9.py`) is tested on synthetic molecules and produces the
expected feature matrix shape and ridge regression output. Real QM9
numbers require downloading ~660MB of data from figshare.

## What is being tested

The question: does the 600-cell ADE eigenspace feature basis, combined
with a kernel ridge readout, achieve competitive accuracy on molecular
property prediction — a domain where the published baselines are
equivariant neural networks (SchNet, DimeNet, NequIP, MACE) that use
gradient descent over millions of learned parameters?

This is a test of the **inductive bias** (icosahedral symmetry as an
approximation to continuous SO(3) equivariance), not the readout or the
training algorithm. If competitive: the fixed geometric basis captures
physically meaningful molecular structure without any message passing
or learned features. If not: the icosahedral sampling does not encode
enough molecular structure without learned distance embedding.

## Dataset

**QM9** (Ramakrishnan et al., 2014): 133,885 small organic molecules
with up to 29 atoms each (H, C, N, O, F). 19 quantum mechanical
properties computed at the B3LYP/6-31G(2df,p) level.

- 3,054 "uncharacterized" molecules excluded (convergence issues).
- Standard split: 110,000 train / 10,000 val / ~10,831 test.
- Random shuffle with seed=42 (matching convention from SchNet papers).

## Feature construction (pinned)

For each molecule, centered on center of mass:

1. **Per-atom direction**: r̂ = r / |r| in S² (atoms at r ≈ 0 get
   uniform angular assignment).
2. **Hopf section lift**: r̂ → q ∈ S³ via the standard Hopf section.
3. **Von-Mises-Fisher soft assignment**: softmax(κ · |q · v_j|) for
   120 600-cell vertices.
4. **Radial weighting**: exp(-|r|² / 2σ²) per atom.
5. **Atom-type channels**: 5 channels (H, C, N, O, F), accumulated
   independently.
6. **ADE eigenspace decomposition**: 293 features per (κ, σ, channel)
   via `extract_features_from_F` in `train_ade_hopf.py`.
7. **Multi-scale**: κ ∈ {3.0, 5.5, 8.0}, σ ∈ {0.5, 1.0, 2.0} Å.
8. **Concatenation**: 293 × 3 kappas × 3 sigmas × 5 channels = 13,185
   features per molecule.
9. **Standardization**: mean/std computed on training set, applied to all.
10. **Readout**: polynomial kernel ridge (degree 2, Nystrom with
    m ∈ {3000, 5000}), alpha swept over {0.01, 0.1, 1.0}.

No step above is learned or tuned after seeing the test set.

## Properties to predict

| Property | QM9 index | Unit | Scale factor |
|----------|-----------|------|-------------|
| gap      | 9         | meV  | ×27211.386  |
| HOMO     | 7         | meV  | ×27211.386  |
| LUMO     | 8         | meV  | ×27211.386  |
| U0       | 12        | meV  | ×27211.386  |
| mu       | 5         | D    | ×1          |
| alpha    | 6         | Bohr³| ×1          |
| Cv       | 16        | cal/(mol K) | ×1   |

## Our baselines (implemented, not cited)

1. **Coulomb matrix + ridge**: sorted eigenvalues of the Coulomb
   matrix (C_ij = Z_i·Z_j / |r_i - r_j|, C_ii = 0.5·Z_i^2.4)
   → same ridge/kernel-ridge readout. This is a standard classical
   molecular fingerprint. If we lose to it, the 600-cell features add
   nothing over a simple distance-based representation.

2. **Random features + ridge**: random (N, 13185) feature matrix with
   the same standardization + kernel ridge. Floor baseline.

## Published baselines (cited, not reimplemented)

| Property | SchNet | DimeNet++ | NequIP | MACE |
|----------|--------|-----------|--------|------|
| gap (meV)| 63     | 32.6      | ~29    | ~28  |
| HOMO     | 41     | 24.6      | ~15    | ~11  |
| LUMO     | 34     | 19.5      | ~13    | ~9   |
| U0       | 14     | 6.3       | ~4     | ~2   |
| mu (D)   | 0.033  | 0.030     | ~0.012 | ~0.010|

Sources: Schütt et al. 2018 (SchNet), Gasteiger et al. 2020 (DimeNet++),
Batzner et al. 2022 (NequIP), Batatia et al. 2022 (MACE).

## Success / null criteria

**Competitive (success):** MAE within 2× of SchNet on at least 3 of 7
properties. This demonstrates that fixed geometric features capture
meaningful molecular structure without message passing or learned
parameters.

**Strong result:** Beat Coulomb-matrix + ridge AND be within 3× of
SchNet on gap. The icosahedral decomposition adds structure beyond
classical distance-based fingerprints.

**Null:** Worse than 5× SchNet on all properties, OR lose to
Coulomb-matrix + ridge on all properties. The icosahedral sampling
does not capture enough molecular structure without learned distance
encoding. Write up as a negative result in FINDINGS.md.

**Sanity check:** We should NOT beat MACE or NequIP on any property.
If we do, investigate for data leak, unit error, or wrong split before
reporting.

## Known risks and honest caveats

1. **Directional encoding misses interatomic distance information.**
   Our pipeline projects atoms to S² by direction from center of mass.
   Distance enters only via the radial Gaussian. Molecular properties
   like HOMO-LUMO gap depend critically on specific interatomic
   distances (bond lengths). Multi-σ mitigates but may not suffice.

2. **Center-of-mass reference frame is chemistry-naive.** Two molecules
   with different bonding but similar radial atom distributions from
   center of mass will look similar in our features. An atom-centered
   frame (each atom as reference, aggregate) would be more principled
   but is a larger implementation lift.

3. **13,185 features with 110k training samples.** Regularization
   matters. Alpha sweep is our only defense against overfitting.

4. **Approximate, not exact, SO(3) equivariance.** The 600-cell has
   icosahedral (60-fold) symmetry, not continuous SO(3). Features
   change slightly under arbitrary rotations. The rotation invariance
   test in `mol_kernel.py` quantifies this breaking.

## What a result would and would not mean

**Would mean:** the 600-cell / ADE / Hopf machinery encodes molecular
structure well enough to compete with (or fall short of, informatively)
gradient-trained equivariant neural networks on a standard benchmark.
Either outcome is useful: competitive = the fixed basis works;
non-competitive = the specific failure mode (missing distance info,
insufficient angular resolution, CoM frame) tells us what to fix.

**Would not mean:**
- "The geometry replaces gradient descent." (We're comparing fixed
  features + ridge to learned features + SGD. They're different
  paradigms with different costs.)
- "Icosahedral symmetry is the right inductive bias for chemistry."
  (One dataset, one split, one readout. Narrow scope.)
- "The chirality claim is vindicated." (Unrelated. See FINDINGS.md.)
