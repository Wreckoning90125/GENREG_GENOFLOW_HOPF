# ================================================================
# Nematic director-field defect dataset.
#
# A 2D nematic director field is a field of UNSIGNED orientations
# theta(x,y) in [0, pi)  (a "headless arrow": n ~ -n). Its point
# defects carry a winding charge
#     q = (1/2pi) * contour integral of grad theta
# which, because theta is defined mod pi, can be a HALF-INTEGER.
# The q = +-1/2 defects are the signature of nematic (director)
# order -- they cannot exist in a polar (vector) field.
#
# Distinguishing +1/2 from -1/2 requires the winding SIGN of the
# director, which lives in the double cover -- i.e. a PROJECTIVE
# representation of the rotation group, classified by H^2(.,U(1)).
# This is the physical incarnation of the cohomological obstruction
# that the Warman-Schafer-Nameki cocycle alpha_N measures.
#
# This module generates such fields (standard defect formula
# theta = q*phi + theta0), with random global orientation gauge
# theta0, defect-position jitter, reflection (chirality) and noise,
# and encodes each pixel's director as a histogram over the 4N bins
# of the cyclic group Z_{4N} -- the per-site Hilbert space C[Z_4N]
# on which the cocycle machinery acts.  Nothing about alpha_N is
# injected into the data: the projective structure is forced by the
# physics (theta mod pi), not by us.
# ================================================================

from __future__ import annotations

import numpy as np

# the four defect charges; the +-1/2 pair is the hard, spinor-sensitive one
CHARGES = (-1.0, -0.5, 0.5, 1.0)
CLASS_NAMES = ("q=-1", "q=-1/2", "q=+1/2", "q=+1")


def director_field(q, H, W, theta0, cx, cy, reflect=False, noise=0.0, rng=None):
    """Director angle field theta(x,y) in [0,pi) for a charge-q defect.

    theta = q * atan2(dy, dx) + theta0   (standard isolated-defect form);
    optional reflection flips chirality (x -> -x), optional von-Mises-ish
    angular noise. Returned mod pi (director identification).
    """
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    dx = xs - cx
    dy = ys - cy
    if reflect:
        dx = -dx
    phi = np.arctan2(dy, dx)
    theta = q * phi + theta0
    if noise and rng is not None:
        theta = theta + noise * rng.standard_normal((H, W))
    return np.mod(theta, np.pi)


def encode_director_histogram(theta, n_bins, kappa=8.0):
    """Encode each pixel's director (theta in [0,pi)) as a soft
    histogram over the 4N bins of Z_{4N} (full circle [0,2pi)).

    The director identification n ~ -n is enforced by depositing mass
    at BOTH theta and theta+pi, so the histogram is invariant under the
    central rotation r^{2N} (the pi rotation) -- exactly the structure
    that makes the half-integer content a projective (spinor) feature.

    Returns array (n_bins, H, W), L1-normalized over bins per pixel.
    """
    H, W = theta.shape
    bin_centers = np.arange(n_bins) * (2 * np.pi / n_bins)  # [0,2pi)
    # angular distance to each bin centre, for theta and theta+pi
    feat = np.zeros((n_bins, H, W))
    for shift in (0.0, np.pi):
        ang = (theta + shift)[None, :, :]                   # (1,H,W)
        d = ang - bin_centers[:, None, None]                # (n_bins,H,W)
        feat += np.exp(kappa * np.cos(d))                   # von Mises kernel
    feat /= feat.sum(axis=0, keepdims=True) + 1e-12
    return feat.astype(np.float32)


def make_sample(label_idx, H, W, n_bins, rng, gauge=True, jitter=True, noise=0.05):
    q = CHARGES[label_idx]
    theta0 = rng.uniform(0, np.pi) if gauge else 0.0
    cx = W / 2 + (rng.uniform(-3, 3) if jitter else 0.0)
    cy = H / 2 + (rng.uniform(-3, 3) if jitter else 0.0)
    reflect = bool(rng.integers(0, 2))
    theta = director_field(q, H, W, theta0, cx, cy, reflect=reflect,
                           noise=noise, rng=rng)
    return encode_director_histogram(theta, n_bins)


def make_dataset(n_per_class, N=4, H=28, W=28, seed=0,
                 gauge=True, jitter=True, noise=0.05):
    """Build a balanced 4-class defect dataset.

    N selects the group D_{4N}; orientation is binned into 4N bins.
    Returns X (n, 4N, H, W) float32, y (n,) int64.
    """
    rng = np.random.default_rng(seed)
    n_bins = 4 * N
    X, y = [], []
    for label in range(len(CHARGES)):
        for _ in range(n_per_class):
            X.append(make_sample(label, H, W, n_bins, rng,
                                 gauge=gauge, jitter=jitter, noise=noise))
            y.append(label)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_director_image_dataset(n_per_class, H=25, W=25, seed=0,
                                gauge=True, jitter=True, noise=0.05):
    """Balanced 4-class defect dataset as 2-channel director IMAGES
    (cos 2theta, sin 2theta) -- the standard headless nematic encoding,
    on which a D_4 group-equivariant CNN operates.  Defect centered (so
    90-deg rotations act cleanly) with optional integer-pixel jitter.

    Returns X (n, 2, H, W) float32, y (n,) int64.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    cx0, cy0 = (W - 1) / 2.0, (H - 1) / 2.0
    for label in range(len(CHARGES)):
        q = CHARGES[label]
        for _ in range(n_per_class):
            theta0 = rng.uniform(0, np.pi) if gauge else 0.0
            dx = int(rng.integers(-2, 3)) if jitter else 0
            dy = int(rng.integers(-2, 3)) if jitter else 0
            # NB: NO random reflection here -- a reflection flips the
            # winding-charge sign (+q <-> -q), which is exactly the label,
            # so reflecting would corrupt it.  Charge IS reflection-odd,
            # which is why a reflection-invariant (linear) equivariant net
            # is architecturally blind to the sign -- the obstruction.
            th = director_field(q, H, W, theta0, cx0 + dx, cy0 + dy,
                                reflect=False, noise=noise, rng=rng)
            img = np.stack([np.cos(2 * th), np.sin(2 * th)]).astype(np.float32)
            X.append(img)
            y.append(label)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def gauge_shift(X, k):
    """Apply a global orientation-gauge rotation r^k to a batch:
    cyclically shift the 4N orientation channels by k bins.

    Physically: rotate every director by 2*pi*k/(4N). The defect charge
    (the label) is invariant under this gauge transform, so it is the
    natural symmetry to test inductive bias against.
    """
    return np.roll(X, shift=k, axis=1)


if __name__ == "__main__":
    X, y = make_dataset(50, N=4, H=28, W=28, seed=1)
    print("dataset:", X.shape, y.shape, "classes:", np.bincount(y))
    print("per-pixel histogram sums ~1:", float(X.sum(1).mean()))
    # sanity: gauge shift by 2N (pi rotation) leaves the director invariant
    N = 4
    Xs = gauge_shift(X, 2 * N)
    print("pi-rotation (r^{2N}) leaves director histogram invariant:",
          f"max diff = {np.max(np.abs(Xs - X)):.2e}")
    # +1/2 vs -1/2 must be distinguishable in the data (else task ill-posed)
    Xp = X[y == 2].mean(0)  # +1/2
    Xm = X[y == 1].mean(0)  # -1/2
    print(f"mean |+1/2 - -1/2| histogram difference: {np.mean(np.abs(Xp - Xm)):.4f}")
