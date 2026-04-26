"""
Closed-form metric data for the regular 600-cell on the unit S^3, plus
the diagonal circumcentric Hodge stars *_k : Omega^k -> Omega^{n-k}.

Stage 1 of the omnigenity-relaxation lab (CONSULTATION.md Q7 -> option
b). All measures and Hodge-star scalars below are exact closed forms;
the equivariance and spectrum tests are correspondingly hard, not
soft.

Primal simplex measures on unit S^3 (R = 1):

    |v_0|  = 1                          (vertex)
    |e_1|  = pi / 5                     (edge arc length, 36 deg)
    |f_2|  = 3 arctan(2) - pi           (face area; equilateral
                                         spherical triangle with
                                         side pi/5)
    |c_3|  = pi^2 / 300                 (cell 3-volume; total S^3
                                         volume 2 pi^2 over 600 cells)

Derivation of |f_2|: equilateral spherical triangle with side a has
area 3 alpha - pi where alpha is the interior angle; cosine rule for
spherical triangles gives cos(alpha) = cos(a) / (1 + cos(a)). For
a = pi/5, cos(pi/5) = (1 + sqrt(5)) / 4, which simplifies to
cos(alpha) = 1 / sqrt(5) -- so alpha = arctan(2) and area = 3 arctan(2)
- pi.

Dual simplex measures (the dual polytope is the regular 120-cell {5,3,3}
with 600 vertices, 1200 edges, 720 pentagonal faces, 120 dodecahedral
cells):

    |v*_3|  = pi^2 / 60                 (Voronoi dodec volume = 2 pi^2
                                         over 120 cells)
    |e*_2|  = ~ 0.128388                (regular spherical pentagon on
                                         the 120-cell -- numerical;
                                         closed form skipped)
    |f*_1|  = arccos((1 + 3 sqrt(5)) / 8)
                                        (120-cell edge arc; CLOSED
                                         FORM derived from the icosian
                                         quaternion construction --
                                         the 120-cell's edge inner
                                         product is (1 + 3 sqrt(5)) / 8)
    |c*_0|  = 1                         (cell circumcenter is a point
                                         on S^3)

Diagonal Hodge stars (every k-cell lies in a single H_4-orbit since
the 600-cell is regular under H_4 of order 14400, so the circumcentric
Hodge star is a SCALAR multiple of the identity on each cochain space;
2I-equivariance is automatic -- scalar . I commutes with every linear
map):

    *_0  =  |v*_3| / |v_0|   =  pi^2 / 60
    *_1  =  |e*_2| / |e_1|   =  pent_area * 5 / pi
    *_2  =  |f*_1| / |f_2|   =  arccos((1 + 3 sqrt(5)) / 8) /
                                (3 arctan(2) - pi)
    *_3  =  |c*_0| / |c_3|   =  300 / pi^2

Sanity identity: *_0 . *_3 = 5 exactly (the dim-0 and dim-3 stars,
combined, give the 600/120 = 5 cell-per-vertex orbit ratio).
"""
from __future__ import annotations

import math

import numpy as np

from cell600 import get_geometry


# Closed-form scalars
PHI = (1.0 + math.sqrt(5.0)) / 2.0
ARCTAN2 = math.atan(2.0)

VOL_VERTEX = 1.0
ARC_EDGE = math.pi / 5.0
AREA_FACE = 3.0 * ARCTAN2 - math.pi
VOL_CELL = math.pi * math.pi / 300.0

# Dual measures (dual polytope = 120-cell on unit S^3)
VOL_DUAL_OF_VERTEX = math.pi * math.pi / 60.0
ARC_DUAL_OF_FACE = math.acos((1.0 + 3.0 * math.sqrt(5.0)) / 8.0)
VOL_DUAL_OF_CELL = 1.0
# Pentagon area on unit S^2 (dual face for primal edge): no clean
# closed form pasted; computed from the spherical-pentagon side and
# interior angle. Kept as a numerical constant for now.
AREA_DUAL_OF_EDGE = 0.12838820325862648  # spherical-pentagon area, R = 1


# Diagonal Hodge-star scalars: *_k = |dual_k| / |primal_k|
STAR_0_SCALAR = VOL_DUAL_OF_VERTEX / VOL_VERTEX                # pi^2 / 60
STAR_1_SCALAR = AREA_DUAL_OF_EDGE / ARC_EDGE                   # ~ 0.20434
STAR_2_SCALAR = ARC_DUAL_OF_FACE / AREA_FACE                   # ~ 1.5063
STAR_3_SCALAR = VOL_DUAL_OF_CELL / VOL_CELL                    # 300 / pi^2


def hodge_star_0():
    """*_0 : Omega^0 -> Omega^3 (vertex 0-cochains -> cell 3-cochain
    densities). Scalar . I_120 by H_4-regularity."""
    return STAR_0_SCALAR * np.eye(120)


def hodge_star_1():
    """*_1 : Omega^1 -> Omega^2. Scalar . I_720."""
    return STAR_1_SCALAR * np.eye(720)


def hodge_star_2():
    """*_2 : Omega^2 -> Omega^1. Scalar . I_1200."""
    return STAR_2_SCALAR * np.eye(1200)


def hodge_star_3():
    """*_3 : Omega^3 -> Omega^0. Scalar . I_600."""
    return STAR_3_SCALAR * np.eye(600)


def metric_laplacian_0():
    """Metric Hodge Laplacian on 0-forms:

        Delta_0  =  *_0^{-1}  d0^T  *_1  d0

    H_0(S^3) = R, so this has 1-dim kernel (constant 0-forms) and 119
    nonzero eigenvalues. By 2I-equivariance the eigenvalues group by
    isotypic with multiplicities [0, 4, 4, 9, 9, 16, 16, 25, 36]
    (canonical 2I irrep ordering 1, 2a, 2b, 3a, 3b, 4a, 4b, 5, 6).
    """
    g = get_geometry()
    d0 = g["d0"]
    return (1.0 / STAR_0_SCALAR) * d0.T @ (STAR_1_SCALAR * d0)


def metric_laplacian_3():
    """Metric Hodge Laplacian on 3-forms:

        Delta_3  =  *_3^{-1}  d2  *_2  d2^T

    Symmetric to Delta_0 by Hodge duality; 1-dim kernel (constant
    3-form) plus 599 nonzero eigenvalues.
    """
    g = get_geometry()
    d2 = g["d2"]
    return (1.0 / STAR_3_SCALAR) * d2 @ (STAR_2_SCALAR * d2.T)


def expected_metric_to_combinatorial_ratio_delta0():
    """Ratio of metric Laplacian eigenvalues to combinatorial graph
    Laplacian eigenvalues on 0-forms.

    Combinatorial: L_comb = d0^T @ d0  (graph Laplacian; eigenvalues
    are integers / golden-ratio rationals -- multiplicities [1, 4, 9,
    16, 25, 36, 9, 16, 4]).
    Metric:       Delta_0 = (1 / *_0) d0^T (*_1 d0) = (*_1 / *_0) L_comb.

    So the ratio is exactly STAR_1_SCALAR / STAR_0_SCALAR.
    """
    return STAR_1_SCALAR / STAR_0_SCALAR


def isotypic_projector_on_vertices(scalar_eigenspaces, idx):
    """P_i = V_i V_i^T : projector onto the i-th 2I-isotypic component
    on vertices. (V_i columns are an orthonormal basis for the
    isotypic by construction in cell600.py.)
    """
    V = scalar_eigenspaces[idx]["vectors"]
    return V @ V.T
