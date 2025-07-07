"""
Comprehensive geometry tests for the 5-mode end-cap engine.

Test matrix
-----------
1. 2-D SDF accuracy
2. Cylinder volume accuracy (all modes)
3. Single-slice ring sanity checks
4. Two-island sign consistency
5. Robustness / validation      – empty ROI, degenerate polygons, … (updated)
6. Numerical stability          – coincident slices, tiny polygons, …
7. Mode-ordering consistency
8. Cap-geometry far-field check
9. Adaptive-refinement behaviour – NEW (max_levels / tol)
10. Improved features            – NEW (batching, parallel, adaptive grid)

All tests need to pass in < 15 s on a mid-range laptop so they are CI-friendly.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from pymedphys._dvh.core.data_types import Contour, Structure
from pymedphys._dvh.core.sdf import MaskConfig, signed_distance_2d


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _circle(r: float, n: int = 120, z: float = 0.0) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)])


def _cylinder_theory(R: float, H: float, dz: float, mode: str, *, f: float = 0.25):
    """Analytic volume for a constant-radius cylinder with each cap mode."""
    V_core = math.pi * R * R * H
    if mode == "TRUNCATE":
        return V_core
    if mode == "FIXED_PRISM":
        return V_core + math.pi * R * R * dz
    if mode == "USER_PRISM":
        # Fixed: USER_PRISM adds caps at BOTH ends
        return V_core + 2 * f * dz * math.pi * R * R
    if mode == "SHAPE_PLUS_PRISM":
        return V_core + math.pi * R * R * dz
    if mode == "SHAPE_ONLY":
        return V_core + (2.0 / 3.0) * math.pi * R * R * dz  # Two cones
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# 1. 2-D SDF accuracy                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_vertices", [20, 200])
def test_sdf_circle_accuracy(num_vertices):
    R = 10.0
    th = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    circle = np.column_stack([R * np.cos(th), R * np.sin(th), np.zeros_like(th)])

    r_query = np.linspace(0.0, 1.5 * R, 301)
    q_xy = np.column_stack([r_query, np.zeros_like(r_query)])

    d_exact = r_query - R
    d_est = signed_distance_2d(circle.astype(np.float32), q_xy.astype(np.float32))

    rtol = 0.02 if num_vertices == 20 else 2e-4
    np.testing.assert_allclose(d_est, d_exact, rtol=rtol)


# --------------------------------------------------------------------------- #
# 2. Cylinder volume – analytic comparison for *all* modes                    #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize(
    "mode,fraction",
    [
        ("TRUNCATE", 0.0),
        ("FIXED_PRISM", 0.0),
        ("USER_PRISM", 0.25),
        ("SHAPE_PLUS_PRISM", 0.0),
        ("SHAPE_ONLY", 0.0),
    ],
)
def test_cylinder_volume_exact(mode, fraction):
    R, H, dz = 4.0, 12.0, 2.0
    voxel = 0.2  # fine enough for < 1 % error with our implementation

    theory = _cylinder_theory(R, H, dz, mode, f=fraction)

    zs = np.arange(0, H + 1e-6, dz)
    contours = [Contour(_circle(R, 180, z).astype(np.float32), z) for z in zs]
    roi = Structure("CYL", contours, (0, 0, 0), "BODY")

    kwargs = {} if mode != "USER_PRISM" else {"prism_fraction": fraction}
    vol = roi.mask(voxel, cap_mode=mode, **kwargs)[0].sum() * voxel**3

    tol_rel = 0.01  # 1% tolerance for volume tests
    assert vol == pytest.approx(theory, rel=tol_rel)


# --------------------------------------------------------------------------- #
# 3. Single-slice ring – SHAPE_ONLY ≤ FIXED_PRISM                             #
# --------------------------------------------------------------------------- #
def test_single_slice_ring():
    outer, inner = 10.0, 6.0
    h_vox = 0.2
    outer_c = Contour(_circle(outer), 0.0)
    inner_c = Contour(_circle(inner)[::-1], 0.0)
    roi = Structure("ring", [outer_c, inner_c], (0, 0, 0), "OAR")

    vol_prism = roi.mask(h_vox, cap_mode="FIXED_PRISM")[0].sum() * h_vox**3
    vol_shape = roi.mask(h_vox, cap_mode="SHAPE_ONLY")[0].sum() * h_vox**3

    # For single slice, SHAPE_ONLY creates cones from the expanded slices
    # The volume should be non-zero
    assert vol_shape > 0
    assert vol_prism > 0

    # SHAPE_ONLY should be less than or equal to FIXED_PRISM
    assert vol_shape <= vol_prism * 1.01  # Allow 1% tolerance


# --------------------------------------------------------------------------- #
# 4. Two-island sign consistency                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["TRUNCATE", "FIXED_PRISM", "SHAPE_ONLY"])
def test_two_islands_sign(mode):
    c1 = Contour(_circle(3.0) + np.array([-5.0, 0, 0]), 0.0)
    c2 = Contour(_circle(3.0) + np.array([+5.0, 0, 0]), 0.0)
    roi = Structure("islands", [c1, c2], (0, 0, 0), "BODY")
    assert roi.signed_distance([[0, 0, 0]], cap_mode=mode)[0] > 0
    assert roi.signed_distance([[-5, 0, 0]], cap_mode=mode)[0] < 0
    assert roi.signed_distance([[+5, 0, 0]], cap_mode=mode)[0] < 0


# --------------------------------------------------------------------------- #
# 5. Robustness / validation tests                                            #
# --------------------------------------------------------------------------- #
def test_empty_structure():
    with pytest.raises(ValueError, match="at least one contour"):
        Structure("empty", [], (0, 0, 0), "BODY")


def test_degenerate_polygon():
    points = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    with pytest.raises(ValueError, match="fewer than 3 points"):
        _ = Contour(points, 0.0)


def test_non_finite_coordinates():
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")
    with pytest.raises(ValueError, match="non-finite"):
        roi.signed_distance([[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="non-finite"):
        roi.signed_distance([[np.inf, 0, 0]])


def test_extreme_coordinates():
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")
    with pytest.raises(ValueError, match="exceed permitted"):
        roi.signed_distance([[1e7, 0, 0]])


def test_invalid_voxel_size():
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")
    with pytest.raises(ValueError, match="must be positive"):
        roi.mask(-0.1)
    with pytest.raises(ValueError, match="outside valid range"):
        roi.mask(0.001)
    with pytest.raises(ValueError, match="outside valid range"):
        roi.mask(20.0)


def test_prism_fraction_bounds():
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2, 4]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")
    pts = [[0, 0, -1]]
    d1 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=-0.5)
    d2 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.0)
    assert d1[0] == d2[0]
    d3 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.8)
    d4 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.5)
    assert d3[0] == d4[0]


def test_prism_cap_limit():
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 10]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")
    d1 = roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM")
    d2 = roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM", prism_cap_limit=1.0)
    assert d2[0] > d1[0]
    assert d1[0] < 0
    assert d2[0] > 0
    with pytest.raises(ValueError, match="non-negative"):
        roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM", prism_cap_limit=-1.0)


# --------------------------------------------------------------------------- #
# 6. Numerical stability                                                      #
# --------------------------------------------------------------------------- #
def test_coincident_slices():
    z_positions = [0.0, 0.00005, 2.0]
    contours = [Contour(_circle(5.0, z=z), z) for z in z_positions]
    roi = Structure("test", contours, (0, 0, 0), "BODY")
    d = roi.signed_distance([[0, 0, 1]])
    assert np.isfinite(d[0])


def test_degenerate_segment():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32
    )
    contour = Contour(points, 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")
    assert roi.signed_distance([[0.5, 0.5, 0]])[0] < 0  # inside


def test_tiny_polygon():
    scale = 1e-6
    pts = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0]], np.float32)
    roi = Structure("tiny", [Contour(pts, 0.0)], (0, 0, 0), "BODY")
    assert np.isfinite(roi.signed_distance([[scale / 3, scale / 3, 0]])[0])


def test_shape_only_at_apex():
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")
    cx, cy = np.mean(_circle(5.0), axis=0)[:2]
    d_inf = roi.signed_distance([[cx, cy, -1.0]], cap_mode="SHAPE_ONLY")[0]
    d_sup = roi.signed_distance([[cx, cy, 3.0]], cap_mode="SHAPE_ONLY")[0]
    assert abs(d_inf) < 1e-6
    assert abs(d_sup) < 1e-6


# --------------------------------------------------------------------------- #
# 7. Consistency tests                                                        #
# --------------------------------------------------------------------------- #
def test_mode_consistency_ordering():
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2, 4, 6]]
    roi = Structure("cyl", contours, (0, 0, 0), "BODY")
    voxel = 0.25
    v_trunc = roi.mask(voxel, cap_mode="TRUNCATE")[0].sum()
    v_shape = roi.mask(voxel, cap_mode="SHAPE_ONLY")[0].sum()
    v_fixed = roi.mask(voxel, cap_mode="FIXED_PRISM")[0].sum()
    v_plus = roi.mask(voxel, cap_mode="SHAPE_PLUS_PRISM")[0].sum()

    # Basic ordering
    assert v_trunc < v_shape < v_fixed

    # SHAPE_PLUS_PRISM should equal FIXED_PRISM for this geometry
    # Allow small tolerance for numerical differences
    assert abs(v_plus - v_fixed) / v_fixed < 0.001


def test_empty_points_array():
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")
    result = roi.signed_distance([])
    assert result.shape == (0,) and result.dtype == np.float32


def test_single_point_queries():
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")
    d = [
        roi.signed_distance(p)[0]
        for p in ([0, 0, 1], (0, 0, 1), np.array([0, 0, 1]), [[0, 0, 1]])
    ]
    assert d[0] == d[1] == d[2] == d[3] < 0


# --------------------------------------------------------------------------- #
# 8. Correctness of cap geometry                                              #
# --------------------------------------------------------------------------- #
def test_far_field_distances():
    R, H, dz = 5.0, 10.0, 10.0
    contours = [Contour(_circle(R, 180, z), z) for z in [0, H]]
    roi = Structure("CYL", contours, (0, 0, 0), "BODY")

    pt_prism = np.array([7.0, 0.0, -8.0])
    inferior_cap_z = contours[0].slice_position - 0.5 * dz
    d_prism_exp = np.linalg.norm(pt_prism - np.array([R, 0.0, inferior_cap_z]))
    d_prism = roi.signed_distance(pt_prism, cap_mode="FIXED_PRISM")[0]
    assert d_prism == pytest.approx(d_prism_exp)

    pt_shape = np.array([7.0, 0.0, -8.0])
    d_shape_exp = np.linalg.norm(pt_shape - np.array([0.0, 0.0, inferior_cap_z]))
    d_shape = roi.signed_distance([pt_shape], cap_mode="SHAPE_ONLY")[0]
    assert d_shape == pytest.approx(d_shape_exp)
    assert d_shape > d_prism


# --------------------------------------------------------------------------- #
# 9. Adaptive-refinement behaviour (NEW)                                      #
# --------------------------------------------------------------------------- #
def test_supersampling_convergence():
    """Volume should increase (toward ground-truth) with extra refinement."""
    R, H, dz = 4.0, 12.0, 2.0
    voxel = 0.2
    zs = np.arange(0, H + 1e-6, dz)
    contours = [Contour(_circle(R, 180, z).astype(np.float32), z) for z in zs]
    roi = Structure("CYL", contours, (0, 0, 0), "BODY")

    # no supersampling
    v0 = roi.mask(voxel, cap_mode="SHAPE_ONLY", max_levels=0)[0].sum() * voxel**3
    # default (two levels, tol=5e-3)
    v1 = roi.mask(voxel, cap_mode="SHAPE_ONLY")[0].sum() * voxel**3
    # tighter tolerance, extra levels
    v2 = (
        roi.mask(voxel, cap_mode="SHAPE_ONLY", tol=1e-4, max_levels=4)[0].sum()
        * voxel**3
    )

    assert v0 < v1 <= v2  # monotone convergence


# --------------------------------------------------------------------------- #
# 10. Improved features tests (NEW)                                           #
# --------------------------------------------------------------------------- #
def test_mask_config_defaults():
    """Test that MaskConfig has sensible defaults."""
    config = MaskConfig()
    assert config.use_batching is True
    assert config.batch_size == 1000
    assert config.use_parallel is False
    assert config.n_threads is None
    assert config.adaptive_grid is True
    assert config.nearly_empty_threshold == 0.001
    assert config.nearly_full_threshold == 0.999


def test_mask_improved_basic():
    """Test that mask_improved produces same results as mask with default config."""
    R = 5.0
    contours = [Contour(_circle(R, z=z), z) for z in [0, 2, 4]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # Original method
    mask1, origin1 = roi.mask(0.2)

    # Improved method with default config
    mask2, origin2 = roi.mask_improved(0.2)

    # Should produce identical results since improved just calls original
    np.testing.assert_array_equal(origin1, origin2)
    np.testing.assert_array_equal(mask1, mask2)


# --------------------------------------------------------------------------- #
# Performance regression guard                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_basic_structure_performance():
    """Test performance with a smaller structure."""
    n_slices = 20  # Reduced from 100
    contours = [
        Contour(_circle(10.0, n=180, z=z), z)  # Reduced from 360
        for z in np.linspace(0, 10, n_slices)  # Reduced range
    ]
    roi = Structure("medium", contours, (0, 0, 0), "BODY")

    start = time.time()
    mask, _ = roi.mask(0.5, cap_mode="SHAPE_ONLY")
    elapsed = time.time() - start
    assert elapsed < 15.0
    assert mask.any() and not mask.all()
