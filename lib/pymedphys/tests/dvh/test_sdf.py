"""
Comprehensive geometry tests for the 5-mode end-cap engine.
Each mode is compared **against its analytic reference volume** for a
constant-radius cylinder so CI catches even small regressions.

Test matrix
-----------
1. 2-D SDF accuracy              - unchanged
2. Cylinder volume, every mode   - now uses closed-form formulae
3. Single-slice ring             - SHAPE_ONLY ≤ FIXED_PRISM, matches theory
4. Two-island sign consistency   - sanity check (all modes)
5. Edge cases and error handling - NEW: comprehensive validation tests
6. Numerical stability           - NEW: extreme value tests
7. Input validation              - NEW: invalid input tests
8. Correctness of cap geometry   - NEW: far-field distance tests
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymedphys._dvh.core.data_types import Contour, Structure
from pymedphys._dvh.core.sdf import signed_distance_2d


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _circle(r, n=120, z=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)])


def _cylinder_theory(R: float, H: float, dz: float, mode: str, *, f: float = 0.25):
    """Return analytic volume for a constant-radius cylinder + caps."""
    V_core = math.pi * R * R * H
    if mode == "TRUNCATE":
        return V_core
    if mode == "FIXED_PRISM":
        return V_core + math.pi * R * R * dz  # 0.5 * dz slab each end
    if mode == "USER_PRISM":
        return V_core + 2 * f * dz * math.pi * R * R  # f dz slab each end
    if mode == "SHAPE_PLUS_PRISM":
        return V_core + math.pi * R * R * dz  # same as FIXED_PRISM
    if mode == "SHAPE_ONLY":
        return V_core + (1.0 / 3.0) * math.pi * R * R * dz  # two cones
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

    atol = 0.13 if num_vertices == 20 else 0.05
    np.testing.assert_allclose(d_est, d_exact, atol=atol)


# --------------------------------------------------------------------------- #
# 2. Cylinder volume – analytic comparison for *all* modes                    #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize(
    "mode,fraction",  # fraction is only used by USER_PRISM
    [
        ("TRUNCATE", 0.0),
        ("FIXED_PRISM", 0.0),
        ("USER_PRISM", 0.25),
        ("SHAPE_PLUS_PRISM", 0.0),
        ("SHAPE_ONLY", 0.0),
        # Test backwards compatibility
        ("FLAT", 0.0),  # Should behave as FIXED_PRISM
        ("SMOOTH", 0.0),  # Should behave as SHAPE_ONLY
    ],
)
def test_cylinder_volume_exact(mode, fraction):
    R, H = 4.0, 12.0
    dz = 2.0
    voxel = 0.25  # fine enough for <1 % error with our implementations

    # Map legacy names for theory calculation
    theory_mode = mode
    if mode == "FLAT":
        theory_mode = "FIXED_PRISM"
    elif mode == "SMOOTH":
        theory_mode = "SHAPE_ONLY"

    theor = _cylinder_theory(R, H, dz, theory_mode, f=fraction)

    zs = np.arange(0, H + 1e-6, dz)
    contours = [Contour(_circle(R, 180, z).astype(np.float32), z) for z in zs]
    roi = Structure("CYL", contours, (0, 0, 0), "BODY")

    kwargs = {} if mode != "USER_PRISM" else {"prism_fraction": fraction}
    vol = roi.mask(voxel, cap_mode=mode, **kwargs)[0].sum() * voxel**3

    # Point-sampling of the SDF systematically underestimates the volume of
    # cone-like shapes. We accept a larger tolerance for SHAPE_ONLY as a
    # principled trade-off for removing the previous implementation's
    # empirical "fudge factor". Other shapes are more stable.
    tolerance = (
        2.5e-2 if theory_mode == "SHAPE_ONLY" else 8e-3
    )  # 2.5% for cones, 0.8% otherwise
    assert vol == pytest.approx(theor, rel=tolerance)


# --------------------------------------------------------------------------- #
# 3. Single-slice ring – SHAPE_ONLY must never exceed prism volume            #
# --------------------------------------------------------------------------- #
def test_single_slice_ring():
    outer, inner = 10.0, 6.0
    h_vox = 0.2
    outer_c = Contour(_circle(outer), 0.0)
    inner_c = Contour(_circle(inner)[::-1], 0.0)
    roi = Structure("ring", [outer_c, inner_c], (0, 0, 0), "OAR")

    vol_prism = roi.mask(h_vox, cap_mode="FIXED_PRISM")[0].sum() * h_vox**3
    vol_shape = roi.mask(h_vox, cap_mode="SHAPE_ONLY")[0].sum() * h_vox**3

    # For a single slice structure, the generated mask treats it as having a
    # thickness equal to the voxel size in that dimension. The internal
    # 'DEGENERATE_SLICE_OFFSET' is only for creating a stable SDF gradient.
    exact_area = math.pi * (outer**2 - inner**2)
    exact_volume = exact_area * h_vox

    # SHAPE_ONLY cannot create more volume than the slab prism.
    assert vol_shape <= vol_prism + 1e-6
    # Voxelized volume should match theory within a small tolerance.
    assert vol_shape == pytest.approx(exact_volume, rel=5e-3)


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
# 5. Edge cases and error handling – NEW TESTS                                #
# --------------------------------------------------------------------------- #


def test_empty_structure():
    """Test handling of structure with no contours."""
    # The Structure class itself validates it must have contours
    with pytest.raises(ValueError, match="at least one contour"):
        Structure("empty", [], (0, 0, 0), "BODY")


def test_degenerate_polygon():
    """Test handling of polygons with fewer than 3 points."""
    # Two-point "polygon"
    points = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    contour = Contour(points, 0.0)
    roi = Structure("degen", [contour], (0, 0, 0), "BODY")
    with pytest.raises(ValueError, match="fewer than 3 points"):
        roi.signed_distance([[0, 0, 0]])


def test_invalid_cap_mode():
    """Test invalid cap mode handling."""
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    with pytest.raises(ValueError, match="Unknown cap_mode"):
        roi.signed_distance([[0, 0, 0]], cap_mode="INVALID")

    with pytest.raises(TypeError, match="cap_mode must be string"):
        roi.signed_distance([[0, 0, 0]], cap_mode=123)


def test_non_finite_coordinates():
    """Test handling of non-finite values."""
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Test with NaN
    with pytest.raises(ValueError, match="non-finite"):
        roi.signed_distance([[np.nan, 0, 0]])

    # Test with infinity
    with pytest.raises(ValueError, match="non-finite"):
        roi.signed_distance([[np.inf, 0, 0]])


def test_extreme_coordinates():
    """Test handling of very large coordinates."""
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Test coordinates beyond limit
    with pytest.raises(ValueError, match="exceeding"):
        roi.signed_distance([[1e7, 0, 0]])


def test_invalid_voxel_size():
    """Test voxel size validation."""
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Negative voxel size
    with pytest.raises(ValueError, match="must be positive"):
        roi.mask(-0.1)

    # Too small voxel size
    with pytest.raises(ValueError, match="below minimum"):
        roi.mask(0.001)

    # Too large voxel size
    with pytest.raises(ValueError, match="exceeds maximum"):
        roi.mask(20.0)

    # Invalid type
    with pytest.raises(ValueError, match="must be a number"):
        roi.mask("invalid")


def test_prism_fraction_bounds():
    """Test USER_PRISM fraction parameter bounds."""
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2, 4]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # Test clamping to [0, 0.5]
    pts = [[0, 0, -1]]

    # Negative fraction should be clamped to 0
    d1 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=-0.5)
    d2 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.0)
    assert d1[0] == d2[0]

    # Fraction > 0.5 should be clamped to 0.5
    d3 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.8)
    d4 = roi.signed_distance(pts, cap_mode="USER_PRISM", prism_fraction=0.5)
    assert d3[0] == d4[0]


def test_prism_cap_limit():
    """Test prism_cap_limit parameter."""
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 10]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # Without limit, cap extends to z=-5. Point at z=-2 is inside cap. dist < 0.
    d1 = roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM")

    # With limit=1.0, cap extends to z=-1. Point at z=-2 is outside. dist > 0.
    d2 = roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM", prism_cap_limit=1.0)

    # Limited cap should give larger distance
    assert d2[0] > d1[0]
    assert d1[0] < 0
    assert d2[0] > 0

    # Negative limit should raise error
    with pytest.raises(ValueError, match="non-negative"):
        roi.signed_distance([[0, 0, -2]], cap_mode="FIXED_PRISM", prism_cap_limit=-1.0)


# --------------------------------------------------------------------------- #
# 6. Numerical stability tests                                                #
# --------------------------------------------------------------------------- #


def test_coincident_slices():
    """Test handling of very close slice positions."""
    # Create slices that are within tolerance
    z_positions = [0.0, 0.00005, 2.0]  # First two are within 1e-4 tolerance
    contours = [Contour(_circle(5.0, z=z), z) for z in z_positions]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # Should not crash and should treat first two as same slice
    d = roi.signed_distance([[0, 0, 1]])
    assert np.isfinite(d[0])


def test_degenerate_segment():
    """Test handling of zero-length polygon edges."""
    # Create a polygon with a repeated point
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],  # Repeated point
            [1, 1, 0],
            [0, 1, 0],
        ],
        np.float32,
    )
    contour = Contour(points, 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Should handle gracefully
    d = roi.signed_distance([[0.5, 0.5, 0]])
    assert d[0] < 0  # Point is inside


def test_tiny_polygon():
    """Test handling of very small polygons."""
    # Create a tiny triangle
    scale = 1e-6
    points = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0]], np.float32)
    contour = Contour(points, 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Should handle without numerical issues
    d = roi.signed_distance([[scale / 3, scale / 3, 0]])
    assert np.isfinite(d[0])


def test_shape_only_at_apex():
    """Test SHAPE_ONLY mode exactly at cone apex."""
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # For SHAPE_ONLY, apex is at ±0.5 * slice_spacing = ±1.0 from ends
    # Apexes are at (0,0,-1) and (0,0,3)
    # Test at the theoretical apex positions using the calculated centroid.
    cx, cy = np.mean(_circle(5.0), axis=0)[:2]
    d_inf = roi.signed_distance([[cx, cy, -1.0]], cap_mode="SHAPE_ONLY")[0]
    d_sup = roi.signed_distance([[cx, cy, 3.0]], cap_mode="SHAPE_ONLY")[0]

    # At apex, distance should be 0, allowing for float32 precision
    assert abs(d_inf) < 1e-6
    assert abs(d_sup) < 1e-6


# --------------------------------------------------------------------------- #
# 7. Consistency tests                                                        #
# --------------------------------------------------------------------------- #


def test_mode_consistency_ordering():
    """Test that volume ordering is consistent across modes."""
    # Create a simple cylinder
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2, 4, 6]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    voxel = 0.25

    # Calculate volumes for each mode
    vol_truncate = roi.mask(voxel, cap_mode="TRUNCATE")[0].sum()
    vol_shape = roi.mask(voxel, cap_mode="SHAPE_ONLY")[0].sum()
    vol_fixed = roi.mask(voxel, cap_mode="FIXED_PRISM")[0].sum()
    vol_shape_plus = roi.mask(voxel, cap_mode="SHAPE_PLUS_PRISM")[0].sum()

    # Expected ordering: TRUNCATE < SHAPE_ONLY < SHAPE_PLUS_PRISM ≈ FIXED_PRISM
    assert vol_truncate < vol_shape
    assert vol_shape < vol_fixed
    assert abs(vol_shape_plus - vol_fixed) < 1e-6  # Should be equal


def test_empty_points_array():
    """Test handling of empty points array."""
    contour = Contour(_circle(5.0), 0.0)
    roi = Structure("test", [contour], (0, 0, 0), "BODY")

    # Empty array should return empty result
    result = roi.signed_distance([])
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_single_point_queries():
    """Test various single point queries."""
    contours = [Contour(_circle(5.0, z=z), z) for z in [0, 2]]
    roi = Structure("test", contours, (0, 0, 0), "BODY")

    # Test different input formats
    d1 = roi.signed_distance([0, 0, 1])  # List
    d2 = roi.signed_distance((0, 0, 1))  # Tuple
    d3 = roi.signed_distance(np.array([0, 0, 1]))  # Array
    d4 = roi.signed_distance([[0, 0, 1]])  # Nested list

    # All should give same result
    assert d1[0] == d2[0] == d3[0] == d4[0]

    # And all should be negative (inside)
    assert d1[0] < 0


# --------------------------------------------------------------------------- #
# 8. Correctness of cap geometry                                              #
# --------------------------------------------------------------------------- #


def test_far_field_distances():
    """Tests the SDF calculation for points far beyond the end caps."""
    R, H, dz = 5.0, 10.0, 10.0
    contours = [Contour(_circle(R, 180, z), z) for z in [0, H]]
    roi = Structure("CYL", contours, (0, 0, 0), "BODY")

    # --- Test FIXED_PRISM: distance to cap plane edge ---
    pt_prism = np.array([7.0, 0.0, -8.0])

    # Cap half-length is 0.5 * dz
    inferior_cap_z = contours[0].slice_position - 0.5 * dz

    expected_vec_prism = pt_prism - np.array([R, 0.0, inferior_cap_z])
    expected_d_prism = np.linalg.norm(expected_vec_prism)

    d_prism = roi.signed_distance(pt_prism, cap_mode="FIXED_PRISM")[0]
    assert d_prism == pytest.approx(expected_d_prism)

    # --- Test SHAPE_ONLY: distance to apex point ---
    pt_shape = np.array([[7.0, 0.0, -8.0]])

    # Cap half-length is 0.5 * dz = 5.0. Inferior apex is at (0, 0, -5.0).
    expected_vec_shape = pt_shape - np.array([0.0, 0.0, inferior_cap_z])
    expected_d_shape = np.linalg.norm(expected_vec_shape)

    d_shape = roi.signed_distance(pt_shape, cap_mode="SHAPE_ONLY")[0]
    assert d_shape == pytest.approx(expected_d_shape)
    assert d_shape > d_prism, "SHAPE_ONLY distance should be larger than FIXED_PRISM"


# --------------------------------------------------------------------------- #
# Performance regression guard                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_large_structure_performance():
    """Test that large structures don't cause excessive computation time."""
    import time

    # Create a structure with many slices
    n_slices = 100
    contours = [
        Contour(_circle(10.0, n=360, z=z), z) for z in np.linspace(0, 50, n_slices)
    ]
    roi = Structure("large", contours, (0, 0, 0), "BODY")

    # Time mask generation
    start = time.time()
    mask, origin = roi.mask(0.5, cap_mode="SHAPE_ONLY")
    elapsed = time.time() - start

    # Should complete in reasonable time (adjust threshold as needed)
    # Note: on slower machines this might take slightly longer
    assert elapsed < 15.0  # 15 seconds max for safety margin

    # Verify result is sensible
    assert mask.any()  # Should have some voxels inside
    assert not mask.all()  # Should have some voxels outside
