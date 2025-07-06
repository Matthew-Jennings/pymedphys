"""
Core-level tests for DVH data-types, DICOM loaders, and validate_case.

All checks live in ONE file to keep discovery simple.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymedphys._dvh.core.data_types import (
    Case,
    Contour,
    CtGeometry,
    DoseGrid,
    Structure,
)
from pymedphys._dvh.core.dicom_loader import (
    load_case,
    load_ct_geometry,
    load_rt_dose,
    load_rt_struct,
    validate_case,
)


# ----------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------
def _square_contour(z: float = 0.0, shift: float = 0.0) -> Contour:
    """1 × 1 mm square in the *xy*-plane, optionally translated +shift in all axes."""
    pts = (
        np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], dtype=np.float32) + shift
    )
    return Contour(pts, z + shift)


# ----------------------------------------------------------------------------------
# Pure DoseGrid tests
# ----------------------------------------------------------------------------------
def test_grid_roundtrip():
    """Verify forward and inverse transforms for several orientation matrices."""

    origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    idx = np.array([4.0, 5.0, 6.0], dtype=np.float32)  # grid coords (i, j, k)
    spacing = 2 * np.ones(3, dtype=np.float32)  # 1 mm spacing in all axes

    # (orientation matrix, expected patient coords for idx = [4,5,6])
    orientation_cases = [
        (
            np.eye(3, dtype=np.float32),  # identity
            np.array([9.0, 12.0, 15.0], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [0, -1, 0],  # 90° about +z
                    [1, 0, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            ),
            np.array([-9.0, 10.0, 15.0], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [1, 0, 0],  # 90° about +x
                    [0, 0, -1],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            ),
            np.array([9.0, -10.0, 13.0], dtype=np.float32),
        ),
    ]

    for orient, expected_patient in orientation_cases:
        dg = DoseGrid(
            values=np.zeros((10, 10, 10), np.float32),
            origin=origin,
            spacing=spacing,
            orientation=orient,
            units="Gy",
        )

        # ---- forward transform -------------------------------------------------
        patient_coords = dg.grid_to_patient(idx)
        np.testing.assert_allclose(patient_coords, expected_patient)

        # ---- inverse (round-trip) check ----------------------------------------
        np.testing.assert_allclose(dg.patient_to_grid(patient_coords), idx)


@pytest.mark.parametrize(
    "bad",
    [
        {"values": np.zeros((5, 5))},
        {"origin": np.zeros(2)},
        {"spacing": np.ones(2)},
        {"orientation": np.eye(2)},
    ],
)
def test_dosegrid_validation(bad):
    base = dict(
        values=np.zeros((5, 5, 5)),
        origin=np.zeros(3),
        spacing=np.ones(3),
        orientation=np.eye(3),
        units="Gy",
    )
    with pytest.raises(ValueError):
        DoseGrid(**{**base, **bad})


# ----------------------------------------------------------------------------------
# Contour + Structure short checks
# ----------------------------------------------------------------------------------
def test_contour_area_and_centroid():
    square = _square_contour().points
    c = Contour(square, 0.0)
    assert c.compute_area() == pytest.approx(1.0)
    np.testing.assert_allclose(c.compute_centroid(), [0.5, 0.5, 0.0])


# ----------------------------------------------------------------------------- #
# Volume-estimation accuracy for a rotated cylinder                             #
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_points, rel_tol",
    [
        (20, 0.02),  # low density, 2 % tolerance
        (200, 0.0002),  # 10× density, 0.02% tolerance
    ],
)
def test_structure_volume_estimate(num_points, rel_tol):
    """
    Build a synthetic cylinder, rotating each axial contour by an extra 1°
    so that slice vertices do not align vertically.  Run with two different
    contour resolutions to check that estimate_volume converges.
    """

    # 1. Cylinder dimensions (mm)
    radius, height = 5.0, 10.0

    # 2. Eleven equally spaced z-positions → Δz = 1 mm
    zs = np.linspace(0, height, 11)

    contours = []
    for i, z in enumerate(zs):
        # 3a. Base angles for this contour
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        # 3b. Rotate slice i by i × 1°
        angles += np.deg2rad(i)

        # 3c. Cartesian coordinates
        pts = np.column_stack(
            [
                radius * np.cos(angles),  # x
                radius * np.sin(angles),  # y
                np.full_like(angles, z),  # z
            ]
        )

        contours.append(Contour(pts.astype(np.float32), z))

    # 4. Structure assembly
    s = Structure("cyl", contours, (255, 0, 0), "PTV")

    # 5. Analytic cylinder volume (mm³)
    exact = np.pi * radius**2 * height

    # 6. Assert estimator accuracy
    assert s.estimate_volume(contours) == pytest.approx(exact, rel=rel_tol)


# ----------------------------------------------------------------------------------
# DICOM loader smoke tests (fixtures defined in tests/conftest.py)
# ----------------------------------------------------------------------------------
def test_loaders_smoke(rt_dose_file: Path, rt_struct_file: Path, ct_dir: Path):
    load_rt_dose(rt_dose_file)
    load_rt_struct(rt_struct_file)
    load_ct_geometry(ct_dir)


# ----------------------------------------------------------------------------------
# validate_case – happy path
# ----------------------------------------------------------------------------------
def test_full_case_and_validation(rt_dose_file, rt_struct_file, ct_dir):
    case = load_case(rt_dose_file, rt_struct_file, ct_dir)
    validate_case(case)  # should not raise


# ------------------------------------------------------------------ #
# Objects shared by all failure-mode cases                           #
# ------------------------------------------------------------------ #
dg_unit = DoseGrid(
    values=np.zeros((10, 10, 10), np.float32),
    origin=np.zeros(3),
    spacing=np.ones(3),
    orientation=np.eye(3, dtype=np.float32),
    units="Gy",
)

ct_mismatch = CtGeometry(  # same shape, *different* orientation
    shape=(10, 10, 10),
    origin=np.zeros(3),
    spacing=np.ones(3),
    orientation=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32),
)


def _case_no_contours() -> Case:
    """Start with 1 contour, then empty list so validation—not dataclass—fails."""
    roi = Structure("empty", [_square_contour()], (0, 0, 0), "PTV")
    roi.contours.clear()
    return Case(None, {"empty": roi})  # no dose grid ⇒ bbox code not reached


def _two_partial_contours() -> list[Contour]:
    """Two slices whose bbox straddles the (0,0) origin → 'extends beyond'."""
    return [_square_contour(shift=-0.5, z=z) for z in (0.0, 3.0)]


# ------------------------------------------------------------------ #
# Parametric table – eight failure cases                             #
# ------------------------------------------------------------------ #
_CASE_BUILDERS = [
    # 1 – missing dose grid
    lambda: Case(
        None, {"roi": Structure("roi", [_square_contour()], (0, 0, 0), "PTV")}
    ),
    # 2 – missing structures
    lambda: Case(dg_unit, {}),
    # 3 – dose ↔ CT orientation mismatch
    lambda: Case(dg_unit, {}, ct_geometry=ct_mismatch),
    # 4 – volume discrepancy >10 %
    lambda: Case(
        dg_unit,
        {
            "big": Structure(
                "big",
                [_square_contour(), _square_contour(3)],
                (0, 0, 0),
                "PTV",
                volume=1e6,
            )
        },
    ),
    # 5 – ROI with zero contours (after mutation)
    _case_no_contours,
    # 6 – single-slice ROI
    lambda: Case(
        dg_unit,
        {"one": Structure("one", [_square_contour()], (0, 0, 0), "PTV")},
    ),
    # 7 – completely outside dose grid
    lambda: Case(
        dg_unit,
        {"far": Structure("far", [_square_contour(shift=1000)], (0, 0, 0), "PTV")},
    ),
    # 8 – partially outside dose grid
    lambda: Case(
        dg_unit,
        {"edge": Structure("edge", _two_partial_contours(), (0, 0, 0), "PTV")},
    ),
]


@pytest.mark.parametrize("case_builder", _CASE_BUILDERS)
def test_validate_case_failure_modes(case_builder):
    """Each builder should raise *some* ValueError when validated."""
    with pytest.raises(ValueError):
        validate_case(case_builder())
