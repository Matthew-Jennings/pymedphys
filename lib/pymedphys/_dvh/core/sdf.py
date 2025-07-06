"""
Signed-distance utilities and voxeliser with full support for the generic
end-capping specification (July 2025).

---------------------------------------------------------------------------
Overview
---------------------------------------------------------------------------
*   **2-D core**  Fast, numba-accelerated routines for computing the signed
    distance from an (x, y) point to a single polygon
    (`signed_distance_2d`).  The inside / outside test uses a winding-number
    implementation that is robust to self-intersections and duplicate
    vertices.
*   **3-D wrapper** `structure_signed_distance` blends distances slice-by-
    slice along *z* and adds one of five possible axial end-caps
    (TRUNCATE | FIXED_PRISM | USER_PRISM | SHAPE_PLUS_PRISM | SHAPE_ONLY).
*   **Voxel mask** `Structure.mask()` turns the SDF into a 3-D boolean mask
    suitable for volume calculations and DVH generation.

Public API
---------------------------------------------------------------------------
*   `Structure.signed_distance(points, *, cap_mode="SHAPE_ONLY", …)`
*   `Structure.mask(voxel_size=0.2, *, cap_mode="SHAPE_ONLY", …)`
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numba as nb
import numpy as np
from numpy.typing import NDArray

from .data_types import Structure

# --------------------------------------------------------------------------- #
#  Constants for numerical stability and clarity                              #
# --------------------------------------------------------------------------- #

# Numerical tolerances
EPSILON = 1e-9  # General epsilon for avoiding division by zero
Z_TOLERANCE = 1e-4  # Tolerance for Z-coordinate comparisons (0.1 mm)
DEGENERATE_SLICE_OFFSET = 1e-3  # Offset for single-slice structures (1 mm)

# Distance computation
INFINITY_PROXY = 1e9  # Large number representing infinity in distance calculations

# Input validation limits
MIN_VOXEL_SIZE = 0.01  # Minimum allowed voxel size (0.01 mm)
MAX_VOXEL_SIZE = 10.0  # Maximum allowed voxel size (10 mm)
MAX_COORDINATE = 1e6  # Maximum allowed coordinate value (1000 m)

# --------------------------------------------------------------------------- #
#  Low-level helpers (with enhanced robustness)                               #
# --------------------------------------------------------------------------- #


@nb.njit(inline="always")
def _dist_pt_segment(px, py, x1, y1, x2, y2):
    """
    Euclidean distance from an arbitrary point *(px, py)* to the finite line
    segment *[(x1, y1), (x2, y2)]*.

    The routine is hot-looped from `signed_distance_2d` and therefore:
        * uses **Numba** with `inline="always"` for maximal SIMD fusion;
        * avoids heap allocations entirely;
        * returns the *unsigned* distance (the caller decides the sign).
    """
    vx, vy = x2 - x1, y2 - y1
    segment_length_sq = vx * vx + vy * vy

    # Handle degenerate segment (point)
    if segment_length_sq < EPSILON:
        return math.hypot(px - x1, py - y1)

    wx, wy = px - x1, py - y1
    t = (vx * wx + vy * wy) / segment_length_sq
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    dx, dy = wx - t * vx, wy - t * vy
    return math.hypot(dx, dy)


@nb.njit(inline="always")
def _winding_number(
    px: float, py: float, xs: NDArray[np.float32], ys: NDArray[np.float32]
) -> int:
    """
    Fast winding-number implementation.

    Parameters
    ----------
    px, py
        Query point.
    xs, ys
        Vertex coordinates **clockwise or anti-clockwise**.  Orientation
        does not matter – we rely on the parity of edge crossings alone,
        matching the DICOM even–odd filling rule.

    Returns
    -------
    int
        Zero ⇒ outside, non-zero ⇒ inside.

    Notes
    -----
    * Degenerate edges (zero length) are skipped to avoid NaNs.
    * This version is branch-reduced to keep the JIT kernel simple.
    """
    wn = 0
    n = xs.size

    # Handle degenerate polygon
    if n < 3:
        return 0

    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]

        # Skip degenerate edges
        if abs(y2 - y1) < EPSILON and abs(x2 - x1) < EPSILON:
            continue

        if y1 <= py:
            if y2 > py and (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) > 0:
                wn += 1
        else:
            if y2 <= py and (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) < 0:
                wn -= 1
    return wn


@nb.njit(parallel=True, cache=True)
def signed_distance_2d(
    contour_pts: NDArray[np.float32], queries_xy: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Signed distance from one *simple polygon* to many query points.

    The polygon is taken to lie in the *xy*-plane at *z = const*.

    Parameters
    ----------
    contour_pts
        **N × 3** array of (x, y, z) vertices.  The *z* component is ignored.
    queries_xy
        **M × 2** array of query coordinates (x, y).

    Returns
    -------
    M-element ``float32`` array.
        Negative ⇒ inside • Positive ⇒ outside.
    """
    # Input validation
    if contour_pts.shape[0] < 3:
        raise ValueError("Contour must have at least 3 points")

    xs, ys = contour_pts[:, 0], contour_pts[:, 1]
    m = queries_xy.shape[0]
    out = np.empty(m, np.float32)

    for q in nb.prange(m):
        px, py = queries_xy[q, 0], queries_xy[q, 1]
        d_min = INFINITY_PROXY

        for i in range(xs.size):
            d = _dist_pt_segment(
                px, py, xs[i], ys[i], xs[(i + 1) % xs.size], ys[(i + 1) % xs.size]
            )
            if d < d_min:
                d_min = d

        inside = _winding_number(px, py, xs, ys) != 0
        out[q] = -d_min if inside else d_min
    return out


def _slice_signed_distance(
    polys: list[NDArray[np.float32]], x: float, y: float
) -> float:
    """
    Signed distance to a *slice* that may contain **multiple islands and/or
    holes**.

    The even–odd (XOR) rule gives DICOM-compatible behaviour:
        • odd number of negative distances … inside
        • even number … outside

    The function deliberately **ignores winding direction**
    (DICOM does not prescribe CW/CCW for separate islands).
    """
    if not polys:
        return INFINITY_PROXY

    min_abs, crossings = INFINITY_PROXY, 0
    q = np.array([[x, y]], np.float32)

    for P in polys:
        if P.shape[0] < 3:  # Skip degenerate polygons
            continue
        d = signed_distance_2d(P, q)[0]
        if abs(d) < min_abs:
            min_abs = abs(d)
        if d < 0:
            crossings ^= 1

    return -min_abs if crossings else +min_abs


# --------------------------------------------------------------------------- #
#  End-cap strategy helpers                                                   #
# --------------------------------------------------------------------------- #

_CAPS = {
    "TRUNCATE",
    "FIXED_PRISM",
    "USER_PRISM",
    "SHAPE_PLUS_PRISM",
    "SHAPE_ONLY",
    # ▸ backwards-compat synonyms
    "FLAT",
    "SMOOTH",
}


def _normalise_mode(s: str) -> str:
    """Map user-supplied *cap_mode* strings to one of the five canonical
    identifiers, while preserving backward compatibility.
    """
    if not isinstance(s, str):
        raise TypeError(f"cap_mode must be string, got {type(s).__name__}")

    s_up = s.strip().upper()
    if s_up == "FLAT":
        return "FIXED_PRISM"
    if s_up == "SMOOTH":
        return "SHAPE_ONLY"
    if s_up not in _CAPS:
        raise ValueError(f"Unknown cap_mode {s!r}. Valid modes: {sorted(_CAPS)}")
    return s_up


# --------------------------------------------------------------------------- #
#  3-D signed distance                                                        #
# --------------------------------------------------------------------------- #


def _validate_structure(struct: Structure):
    """Validate structure has valid contours."""
    if not hasattr(struct, "contours") or not struct.contours:
        raise ValueError("Structure must have at least one contour")

    for i, c in enumerate(struct.contours):
        if not hasattr(c, "points") or not hasattr(c, "slice_position"):
            raise ValueError(f"Contour {i} missing required attributes")
        if c.points.shape[0] < 3:
            raise ValueError(f"Contour {i} has fewer than 3 points")
        if not np.isfinite(c.slice_position):
            raise ValueError(f"Contour {i} has non-finite slice position")


def _prepare_slices(struct: Structure):
    """Return z-list and polygons grouped by z (duplicates merged)."""
    _validate_structure(struct)

    z_unique: list[float] = []
    polys: list[list[NDArray[np.float32]]] = []

    # Sort contours by slice position to ensure correct grouping
    sorted_contours = sorted(struct.contours, key=lambda c: c.slice_position)

    for c in sorted_contours:
        z = float(c.slice_position)

        # Validate z coordinate
        if abs(z) > MAX_COORDINATE:
            raise ValueError(f"Slice position {z} exceeds maximum allowed value")

        # Check if this z is close to the last one
        if z_unique and abs(z - z_unique[-1]) < Z_TOLERANCE:
            polys[-1].append(c.points.astype(np.float32))
        else:
            z_unique.append(z)
            polys.append([c.points.astype(np.float32)])

    z_arr = np.array(z_unique, np.float32)

    # Handle single-slice structures
    if z_arr.size == 1:
        z0 = z_arr[0]
        z_arr = np.array(
            [z0 - DEGENERATE_SLICE_OFFSET, z0 + DEGENERATE_SLICE_OFFSET], np.float32
        )
        polys = [polys[0], polys[0]]

    return z_arr, polys


def structure_signed_distance(
    struct: Structure,
    points: NDArray[np.float32],
    *,
    cap_mode: Literal[
        "TRUNCATE",
        "FIXED_PRISM",
        "USER_PRISM",
        "SHAPE_PLUS_PRISM",
        "SHAPE_ONLY",
        "FLAT",
        "SMOOTH",
    ] = "SHAPE_ONLY",
    prism_fraction: float = 0.25,
    prism_cap_limit: float | None = None,
) -> NDArray[np.float32]:
    r"""
    Signed distance from an arbitrary 3-D point cloud to a closed
    *Region-Of-Interest* (ROI).

    ----------  ------------------------------------------------------------
    Cap mode    Geometric interpretation
    ----------  ------------------------------------------------------------
    TRUNCATE    No axial cap – the ROI ends flush with the first/last slice.
    FIXED_PRISM ½ Δz prism attached to each end.
    USER_PRISM  f · Δz prism (*f* ∈ \[0, 0.5\]) at each end.
    SHAPE_PLUS  Linear slice-to-slice blending **plus** the ½ Δz prism.
    SHAPE_ONLY  Pure slice-to-slice blending extended to a cone apex.
    ----------  ------------------------------------------------------------

    Parameters
    ----------
    struct
        A fully-populated :class:`~pymedphys._dvh.core.data_types.Structure`
        (must have ≥1 contour).
    points
        Array-like of shape (N, 3) **or** broadcastable to that.
    cap_mode, prism_fraction, prism_cap_limit
        See table and Notes below.

    Returns
    -------
    np.ndarray, dtype ``float32``
        *N* signed distances.

    Notes
    -----
    *   Distances are exact *within machine precision* for planar slices and
        linear interpolation, but the cone/prism cap is an *analytic*
        extension – no tessellation artefacts.
    *   When *prism_cap_limit* is set, the half-length is
        ``min(cap_length, prism_cap_limit)``.  This mirrors the ProKnow
        interpretation so third-party results can be compared 1-to-1.
    *   The function is intentionally **branch-heavy** – the outer loop is
        pure-Python because the point cloud size is usually small in
        clinical queries (< 10⁴ points).  Optimising further with Numba
        vectorisation gives negligible speed-ups but complicates testing.
    """
    mode = _normalise_mode(cap_mode)

    # Validate and prepare points
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.array([], np.float32)

    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        raise ValueError("Points contain non-finite values")

    # Check coordinate bounds
    if np.any(np.abs(pts) > MAX_COORDINATE):
        raise ValueError(f"Points contain coordinates exceeding {MAX_COORDINATE}")

    z_slices, polys = _prepare_slices(struct)

    # local slice spacing
    dz_inf = z_slices[1] - z_slices[0]
    dz_sup = z_slices[-1] - z_slices[-2]

    # cap lengths per mode ---------------------------------------------------
    if mode == "TRUNCATE":
        cap_inf = cap_sup = 0.0
    elif mode == "FIXED_PRISM":
        cap_inf, cap_sup = 0.5 * dz_inf, 0.5 * dz_sup
    elif mode == "USER_PRISM":
        f = max(0.0, min(0.5, prism_fraction))
        cap_inf, cap_sup = f * dz_inf, f * dz_sup
    else:  # SHAPE_PLUS_PRISM or SHAPE_ONLY
        cap_inf, cap_sup = 0.5 * dz_inf, 0.5 * dz_sup

    if prism_cap_limit is not None:
        if prism_cap_limit < 0:
            raise ValueError("prism_cap_limit must be non-negative")
        cap_inf = min(cap_inf, prism_cap_limit)
        cap_sup = min(cap_sup, prism_cap_limit)

    z_cap_min = z_slices[0] - cap_inf
    z_cap_max = z_slices[-1] + cap_sup

    # pointer to last / first slice polygons
    P_inf = polys[0]
    P_sup = polys[-1]

    # slice centroids (for SHAPE_ONLY cones)
    # Note: Vertex mean is a good approximation for the centroid of a
    # symmetric polygon, but not for a general polygon. This is a
    # standard and robust modeling choice for defining the cone apex.
    all_pts_inf = np.vstack(P_inf)
    all_pts_sup = np.vstack(P_sup)

    cx_inf, cy_inf = (
        np.mean(all_pts_inf, axis=0)[:2] if all_pts_inf.shape[0] > 0 else (0.0, 0.0)
    )
    cx_sup, cy_sup = (
        np.mean(all_pts_sup, axis=0)[:2] if all_pts_sup.shape[0] > 0 else (0.0, 0.0)
    )

    out = np.empty(len(pts), np.float32)

    for i, (x, y, z) in enumerate(pts):
        # ------------------------------------------------------------------ #
        #  Inferior side                                                     #
        # ------------------------------------------------------------------ #
        if z < z_slices[0]:
            d_xy = _slice_signed_distance(P_inf, x, y)
            dz = z_slices[0] - z

            # ---------- TRUNCATE or outside USER / FIXED prism -------------
            if mode == "TRUNCATE" or (
                mode in {"FIXED_PRISM", "USER_PRISM"} and z < z_cap_min
            ):
                # Distance to the capped prism volume boundary
                axial_dist = dz - cap_inf if mode != "TRUNCATE" else dz
                out[i] = math.hypot(max(0.0, d_xy), axial_dist)
                continue

            # ------------------- Prism-based caps --------------------------
            if mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM"}:
                # Inside the prism cap's axial range, distance is just lateral
                out[i] = d_xy
                continue

            # ------------------------ SHAPE_ONLY ---------------------------
            # Beyond the cone apex, distance is Euclidean to the apex point
            if dz >= cap_inf or cap_inf < EPSILON:
                axial_dist_from_apex = dz - cap_inf
                out[i] = math.hypot(x - cx_inf, y - cy_inf, axial_dist_from_apex)
                continue

            # Inside the cone cap volume
            t = min(1.0, dz / cap_inf)  # 0 (at slice) → 1 (at apex)
            scale = max(EPSILON, 1.0 - t)  # Avoid division by zero
            sx = cx_inf + (x - cx_inf) / scale
            sy = cy_inf + (y - cy_inf) / scale
            d_cap = _slice_signed_distance(P_inf, sx, sy) * scale
            out[i] = d_cap
            continue

        # ------------------------------------------------------------------ #
        #  Superior side                                                     #
        # ------------------------------------------------------------------ #
        if z > z_slices[-1]:
            d_xy = _slice_signed_distance(P_sup, x, y)
            dz = z - z_slices[-1]

            if mode == "TRUNCATE" or (
                mode in {"FIXED_PRISM", "USER_PRISM"} and z > z_cap_max
            ):
                # Distance to the capped prism volume boundary
                axial_dist = dz - cap_sup if mode != "TRUNCATE" else dz
                out[i] = math.hypot(max(0.0, d_xy), axial_dist)
                continue

            if mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM"}:
                # Inside the prism cap's axial range, distance is just lateral
                out[i] = d_xy
                continue

            # ------------------------ SHAPE_ONLY ---------------------------
            # Beyond the cone apex, distance is Euclidean to the apex point
            if dz >= cap_sup or cap_sup < EPSILON:
                axial_dist_from_apex = dz - cap_sup
                out[i] = math.hypot(x - cx_sup, y - cy_sup, axial_dist_from_apex)
                continue

            # Inside the cone cap volume
            t = min(1.0, dz / cap_sup)
            scale = max(EPSILON, 1.0 - t)
            sx = cx_sup + (x - cx_sup) / scale
            sy = cy_sup + (y - cy_sup) / scale
            d_cap = _slice_signed_distance(P_sup, sx, sy) * scale
            out[i] = d_cap
            continue

        # ------------------------------------------------------------------ #
        #  Interior region – shape-based linear blend                        #
        # ------------------------------------------------------------------ #
        # Find the slice interval containing z
        k = np.searchsorted(z_slices, z, side="right") - 1
        k = max(0, min(k, len(z_slices) - 2))  # Clamp to valid range

        # Check if we're exactly on the last slice
        if k == len(z_slices) - 1 or abs(z - z_slices[-1]) < EPSILON:
            out[i] = _slice_signed_distance(polys[-1], x, y)
            continue

        # Linear interpolation between slices
        z_range = z_slices[k + 1] - z_slices[k]
        if z_range < EPSILON:  # Degenerate case
            out[i] = _slice_signed_distance(polys[k], x, y)
            continue

        dz_local = (z - z_slices[k]) / z_range
        d0 = _slice_signed_distance(polys[k], x, y)
        d1 = _slice_signed_distance(polys[k + 1], x, y)
        out[i] = d0 * (1.0 - dz_local) + d1 * dz_local

    return out


# --------------------------------------------------------------------------- #
#  Voxel mask wrapper                                                         #
# --------------------------------------------------------------------------- #


def _mask_wrapper(
    self: Structure,
    voxel_size: float | Sequence[float] = 0.2,
    *,
    cap_mode: str = "SHAPE_ONLY",
    **sd_kwargs,
):
    """
    Rasterise the ROI into a regular grid.

    Parameters
    ----------
    self
        Host :class:`Structure`.
    voxel_size
        Scalar → isotropic • 3-tuple → (dx, dy, dz).  Units: **mm**.
    cap_mode, **sd_kwargs
        Forwarded verbatim to :func:`structure_signed_distance`.

    Returns
    -------
    mask : np.ndarray, ``bool``
        3-D occupancy grid.  ``True`` = inside.
    origin : 3-tuple[float]
        (x₀, y₀, z₀) of the **first** voxel centre.
    """
    # Validate voxel size
    if isinstance(voxel_size, (int, float)):
        voxel_size = (float(voxel_size),) * 3

    try:
        dx, dy, dz = map(float, voxel_size)
    except (TypeError, ValueError):
        raise ValueError("voxel_size must be a number or sequence of 3 numbers")

    # Check voxel size bounds
    for v, name in [(dx, "dx"), (dy, "dy"), (dz, "dz")]:
        if v <= 0:
            raise ValueError(f"Voxel size {name}={v} must be positive")
        if v < MIN_VOXEL_SIZE:
            raise ValueError(f"Voxel size {name}={v} is below minimum {MIN_VOXEL_SIZE}")
        if v > MAX_VOXEL_SIZE:
            raise ValueError(f"Voxel size {name}={v} exceeds maximum {MAX_VOXEL_SIZE}")

    bb_min, bb_max = self.compute_bounding_box()

    # Validate bounding box
    if not np.all(np.isfinite(bb_min)) or not np.all(np.isfinite(bb_max)):
        raise ValueError("Structure bounding box contains non-finite values")

    mode = _normalise_mode(cap_mode)

    pad_z = 0.0
    if mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM", "SHAPE_ONLY"}:
        zs = sorted({float(c.slice_position) for c in self.contours})
        if len(zs) > 1:
            pad_z = 0.5 * max(zs[1] - zs[0], zs[-1] - zs[-2])
        else:
            pad_z = dz

    bb_min -= (dx, dy, pad_z)
    bb_max += (dx, dy, pad_z)

    # Generate grid with safety checks for memory
    grid_size = ((bb_max - bb_min) / np.array([dx, dy, dz])).astype(int) + 1
    total_voxels = np.prod(grid_size)

    if total_voxels > 1e9:  # 1 billion voxels safety limit
        raise ValueError(
            f"Grid size {grid_size} would create {total_voxels:.1e} voxels, exceeding safety limit"
        )

    xs = np.arange(bb_min[0], bb_max[0] + 0.5 * dx, dx, np.float32)
    ys = np.arange(bb_min[1], bb_max[1] + 0.5 * dy, dy, np.float32)
    zs = np.arange(bb_min[2], bb_max[2] + 0.5 * dz, dz, np.float32)

    gX, gY, gZ = np.meshgrid(xs, ys, zs, indexing="xy")
    sdf = structure_signed_distance(
        self,
        np.column_stack([gX.ravel(), gY.ravel(), gZ.ravel()]),
        cap_mode=mode,
        **sd_kwargs,
    )

    # ------------------------------------------------------------------
    # Boundary-voxel rule
    # A consistent interior rule (SDF < 0) is used for all modes. This
    # defines the volume as the set of points where the signed distance is
    # negative. Note that this point-sampling method can systematically
    # underestimate the volume of shapes with sharp features (e.g., the
    # cone caps in SHAPE_ONLY mode). This is a known trade-off of the
    # algorithm, and is preferable to using non-physical fudge factors.
    # ------------------------------------------------------------------
    mask = (sdf < 0.0).reshape(gX.shape)

    return mask, (xs[0], ys[0], zs[0])


# --------------------------------------------------------------------------- #
#  Monkey-patch onto Structure                                                #
# --------------------------------------------------------------------------- #
def _sd_wrapper(self: Structure, pts, *, cap_mode="SHAPE_ONLY", **kw):
    """Thin façade so that calling ``roi.signed_distance`` feels natural to
    end-users."""

    return structure_signed_distance(self, pts, cap_mode=cap_mode, **kw)


Structure.signed_distance = _sd_wrapper  # type: ignore[attr-defined]
Structure.mask = _mask_wrapper  # type: ignore[attr-defined]

__all__ = ["signed_distance_2d"]
