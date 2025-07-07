"""
Signed-distance utilities and voxeliser
======================================

Fully supports the “generic” five-mode axial end-capping specification
(July 2025).

---------------------------------------------------------------------------
Overview
---------------------------------------------------------------------------
* **2-D core**   `signed_distance_2d()` – fast Numba kernel that returns the
  signed distance from a point to a *single* polygon, using a winding-number
  test that is robust to duplicate vertices & self-intersections.

* **3-D wrapper** `structure_signed_distance()` interpolates slice-by-slice
  SDFs and attaches one of five possible end-caps
  (`TRUNCATE | FIXED_PRISM | USER_PRISM | SHAPE_PLUS_PRISM | SHAPE_ONLY`).

* **Voxel mask** `Structure.mask()` converts the 3-D SDF into an occupancy
  grid with *adaptive supersampling* so that small structures do not vanish
  at coarse voxel sizes.

Public API
---------------------------------------------------------------------------
* `Structure.signed_distance(points, *, cap_mode="SHAPE_ONLY", …)`
* `Structure.mask(voxel_size=0.2, *, cap_mode="SHAPE_ONLY", …)`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numba as nb
import numpy as np
from numba.typed import List as NbList
from numpy.typing import NDArray

# Local import
from .data_types import Structure

# --------------------------------------------------------------------------- #
#  Numerical constants                                                        #
# --------------------------------------------------------------------------- #

# Numerical tolerances
EPSILON = 1e-9  # Avoid divide-by-zero & FP glitches
Z_TOLERANCE = 1e-4  # Two contours belong to same slice if |Δz| < 0.1 mm
DEGENERATE_SLICE_OFFSET = 0.5  # mm – gives single-slice ROIs a tangible thickness

# Distance / bounding-box helpers
INFINITY_PROXY = 1e9
MIN_VOXEL_SIZE = 0.01  # 10 µm
MAX_VOXEL_SIZE = 10.0  # 10 mm
MAX_COORDINATE = 1e6  # 1 km – far outside any DICOM frame

# --------------------------------------------------------------------------- #
#  Advanced-feature configuration                                             #
# --------------------------------------------------------------------------- #


@dataclass
class MaskConfig:
    """Configuration for the (experimental) `mask_improved()` façade."""

    use_batching: bool = True
    batch_size: int = 1000
    use_parallel: bool = False
    n_threads: Optional[int] = None
    adaptive_grid: bool = True
    adaptive_threshold: float = 0.2
    min_adaptive_grid: int = 3
    max_adaptive_grid: int = 7
    nearly_empty_threshold: float = 0.001
    nearly_full_threshold: float = 0.999


# --------------------------------------------------------------------------- #
#  Low-level 2-D SDF helpers                                                  #
# --------------------------------------------------------------------------- #


@nb.njit(inline="always", fastmath=True)
def _dist_pt_segment(px, py, x1, y1, x2, y2):
    """Euclidean distance from (px, py) to the finite segment [(x1, y1)-(x2, y2)]."""
    vx, vy = x2 - x1, y2 - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < EPSILON:
        return math.hypot(px - x1, py - y1)

    wx, wy = px - x1, py - y1
    t = (vx * wx + vy * wy) / seg_len2
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    dx, dy = wx - t * vx, wy - t * vy
    return math.hypot(dx, dy)


@nb.njit(inline="always", fastmath=True)
def _winding_number(
    px: float, py: float, xs: NDArray[np.float32], ys: NDArray[np.float32]
) -> int:
    """Even–odd winding number (0 → outside, non-zero → inside)."""
    wn = 0
    n = xs.size
    if n < 3:
        return 0

    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]

        # Skip degenerate edges
        if abs(y2 - y1) < EPSILON and abs(x2 - x1) < EPSILON:
            continue

        if y1 <= py < y2:
            if (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) > 0:
                wn += 1
        elif y2 <= py < y1:
            if (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) < 0:
                wn -= 1
    return wn


@nb.njit(parallel=True, fastmath=True, cache=True)
def signed_distance_2d(
    contour_pts: NDArray[np.float32], queries_xy: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Signed distance from a **simple polygon** to many query points in the *xy*
    plane. Negative → inside, positive → outside.
    """
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


# A Numba-friendly variant that accepts a *typed list* of polygons ------------
@nb.njit(cache=True, fastmath=True)
def _slice_signed_distance_nb(polys: NbList, x: float, y: float) -> float:
    """
    Distance to an ROI *slice* that may contain multiple islands & holes
    (DICOM even–odd fill rule).
    """
    min_abs, crossings = INFINITY_PROXY, 0
    q = np.array([[x, y]], np.float32)

    for P in polys:  # typed list → zero Python overhead
        if P.shape[0] < 3:
            continue
        d = signed_distance_2d(P, q)[0]
        if abs(d) < min_abs:
            min_abs = abs(d)
        if d < 0:
            crossings ^= 1

    return -min_abs if crossings else +min_abs


# --------------------------------------------------------------------------- #
#  3-D signed distance                                                        #
# --------------------------------------------------------------------------- #

_CAPS = {
    "TRUNCATE",
    "FIXED_PRISM",
    "USER_PRISM",
    "SHAPE_PLUS_PRISM",
    "SHAPE_ONLY",
}


def _validate_structure(struct: Structure):
    if not getattr(struct, "contours", None):
        raise ValueError("Structure must have at least one contour")

    for i, c in enumerate(struct.contours):
        if not hasattr(c, "points") or not hasattr(c, "slice_position"):
            raise ValueError(f"Contour {i} missing required attributes")
        if c.points.shape[0] < 3:
            raise ValueError(f"Contour {i} has fewer than 3 points")
        if not np.isfinite(c.slice_position):
            raise ValueError(f"Contour {i} has non-finite slice position")


def _prepare_slices(struct: Structure):
    """Sort contours by z and merge polygons that belong to the same slice."""
    _validate_structure(struct)

    z_unique: list[float] = []
    polys: list[list[NDArray[np.float32]]] = []

    for c in sorted(struct.contours, key=lambda c: c.slice_position):
        z = float(c.slice_position)

        if abs(z) > MAX_COORDINATE:
            raise ValueError(f"Slice position {z} exceeds maximum allowed value")

        if z_unique and abs(z - z_unique[-1]) < Z_TOLERANCE:
            polys[-1].append(c.points.astype(np.float32))
        else:
            z_unique.append(z)
            polys.append([c.points.astype(np.float32)])

    z_arr = np.array(z_unique, np.float32)

    # Single-slice ROI → duplicate with ±0.5 mm offset so interpolation works
    if z_arr.size == 1:
        z0 = z_arr[0]
        z_arr = np.array(
            [z0 - DEGENERATE_SLICE_OFFSET, z0 + DEGENERATE_SLICE_OFFSET], np.float32
        )
        polys = [polys[0], polys[0]]

    # Convert to Numba typed list of typed lists
    nb_polys = NbList()
    for slice_polys in polys:
        inner = NbList()
        for P in slice_polys:
            inner.append(P)
        nb_polys.append(inner)

    return z_arr, nb_polys


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
    ] = "SHAPE_ONLY",
    prism_fraction: float = 0.25,
    prism_cap_limit: float | None = None,
) -> NDArray[np.float32]:
    """
    Signed distance from a point cloud to a **closed** ROI.

    Cap-mode semantics (identical to ProKnow):

    * **TRUNCATE**    ROI ends flush with first / last slice.
    * **FIXED_PRISM**  adds a ½ Δz prism at each end.
    * **USER_PRISM**  adds *f·Δz* prism at each end (0 ≤ *f* ≤ 0.5).
    * **SHAPE_PLUS_PRISM** linear blend + ½ Δz prism (best of both worlds).
    * **SHAPE_ONLY**  linear blend toward a cone apex (no flat prism).
    """
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.empty(0, np.float32)
    if not np.all(np.isfinite(pts)):
        raise ValueError("Points contain non-finite values")
    if np.any(np.abs(pts) > MAX_COORDINATE):
        raise ValueError("Point coordinates exceed allowed range")

    z_slices, polys = _prepare_slices(struct)

    dz_inf = z_slices[1] - z_slices[0]
    dz_sup = z_slices[-1] - z_slices[-2]

    # Cap lengths -------------------------------------------------------------
    if cap_mode == "TRUNCATE":
        cap_inf = cap_sup = 0.0
    elif cap_mode == "FIXED_PRISM":
        cap_inf, cap_sup = 0.5 * dz_inf, 0.5 * dz_sup
    elif cap_mode == "USER_PRISM":
        f = max(0.0, min(0.5, prism_fraction))
        cap_inf, cap_sup = f * dz_inf, f * dz_sup
    else:  # SHAPE_PLUS_PRISM | SHAPE_ONLY
        cap_inf, cap_sup = 0.5 * dz_inf, 0.5 * dz_sup

    if prism_cap_limit is not None:
        if prism_cap_limit < 0:
            raise ValueError("prism_cap_limit must be non-negative")
        cap_inf = min(cap_inf, prism_cap_limit)
        cap_sup = min(cap_sup, prism_cap_limit)

    z_cap_min = z_slices[0] - cap_inf
    z_cap_max = z_slices[-1] + cap_sup

    # End-slice centroids (cone apex for SHAPE_ONLY)
    all_pts_inf = np.vstack(polys[0])
    all_pts_sup = np.vstack(polys[-1])
    cx_inf, cy_inf = np.mean(all_pts_inf, axis=0)[:2]
    cx_sup, cy_sup = np.mean(all_pts_sup, axis=0)[:2]

    # --------------------------------------------------------------------- #
    #  Main loop – still in Python, but the expensive                        #
    #  slice-distance calls are now Numba-accelerated.                       #
    # --------------------------------------------------------------------- #
    out = np.empty(len(pts), np.float32)

    for i, (x, y, z) in enumerate(pts):
        # ---------------------------- inferior side -----------------------
        if z < z_slices[0]:
            d_xy = _slice_signed_distance_nb(polys[0], x, y)
            dz = z_slices[0] - z

            if cap_mode == "TRUNCATE" or (
                cap_mode in {"FIXED_PRISM", "USER_PRISM"} and z <= z_cap_min
            ):
                axial = dz - cap_inf if cap_mode != "TRUNCATE" else dz
                out[i] = math.hypot(max(0.0, d_xy), axial)
                continue

            if cap_mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM"}:
                out[i] = d_xy
                continue

            # SHAPE_ONLY  – cone apex
            if dz >= cap_inf or cap_inf < EPSILON:
                out[i] = math.hypot(x - cx_inf, y - cy_inf, dz - cap_inf)
                continue

            t = min(1.0, dz / cap_inf)
            scale = max(EPSILON, 1.0 - t)
            sx = cx_inf + (x - cx_inf) / scale
            sy = cy_inf + (y - cy_inf) / scale
            d_cap = _slice_signed_distance_nb(polys[0], sx, sy) * scale
            out[i] = d_cap
            continue

        # ---------------------------- superior side -----------------------
        if z > z_slices[-1]:
            d_xy = _slice_signed_distance_nb(polys[-1], x, y)
            dz = z - z_slices[-1]

            if cap_mode == "TRUNCATE" or (
                cap_mode in {"FIXED_PRISM", "USER_PRISM"} and z >= z_cap_max
            ):
                axial = dz - cap_sup if cap_mode != "TRUNCATE" else dz
                out[i] = math.hypot(max(0.0, d_xy), axial)
                continue

            if cap_mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM"}:
                out[i] = d_xy
                continue

            if dz >= cap_sup or cap_sup < EPSILON:
                out[i] = math.hypot(x - cx_sup, y - cy_sup, dz - cap_sup)
                continue

            t = min(1.0, dz / cap_sup)
            scale = max(EPSILON, 1.0 - t)
            sx = cx_sup + (x - cx_sup) / scale
            sy = cy_sup + (y - cy_sup) / scale
            d_cap = _slice_signed_distance_nb(polys[-1], sx, sy) * scale
            out[i] = d_cap
            continue

        # ---------------------------- interior ----------------------------
        k = np.searchsorted(z_slices, z, side="right") - 1
        k = max(0, min(k, len(z_slices) - 2))

        if k == len(z_slices) - 1 or abs(z - z_slices[-1]) < EPSILON:
            out[i] = _slice_signed_distance_nb(polys[-1], x, y)
            continue

        z_range = z_slices[k + 1] - z_slices[k]
        if z_range < EPSILON:
            out[i] = _slice_signed_distance_nb(polys[k], x, y)
            continue

        dz_local = (z - z_slices[k]) / z_range
        d0 = _slice_signed_distance_nb(polys[k], x, y)
        d1 = _slice_signed_distance_nb(polys[k + 1], x, y)
        out[i] = d0 * (1.0 - dz_local) + d1 * dz_local

    return out


# --------------------------------------------------------------------------- #
#  Voxel-mask wrapper                                                         #
# --------------------------------------------------------------------------- #


def _mask_wrapper(
    self: Structure,
    voxel_size: float | Sequence[float] = 0.2,
    *,
    cap_mode: str = "SHAPE_ONLY",
    tol: float | None = 2e-3,  # target |Δfill| per voxel (0.2 %)
    max_levels: int = 3,  # allow one extra refinement pass
    min_grid: int = 3,
    **sd_kwargs,
):
    """
    Rasterise the ROI into a regular grid **with adaptive supersampling**.

    Returns
    -------
    mask   : np.ndarray[float32]
        Fractional occupancy (0 – 1) for every voxel.
    origin : (x0, y0, z0)
        World co-ordinates of voxel [0, 0, 0] centre.
    """
    # ------------------------------------------------------------------ #
    # 1  Sanity-check & build the voxel grid                             #
    # ------------------------------------------------------------------ #
    if isinstance(voxel_size, (int, float)):
        voxel_size = (float(voxel_size),) * 3
    dx, dy, dz = map(float, voxel_size)

    for v, name in zip((dx, dy, dz), "xyz"):
        if v <= 0:
            raise ValueError(f"{name} voxel dimension must be positive")
        if not (MIN_VOXEL_SIZE <= v <= MAX_VOXEL_SIZE):
            raise ValueError(
                f"{name} voxel dimension {v} outside valid range "
                f"[{MIN_VOXEL_SIZE}, {MAX_VOXEL_SIZE}] mm"
            )

    bb_min, bb_max = self.compute_bounding_box()

    # Pad Z for cap-modes that protrude beyond the first / last slice
    pad_z = 0.0
    if cap_mode in {"FIXED_PRISM", "USER_PRISM", "SHAPE_PLUS_PRISM", "SHAPE_ONLY"}:
        zs = sorted({float(c.slice_position) for c in self.contours})
        pad_z = 0.5 * (zs[1] - zs[0]) if len(zs) > 1 else dz

    bb_min -= (dx, dy, pad_z)
    bb_max += (dx, dy, pad_z)

    xs = np.arange(bb_min[0] + 0.5 * dx, bb_max[0] - 0.5 * dx + 1e-9, dx, np.float32)
    ys = np.arange(bb_min[1] + 0.5 * dy, bb_max[1] - 0.5 * dy + 1e-9, dy, np.float32)
    zs = np.arange(bb_min[2] + 0.5 * dz, bb_max[2] - 0.5 * dz + 1e-9, dz, np.float32)

    # ---------------------- NEW: grid shift so every slice plane aligns -----
    for z0 in sorted({float(c.slice_position) for c in self.contours}):
        off = (z0 - zs[0]) % dz
        if 1e-6 < off < dz - 1e-6:
            zs += dz - off  # rigid shift – keeps spacing identical
            break
    # -----------------------------------------------------------------------

    gX, gY, gZ = np.meshgrid(xs, ys, zs, indexing="xy")
    centres = np.column_stack([gX.ravel(), gY.ravel(), gZ.ravel()])

    # ------------------------------------------------------------------ #
    # 2  Initial SDF at voxel centres                                    #
    # ------------------------------------------------------------------ #
    sdf = structure_signed_distance(self, centres, cap_mode=cap_mode, **sd_kwargs)

    half_diag = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
    deep_in = sdf < -half_diag
    deep_out = sdf > half_diag
    boundary = ~(deep_in | deep_out)

    mask = np.zeros_like(sdf, dtype=np.float32)
    mask[deep_in] = 1.0
    mask[deep_out] = 0.0
    if not boundary.any():  # trivial ROI (fits in one voxel)
        return mask.reshape(gX.shape), (xs[0], ys[0], zs[0])

    # ------------------------------------------------------------------ #
    # 3  Adaptive supersampling                                          #
    # ------------------------------------------------------------------ #
    ctrs = centres[boundary]
    idx = np.flatnonzero(boundary)
    fill_prev = np.full(ctrs.shape[0], 0.5, np.float32)  # dummy

    if cap_mode == "SHAPE_PLUS_PRISM":
        min_grid = max(min_grid, 5)  # a tad denser for this mode

    grid = max(3, min_grid | 1)  # odd integer ≥ 3
    level = 0

    while level < max_levels:
        # Build regular (grid³) offsets in voxel space
        g = np.linspace(-0.5, 0.5, grid, dtype=np.float32)
        offsets = (
            np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
            .reshape(-1, 3, order="C")
            .astype(np.float32)
        )

        pts = (ctrs[:, None, :] + offsets[None, :, :] * (dx, dy, dz)).reshape(-1, 3)
        sdf_sub = structure_signed_distance(self, pts, cap_mode=cap_mode, **sd_kwargs)
        fill = (sdf_sub < 0.0).reshape(-1, offsets.shape[0]).mean(1).astype(np.float32)

        mask[idx] = fill

        # Convergence check -------------------------------------------------
        if tol is not None:
            delta = np.abs(fill - fill_prev)
            unfinished = delta > tol
        else:
            unfinished = np.ones_like(fill, dtype=bool)

        if not unfinished.any():
            break  # ✓ all boundary voxels converged

        # Prepare next round -----------------------------------------------
        level += 1
        if level >= max_levels:
            break

        ctrs = ctrs[unfinished]
        idx = idx[unfinished]
        fill_prev = fill[unfinished]
        grid += 2  # 3 → 5 → 7 → …

    return mask.reshape(gX.shape), (xs[0], ys[0], zs[0])


# --------------------------------------------------------------------------- #
#  Experimental façade (kept for compatibility)                               #
# --------------------------------------------------------------------------- #


def _mask_wrapper_improved(
    self: Structure,
    voxel_size: float | Sequence[float] = 0.2,
    *,
    cap_mode: str = "SHAPE_ONLY",
    tol: float | None = 2e-3,
    max_levels: int = 3,
    min_grid: int = 3,
    config: Optional[MaskConfig] = None,  # noqa: D401 – simple pass-through
    **sd_kwargs,
):
    """Improved mask wrapper (currently forwards to `_mask_wrapper`)."""
    return _mask_wrapper(
        self,
        voxel_size,
        cap_mode=cap_mode,
        tol=tol,
        max_levels=max_levels,
        min_grid=min_grid,
        **sd_kwargs,
    )


# --------------------------------------------------------------------------- #
#  Monkey-patch onto `Structure`                                              #
# --------------------------------------------------------------------------- #


def _sd_wrapper(self: Structure, pts, *, cap_mode="SHAPE_ONLY", **kw):
    """`roi.signed_distance(…)` façade."""
    return structure_signed_distance(self, pts, cap_mode=cap_mode, **kw)


Structure.signed_distance = _sd_wrapper  # type: ignore[attr-defined]
Structure.mask = _mask_wrapper  # type: ignore[attr-defined]
Structure.mask_improved = _mask_wrapper_improved  # type: ignore[attr-defined]

__all__ = ["signed_distance_2d", "structure_signed_distance", "MaskConfig"]
