"""pymedphys._dvh.core.data_types
================================

Canonical data structures used by the dose-volume-histogram (DVH) reference
implementation.

The objects in this module are intentionally **thin, immutable value types**
(`@dataclass(slots=True, frozen=True)`) and contain *no* heavy-weight
algorithms beyond basic geometry helpers.  They are designed to be created
once and passed around freely, without hidden state mutations, so that
testing and reasoning about downstream code becomes easier.

Coordinate conventions
----------------------
* All physical positions are **patient-based DICOM coordinates** (x, y, z) in
  millimetres.
* Voxel indices are zero-based integer triples (i, j, k) that map to
  *(column, row, frame)* in DICOM terminology.
* ``orientation`` is a 3 × 3 **direction-cosine matrix** whose **columns** are
  the x-, y- and z-unit vectors of the grid axes (the standard DICOM
  definition).  The matrix must therefore be orthonormal.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

DOSE_UNITS_SUPPORTED = ("Gy", "Relative")


# -----------------------------------------------------------------------------#
# DoseGrid                                                                     #
# -----------------------------------------------------------------------------#
@dataclass(slots=True, frozen=True)
class DoseGrid:
    """
    A 3-D dose lattice plus full patient-space geometry.

    Parameters
    ----------
    values
        Dose values in **Gy** or relative units.  Shape ``(nx, ny, nz)``.
    origin
        World-space coordinates *(x, y, z)* of voxel ``(0, 0, 0)``, **mm**.
    spacing
        Physical voxel dimensions ``(dx, dy, dz)`` in **mm**.
    orientation
        3 × 3 direction-cosine matrix; columns are *x̂, ŷ, ẑ*.
    units
        Either ``"Gy"`` or ``"Relative"`` (case-insensitive).

    Notes
    -----
    The dataclass is frozen; internal type coercion inside ``__post_init__``
    therefore uses ``object.__setattr__`` instead of direct assignment.
    """

    values: NDArray[np.float32]
    origin: NDArray[np.float32]
    spacing: NDArray[np.float32]
    orientation: NDArray[np.float32]
    units: str

    # ---------------------------------------------------------------------#
    # Validation                                                            #
    # ---------------------------------------------------------------------#
    def __post_init__(self) -> None:
        """Run shape, orthogonality and unit checks after initialisation."""
        if self.values.ndim != 3:
            raise ValueError(f"Dose values must be 3-D, got {self.values.ndim}-D")

        if self.origin.shape != (3,):
            raise ValueError(f"origin must be (3,), got {self.origin.shape}")

        if self.spacing.shape != (3,):
            raise ValueError(f"spacing must be (3,), got {self.spacing.shape}")

        if self.orientation.shape != (3, 3):
            raise ValueError("orientation must be 3 × 3")

        if not np.allclose(self.orientation @ self.orientation.T, np.eye(3), atol=1e-6):
            raise ValueError("orientation matrix must be orthonormal")

        # Normalise units and data types (must use object.__setattr__ because frozen=True)
        object.__setattr__(self, "units", self.units.title())
        if self.units not in DOSE_UNITS_SUPPORTED:
            raise ValueError(
                f"units must be one of {', '.join(DOSE_UNITS_SUPPORTED)}; got {self.units!r}"
            )

        object.__setattr__(self, "values", self.values.astype(np.float32, copy=False))
        object.__setattr__(self, "origin", self.origin.astype(np.float32, copy=False))
        object.__setattr__(self, "spacing", self.spacing.astype(np.float32, copy=False))
        object.__setattr__(
            self, "orientation", self.orientation.astype(np.float32, copy=False)
        )

    # ---------------------------------------------------------------------#
    # Coordinate transforms                                                 #
    # ---------------------------------------------------------------------#
    def grid_to_patient(self, indices: Sequence) -> NDArray[np.float32]:
        """
        Convert integer or floating-point voxel indices to patient coordinates.

        Parameters
        ----------
        indices
            ``(..., 3)`` array of ``(i, j, k)`` indices.

        Returns
        -------
        numpy.ndarray
            ``(..., 3)`` array of *(x, y, z)* coordinates, **mm**.
        """
        scaled = np.asarray(indices, np.float32) * self.spacing
        return (scaled @ self.orientation.T) + self.origin

    def patient_to_grid(self, coords: NDArray) -> NDArray[np.float32]:
        """
        Convert patient coordinates back to fractional voxel indices.

        Parameters
        ----------
        coords
            ``(..., 3)`` array of *(x, y, z)* points in **mm**.

        Returns
        -------
        numpy.ndarray
            ``(..., 3)`` array of ``(i, j, k)`` indices (float).
        """
        translated = np.asarray(coords, np.float32) - self.origin
        return (translated @ self.orientation) / self.spacing


# -----------------------------------------------------------------------------#
# Contour                                                                      #
# -----------------------------------------------------------------------------#
@dataclass(slots=True, frozen=True)
class Contour:
    """
    One closed planar contour lying on a single *z*-slice.

    Parameters
    ----------
    points
        ``(n, 3)`` array of *(x, y, z)* vertices, **mm**.
    slice_position
        Z-coordinate of the plane (redundant but convenient).

    Raises
    ------
    ValueError
        If points are not 2-D, not Nx3, or have inconsistent *z*.
    """

    points: NDArray[np.float32]
    slice_position: float

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (n,3); got {self.points.shape}")

        object.__setattr__(self, "points", self.points.astype(np.float32, copy=False))

        if not np.allclose(self.points[:, 2], self.slice_position, atol=1e-3):
            raise ValueError("all contour points must share the same z value")

    # ---------------------------------------------------------------------#
    # Geometry helpers                                                     #
    # ---------------------------------------------------------------------#
    def compute_area(self) -> float:
        """
        Return the 2-D polygonal area using the shoelace formula.

        Returns
        -------
        float
            Area in **mm²**.  Degenerates to 0 for < 3 vertices.

        References
        ----------
        * Wikipedia contributors, “*Shoelace formula*,” *Wikipedia,
          The Free Encyclopedia*,
          https://en.wikipedia.org/wiki/Shoelace_formula
        """
        if len(self.points) < 3:
            return 0.0

        x, y = self.points[:, 0], self.points[:, 1]
        return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def compute_centroid(self) -> NDArray[np.float32]:
        """
        Centroid of the polygonal area.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` vector *(x̄, ȳ, z̄)* in **mm**.  If the contour is empty,
            x̄ = ȳ = 0.
        """
        if len(self.points) == 0:
            return np.array([0.0, 0.0, self.slice_position], dtype=np.float32)

        centroid_xy = self.points[:, :2].mean(axis=0)
        return np.array([*centroid_xy, self.slice_position], dtype=np.float32)


# -----------------------------------------------------------------------------#
# Structure                                                                    #
# -----------------------------------------------------------------------------#
@dataclass
class Structure:
    """
    A labelled ROI consisting of an ordered stack of **Contour** objects.

    Parameters
    ----------
    name
        ROI name as stored in DICOM.
    contours
        List of :class:`Contour` objects.  Must contain **≥ 1** element.
    color
        Display colour *(R, G, B)*, 0–255.
    type
        DICOM ``RTROIInterpretedType`` (e.g. ``"PTV"``, ``"ORGAN"``).
    volume
        If present in the RT-STRUCT, the documented volume (mm³).

    Notes
    -----
    Contours are automatically sorted by ``slice_position`` on initialisation.
    """

    name: str
    contours: List[Contour]
    color: Tuple[int, int, int]
    type: str
    volume: Optional[float] = None

    # ---------------------------------------------------------------------#
    # Validation                                                            #
    # ---------------------------------------------------------------------#
    def __post_init__(self) -> None:
        if not self.contours:
            raise ValueError("Structure must contain at least one contour")
        self.contours.sort(key=lambda c: c.slice_position)

    # ---------------------------------------------------------------------#
    # Derived quantities                                                    #
    # ---------------------------------------------------------------------#
    def compute_bounding_box(self) -> Tuple[NDArray, NDArray]:
        """
        Axis-aligned bounding box that encloses **all** vertices.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``(min_xyz, max_xyz)``, each ``(3,)`` **mm** arrays.
        """
        all_pts = np.vstack([c.points for c in self.contours])
        return all_pts.min(axis=0), all_pts.max(axis=0)

    # .....................................................................#
    @staticmethod
    def estimate_volume(contours: Sequence[Contour]) -> float:
        """
        Pairwise prismoidal volume estimate between successive contours.

        Parameters
        ----------
        contours
            Ordered list of contour slices (ascending or descending *z*).

        Returns
        -------
        float
            Total volume in **mm³**.

        See Also
        --------
        :meth:`compute_bounding_box`
        """
        if len(contours) < 2:
            return 0.0

        areas = [c.compute_area() for c in contours]
        zs = [c.slice_position for c in contours]

        vol = 0.0
        for (a1, z1), (a2, z2) in zip(zip(areas, zs), zip(areas[1:], zs[1:])):
            h = abs(z2 - z1)
            vol += h / 3 * (a1 + a2 + np.sqrt(a1 * a2))
        return vol


# -----------------------------------------------------------------------------#
# CtGeometry                                                                   #
# -----------------------------------------------------------------------------#
@dataclass(slots=True, frozen=True)
class CtGeometry:
    """
    Geometry metadata for a CT volume **without** pixel data.

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` voxel counts.
    origin
        (x, y, z) of voxel ``(0, 0, 0)``, **mm**.
    spacing
        ``(dx, dy, dz)`` in **mm**.
    orientation
        Orthonormal 3 × 3 direction-cosine matrix.
    """

    shape: tuple[int, int, int]
    origin: NDArray[np.float32]
    spacing: NDArray[np.float32]
    orientation: NDArray[np.float32]

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ValueError("shape must have three elements (nx, ny, nz)")

        object.__setattr__(self, "origin", self.origin.astype(np.float32, copy=False))
        object.__setattr__(self, "spacing", self.spacing.astype(np.float32, copy=False))
        object.__setattr__(
            self, "orientation", self.orientation.astype(np.float32, copy=False)
        )

        if not np.allclose(self.orientation @ self.orientation.T, np.eye(3), atol=1e-6):
            raise ValueError("orientation matrix must be orthonormal")


# -----------------------------------------------------------------------------#
# Case                                                                         #
# -----------------------------------------------------------------------------#
@dataclass
class Case:
    """
    Aggregates all objects required to describe a radiotherapy *case*.

    Parameters
    ----------
    dose_grid
        Primary dose distribution (may be ``None`` for structure-only QA).
    structures
        Mapping ``name → Structure``.
    ct_geometry
        Optional CT geometry for coordinate consistency checks.
    metadata
        Free-form dictionary for extra information (filepaths, timestamps …).

    Notes
    -----
    *Only* quick sanity checks are implemented here; heavy QA logic lives
    elsewhere.
    """

    dose_grid: Optional[DoseGrid]
    structures: Dict[str, Structure]
    ct_geometry: Optional[CtGeometry] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------#
    # Mutators                                                              #
    # ---------------------------------------------------------------------#
    def add_structure(self, structure: Structure) -> None:
        """Insert or overwrite a structure in :pyattr:`structures`."""
        self.structures[structure.name] = structure

    # ---------------------------------------------------------------------#
    # Consistency QA                                                        #
    # ---------------------------------------------------------------------#
    def check_consistency(self) -> List[str]:
        """
        Run inexpensive geometry/metadata sanity checks.

        Returns
        -------
        list[str]
            Human-readable warning messages.  Empty ⇒ no detected issues.
        """
        warnings: List[str] = []

        # ------------------------------------------------------------#
        # Presence checks                                             #
        # ------------------------------------------------------------#
        if self.dose_grid is None:
            warnings.append("No dose grid loaded")

        if not self.structures:
            warnings.append("No structures loaded")

        # ------------------------------------------------------------#
        # Dose ↔ CT orientation                                       #
        # ------------------------------------------------------------#
        if self.dose_grid and self.ct_geometry:
            if not np.allclose(
                self.dose_grid.orientation,
                self.ct_geometry.orientation,
                atol=1e-6,
            ):
                warnings.append(
                    "Dose grid and CT geometry have different orientation matrices"
                )

        # ------------------------------------------------------------#
        # Structure volume plausibility                               #
        # ------------------------------------------------------------#
        for name, s in self.structures.items():
            if s.volume is not None:
                est = Structure.estimate_volume(s.contours)
                if est > 0:
                    diff_pct = abs(s.volume - est) / s.volume * 100
                    if diff_pct > 10:
                        warnings.append(
                            f"Structure '{name}': documented volume differs "
                            f"from contour-derived volume by {diff_pct:.1f} %"
                        )

            if not s.contours:
                warnings.append(f"Structure '{name}' has no contours")
            elif len(s.contours) == 1:
                warnings.append(f"Structure '{name}' has only one contour slice")

        # ------------------------------------------------------------#
        # Structure bounding boxes versus dose grid extent            #
        # ------------------------------------------------------------#
        if self.dose_grid is not None:
            dose_min = self.dose_grid.origin
            dose_max = self.dose_grid.origin + self.dose_grid.spacing * np.array(
                self.dose_grid.values.shape
            )

            for name, s in self.structures.items():
                bb_min, bb_max = s.compute_bounding_box()
                if np.any(bb_max < dose_min) or np.any(bb_min > dose_max):
                    warnings.append(
                        f"Structure '{name}' lies completely outside the dose grid"
                    )
                elif np.any(bb_min < dose_min) or np.any(bb_max > dose_max):
                    warnings.append(
                        f"Structure '{name}' extends beyond dose grid boundaries"
                    )

        return warnings
