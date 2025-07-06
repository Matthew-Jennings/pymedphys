"""DICOM loading functionality for RT Structure Sets, RT Dose, and CT images."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pydicom
from numpy.typing import NDArray

from .data_types import Case, Contour, CtGeometry, DoseGrid, Structure

logger = logging.getLogger(__name__)

CUBIC_CM_to_CUBIC_MM = 1000

__all__ = [
    "load_rt_dose",
    "load_rt_struct",
    "load_ct_geometry",
    "load_case",
    "validate_case",
]


def load_rt_dose(filepath: os.PathLike) -> DoseGrid:
    """
    Parse a DICOM **RTDOSE** file into a :class:`~pymedphys._dvh.core.data_types.DoseGrid`.

    Parameters
    ----------
    filepath
        Path-like object (``str`` or ``Path``) pointing to a single RT-Dose
        DICOM file.

    Returns
    -------
    DoseGrid
        Float-32 dose lattice with origin, spacing and orientation fully
        validated.

    Raises
    ------
    ValueError
        If the object is not RTDOSE, has non-positive ``DoseGridScaling`` or
        inconsistent slice spacing/orientation.
    pydicom.errors.InvalidDicomError
        If the file cannot be read by *pydicom*.

    Notes
    -----
    A “metadata-only” read (`stop_before_pixels=True`) is performed first to
    catch errors *before* loading the potentially large pixel array.
    """

    ds = pydicom.dcmread(
        filepath,
        stop_before_pixels=True,
        specific_tags=(
            "Modality",
            "DoseGridScaling",
            "ImageOrientationPatient",
            "GridFrameOffsetVector",
        ),
    )

    # Verify this is an RT Dose file
    if ds.Modality != "RTDOSE":
        raise ValueError(f"File is not an RT Dose (SOP Class: {ds.SOPClassUID})")

    if ds.DoseGridScaling <= 0:
        raise ValueError(
            f"Invalid DoseGridScaling value: {ds.DoseGridScaling}. Must be positive."
        )

    if not hasattr(ds, "GridFrameOffsetVector") or len(ds.GridFrameOffsetVector) <= 1:
        raise ValueError("GridFrameOffsetVector not found or insufficient data")

    z_spacing_all = np.diff(ds.GridFrameOffsetVector)
    if not np.allclose(z_spacing_all, z_spacing_all[0]):
        raise ValueError("Inconsistent z-spacing in GridFrameOffsetVector")
    z_spacing = z_spacing_all[0]

    orientation = _orientation_from_iop(ds.ImageOrientationPatient)

    ds = pydicom.dcmread(filepath)

    # Extract geometry
    origin = np.array(ds.ImagePositionPatient, dtype=np.float32)

    # Want column spacing (typically x), then row spacing (typically y).
    spacing = np.array(
        [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), abs(z_spacing)],
        dtype=np.float32,
    )

    # Extract dose data
    dose_values = ds.pixel_array * ds.DoseGridScaling

    # Transpose to get (x, y, z) indexing
    dose_values = np.transpose(dose_values, (2, 1, 0))

    return DoseGrid(
        values=dose_values,
        origin=origin,
        spacing=spacing,
        orientation=orientation,
        units=ds.DoseUnits,
    )


def load_rt_struct(
    filepath: os.PathLike, roi_names: Optional[List[str]] = None
) -> dict[str, Structure]:
    """
    Convert an **RTSTRUCT** file into a mapping ``name → Structure``.

    Parameters
    ----------
    filepath
        Location of the RT-Structure-Set DICOM.
    roi_names
        List of ROI names to keep. ``None`` ⇒ load **all** ROIs.

    Returns
    -------
    dict[str, Structure]
        Each :class:`Structure` contains its slice-sorted contour list.

    Warns
    -----
    logger.warning
        Non-planar or malformed contours are skipped with a warning.

    Raises
    ------
    ValueError
        If the file is not RTSTRUCT.
    """

    ds = pydicom.dcmread(filepath)

    # Verify this is an RT Structure Set
    if ds.Modality != "RTSTRUCT":
        raise ValueError(
            f"File is not an RT Structure Set (SOP Class: {ds.SOPClassUID})"
        )

    structures = {}

    # Build ROI number to name mapping
    roi_names_map = {}
    roi_volumes = {}
    roi_colors = {}
    roi_interpreted_types = {}

    for roi_seq in ds.StructureSetROISequence:
        roi_number = roi_seq.ROINumber
        roi_name = roi_seq.ROIName
        roi_names_map[roi_number] = roi_name

        # Get volume if available
        if hasattr(roi_seq, "ROIVolume"):
            roi_volumes[roi_name] = float(roi_seq.ROIVolume) * CUBIC_CM_to_CUBIC_MM

    # Get ROI Interpreted Type
    for roi_obs in ds.RTROIObservationsSequence:
        roi_number = roi_obs.ReferencedROINumber
        roi_name = roi_names_map.get(roi_number)
        if roi_name:
            roi_interpreted_types[roi_name] = roi_obs.RTROIInterpretedType

    # Process ROI contours
    for roi_contour in ds.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        roi_name = roi_names_map.get(roi_number)

        if roi_name is None:
            continue

        # Skip if not in requested list
        if roi_names is not None and roi_name not in roi_names:
            continue

        roi_colors[roi_name] = tuple(int(c) for c in roi_contour.ROIDisplayColor)

        if not hasattr(roi_contour, "ContourSequence"):
            if roi_name in roi_names:
                logger.warning(f"No contours found for requested ROI '{roi_name}'")
            else:
                logger.debug(f"No contours found for ROI '{roi_name}', skipping")
            continue

        contours = _extract_contours(roi_contour, roi_name)

        if contours:
            structure = Structure(
                name=roi_name,
                contours=contours,
                color=roi_colors[roi_name],
                type=roi_interpreted_types[roi_name],
                volume=roi_volumes.get(roi_name),
            )
            structures[roi_name] = structure

    return structures


def _extract_contours(roi_contour, roi_name) -> list[Contour]:
    contours = []
    for contour in roi_contour.ContourSequence:
        # Check contour type
        if contour.ContourGeometricType != "CLOSED_PLANAR":
            logger.warning(f"Skipping non-planar contour in {roi_name}")
            continue

        # Extract contour points
        contour_data = np.array(contour.ContourData, dtype=np.float32)
        n_points = int(contour.NumberOfContourPoints)

        if len(contour_data) != 3 * n_points:
            logger.warning(f"Contour data size mismatch in {roi_name}")
            continue

        # Reshape to Nx3 array
        points = contour_data.reshape((n_points, 3))

        # Get slice position (should be consistent for all points)
        if not np.allclose(points[:, 2], points[0, 2]):
            logger.warning(
                f"Contour points in {roi_name} have inconsistent z-coordinates"
            )
            continue
        slice_position = points[0, 2]

        contour = Contour(points=points, slice_position=slice_position)
        contours.append(contour)
    return contours


def load_ct_geometry(dirpath: os.PathLike) -> CtGeometry:
    """
    Read geometry (no pixels) from a folder containing a single CT series.

    Parameters
    ----------
    dirpath
        Directory with one *.dcm* CT image series.  Mixed modalities are
        ignored.

    Returns
    -------
    CtGeometry
        Shape, origin, spacing and orientation of the CT stack.

    Raises
    ------
    ValueError
        If no CT slices are found, multiple series are detected, or metadata
        (rows/columns/pixel spacing/ slice spacing) are inconsistent.
    """

    dirpath = Path(dirpath)

    if not dirpath.is_dir():
        raise ValueError("Provided path must be a directory")

    # Load all DICOM files in directory
    dicom_files = sorted(dirpath.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted(dirpath.glob("*"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dirpath}")

    # Read all slices
    slice_candidates = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(
                f,
                stop_before_pixels=True,
                specific_tags=("Modality", "SeriesInstanceUID"),
            )
            if ds.Modality != "CT":
                continue
            slice_candidates.append((f, ds.SeriesInstanceUID))
        except (pydicom.errors.InvalidDicomError, AttributeError):
            continue

    if not slice_candidates:
        raise ValueError("No valid CT slices found")

    for f, series_instance_uid in slice_candidates[1:]:
        if series_instance_uid != slice_candidates[0][1]:
            raise ValueError(
                f"The DICOM CT image slice in {f.name} is from a different series than the one in {slice_candidates[0].name} ({series_instance_uid} vs. {slice_candidates[0][1]})"
            )

    slices = [pydicom.dcmread(f, stop_before_pixels=True) for f, _ in slice_candidates]

    # Sort by ImagePositionPatient z-coordinate
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # # Stack into 3D array
    # ct_values = np.stack([s.pixel_array for s in slices], axis=0)

    # Use first slice for geometry
    ds = slices[0]

    num_rows_all = [s.Rows for s in slices]
    if not all(nr == num_rows_all[0] for nr in num_rows_all[1:]):
        raise ValueError("Inconsistent number of rows between CT slices")

    num_cols_all = [s.Columns for s in slices]
    if not all(nc == num_cols_all[0] for nc in num_cols_all[1:]):
        raise ValueError("Inconsistent number of columns between CT slices")

    x_spacing_all = np.array(
        [float(s.PixelSpacing[1]) for s in slices], dtype=np.float32
    )
    y_spacing_all = np.array(
        [float(s.PixelSpacing[0]) for s in slices], dtype=np.float32
    )

    if not np.allclose(x_spacing_all[1:], x_spacing_all[0]):
        raise ValueError("Inconsistent pixel x spacing between CT slices")

    if not np.allclose(y_spacing_all[1:], y_spacing_all[0]):
        raise ValueError("Inconsistent pixel y spacing between CT slices")

    # Calculate actual z-spacing from slice positions
    z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
    z_spacing_all = np.diff(z_positions)
    if not np.allclose(z_spacing_all[1:], z_spacing_all[0]):
        raise ValueError("Inconsistent spacing between CT slices.")

    # # Apply rescale slope and intercept to get HU values
    # if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
    #     ct_values = ct_values * ds.RescaleSlope + ds.RescaleIntercept

    # Extract geometry
    origin = np.array(ds.ImagePositionPatient, dtype=np.float32)
    spacing = np.array(
        [x_spacing_all[0], y_spacing_all[0], abs(z_spacing_all[0])], dtype=np.float32
    )
    shape = (
        int(ds.Columns),
        int(ds.Rows),
        len(slices),
    )

    orientation = _orientation_from_iop(ds.ImageOrientationPatient)

    # Transpose to (x, y, z) indexing
    # ct_values = np.transpose(ct_values, (2, 1, 0))

    return CtGeometry(
        shape=shape, origin=origin, spacing=spacing, orientation=orientation
    )


def load_case(
    rt_dose_path: os.PathLike,
    rt_struct_path: os.PathLike,
    ct_path: Optional[os.PathLike] = None,
    roi_names: Optional[List[str]] = None,
) -> Case:
    """
    Convenience wrapper that calls the three loaders above and returns a
    fully-populated :class:`Case`.

    Parameters
    ----------
    rt_dose_path
        RT-Dose file.
    rt_struct_path
        RT-Structure-Set file.
    ct_path
        *Optional.* Folder or file of the corresponding CT series.  If omitted,
        only dose/structures are loaded.
    roi_names
        Optional subset of structures to import.

    Returns
    -------
    Case
        Ready-to-use container object.

    Notes
    -----
    A warning is logged for every *consistency* issue found by
    :py:meth:`Case.check_consistency`.
    """

    rt_dose_path = Path(rt_dose_path)
    rt_struct_path = Path(rt_struct_path)

    # Load RT Dose
    if not rt_dose_path.exists():
        raise ValueError(f"RT Dose file not found: {rt_dose_path}")
    dose_grid = load_rt_dose(rt_dose_path)
    logger.info(f"Loaded RT Dose: shape={dose_grid.values.shape}")

    # Load RT Structure Set
    if not rt_struct_path.exists():
        raise ValueError(f"RT Structure Set file not found: {rt_struct_path}")
    structures = load_rt_struct(rt_struct_path, roi_names)
    logger.info(f"Loaded {len(structures)} structures")

    case = Case(dose_grid=dose_grid, structures=structures)

    # Load CT
    if ct_path is not None:
        ct_path = Path(ct_path)

        if ct_path.exists():
            case.ct_geometry = load_ct_geometry(ct_path)
            logger.info(f"Loaded CT: shape={case.ct_geometry.shape}")

    # Run consistency checks
    warnings = case.check_consistency()
    for warning in warnings:
        logger.warning(warning)

    return case


def _orientation_from_iop(iop: Sequence[float]) -> NDArray[np.float32]:
    """
    Build an orthonormal orientation matrix from ``ImageOrientationPatient``.

    Parameters
    ----------
    iop
        Length-6 sequence ``[row_x, row_y, row_z, col_x, col_y, col_z]``.

    Returns
    -------
    numpy.ndarray
        3 × 3 direction-cosine matrix whose columns are *(x̂, ŷ, ẑ)*.

    Raises
    ------
    ValueError
        If the resulting matrix is not orthonormal within 1 µrad.
    """

    row, col = np.array(iop[:3]), np.array(iop[3:])
    norm = np.cross(row, col)
    orient = np.column_stack([row, col, norm]).astype(np.float32)
    if not np.allclose(orient @ orient.T, np.eye(3), atol=1e-6):
        raise ValueError("ImageOrientationPatient not orthonormal")
    return orient


def validate_case(case: Case) -> None:
    """Raise ``ValueError`` if :pyattr:`Case.check_consistency` returns warnings."""
    warnings = case.check_consistency()
    if warnings:
        raise ValueError("Case failed validation:\n  • " + "\n  • ".join(warnings))
