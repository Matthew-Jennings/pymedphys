"""PyTest local plugins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

SKIPPING_CONFIG = {
    "slow": {
        "options": ["--run-only-slow", "--slow"],
        "help": "run the slow tests",
        "description": "mark test as slow to run",
        "skip_otherwise": True,
    },
    "cypress": {
        "options": ["--run-only-yarn", "--cypress"],
        "help": "run the cypress tests",
        "description": "mark test as using cypress",
        "skip_otherwise": True,
    },
    "pydicom": {
        "options": ["--run-only-pydicom", "--pydicom"],
        "help": "run only the tests that use pydicom",
        "description": "mark test as using pydicom",
        "skip_otherwise": False,
    },
    "mosaiqdb": {
        "options": ["--run-only-mosaiqdb", "--mosaiqdb"],
        "help": "run only the tests that use mosaiq db",
        "description": "mark test as using mosaiq db",
        "skip_otherwise": True,
    },
    "anthropic_key": {
        "options": ["--run-only-anthropic", "--anthropic"],
        "help": "run only the tests that use Anthropic API",
        "description": "mark test as requiring an Anthropic API key",
        "skip_otherwise": True,
    },
    "all": {
        "options": ["--run-all-tests", "--all"],
        "help": "run all tests",
        "description": "run all tests that would normally be skipped due to a configuration flag",
        "skip_otherwise": False,
    },
}


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
def pytest_addoption(parser):
    for _, skip_item in SKIPPING_CONFIG.items():
        for option in skip_item["options"]:
            parser.addoption(
                option, action="store_true", default=False, help=skip_item["help"]
            )


def pytest_configure(config):
    for key, skip_item in SKIPPING_CONFIG.items():
        config.addinivalue_line("markers", f"{key}: {skip_item['description']}")


def pytest_collection_modifyitems(config, items):
    for option in SKIPPING_CONFIG["all"]["options"]:
        if config.getoption(option):
            return

    for key, skip_item in SKIPPING_CONFIG.items():
        this_option_set = False
        provided_option = ""

        for option in skip_item["options"]:
            if config.getoption(option):
                this_option_set = True
                provided_option = option
                break

        if not this_option_set:
            if skip_item["skip_otherwise"]:
                skip = pytest.mark.skip(
                    reason=f"need {skip_item['options'][-1]} option to run"
                )

                for item in items:
                    if key in item.keywords:
                        item.add_marker(skip)
        else:
            skip = pytest.mark.skip(reason=f"since {provided_option} was passed")
            for item in items:
                if key not in item.keywords:
                    item.add_marker(skip)


def pytest_ignore_collect(collection_path, config):  # pylint: disable = unused-argument
    """return True to prevent considering this collection_path for collection.

    This hook is consulted for all files and directories prior to
    calling more specific hooks.
    """

    relative_path = os.path.relpath(str(collection_path), os.path.dirname(__file__))
    relative_path_list = relative_path.split(os.path.sep)

    return (
        (len(relative_path_list) > 1 and relative_path_list[0] == "examples")
        or "node_modules" in relative_path_list
        or "site-packages" in relative_path_list
        or "_build" in relative_path_list
        or ("_bundle" in relative_path_list and "python" in relative_path_list)
        or (
            config.getoption("--doctest-modules")
            and (
                "_gamma" in relative_path_list
                or "tests" in relative_path_list
                or "_imports" in relative_path_list
                or "_experimental" in relative_path_list
            )
        )
    )


# --------------------------------------------------------------------------- #
# Internal DICOM factory                                                      #
# --------------------------------------------------------------------------- #
class _MakeDICOM:
    """Tiny factory that writes minimal—but valid—DICOM objects to disk."""

    # ..................................................................... #
    @staticmethod
    def rt_dose(
        fp: Path, *, shape: Sequence[int] = (5, 10, 10), dose_gy: float = 10.0
    ) -> None:
        fm = pydicom.dataset.FileMetaDataset()
        fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"  # RTDOSE
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = "1.2.840.10008.1.2.1"
        fm.ImplementationClassUID = generate_uid()

        ds = FileDataset(str(fp), {}, file_meta=fm, preamble=b"\0" * 128)
        ds.Modality = "RTDOSE"
        ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [2.0, 2.0]
        ds.GridFrameOffsetVector = list(np.arange(shape[0]) * 3.0)

        ds.DoseUnits = "GY"
        ds.DoseGridScaling = 0.01
        ds.Rows, ds.Columns, ds.NumberOfFrames = shape[1], shape[2], shape[0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        arr = np.full(shape, int(dose_gy / ds.DoseGridScaling), dtype=np.uint16)
        ds.PixelData = arr.tobytes()
        ds.save_as(fp, write_like_original=False)

    # ..................................................................... #
    @staticmethod
    def rt_struct(fp: Path, *, names: Sequence[str] = ("PTV", "OAR")) -> None:
        """Write a minimal RT-STRUCT whose contours lie fully inside the test dose grid."""
        fm = pydicom.dataset.FileMetaDataset()
        fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = "1.2.840.10008.1.2.1"

        ds = FileDataset(str(fp), {}, file_meta=fm, preamble=b"\0" * 128)
        ds.Modality = "RTSTRUCT"
        ds.StructureSetLabel = "pytest-set"
        ds.StructureSetROISequence = []
        ds.ROIContourSequence = []
        ds.RTROIObservationsSequence = []

        colours = [(255, 0, 0), (0, 255, 0)]
        for i, name in enumerate(names, start=1):
            # ‒‒ ROI header ------------------------------------------------
            ss_roi = Dataset()
            ss_roi.ROINumber, ss_roi.ROIName = i, name
            ds.StructureSetROISequence.append(ss_roi)

            # ‒‒ Observation ---------------------------------------------
            roi_obs = Dataset()
            roi_obs.ReferencedROINumber = i
            roi_obs.RTROIInterpretedType = "PTV" if "PTV" in name else "ORGAN"
            ds.RTROIObservationsSequence.append(roi_obs)

            # ‒‒ Contours -------------------------------------------------
            roi_con = Dataset()
            roi_con.ReferencedROINumber = i
            roi_con.ROIDisplayColor = list(colours[(i - 1) % 2])
            roi_con.ContourSequence = []

            # rectangle 0‥8 mm  (fully inside 0‥20 mm dose grid)
            s = 8.0
            for z in (0.0, 3.0, 6.0):
                c = Dataset()
                c.ContourGeometricType = "CLOSED_PLANAR"
                c.NumberOfContourPoints = 4
                c.ContourData = [
                    0.0,
                    0.0,
                    z,
                    s,
                    0.0,
                    z,
                    s,
                    s,
                    z,
                    0.0,
                    s,
                    z,
                ]
                roi_con.ContourSequence.append(c)

            ds.ROIContourSequence.append(roi_con)

        ds.save_as(fp, write_like_original=False)

    # ..................................................................... #
    @staticmethod
    def ct_series(dir_: Path, *, z_positions: Sequence[float] = (0.0, 2.0)) -> None:
        dir_.mkdir(exist_ok=True)
        series_instance_uid = generate_uid()

        for idx, z in enumerate(z_positions):
            fm = pydicom.dataset.FileMetaDataset()
            fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image
            fm.MediaStorageSOPInstanceUID = generate_uid()
            fm.TransferSyntaxUID = "1.2.840.10008.1.2.1"

            ds = FileDataset(
                str(dir_ / f"ct_{idx:03d}.dcm"), {}, file_meta=fm, preamble=b"\0" * 128
            )
            ds.Modality = "CT"
            ds.SeriesInstanceUID = series_instance_uid
            ds.ImagePositionPatient = [0.0, 0.0, z]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.PixelSpacing = [1.0, 1.0]
            ds.Rows = ds.Columns = 10
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = -1000.0
            ds.RescaleSlope = 1.0
            ds.PixelData = np.full((10, 10), 1000, dtype=np.int16).tobytes()
            ds.save_as(dir_ / f"ct_{idx:03d}.dcm", write_like_original=False)


# --------------------------------------------------------------------------- #
# Pytest fixtures                                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rt_dose_file(tmp_path_factory):
    fp = tmp_path_factory.mktemp("dcm") / "dose.dcm"
    _MakeDICOM.rt_dose(fp, shape=(5, 10, 10), dose_gy=20.0)
    return fp


@pytest.fixture(scope="module")
def rt_struct_file(tmp_path_factory):
    fp = tmp_path_factory.mktemp("dcm") / "struct.dcm"
    _MakeDICOM.rt_struct(fp, names=("PTV", "OAR"))
    return fp


@pytest.fixture(scope="module")
def ct_dir(tmp_path_factory):
    dir_ = tmp_path_factory.mktemp("ct")
    _MakeDICOM.ct_series(dir_, z_positions=(0.0, 2.5, 5.0))
    return dir_
