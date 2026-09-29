"""
src/calibration/optical_calibration.py
=========================================
One-time microscope optical calibration (µm/pixel) — stage-micrometer
based, per exact hardware configuration.

SCOPE
-----
This module contains ONLY pure calculation/validation logic and the
`CalibrationRecord` data shape. It has zero dependency on Streamlit,
OpenCV camera I/O, or the database layer — `app.py` handles capturing
the calibration frame and driving the UI, `db_manager.py` handles
persistence, and this module is the single place the actual
µm/pixel = known_distance_um / pixel_distance arithmetic and its
validation live, so neither of those layers duplicates the formula.

SCIENTIFIC PRINCIPLE
---------------------
Optical calibration (µm/pixel — how many real-world microns one pixel
represents) is a DIFFERENT concept from effective sample depth (how
deep the observed liquid layer is). This module only ever computes
and stores the former. It has no knowledge of chambers, coverslips,
sample volume, or depth — see `src.analysis.concentration` for that,
which accepts a *resolved* µm/pixel value from this module rather
than ever deriving one itself.

Never invents a value: every public function here either returns a
real, calculated number from real inputs, or raises/reports a validation
failure. There is no fallback constant anywhere in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

#: Heuristic sanity bounds for a computed µm/pixel result -- these
#: exist only to catch obvious data-entry/measurement mistakes (e.g.
#: clicking the same point twice, or entering the wrong known
#: distance), not as a scientific claim about what values are
#: possible. Ordinary light-microscope setups fall well within this
#: very wide range; a result outside it is treated as suspect and
#: rejected rather than silently saved.
MIN_REASONABLE_UM_PER_PIXEL: float = 0.01
MAX_REASONABLE_UM_PER_PIXEL: float = 50.0

#: Minimum pixel separation between the two selected calibration
#: points before a measurement is considered usable. Below this,
#: rounding/click-precision error dominates the result.
MIN_PIXEL_DISTANCE: float = 5.0


class CalibrationStatus:
    """Status of one saved calibration record."""
    VALIDATED = "validated"     # passed all QC checks at save time
    REJECTED = "rejected"       # failed QC -- never actually saved, kept only for audit/logging by callers that choose to log rejections
    SUPERSEDED = "superseded"   # replaced by a newer calibration for the same configuration


@dataclass
class CalibrationRecord:
    """
    One microscope optical calibration measurement, tied to an EXACT
    hardware configuration. Two records with different `magnification`
    (or any other configuration field) are never interchangeable — see
    `build_configuration_key`.

    Multiple records may exist for the SAME configuration (repeat
    measurements) — see `CalibrationRepeatability` below. The
    persistence layer (`db_manager.py`) never overwrites an existing
    record; each save creates a new one, and "the current calibration"
    for a configuration is defined as the most recent record for it.
    """
    microscope: str
    camera: str
    magnification: str
    camera_resolution: str
    digital_zoom: str                 # "" / "None" if not applicable
    known_distance_um: float
    measured_pixel_distance: float
    microns_per_pixel: float
    image_width_px: int = 0            # ACTUAL captured calibration image dimensions —
    image_height_px: int = 0           # may differ from the nominal camera_resolution setting
    operator: str = ""
    calibration_status: str = CalibrationStatus.VALIDATED
    software_version: str = ""
    created_at: str = ""
    notes: str = ""
    calibration_id: Optional[str] = None

    def configuration_key(self) -> Tuple[str, str, str, str, str]:
        return build_configuration_key(
            self.microscope, self.camera, self.magnification,
            self.camera_resolution, self.digital_zoom,
        )


@dataclass
class CalibrationRepeatability:
    """
    Transparent repeatability statistics across repeat calibration
    measurements of the SAME exact configuration.

    NO acceptance threshold is applied or implied anywhere in this
    module — these numbers are reported as-is for the operator/lab to
    judge against whatever threshold their own protocol establishes
    (if any). `std_dev_um_per_pixel`/`cv_percent` are `None` (never a
    fabricated 0.0) when fewer than 2 measurements exist — with a
    single measurement, repeatability has genuinely not been assessed
    yet, which is different from "assessed and found to be zero
    variation".
    """
    n_measurements: int
    mean_um_per_pixel: Optional[float]
    std_dev_um_per_pixel: Optional[float]
    cv_percent: Optional[float]
    values: List[float]

    def to_dict(self) -> dict:
        return {
            "n_measurements": self.n_measurements,
            "mean_um_per_pixel": self.mean_um_per_pixel,
            "std_dev_um_per_pixel": self.std_dev_um_per_pixel,
            "cv_percent": self.cv_percent,
            "values": list(self.values),
        }


def calculate_calibration_repeatability(
    microns_per_pixel_values: List[float],
) -> CalibrationRepeatability:
    """
    Compute mean / SD / CV across repeated µm/pixel measurements for
    the same exact configuration. Pure statistics only — see
    `CalibrationRepeatability` docstring for why SD/CV are `None`
    (not 0.0) below n=2.
    """
    import statistics as _stats

    values = [v for v in microns_per_pixel_values if v is not None and v > 0]
    n = len(values)
    if n == 0:
        return CalibrationRepeatability(0, None, None, None, [])
    if n == 1:
        return CalibrationRepeatability(1, values[0], None, None, values)

    mean_val = _stats.mean(values)
    sd = _stats.stdev(values)
    cv = (sd / mean_val * 100.0) if mean_val > 0 else None
    return CalibrationRepeatability(n, mean_val, sd, cv, values)


def build_configuration_key(
    microscope: str,
    camera: str,
    magnification: str,
    camera_resolution: str,
    digital_zoom: str = "",
) -> Tuple[str, str, str, str, str]:
    """
    Canonical identity for "the exact optical configuration a
    calibration applies to". Two configurations are equivalent only if
    every one of these five fields matches exactly -- e.g. 20x is
    NEVER treated as equivalent to 40x, and a different camera
    resolution is NEVER treated as equivalent to another, even on the
    same microscope/camera.

    Values are stripped and empty digital_zoom is normalised to
    "None" so lookups are consistent regardless of whether a caller
    passed "" or None-as-string.
    """
    zoom = (digital_zoom or "").strip() or "None"
    return (
        (microscope or "").strip(),
        (camera or "").strip(),
        (magnification or "").strip(),
        (camera_resolution or "").strip(),
        zoom,
    )


def measure_pixel_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Euclidean pixel distance between two operator-identified endpoints
    of a known stage-micrometer interval.

    Parameters
    ----------
    x1, y1, x2, y2 : float   pixel coordinates in the calibration image

    Returns
    -------
    float   distance in pixels (always >= 0)
    """
    return math.hypot(x2 - x1, y2 - y1)


def calculate_microns_per_pixel(known_distance_um: float, pixel_distance: float) -> float:
    """
    The one and only place this formula is implemented:

        microns_per_pixel = known_distance_um / pixel_distance

    Parameters
    ----------
    known_distance_um : float   the stage micrometer's marked distance, must be > 0
    pixel_distance : float      measured pixel separation, must be > 0

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If either input is not strictly positive.
    """
    if known_distance_um <= 0:
        raise ValueError("known_distance_um must be > 0")
    if pixel_distance <= 0:
        raise ValueError("pixel_distance must be > 0")
    return known_distance_um / pixel_distance


def validate_calibration_inputs(
    known_distance_um: float,
    pixel_distance: float,
) -> List[str]:
    """
    Quality-control checks run BEFORE a calibration is allowed to be
    saved (never after — an invalid calibration is never persisted).

    Returns
    -------
    list of str   human-readable problems found; empty list means the
        inputs are acceptable to compute and save a calibration from.
        Never raises -- callers show these as an error list and refuse
        to save when non-empty.
    """
    problems: List[str] = []

    if known_distance_um is None or known_distance_um <= 0:
        problems.append("Known stage-micrometer distance must be a positive number of µm.")

    if pixel_distance is None or pixel_distance <= 0:
        problems.append(
            "Measured pixel distance must be greater than zero — the two selected "
            "points cannot be the same point."
        )
    elif pixel_distance < MIN_PIXEL_DISTANCE:
        problems.append(
            f"Selected points are only {pixel_distance:.1f} px apart "
            f"(minimum {MIN_PIXEL_DISTANCE:g} px) — pick two points further apart "
            "for a reliable measurement."
        )

    if not problems:
        try:
            result = calculate_microns_per_pixel(known_distance_um, pixel_distance)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if not (MIN_REASONABLE_UM_PER_PIXEL <= result <= MAX_REASONABLE_UM_PER_PIXEL):
                problems.append(
                    f"Calculated value ({result:.4f} µm/pixel) is outside the "
                    f"plausible range ({MIN_REASONABLE_UM_PER_PIXEL:g}-"
                    f"{MAX_REASONABLE_UM_PER_PIXEL:g} µm/pixel) for a light "
                    "microscope setup — check the known distance and the "
                    "selected points and try again."
                )

    return problems


def build_calibration_record(
    microscope: str,
    camera: str,
    magnification: str,
    camera_resolution: str,
    known_distance_um: float,
    pixel_distance: float,
    digital_zoom: str = "",
    image_width_px: int = 0,
    image_height_px: int = 0,
    operator: str = "",
    software_version: str = "",
    notes: str = "",
) -> CalibrationRecord:
    """
    Validate inputs and build a `CalibrationRecord` ready to save.

    Parameters
    ----------
    image_width_px, image_height_px : int
        The ACTUAL pixel dimensions of the captured calibration image
        (not the nominal camera_resolution setting — these can differ
        if the image was cropped/resized/binned). Stored for audit and
        mismatch-detection purposes; see
        `check_image_dimensions_match`.
    operator : str
        Who performed this specific calibration measurement.

    Raises
    ------
    ValueError
        If `validate_calibration_inputs` reports any problem — the
        caller should call that function itself first to show the
        error list to the operator; this raises with the joined
        messages as a convenience for non-UI callers (e.g. tests).
    """
    problems = validate_calibration_inputs(known_distance_um, pixel_distance)
    if problems:
        raise ValueError("; ".join(problems))

    microns_per_pixel = calculate_microns_per_pixel(known_distance_um, pixel_distance)

    return CalibrationRecord(
        microscope=microscope,
        camera=camera,
        magnification=magnification,
        camera_resolution=camera_resolution,
        digital_zoom=(digital_zoom or "").strip() or "None",
        known_distance_um=known_distance_um,
        measured_pixel_distance=pixel_distance,
        microns_per_pixel=microns_per_pixel,
        image_width_px=int(image_width_px or 0),
        image_height_px=int(image_height_px or 0),
        operator=operator,
        calibration_status=CalibrationStatus.VALIDATED,
        software_version=software_version,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        notes=notes,
    )


def check_image_dimensions_match(
    calibration_image_width_px: int,
    calibration_image_height_px: int,
    analyzed_image_width_px: int,
    analyzed_image_height_px: int,
) -> Optional[str]:
    """
    Scientific-safety check (task requirement: "If an image is resized
    or cropped before physical measurement, the physical scale must be
    transformed correctly or the appropriate calibration must be
    used"). Compares the ACTUAL pixel dimensions the calibration was
    measured against to the ACTUAL dimensions of the video/image
    currently being analyzed.

    This does not attempt to auto-correct a mismatch (that would risk
    silently applying an unverified transformation) — it only reports
    one so the caller can warn the operator rather than silently
    trusting a calibration that may no longer apply.

    Parameters
    ----------
    calibration_image_width_px, calibration_image_height_px : int
        From the saved `CalibrationRecord`. 0 means "not recorded"
        (older calibrations, or built without image dimensions) — in
        that case this function returns None (nothing to compare
        against), it does not treat 0 as a mismatch.
    analyzed_image_width_px, analyzed_image_height_px : int
        The actual frame/video dimensions being analyzed now.

    Returns
    -------
    str | None   a warning message, or None if dimensions match (or
        the calibration has no recorded dimensions to compare).
    """
    if calibration_image_width_px <= 0 or calibration_image_height_px <= 0:
        return None
    if analyzed_image_width_px <= 0 or analyzed_image_height_px <= 0:
        return None

    if (calibration_image_width_px, calibration_image_height_px) != (
        analyzed_image_width_px, analyzed_image_height_px,
    ):
        return (
            f"Calibration image size ({calibration_image_width_px}×"
            f"{calibration_image_height_px} px) does not match the "
            f"analyzed image size ({analyzed_image_width_px}×"
            f"{analyzed_image_height_px} px) — the saved µm/pixel value "
            f"may not apply if the image has been cropped, resized, or "
            f"binned differently since calibration."
        )
    return None