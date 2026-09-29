"""
tests/test_optical_calibration.py
====================================
Unit tests for src/calibration/optical_calibration.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from src.calibration import optical_calibration as oc


# ── 1. µm/pixel calculation ──────────────────────────────────────
def test_worked_example_100um_250px():
    assert oc.calculate_microns_per_pixel(100.0, 250.0) == pytest.approx(0.40)


def test_measure_pixel_distance_horizontal():
    assert oc.measure_pixel_distance(100, 100, 350, 100) == pytest.approx(250.0)


def test_measure_pixel_distance_diagonal():
    # 3-4-5 triangle
    assert oc.measure_pixel_distance(0, 0, 3, 4) == pytest.approx(5.0)


# ── 2. Invalid zero pixel distance rejected ───────────────────────
def test_zero_pixel_distance_rejected():
    with pytest.raises(ValueError):
        oc.calculate_microns_per_pixel(100.0, 0.0)


def test_zero_pixel_distance_in_validation():
    problems = oc.validate_calibration_inputs(100.0, 0.0)
    assert len(problems) > 0


# ── 3. Negative/invalid values rejected ───────────────────────────
def test_negative_known_distance_rejected():
    with pytest.raises(ValueError):
        oc.calculate_microns_per_pixel(-5.0, 100.0)


def test_negative_pixel_distance_rejected():
    with pytest.raises(ValueError):
        oc.calculate_microns_per_pixel(100.0, -50.0)


def test_negative_values_in_validation():
    assert len(oc.validate_calibration_inputs(-1.0, 100.0)) > 0
    assert len(oc.validate_calibration_inputs(100.0, -1.0)) > 0


def test_too_close_points_rejected():
    problems = oc.validate_calibration_inputs(100.0, 2.0)
    assert len(problems) > 0


def test_implausible_ratio_rejected():
    # 100 um over 100000 px would be an absurdly small um/pixel
    problems = oc.validate_calibration_inputs(100.0, 100000.0)
    assert len(problems) > 0


def test_valid_inputs_produce_no_problems():
    assert oc.validate_calibration_inputs(100.0, 250.0) == []


# ── 4. Calibration associated with correct configuration ──────────
def test_configuration_key_distinguishes_magnification():
    k1 = oc.build_configuration_key("Scope A", "Cam1", "20×", "1280×720")
    k2 = oc.build_configuration_key("Scope A", "Cam1", "40×", "1280×720")
    assert k1 != k2


def test_configuration_key_distinguishes_resolution():
    k1 = oc.build_configuration_key("Scope A", "Cam1", "20×", "1280×720")
    k2 = oc.build_configuration_key("Scope A", "Cam1", "20×", "640×480")
    assert k1 != k2


def test_configuration_key_normalizes_empty_zoom():
    k1 = oc.build_configuration_key("Scope A", "Cam1", "20×", "1280×720", "")
    k2 = oc.build_configuration_key("Scope A", "Cam1", "20×", "1280×720", None)
    assert k1 == k2
    assert k1[-1] == "None"


def test_calibration_record_configuration_key_matches_builder():
    record = oc.build_calibration_record(
        microscope="Scope A", camera="Cam1", magnification="20×",
        camera_resolution="1280×720", known_distance_um=100.0, pixel_distance=250.0,
    )
    assert record.configuration_key() == oc.build_configuration_key(
        "Scope A", "Cam1", "20×", "1280×720",
    )


# ── 5/6. Saved calibration load + different configuration isolation
#          (DB-level — covered in tests/test_db_manager.py's
#          calibration tests; here we only test the pure-math layer
#          that decides "is this the exact same configuration".) ────
def test_build_calibration_record_raises_on_invalid_inputs():
    with pytest.raises(ValueError):
        oc.build_calibration_record(
            microscope="Scope A", camera="Cam1", magnification="20×",
            camera_resolution="1280×720", known_distance_um=100.0, pixel_distance=0.0,
        )


def test_build_calibration_record_never_invents_a_value():
    # Explicitly confirms no default/example value (e.g. 0.40) is
    # ever returned independent of the actual inputs.
    r1 = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    r2 = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=200.0, pixel_distance=250.0,
    )
    assert r1.microns_per_pixel != r2.microns_per_pixel
    assert r2.microns_per_pixel == pytest.approx(0.80)


# ══════════════════════════════════════════════════════════════
# Repeat measurements / repeatability statistics
# ══════════════════════════════════════════════════════════════

def test_repeatability_multiple_measurements():
    rep = oc.calculate_calibration_repeatability([0.40, 0.41, 0.39])
    assert rep.n_measurements == 3
    assert rep.mean_um_per_pixel == pytest.approx(0.40)
    assert rep.std_dev_um_per_pixel is not None
    assert rep.cv_percent is not None


def test_repeatability_single_measurement_has_no_sd_or_cv():
    """n=1 means repeatability has NOT been assessed — SD/CV must be
    None, never a fabricated 0.0 implying perfect repeatability."""
    rep = oc.calculate_calibration_repeatability([0.40])
    assert rep.n_measurements == 1
    assert rep.mean_um_per_pixel == pytest.approx(0.40)
    assert rep.std_dev_um_per_pixel is None
    assert rep.cv_percent is None


def test_repeatability_empty():
    rep = oc.calculate_calibration_repeatability([])
    assert rep.n_measurements == 0
    assert rep.mean_um_per_pixel is None
    assert rep.std_dev_um_per_pixel is None


def test_repeatability_ignores_invalid_values():
    rep = oc.calculate_calibration_repeatability([0.40, None, 0.42, -1.0, 0.0])
    assert rep.n_measurements == 2


def test_repeatability_no_acceptance_threshold_is_applied():
    """Wildly variable measurements must still be reported as-is —
    this module never applies a pass/fail acceptance threshold."""
    rep = oc.calculate_calibration_repeatability([0.1, 5.0, 0.3])
    assert rep.n_measurements == 3
    assert rep.cv_percent is not None
    # No 'passed'/'acceptable' flag exists on the result at all.
    assert not hasattr(rep, "passed")
    assert not hasattr(rep, "acceptable")


# ══════════════════════════════════════════════════════════════
# Actual image dimensions / scale-change detection
# ══════════════════════════════════════════════════════════════

def test_image_dimension_mismatch_detected():
    warning = oc.check_image_dimensions_match(1280, 720, 640, 480)
    assert warning is not None
    assert "does not match" in warning


def test_image_dimension_match_no_warning():
    assert oc.check_image_dimensions_match(1280, 720, 1280, 720) is None


def test_image_dimension_unrecorded_gives_no_false_alarm():
    """Older calibrations have 0/0 recorded dimensions — that must not
    be reported as a mismatch."""
    assert oc.check_image_dimensions_match(0, 0, 1280, 720) is None
    assert oc.check_image_dimensions_match(1280, 720, 0, 0) is None


def test_calibration_record_stores_actual_image_dimensions_and_operator():
    record = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
        image_width_px=1920, image_height_px=1080, operator="Lab Tech A",
    )
    # Deliberately different from the nominal camera_resolution, to
    # prove the ACTUAL captured dimensions are what get stored.
    assert record.image_width_px == 1920
    assert record.image_height_px == 1080
    assert record.operator == "Lab Tech A"


def test_worked_example_100um_500px_gives_0_2():
    """Second worked example from the spec: 100 µm / 500 px = 0.2."""
    assert oc.calculate_microns_per_pixel(100.0, 500.0) == pytest.approx(0.2)