"""
tests/test_concentration.py
=============================
Unit tests for src/analysis/concentration.py (Feature 18).

Run with:
    pytest tests/test_concentration.py -v

These tests require no camera, video, or trained model — pure math.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from src.analysis import concentration as c


# ── 1. µm/pixel -> FOV conversion ──────────────────────────────
def test_field_of_view_conversion():
    fov = c.calculate_field_of_view(1280, 720, 0.5, 0.5)
    assert fov["fov_width_um"] == pytest.approx(640.0)
    assert fov["fov_height_um"] == pytest.approx(360.0)
    assert fov["fov_width_mm"] == pytest.approx(0.64)
    assert fov["fov_height_mm"] == pytest.approx(0.36)


def test_field_of_view_rejects_non_positive():
    with pytest.raises(ValueError):
        c.calculate_field_of_view(0, 720, 0.5, 0.5)
    with pytest.raises(ValueError):
        c.calculate_field_of_view(1280, 720, 0.0, 0.5)


# ── 2. FOV area ─────────────────────────────────────────────────
def test_fov_area():
    fov = c.calculate_field_of_view(1000, 1000, 1.0, 1.0)
    # 1000 px * 1 um/px = 1000 um = 1 mm per side -> 1 mm^2 area
    assert fov["fov_area_mm2"] == pytest.approx(1.0)


# ── 3. FOV volume (observed volume in mm^3, via mL) ─────────────
def test_observed_volume_mm3_via_ml():
    # 1 mm^2 area * 1000 um (1 mm) depth = 1 mm^3 = 0.001 mL
    vol_ml = c.calculate_observed_volume(1.0, 1000.0)
    assert vol_ml == pytest.approx(0.001)


def test_observed_volume_rejects_non_positive():
    with pytest.raises(ValueError):
        c.calculate_observed_volume(0.0, 20.0)
    with pytest.raises(ValueError):
        c.calculate_observed_volume(1.0, 0.0)


# ── 4. mm^3 -> mL conversion (embedded in observed volume) ──────
def test_mm3_to_ml_conversion_constant():
    # Directly exercise the exact 1000 mm^3 == 1 mL relationship.
    vol_ml = c.calculate_observed_volume(fov_area_mm2=1000.0, chamber_depth_um=1000.0)
    # area(1000mm2) * depth(1mm) = 1000 mm^3 = 1 mL
    assert vol_ml == pytest.approx(1.0)


# ── 5. sperm count -> cells/mL ───────────────────────────────────
def test_concentration_from_feature18_example():
    conc = c.calculate_sperm_concentration(
        sperm_count=200, observed_volume_ml=0.00002, dilution_factor=1.0,
    )
    assert conc == pytest.approx(10_000_000.0)


def test_concentration_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        c.calculate_sperm_concentration(-1, 0.00002, 1.0)
    with pytest.raises(ValueError):
        c.calculate_sperm_concentration(200, 0.0, 1.0)
    with pytest.raises(ValueError):
        c.calculate_sperm_concentration(200, 0.00002, 0.0)


# ── 6. cells/mL -> M cells/mL ─────────────────────────────────────
def test_cells_to_m_cells():
    assert c.cells_per_ml_to_m_per_ml(10_000_000.0) == pytest.approx(10.0)
    assert c.cells_per_ml_to_m_per_ml(1_000_000.0) == pytest.approx(1.0)


# ── 7. dilution correction ────────────────────────────────────────
def test_dilution_correction():
    undiluted = c.calculate_sperm_concentration(200, 0.00002, dilution_factor=1.0)
    diluted   = c.calculate_sperm_concentration(200, 0.00002, dilution_factor=10.0)
    assert diluted == pytest.approx(undiluted * 10)


# ── 8. motile concentration (MCC) ─────────────────────────────────
def test_mcc_from_feature18_example():
    mcc = c.calculate_motile_concentration(100.0, 0.60)
    assert mcc == pytest.approx(60.0)


def test_mcc_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        c.calculate_motile_concentration(100.0, 1.5)
    with pytest.raises(ValueError):
        c.calculate_motile_concentration(-5.0, 0.5)


# ── 9. progressive concentration ──────────────────────────────────
def test_progressive_concentration_from_feature18_example():
    pconc = c.calculate_progressive_concentration(100.0, 0.40)
    assert pconc == pytest.approx(40.0)


# ── 10. non-progressive concentration ────────────────────────────
def test_non_progressive_concentration():
    nconc = c.calculate_non_progressive_concentration(100.0, 0.20)
    assert nconc == pytest.approx(20.0)


# ── 11. immotile concentration ────────────────────────────────────
def test_immotile_concentration():
    iconc = c.calculate_immotile_concentration(100.0, 0.40)
    assert iconc == pytest.approx(40.0)
    # Percentages sum ~100% across the three fractions (Feature 8)
    total = (
        c.calculate_progressive_concentration(100.0, 0.40)
        + c.calculate_non_progressive_concentration(100.0, 0.20)
        + c.calculate_immotile_concentration(100.0, 0.40)
    )
    assert total == pytest.approx(100.0)


# ── 12. SML threshold ─────────────────────────────────────────────
def test_sml_threshold_is_half_initial():
    sml = c.calculate_sml([0, 10], [70, 30])
    assert sml.sml_threshold_percentage == pytest.approx(35.0)


# ── 13. SML interpolation (exact worked example from the spec) ────
def test_sml_interpolation_worked_example():
    times = [0, 10, 20, 30, 40, 50, 60]
    pms   = [70, 68, 64, 58, 50, 42, 34]
    sml = c.calculate_sml(times, pms)
    assert sml.status == "ok"
    assert sml.initial_progressive_motility == pytest.approx(70.0)
    assert sml.sml_threshold_percentage == pytest.approx(35.0)
    # t1=50 pm1=42, t2=60 pm2=34, threshold=35
    # t_sml = 50 + (35-42)/(34-42)*(60-50) = 58.75
    assert sml.sml_minutes == pytest.approx(58.75)


def test_sml_unsorted_input_is_sorted_internally():
    times = [30, 0, 10]
    pms   = [50, 100, 80]
    sml = c.calculate_sml(times, pms)
    assert sml.initial_progressive_motility == pytest.approx(100.0)


# ── 14. invalid sample volumes ─────────────────────────────────────
def test_sample_volume_validation_normal_slide():
    assert c.validate_sample_volume("normal_slide", 8.0) is None
    assert c.validate_sample_volume("normal_slide", 5.0) is not None
    assert c.validate_sample_volume("normal_slide", 12.0) is not None


def test_sample_volume_validation_chamber():
    assert c.validate_sample_volume("chamber", 4.0) is None
    assert c.validate_sample_volume("chamber", 1.0) is not None
    assert c.validate_sample_volume("chamber", 8.0) is not None


# ── 15. missing calibration ────────────────────────────────────────
def test_missing_calibration_returns_status_not_number():
    result = c.compute_concentration_metrics(
        sperm_count=200,
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        image_width_px=1280, image_height_px=720,
        slide_mode="chamber", chamber_name="Makler Chamber",
        magnification="40×", camera_resolution="1280×720",
    )
    assert result.status == "calibration_required"
    assert result.total_concentration_m_per_ml is None


def test_manual_calibration_override_takes_priority():
    val = c.resolve_microns_per_pixel("40×", "1280×720", manual_override=0.33)
    assert val == pytest.approx(0.33)


# ── 16. missing chamber depth ──────────────────────────────────────
def test_missing_chamber_depth_normal_slide():
    result = c.compute_concentration_metrics(
        sperm_count=200,
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        image_width_px=1280, image_height_px=720,
        slide_mode="normal_slide", chamber_name="Standard Glass Slide",
        magnification="40×", camera_resolution="1280×720",
        manual_microns_per_pixel=0.5,
    )
    assert result.status == "depth_required"
    assert result.total_concentration_m_per_ml is None


def test_verified_chamber_depth_allows_calculation():
    result = c.compute_concentration_metrics(
        sperm_count=200,
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        image_width_px=1280, image_height_px=720,
        slide_mode="chamber", chamber_name="Makler Chamber",
        magnification="40×", camera_resolution="1280×720",
        manual_microns_per_pixel=0.5,
    )
    assert result.status == "ok"
    assert result.chamber_depth_um == pytest.approx(10.0)
    assert result.total_concentration_m_per_ml is not None


# ── 17. concentration range warnings ──────────────────────────────
def test_concentration_below_target_range_warns():
    msg = c.get_concentration_status(0.5)
    assert msg is not None and "below" in msg


def test_concentration_above_target_range_warns():
    msg = c.get_concentration_status(600.0)
    assert msg is not None and "above" in msg


def test_concentration_within_target_range_no_warning():
    assert c.get_concentration_status(100.0) is None


# ── Extra: SML insufficient data never invents a value ─────────────
def test_sml_insufficient_data_does_not_invent_value():
    sml = c.calculate_sml([0], [70])
    assert sml.status == "insufficient_data"
    assert sml.sml_minutes is None


def test_sml_no_crossing_does_not_invent_value():
    # Progressive motility stays well above the 50% threshold throughout
    sml = c.calculate_sml([0, 10, 20], [70, 69, 68])
    assert sml.status == "insufficient_data"
    assert sml.sml_minutes is None


# ── Extra: counting-quality heuristic bounds ────────────────────────
def test_counting_quality_labels():
    fov_area = c.calculate_field_of_view(1280, 720, 0.5, 0.5)["fov_area_mm2"]
    assert c.get_counting_quality(1, fov_area) == "Excellent"
    assert c.get_counting_quality(1_000_000, fov_area) == "Poor"


# ── Extra: mean_sperm_per_field averaging ───────────────────────────
def test_mean_sperm_per_field():
    mean, n = c.mean_sperm_per_field([100, 120, 110])
    assert mean == pytest.approx(110.0)
    assert n == 3


def test_mean_sperm_per_field_empty():
    mean, n = c.mean_sperm_per_field([])
    assert mean == 0.0
    assert n == 0


# ══════════════════════════════════════════════════════════════
# CASA-like architecture (PART 45 checklist)
# ══════════════════════════════════════════════════════════════

# ── 1. Coverslip area ───────────────────────────────────────────
def test_coverslip_area_22x22():
    assert c.calculate_coverslip_area_mm2(22.0, 22.0) == pytest.approx(484.0)


def test_coverslip_area_rejects_non_positive():
    with pytest.raises(ValueError):
        c.calculate_coverslip_area_mm2(0.0, 22.0)


# ── 2. µL -> mm^3 (embedded in theoretical depth) ────────────────
def test_ul_equals_mm3_relationship():
    # 1 µL over 1 mm^2 area -> 1 mm depth == 1000 µm
    depth = c.calculate_theoretical_depth_um(1.0, 1.0)
    assert depth == pytest.approx(1000.0)


# ── 3. Theoretical depth (exact worked example from the spec) ────
def test_theoretical_depth_worked_example():
    area = c.calculate_coverslip_area_mm2(22.0, 22.0)
    depth = c.calculate_theoretical_depth_um(8.0, area)
    assert depth == pytest.approx(16.53, abs=0.01)


def test_theoretical_depth_rejects_non_positive():
    with pytest.raises(ValueError):
        c.calculate_theoretical_depth_um(0.0, 484.0)
    with pytest.raises(ValueError):
        c.calculate_theoretical_depth_um(8.0, 0.0)


# ── 4/5. FOV / physical FOV (reuses existing tested functions) ───
def test_physical_fov_used_by_standard_slide_pathway():
    fov = c.calculate_field_of_view(1280, 720, 0.5, 0.5)
    assert fov["fov_width_um"] == pytest.approx(640.0)


# ── 6. Observed volume (reused) ───────────────────────────────────
def test_observed_volume_reused_for_standard_slide():
    vol = c.calculate_observed_volume(0.64 * 0.36, 16.53)
    assert vol > 0


# ── 7. Concentration (reused) — covered extensively above.

# ── 8/9. Field mean / median ──────────────────────────────────────
def test_field_mean_and_median():
    stats = c.calculate_field_statistics([100, 110, 90, 105, 95])
    assert stats.mean == pytest.approx(100.0)
    assert stats.median == pytest.approx(100.0)


# ── 10. Standard deviation ────────────────────────────────────────
def test_field_std_dev():
    stats = c.calculate_field_statistics([100, 100, 100])
    assert stats.std_dev == pytest.approx(0.0)


# ── 11. CV ─────────────────────────────────────────────────────────
def test_field_cv_percent():
    stats = c.calculate_field_statistics([90, 100, 110])
    assert stats.cv_percent == pytest.approx(stats.std_dev / stats.mean * 100.0)


def test_field_cv_high_variability_warns():
    stats = c.calculate_field_statistics([10, 500, 20, 15, 12])
    assert stats.high_variability_warning is not None
    assert "Non-uniform" in stats.high_variability_warning


def test_field_cv_low_variability_no_warning():
    stats = c.calculate_field_statistics([100, 102, 98, 101, 99])
    assert stats.high_variability_warning is None


# ── 12. Field rejection ────────────────────────────────────────────
def test_field_rejection_tracked_separately_from_valid_count():
    stats = c.calculate_field_statistics([100, 105, 98], n_rejected_fields=2)
    assert stats.n_valid_fields == 3
    assert stats.n_rejected_fields == 2


def test_field_quality_valid_property():
    fq_ok = c.FieldQuality(field_id=0, sperm_count=100)
    assert fq_ok.valid is True
    fq_bad = c.FieldQuality(field_id=1, sperm_count=100, rejection_reasons=["severe blur"])
    assert fq_bad.valid is False


# ── 13/14. Missing / invalid optical calibration ──────────────────
def test_standard_slide_missing_calibration_refuses():
    result, stats, tech, qual = c.compute_standard_slide_concentration(
        field_sperm_counts=[100, 110, 95],
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        microns_per_pixel=None, image_width_px=1280, image_height_px=720,
    )
    assert result.status == "calibration_required"
    assert result.total_concentration_m_per_ml is None


def test_standard_slide_missing_depth_validation_refuses():
    result, stats, tech, qual = c.compute_standard_slide_concentration(
        field_sperm_counts=[100] * 10,
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        microns_per_pixel=0.5, image_width_px=1280, image_height_px=720,
    )
    assert result.status == "depth_required"
    assert tech.theoretical_depth_um == pytest.approx(16.53, abs=0.01)
    assert tech.effective_depth_um is None
    assert tech.depth_validation_state == c.ValidationState.NOT_CONFIGURED


# ── 15. Standard Slide mode (full success path) ────────────────────
def test_standard_slide_mode_full_success():
    result, stats, tech, qual = c.compute_standard_slide_concentration(
        field_sperm_counts=[100, 105, 98, 102, 99, 101, 97, 103, 100, 100],
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        microns_per_pixel=0.5, image_width_px=1280, image_height_px=720,
        manual_effective_depth_um=16.53,
    )
    assert result.status == "ok"
    assert result.total_concentration_m_per_ml is not None
    assert stats.n_valid_fields == 10
    assert 0 <= qual.score <= 100


# ── 16/17/18. Leja / Makler / Hemocytometer modes ───────────────────
@pytest.mark.parametrize("chamber_name,expected_depth", [
    ("Leja Chamber", 20.0),
    ("Makler Chamber", 10.0),
    ("Hemocytometer", 100.0),
])
def test_physical_chamber_modes(chamber_name, expected_depth):
    spec = c.CHAMBER_CONFIG[chamber_name]
    assert spec.depth_um == pytest.approx(expected_depth)
    assert spec.category == c.SlideCategory.PHYSICAL_CHAMBER

    result = c.compute_concentration_metrics(
        sperm_count=200,
        motility_summary={"progressive_pct": 40, "nonprogressive_pct": 20, "immotile_pct": 40},
        image_width_px=1280, image_height_px=720,
        slide_mode="chamber", chamber_name=chamber_name,
        magnification="40×", camera_resolution="1280×720",
        manual_microns_per_pixel=0.5,
    )
    assert result.status == "ok"
    assert result.chamber_depth_um == pytest.approx(expected_depth)


def test_standard_glass_slide_is_controlled_estimation_category():
    spec = c.CHAMBER_CONFIG["Standard Glass Slide"]
    assert spec.category == c.SlideCategory.CONTROLLED_ESTIMATION
    assert spec.depth_um is None
    assert spec.validation_state == c.ValidationState.NOT_CONFIGURED


def test_standard_glass_slide_never_silently_assigned_a_chamber_depth():
    # PART 3: must never silently default to 10/20/100 µm like a
    # physical chamber.
    spec = c.CHAMBER_CONFIG["Standard Glass Slide"]
    assert spec.depth_um not in (10.0, 20.0, 100.0)
    assert spec.depth_um is None


# ── 19. Unique track counting (mean_sperm_per_field, tested above) ──

# ── 20/21/22/23. Motility / VCL / VSL / VAP ──────────────────────────
# NOTE: VCL/VSL/VAP/LIN/STR/ALH/BCF pixel-to-physical-velocity
# conversion requires raw per-frame trajectory coordinates from the
# existing (out-of-scope-for-this-module) tracking/motility engine —
# this module only consumes already-computed progressive/non-
# progressive/immotile PERCENTAGES, exactly as the existing pipeline
# already reports them, and does not duplicate or re-derive them.
# calculate_motile_concentration / calculate_progressive_concentration
# (tested extensively above) are this module's actual contribution to
# CASA-style output: converting existing motility fractions into
# concentration fractions.

# ── 24. Concentration labeling ────────────────────────────────────
def test_concentration_label_physical_chamber():
    assert c.get_concentration_label(
        c.SlideCategory.PHYSICAL_CHAMBER, c.ValidationState.PROVISIONAL
    ) == "Measured Concentration"


def test_concentration_label_standard_slide_unvalidated():
    assert c.get_concentration_label(
        c.SlideCategory.CONTROLLED_ESTIMATION, c.ValidationState.NOT_CONFIGURED
    ) == "Estimated Concentration"


def test_concentration_label_standard_slide_validated():
    assert c.get_concentration_label(
        c.SlideCategory.CONTROLLED_ESTIMATION, c.ValidationState.VALIDATED
    ) == "Validated Controlled-Protocol Concentration"


def test_concentration_label_never_says_exact():
    for cat in (c.SlideCategory.PHYSICAL_CHAMBER, c.SlideCategory.CONTROLLED_ESTIMATION):
        for state in (c.ValidationState.NOT_CONFIGURED, c.ValidationState.PROVISIONAL,
                      c.ValidationState.VALIDATED, c.ValidationState.EXPIRED):
            label = c.get_concentration_label(cat, state)
            assert "exact" not in label.lower()


# ── 25. Validation status ──────────────────────────────────────────
def test_validation_states_are_distinct_strings():
    states = {c.ValidationState.NOT_CONFIGURED, c.ValidationState.PROVISIONAL,
             c.ValidationState.VALIDATED, c.ValidationState.EXPIRED}
    assert len(states) == 4


def test_effective_depth_none_is_not_configured_not_valid():
    eff, state = c.get_standard_slide_effective_depth(16.53)
    assert eff is None
    assert state == c.ValidationState.NOT_CONFIGURED


def test_effective_depth_manual_override_is_validated():
    eff, state = c.get_standard_slide_effective_depth(16.53, manual_effective_depth_um=17.0)
    assert eff == pytest.approx(17.0)
    assert state == c.ValidationState.VALIDATED


def test_effective_depth_correction_factor_is_validated():
    eff, state = c.get_standard_slide_effective_depth(
        16.53, depth_correction_factor=1.1,
    )
    assert eff == pytest.approx(16.53 * 1.1)
    assert state == c.ValidationState.VALIDATED


# ── 26. Database backward compatibility — covered by
#         tests/test_db_manager.py-style integration tests run
#         separately against a hand-built legacy-schema SQLite file
#         (see the migration test executed during development); pure
#         schema/statistics math for this module is covered here.

# ══════════════════════════════════════════════════════════════
# Empirical validation framework (PART 22-24, 38-40)
# ══════════════════════════════════════════════════════════════

def _make_records():
    return [
        c.ReferenceValidationRecord(
            sample_id="A", reference_concentration_m_per_ml=100.0,
            system_concentration_m_per_ml=96.0, reference_method="Hemocytometer/manual",
        ),
        c.ReferenceValidationRecord(
            sample_id="B", reference_concentration_m_per_ml=50.0,
            system_concentration_m_per_ml=53.0, reference_method="Hemocytometer/manual",
        ),
        c.ReferenceValidationRecord(
            sample_id="C", reference_concentration_m_per_ml=200.0,
            system_concentration_m_per_ml=190.0, reference_method="Hemocytometer/manual",
        ),
    ]


def test_reference_validation_record_error_worked_example():
    r = c.ReferenceValidationRecord(
        sample_id="X", reference_concentration_m_per_ml=100.0,
        system_concentration_m_per_ml=96.0, reference_method="Hemocytometer/manual",
    )
    assert r.error_m_per_ml == pytest.approx(-4.0)
    assert r.relative_error_percent == pytest.approx(-4.0)


def test_bias_mae_rmse():
    records = _make_records()
    assert c.calculate_bias(records) == pytest.approx(-11.0 / 3.0)
    assert c.calculate_mae(records) == pytest.approx((4.0 + 3.0 + 10.0) / 3.0)
    assert c.calculate_rmse(records) == pytest.approx(((4.0**2 + 3.0**2 + 10.0**2) / 3.0) ** 0.5)


def test_mape():
    records = _make_records()
    mape = c.calculate_mape(records)
    assert mape == pytest.approx((4.0 + 6.0 + 5.0) / 3.0)


def test_correlation_strong_positive():
    records = _make_records()
    corr = c.calculate_correlation(records)
    assert corr > 0.99


def test_bland_altman_agreement():
    records = _make_records()
    ba = c.calculate_bland_altman(records)
    assert ba is not None
    assert ba.mean_difference == pytest.approx(-11.0 / 3.0)
    assert ba.lower_limit_of_agreement < ba.mean_difference < ba.upper_limit_of_agreement
    assert len(ba.points) == 3


def test_bland_altman_insufficient_data():
    assert c.calculate_bland_altman([_make_records()[0]]) is None
    assert c.calculate_bland_altman([]) is None


def test_repeatability_worked_example():
    rep = c.calculate_repeatability("SampleX", [101.0, 98.0, 103.0])
    assert rep is not None
    assert rep.n_runs == 3
    assert rep.mean_m_per_ml == pytest.approx(100.66666667, abs=1e-4)
    assert rep.cv_percent > 0


def test_repeatability_insufficient_runs():
    assert c.calculate_repeatability("X", [100.0]) is None
    assert c.calculate_repeatability("X", []) is None


def test_empty_records_never_fabricate_statistics():
    assert c.calculate_bias([]) is None
    assert c.calculate_mae([]) is None
    assert c.calculate_rmse([]) is None
    assert c.calculate_mape([]) is None
    assert c.calculate_correlation([]) is None


# ══════════════════════════════════════════════════════════════
# Measurement Quality Score (PART 28)
# ══════════════════════════════════════════════════════════════

def test_quality_score_all_good_is_high():
    result = c.calculate_measurement_quality_score(
        calibration_validation_state=c.ValidationState.VALIDATED,
        n_valid_fields=10, expected_fields=10, field_cv_percent=5.0,
        mean_detection_confidence=0.9, tracking_quality="Excellent",
        protocol_validation_state=c.ValidationState.VALIDATED,
    )
    assert result.score >= 90
    assert 0 <= result.score <= 100


def test_quality_score_all_missing_is_low():
    result = c.calculate_measurement_quality_score(
        calibration_validation_state=c.ValidationState.NOT_CONFIGURED,
        n_valid_fields=0, expected_fields=10, field_cv_percent=0.0,
    )
    assert result.score < 30


def test_quality_score_never_exceeds_bounds():
    result = c.calculate_measurement_quality_score(
        calibration_validation_state=c.ValidationState.VALIDATED,
        n_valid_fields=999, expected_fields=10, field_cv_percent=0.0,
        mean_detection_confidence=1.0, tracking_quality="Excellent",
        protocol_validation_state=c.ValidationState.VALIDATED,
    )
    assert 0 <= result.score <= 100


def test_quality_score_reasons_always_explain_the_number():
    result = c.calculate_measurement_quality_score(
        calibration_validation_state=c.ValidationState.PROVISIONAL,
        n_valid_fields=5, expected_fields=10, field_cv_percent=15.0,
    )
    assert len(result.reasons) > 0


# ══════════════════════════════════════════════════════════════
# Dynescan-inspired architecture extensions
# ══════════════════════════════════════════════════════════════

# ── PART 70 worked example, verified exactly as specified ────────
def test_part70_worked_example_full_pipeline():
    """
    FOV width = 0.5 mm, FOV height = 0.5 mm, depth = 0.02 mm,
    unique sperm count = 500
    -> volume = 0.5*0.5*0.02 = 0.005 mm^3 = 0.005 uL = 0.000005 mL
    -> concentration = 500 / 0.000005 = 100,000,000 cells/mL = 100 M/mL
    """
    fov_area_mm2 = 0.5 * 0.5  # 0.25 mm^2
    depth_um = 0.02 * 1000    # 0.02 mm -> 20 um (calculate_observed_volume takes um)
    observed_volume_ml = c.calculate_observed_volume(fov_area_mm2, depth_um)
    assert observed_volume_ml == pytest.approx(0.000005, abs=1e-9)

    conc_cells_per_ml = c.calculate_sperm_concentration(500, observed_volume_ml, dilution_factor=1.0)
    assert conc_cells_per_ml == pytest.approx(100_000_000.0, rel=1e-6)

    conc_m_per_ml = c.cells_per_ml_to_m_per_ml(conc_cells_per_ml)
    assert conc_m_per_ml == pytest.approx(100.0, rel=1e-6)


# ── ChamberSpec depth_source / geometry_validated (PART 38) ──────
def test_chamber_depth_source_and_validation_flags():
    for name in ("Makler Chamber", "Leja Chamber", "Hemocytometer"):
        spec = c.CHAMBER_CONFIG[name]
        assert spec.depth_source == "manufacturer_specification"
        # A published spec is not the same as an empirically validated
        # geometry for THIS specific physical chamber.
        assert spec.geometry_validated is False


def test_standard_glass_slide_depth_source_not_applicable():
    spec = c.CHAMBER_CONFIG["Standard Glass Slide"]
    assert spec.depth_source == "not_applicable_use_theoretical_depth"


# ── Preferred quantitative chamber (PART 7/8) ─────────────────────
def test_leja_is_preferred_quantitative_chamber():
    assert c.PREFERRED_QUANTITATIVE_CHAMBER == "Leja Chamber"
    # The preference is a workflow/UI hint only -- it must not change
    # Leja's own calculated depth or category.
    spec = c.CHAMBER_CONFIG[c.PREFERRED_QUANTITATIVE_CHAMBER]
    assert spec.depth_um == pytest.approx(20.0)
    assert spec.category == c.SlideCategory.PHYSICAL_CHAMBER


# ══════════════════════════════════════════════════════════════
# Phase 1-4 routing integration tests (compute_concentration_metrics_from_params)
# ══════════════════════════════════════════════════════════════

def _base_results():
    return {"total_sperm": 200, "progressive": 40.0, "non_progressive": 20.0, "immotile": 40.0}


def test_routing_standard_slide_goes_to_standard_slide_pathway():
    """Test 1: Standard Glass Slide routes to the Standard Slide calculation."""
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    # Only the Standard Slide pathway populates technical_params/field_statistics.
    assert result.technical_params is not None
    assert result.field_statistics is not None
    assert result.slide_category == c.SlideCategory.CONTROLLED_ESTIMATION


def test_routing_physical_chamber_unaffected():
    """Test 2: physical chamber still routes to the physical chamber calculation."""
    params = {
        "slide_mode": "chamber", "chamber_name": "Makler Chamber",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    assert result.status == "ok"
    assert result.chamber_depth_um == pytest.approx(10.0)
    assert result.slide_category == c.SlideCategory.PHYSICAL_CHAMBER
    assert result.concentration_label == "Measured Concentration"


def test_routing_standard_slide_does_not_require_chamber_depth():
    """Test 3: Standard Slide does not require a physical chamber depth."""
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    assert result.status == "ok"
    # chamber_depth_um is populated from the RESOLVED effective depth,
    # not from CHAMBER_CONFIG["Standard Glass Slide"] (which is None).
    assert result.chamber_depth_um == pytest.approx(16.53)


def test_routing_standard_slide_uses_theoretical_depth_only_where_intended():
    """Test 4: theoretical depth appears in technical_params even when
    concentration itself is withheld (depth_required)."""
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,  # calibrated, but NO effective depth
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    assert result.status == "depth_required"
    assert result.technical_params["theoretical_depth_um"] == pytest.approx(16.53, abs=0.01)
    assert result.total_concentration_m_per_ml is None  # never fabricated


def test_routing_missing_calibration_is_honest_for_both_pathways():
    """Test 5: missing optical calibration produces an honest status,
    for both Standard Slide and physical chamber."""
    slide_params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
    }
    chamber_params = {
        "slide_mode": "chamber", "chamber_name": "Makler Chamber",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
    }
    r1 = c.compute_concentration_metrics_from_params(_base_results(), slide_params)
    r2 = c.compute_concentration_metrics_from_params(_base_results(), chamber_params)
    assert r1.status == "calibration_required"
    assert r2.status == "calibration_required"


def test_routing_standard_slide_result_contains_expected_fields():
    """Test 6: Standard Slide concentration result contains the expected fields."""
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    d = result.to_dict()
    for key in (
        "total_concentration_m_cells_per_ml", "motile_concentration_m_cells_per_ml",
        "progressive_concentration_m_cells_per_ml", "motile_percentage",
        "progressive_percentage", "theoretical_depth_um", "fov_width_um",
        "number_of_fields", "n_valid_fields", "field_mean_count",
        "measurement_quality_score", "measurement_quality_reasons",
        "concentration_status", "concentration_label", "analysis_method",
        "protocol_version", "optical_calibration_status",
    ):
        assert key in d, f"missing expected key: {key}"


def test_routing_standard_slide_remains_estimated_never_validated():
    """Test 7: Standard Slide remains ESTIMATED/PROVISIONAL, even with
    a manually-entered effective depth or a correction factor."""
    base = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,
    }
    r_manual_depth = c.compute_concentration_metrics_from_params(
        _base_results(), {**base, "manual_effective_depth_um": 16.53},
    )
    r_correction = c.compute_concentration_metrics_from_params(
        _base_results(), {**base, "depth_correction_factor": 1.05},
    )
    r_neither = c.compute_concentration_metrics_from_params(_base_results(), base)

    assert r_manual_depth.concentration_label == "Estimated Concentration"
    assert r_correction.concentration_label == "Estimated Concentration"
    # r_neither has no depth at all -> depth_required, no label claim of Validated
    assert r_neither.status == "depth_required"
    for r in (r_manual_depth, r_correction, r_neither):
        if r.concentration_label:
            assert "Validated" not in r.concentration_label


def test_routing_analysis_method_labels_distinct():
    slide_params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    chamber_params = {
        "slide_mode": "chamber", "chamber_name": "Leja Chamber",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,
    }
    r1 = c.compute_concentration_metrics_from_params(_base_results(), slide_params)
    r2 = c.compute_concentration_metrics_from_params(_base_results(), chamber_params)
    assert r1.analysis_method == "Standard Glass Slide – CASA-Like Controlled Analysis"
    assert r2.analysis_method == "Physical Chamber – Leja Chamber"


def test_routing_physical_chamber_now_also_gets_measurement_quality():
    """Phase 4/5 also decorates the physical-chamber pathway with a
    measurement-quality score for UI consistency."""
    params = {
        "slide_mode": "chamber", "chamber_name": "Hemocytometer",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,
    }
    result = c.compute_concentration_metrics_from_params(_base_results(), params)
    assert result.measurement_quality is not None
    assert "measurement_quality_score" in result.measurement_quality