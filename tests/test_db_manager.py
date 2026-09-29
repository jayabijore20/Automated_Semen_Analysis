"""
tests/test_db_manager.py
===========================
Database integration tests for db_manager.py — migration/backward
compatibility (PART 8/Phase 6-7) and the Standard Slide vs physical
chamber save/load round trip.

These tests build real SQLite files (in a temp directory) and drive
the actual DatabaseManager class — not mocks — so they exercise the
genuine ALTER TABLE migration path, not just the in-memory dataclasses.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
from src.calibration import optical_calibration as oc

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.analysis import concentration as conc


_TEST_CONFIG = {"paths": {"outputs": {"plots": "/tmp", "tracking": "/tmp"}}}

# ── Pre-Phase-6 schema (frozen snapshot, deliberately NOT imported
#    from db_manager.py, so this test proves migration FROM that exact
#    older shape rather than testing db_manager.py against itself). ──
_PRE_PHASE6_DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, sample_id TEXT DEFAULT '', animal_id TEXT DEFAULT '',
    species TEXT DEFAULT '', breed TEXT DEFAULT '', sample_type TEXT DEFAULT '',
    collection_date TEXT DEFAULT '', operator TEXT DEFAULT '', lab_name TEXT DEFAULT '',
    chamber TEXT DEFAULT '', microscope_type TEXT DEFAULT '', magnification TEXT DEFAULT '',
    staining_type TEXT DEFAULT '', camera_resolution TEXT DEFAULT '', camera_fps TEXT DEFAULT '',
    session_type TEXT DEFAULT 'single', capture_time TEXT DEFAULT '', interval_min INTEGER DEFAULT 0,
    duration_min INTEGER DEFAULT 0, n_captures INTEGER DEFAULT 1, created_at TEXT DEFAULT '',
    notes TEXT DEFAULT '', slide_mode TEXT, sample_volume_ul REAL, dilution_factor REAL,
    number_of_counting_fields INTEGER, microns_per_pixel_x REAL, microns_per_pixel_y REAL,
    chamber_depth_um REAL, sml_minutes REAL, sml_threshold_percentage REAL,
    initial_progressive_motility REAL
);
"""
_PRE_PHASE6_DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS session_results (
    result_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, capture_index INTEGER DEFAULT 0,
    elapsed_min REAL DEFAULT 0.0, analysis_json TEXT DEFAULT '{}', report_pdf TEXT DEFAULT '',
    report_csv TEXT DEFAULT '', report_json TEXT DEFAULT '', annotated_video TEXT DEFAULT '',
    trend_report_json TEXT DEFAULT '{}', trend_pdf TEXT DEFAULT '', trend_csv TEXT DEFAULT '',
    who_score REAL DEFAULT 0.0, who_category TEXT DEFAULT '', progressive_pct REAL DEFAULT 0.0,
    viability_pct REAL DEFAULT 0.0, total_sperm INTEGER DEFAULT 0, created_at TEXT DEFAULT '',
    total_concentration_m_ml REAL, motile_concentration_m_ml REAL,
    progressive_concentration_m_ml REAL, concentration_status TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""


def _insert_named(conn: sqlite3.Connection, table: str, data: Dict[str, Any]) -> None:
    cols = ",".join(data.keys())
    qs = ",".join("?" for _ in data)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(data.values()))


def _build_legacy_db(db_path: Path) -> str:
    """Create a pre-Phase-6 schema DB with one legacy session; returns its session_id."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(_PRE_PHASE6_DDL_SESSIONS)
    conn.execute(_PRE_PHASE6_DDL_RESULTS)

    sid, rid = str(uuid.uuid4()), str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _insert_named(conn, "sessions", {
        "session_id": sid, "sample_id": "PRE_PHASE6_SAMPLE", "animal_id": "A1",
        "species": "Cattle", "breed": "Holstein", "sample_type": "Fresh Semen",
        "collection_date": "2026-01-01", "operator": "Op1", "lab_name": "Lab1",
        "chamber": "Leja Chamber", "microscope_type": "Brightfield", "magnification": "40×",
        "staining_type": "None", "camera_resolution": "1280×720", "camera_fps": "30 FPS",
        "session_type": "single", "capture_time": now, "interval_min": 0, "duration_min": 0,
        "n_captures": 1, "created_at": now, "notes": "",
        "slide_mode": "chamber", "sample_volume_ul": 5.0, "dilution_factor": 1.0,
        "number_of_counting_fields": 1, "microns_per_pixel_x": 0.5, "microns_per_pixel_y": 0.5,
        "chamber_depth_um": 20.0, "sml_minutes": None, "sml_threshold_percentage": None,
        "initial_progressive_motility": None,
    })
    _insert_named(conn, "session_results", {
        "result_id": rid, "session_id": sid, "capture_index": 0, "elapsed_min": 0.0,
        "analysis_json": json.dumps({"total_sperm": 150}), "report_pdf": "", "report_csv": "",
        "report_json": "", "annotated_video": "", "trend_report_json": "{}", "trend_pdf": "",
        "trend_csv": "", "who_score": 80.0, "who_category": "Good", "progressive_pct": 35.0,
        "viability_pct": 65.0, "total_sperm": 150, "created_at": now,
        "total_concentration_m_ml": 50.0, "motile_concentration_m_ml": 30.0,
        "progressive_concentration_m_ml": 20.0, "concentration_status": "ok",
    })
    conn.commit()
    conn.close()
    return sid


# ══════════════════════════════════════════════════════════════
# Migration / backward compatibility
# ══════════════════════════════════════════════════════════════

def test_migration_preserves_legacy_session(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_sid = _build_legacy_db(db_path)

    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    sessions = db.list_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0]["sample_id"] == "PRE_PHASE6_SAMPLE"

    detail = db.get_session_detail(legacy_sid)
    assert detail["session"]["chamber"] == "Leja Chamber"
    assert detail["results"][0]["total_sperm"] == 150


def test_migration_adds_new_columns_as_none_for_legacy_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_sid = _build_legacy_db(db_path)
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    detail = db.get_session_detail(legacy_sid)
    # New Phase-6 columns must exist and be honestly None -- never a
    # fabricated default like "" or 0.
    for key in ("analysis_method", "protocol_version", "theoretical_depth_um",
               "effective_depth_um", "depth_correction_factor",
               "optical_calibration_status"):
        assert key in detail["session"]
        assert detail["session"][key] is None

    for key in ("concentration_label", "measurement_quality_score",
               "measurement_quality_reasons", "valid_fields",
               "field_mean", "field_median", "field_sd", "field_cv_percent"):
        assert key in detail["results"][0]
        assert detail["results"][0][key] is None


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "legacy.db"
    _build_legacy_db(db_path)
    DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))
    # Second open must not error (columns already exist).
    db2 = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))
    assert len(db2.list_sessions(limit=10)) == 1


def test_new_session_coexists_with_migrated_legacy_session(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_sid = _build_legacy_db(db_path)
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    results = {"total_sperm": 300, "progressive": 55.0, "non_progressive": 15.0, "immotile": 30.0}
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "20×", "camera_resolution": "640×480",
        "image_width_px": 640, "image_height_px": 480,
        "dilution_factor": 1.0, "number_of_counting_fields": 10,
        "manual_microns_per_pixel": 0.4,
    }
    conc_result = conc.compute_concentration_metrics_from_params(results, params)
    results.update(conc_result.to_dict())
    new_sid = db.save_single_session(
        meta={"sample_id": "NEW", "chamber": "Standard Glass Slide",
             "magnification": "20×", "camera_resolution": "640×480",
             "slide_mode": "normal_slide"},
        results=results,
    )

    all_sessions = db.list_sessions(limit=10)
    assert len(all_sessions) == 2
    assert {s["session_id"] for s in all_sessions} == {legacy_sid, new_sid}


def test_delete_still_works_after_migration(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy_sid = _build_legacy_db(db_path)
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    assert db.delete_session(legacy_sid) is True
    assert db.list_sessions(limit=10) == []


# ══════════════════════════════════════════════════════════════
# Standard Slide vs physical chamber save/load round trip
# ══════════════════════════════════════════════════════════════

def _standard_slide_results_and_meta():
    results = {
        "total_sperm": 200, "progressive": 40.0, "non_progressive": 20.0, "immotile": 40.0,
    }
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "dilution_factor": 1.0, "number_of_counting_fields": 10,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    conc_result = conc.compute_concentration_metrics_from_params(results, params)
    results.update(conc_result.to_dict())
    meta = {
        "sample_id": "STDSLIDE-001", "chamber": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "slide_mode": "normal_slide", "sample_volume_ul": 8.0,
        "manual_microns_per_pixel": 0.5, "manual_effective_depth_um": 16.53,
    }
    return results, meta


def test_standard_slide_round_trip_all_fields(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()

    sid = db.save_single_session(meta=meta, results=results)
    detail = db.get_session_detail(sid)
    sess, res = detail["session"], detail["results"][0]

    assert sess["analysis_method"] == "Standard Glass Slide – CASA-Like Controlled Analysis"
    assert sess["protocol_version"] == conc.STANDARD_SLIDE_PROTOCOL.protocol_version
    assert sess["theoretical_depth_um"] == pytest.approx(16.53, abs=0.01)
    assert sess["effective_depth_um"] == pytest.approx(16.53)
    assert sess["optical_calibration_status"] == "provisional"

    assert res["concentration_status"] == "ok"
    assert res["concentration_label"] == "Estimated Concentration"  # NEVER "Validated"
    assert res["valid_fields"] == 1
    assert res["measurement_quality_score"] is not None
    reasons = json.loads(res["measurement_quality_reasons"])
    assert isinstance(reasons, list) and len(reasons) > 0

    # Full JSON blob also carries every field (belt-and-suspenders).
    aj = res["analysis_json"]
    assert aj["concentration_label"] == "Estimated Concentration"
    assert aj["theoretical_depth_um"] == pytest.approx(16.53, abs=0.01)


def test_standard_slide_label_never_validated_even_with_manual_depth(tmp_path):
    """Scientific-honesty regression test: a manually-entered effective
    depth must never promote the label to 'Validated...'."""
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()
    sid = db.save_single_session(meta=meta, results=results)
    detail = db.get_session_detail(sid)
    assert "Validated" not in detail["results"][0]["concentration_label"]
    assert detail["results"][0]["concentration_label"] == "Estimated Concentration"


def test_physical_chamber_round_trip_unaffected_by_phase6(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results = {"total_sperm": 200, "progressive": 40.0, "non_progressive": 20.0, "immotile": 40.0}
    params = {
        "slide_mode": "chamber", "chamber_name": "Makler Chamber",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "dilution_factor": 1.0, "number_of_counting_fields": 1,
        "manual_microns_per_pixel": 0.5,
    }
    conc_result = conc.compute_concentration_metrics_from_params(results, params)
    results.update(conc_result.to_dict())
    sid = db.save_single_session(
        meta={"sample_id": "MAKLER-1", "chamber": "Makler Chamber",
             "magnification": "40×", "camera_resolution": "1280×720",
             "slide_mode": "chamber", "manual_microns_per_pixel": 0.5},
        results=results,
    )
    detail = db.get_session_detail(sid)
    assert detail["session"]["analysis_method"] == "Physical Chamber – Makler Chamber"
    assert detail["results"][0]["concentration_label"] == "Measured Concentration"
    assert detail["session"]["chamber_depth_um"] == pytest.approx(10.0)


def test_view_search_and_comparison_paths_unaffected(tmp_path):
    """Standard existing-functionality smoke check (Phase 6 mandate:
    do not break view / search / comparison)."""
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()
    sid = db.save_single_session(meta=meta, results=results)

    # View
    assert db.get_session_detail(sid) is not None
    # Search
    found = db.search_by_sample_id("STDSLIDE")
    assert any(s["session_id"] == sid for s in found)
    # Comparison-style multi-fetch (list_sessions is what the
    # comparison UI multi-selects from)
    all_sessions = db.list_sessions(limit=10)
    assert len(all_sessions) == 1


# ══════════════════════════════════════════════════════════════
# Fresh database creation (isolated from migration)
# ══════════════════════════════════════════════════════════════

def test_fresh_database_creates_expected_schema(tmp_path):
    """A brand-new DB (no legacy file) must get the full current
    schema directly from _DDL_SESSIONS/_DDL_RESULTS -- this is
    distinct from the migration path (which ALTERs an existing table)
    and must be checked on its own."""
    db_path = tmp_path / "brand_new.db"
    assert not db_path.exists()

    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    session_cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    result_cols = {r[1] for r in conn.execute("PRAGMA table_info(session_results)").fetchall()}
    conn.close()

    for col in ("session_id", "sample_id", "chamber", "slide_mode",
               "protocol_version", "analysis_method", "theoretical_depth_um",
               "effective_depth_um", "depth_correction_factor",
               "optical_calibration_status"):
        assert col in session_cols, f"missing session column: {col}"

    for col in ("result_id", "session_id", "analysis_json",
               "concentration_status", "concentration_label",
               "measurement_quality_score", "measurement_quality_reasons",
               "valid_fields", "field_mean", "field_median", "field_sd",
               "field_cv_percent"):
        assert col in result_cols, f"missing session_results column: {col}"

    assert db.count_sessions() == 0
    assert db.list_sessions(limit=10) == []


def test_fresh_database_is_actually_empty(tmp_path):
    """A freshly created database must contain zero sessions -- this
    would fail loudly if _init_db ever accidentally seeded data."""
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "brand_new.db"))
    assert db.count_sessions() == 0


# ══════════════════════════════════════════════════════════════
# depth_correction_factor pathway (distinct from manual_effective_depth_um)
# ══════════════════════════════════════════════════════════════

def test_standard_slide_via_depth_correction_factor(tmp_path):
    """Effective depth can also be resolved via a validated correction
    factor instead of a directly-entered depth -- must round-trip
    correctly and must NOT claim 'Validated' either."""
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))

    results = {"total_sperm": 200, "progressive": 40.0, "non_progressive": 20.0, "immotile": 40.0}
    params = {
        "slide_mode": "normal_slide", "chamber_name": "Standard Glass Slide",
        "magnification": "40×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "manual_microns_per_pixel": 0.5,
        "depth_correction_factor": 1.1,  # theoretical_depth * 1.1, NOT a manual depth
    }
    conc_result = conc.compute_concentration_metrics_from_params(results, params)
    results.update(conc_result.to_dict())

    theoretical = conc.calculate_theoretical_depth_um(
        conc.STANDARD_SLIDE_PROTOCOL.sample_volume_ul,
        conc.calculate_coverslip_area_mm2(
            conc.STANDARD_SLIDE_PROTOCOL.coverslip_length_mm,
            conc.STANDARD_SLIDE_PROTOCOL.coverslip_width_mm,
        ),
    )
    expected_effective = theoretical * 1.1

    sid = db.save_single_session(
        meta={"sample_id": "CORR-FACTOR-001", "chamber": "Standard Glass Slide",
             "magnification": "40×", "camera_resolution": "1280×720",
             "slide_mode": "normal_slide", "depth_correction_factor": 1.1},
        results=results,
    )
    detail = db.get_session_detail(sid)
    sess, res = detail["session"], detail["results"][0]

    assert sess["theoretical_depth_um"] == pytest.approx(theoretical, abs=0.01)
    assert sess["effective_depth_um"] == pytest.approx(expected_effective, abs=0.01)
    # NOTE: save_single_session reads depth_correction_factor from
    # `results` (results.get("depth_correction_factor") -- populated
    # by StandardSlideTechnicalParams.to_dict() via
    # compute_concentration_metrics_from_params), NOT from `meta`.
    # The `meta` dict here also carries it, but that copy is unused by
    # save_single_session for this particular field -- verified by
    # reading db_manager.py directly, not assumed.
    assert sess["depth_correction_factor"] == pytest.approx(1.1)
    assert res["concentration_label"] == "Estimated Concentration"


# ══════════════════════════════════════════════════════════════
# Field statistics — explicit values, not just presence
# ══════════════════════════════════════════════════════════════

def test_field_statistics_values_for_single_field_run(tmp_path):
    """With the current single-video pipeline, exactly one field is
    ever observed per run -- field_sd and field_cv_percent must
    honestly be 0.0 (not None, not a fabricated non-zero spread),
    and field_mean/field_median must equal the single observed count."""
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()

    sid = db.save_single_session(meta=meta, results=results)
    res = db.get_session_detail(sid)["results"][0]

    assert res["valid_fields"] == 1
    assert res["field_mean"] == pytest.approx(200.0)    # total_sperm from _standard_slide_results_and_meta
    assert res["field_median"] == pytest.approx(200.0)
    assert res["field_sd"] == pytest.approx(0.0)
    assert res["field_cv_percent"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════
# Microscope calibration table (new, purely additive)
# ══════════════════════════════════════════════════════════════

def test_calibration_fresh_table_created(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(microscope_calibrations)").fetchall()}
    conn.close()
    for col in ("calibration_id", "microscope", "camera", "magnification",
               "camera_resolution", "digital_zoom", "known_distance_um",
               "measured_pixel_distance", "microns_per_pixel"):
        assert col in cols


def test_calibration_save_and_load_exact_configuration(tmp_path):
    """Test 4/5: calibration is associated with, and loadable by, the
    correct microscope/camera/magnification/resolution."""
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    record = oc.build_calibration_record(
        microscope="Pongnas C21ED4SXZA", camera="Default USB Camera",
        magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    db.save_calibration(asdict(record))

    found = db.get_calibration("Pongnas C21ED4SXZA", "Default USB Camera", "20×", "1280×720")
    assert found is not None
    assert found["microns_per_pixel"] == pytest.approx(0.40)


def test_calibration_different_configuration_not_reused(tmp_path):
    """Test 6: a different microscope configuration does not
    incorrectly reuse another configuration's calibration."""
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    record = oc.build_calibration_record(
        microscope="Pongnas C21ED4SXZA", camera="Default USB Camera",
        magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    db.save_calibration(asdict(record))

    # Different magnification
    assert db.get_calibration("Pongnas C21ED4SXZA", "Default USB Camera", "40×", "1280×720") is None
    # Different resolution
    assert db.get_calibration("Pongnas C21ED4SXZA", "Default USB Camera", "20×", "640×480") is None
    # Different microscope
    assert db.get_calibration("Other Scope", "Default USB Camera", "20×", "1280×720") is None


def test_calibration_recalibration_preserves_history(tmp_path):
    """Re-calibrating the same configuration must PRESERVE the prior
    measurement as history (task requirement: "Do not overwrite the
    only calibration record"), while `get_calibration` resolves to the
    most recent one."""
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    r1 = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    db.save_calibration(asdict(r1))
    r2 = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=200.0,
    )
    db.save_calibration(asdict(r2))

    # BOTH measurements are retained.
    history = db.get_calibration_history("A", "B", "20×", "1280×720")
    assert len(history) == 2
    assert len(db.list_calibrations()) == 2

    # The CURRENT calibration is the most recent one.
    current = db.get_calibration("A", "B", "20×", "1280×720")
    assert current["microns_per_pixel"] == pytest.approx(0.5)

    # And only one row per configuration is reported as "current".
    assert len(db.list_current_calibrations()) == 1


def test_calibration_delete(tmp_path):
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    record = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    cal_id = db.save_calibration(asdict(record))
    assert len(db.list_calibrations()) == 1
    assert db.delete_calibration(cal_id) is True
    assert db.list_calibrations() == []
    assert db.delete_calibration(cal_id) is False  # already gone


def test_calibration_table_does_not_affect_existing_sessions(tmp_path):
    """Test 7: existing session records remain intact when the
    calibration table is created/used alongside them."""
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()
    sid = db.save_single_session(meta=meta, results=results)

    record = oc.build_calibration_record(
        microscope="A", camera="B", magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    db.save_calibration(asdict(record))

    # Session must still load correctly, unaffected by the calibration table.
    detail = db.get_session_detail(sid)
    assert detail["session"]["sample_id"] == "STDSLIDE-001"
    assert len(db.list_sessions(limit=10)) == 1
    assert len(db.list_calibrations()) == 1


def test_calibration_flows_into_concentration_with_validated_status(tmp_path):
    """End-to-end: a saved calibration resolves through
    compute_concentration_metrics_from_params with VALIDATED status,
    distinct from the PROVISIONAL status a legacy table/manual value
    would produce."""
    from dataclasses import asdict
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    record = oc.build_calibration_record(
        microscope="Pongnas C21ED4SXZA", camera="Default USB Camera",
        magnification="20×", camera_resolution="1280×720",
        known_distance_um=100.0, pixel_distance=250.0,
    )
    db.save_calibration(asdict(record))
    found = db.get_calibration("Pongnas C21ED4SXZA", "Default USB Camera", "20×", "1280×720")

    results = {"total_sperm": 200, "progressive": 40.0, "non_progressive": 20.0, "immotile": 40.0}
    params = {
        "slide_mode": "chamber", "chamber_name": "Makler Chamber",
        "magnification": "20×", "camera_resolution": "1280×720",
        "image_width_px": 1280, "image_height_px": 720,
        "saved_calibration_um_per_pixel": found["microns_per_pixel"],
    }
    result = conc.compute_concentration_metrics_from_params(results, params)
    assert result.status == "ok"
    assert result.optical_calibration_status == "validated"


# ══════════════════════════════════════════════════════════════
# Calibration history, repeatability, and legacy-index migration
# ══════════════════════════════════════════════════════════════

def _save_cal(db, px, microscope="A", camera="B", magnification="20×",
              resolution="1280×720", **kw):
    from dataclasses import asdict
    rec = oc.build_calibration_record(
        microscope=microscope, camera=camera, magnification=magnification,
        camera_resolution=resolution, known_distance_um=100.0, pixel_distance=px, **kw,
    )
    return db.save_calibration(asdict(rec))


def test_calibration_history_accumulates(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    for px in (250.0, 248.0, 252.0):
        _save_cal(db, px)
    history = db.get_calibration_history("A", "B", "20×", "1280×720")
    assert len(history) == 3


def test_calibration_history_isolated_per_configuration(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    _save_cal(db, 250.0, magnification="20×")
    _save_cal(db, 500.0, magnification="40×")
    assert len(db.get_calibration_history("A", "B", "20×", "1280×720")) == 1
    assert len(db.get_calibration_history("A", "B", "40×", "1280×720")) == 1


def test_get_calibration_returns_most_recent(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    _save_cal(db, 200.0, notes="first")
    _save_cal(db, 400.0, notes="second")
    _save_cal(db, 500.0, notes="LAST")
    current = db.get_calibration("A", "B", "20×", "1280×720")
    assert current["notes"] == "LAST"
    assert current["microns_per_pixel"] == pytest.approx(0.2)


def test_calibration_repeatability_via_db(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    for px in (250.0, 248.0, 252.0):
        _save_cal(db, px)
    rep = db.get_calibration_repeatability("A", "B", "20×", "1280×720")
    assert rep["n_measurements"] == 3
    assert rep["std_dev_um_per_pixel"] is not None
    assert rep["cv_percent"] is not None


def test_calibration_repeatability_single_measurement_via_db(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    _save_cal(db, 250.0)
    rep = db.get_calibration_repeatability("A", "B", "20×", "1280×720")
    assert rep["n_measurements"] == 1
    assert rep["std_dev_um_per_pixel"] is None


def test_list_current_calibrations_one_row_per_configuration(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    _save_cal(db, 250.0, magnification="20×")
    _save_cal(db, 248.0, magnification="20×")   # repeat of same config
    _save_cal(db, 500.0, magnification="40×")   # different config
    assert len(db.list_calibrations()) == 3          # full history
    assert len(db.list_current_calibrations()) == 2  # one per configuration


def test_calibration_stores_image_dimensions_and_operator(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    _save_cal(db, 250.0, image_width_px=1280, image_height_px=720, operator="Tech A")
    current = db.get_calibration("A", "B", "20×", "1280×720")
    assert current["image_width_px"] == 1280
    assert current["image_height_px"] == 720
    assert current["operator"] == "Tech A"


def test_legacy_unique_index_migrated_to_allow_history(tmp_path):
    """A database created by the earlier (overwrite-on-save) version
    must migrate to history-preserving behaviour WITHOUT losing the
    calibration row it already had."""
    import uuid
    from datetime import datetime

    db_path = tmp_path / "legacy_cal.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
    CREATE TABLE microscope_calibrations (
        calibration_id TEXT PRIMARY KEY, microscope TEXT NOT NULL, camera TEXT NOT NULL,
        magnification TEXT NOT NULL, camera_resolution TEXT NOT NULL,
        digital_zoom TEXT NOT NULL DEFAULT 'None', known_distance_um REAL NOT NULL,
        measured_pixel_distance REAL NOT NULL, microns_per_pixel REAL NOT NULL,
        calibration_status TEXT NOT NULL DEFAULT 'validated',
        software_version TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT ''
    );""")
    conn.execute(
        "CREATE UNIQUE INDEX idx_calibration_config ON microscope_calibrations"
        "(microscope, camera, magnification, camera_resolution, digital_zoom);"
    )
    conn.execute(
        "INSERT INTO microscope_calibrations (calibration_id,microscope,camera,"
        "magnification,camera_resolution,digital_zoom,known_distance_um,"
        "measured_pixel_distance,microns_per_pixel,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), "A", "B", "20×", "1280×720", "None", 100.0, 250.0, 0.4,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    # The pre-existing calibration survived the migration.
    assert len(db.list_calibrations()) == 1

    # The index is no longer UNIQUE.
    conn2 = sqlite3.connect(str(db_path))
    idx = {r[0]: r[1] for r in conn2.execute(
        "SELECT name, \"unique\" FROM pragma_index_list('microscope_calibrations')"
    ).fetchall()}
    conn2.close()
    assert idx.get("idx_calibration_config") == 0

    # History now accumulates rather than overwriting.
    _save_cal(db, 200.0)
    assert len(db.list_calibrations()) == 2


def test_legacy_calibration_table_gains_new_columns(tmp_path):
    import uuid
    from datetime import datetime

    db_path = tmp_path / "legacy_cal2.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
    CREATE TABLE microscope_calibrations (
        calibration_id TEXT PRIMARY KEY, microscope TEXT NOT NULL, camera TEXT NOT NULL,
        magnification TEXT NOT NULL, camera_resolution TEXT NOT NULL,
        digital_zoom TEXT NOT NULL DEFAULT 'None', known_distance_um REAL NOT NULL,
        measured_pixel_distance REAL NOT NULL, microns_per_pixel REAL NOT NULL,
        calibration_status TEXT NOT NULL DEFAULT 'validated',
        software_version TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT ''
    );""")
    conn.commit()
    conn.close()

    DatabaseManager(config=_TEST_CONFIG, db_path=str(db_path))

    conn2 = sqlite3.connect(str(db_path))
    cols = {r[1] for r in conn2.execute("PRAGMA table_info(microscope_calibrations)").fetchall()}
    conn2.close()
    assert {"image_width_px", "image_height_px", "operator"} <= cols


def test_calibration_history_does_not_disturb_sessions(tmp_path):
    db = DatabaseManager(config=_TEST_CONFIG, db_path=str(tmp_path / "fresh.db"))
    results, meta = _standard_slide_results_and_meta()
    sid = db.save_single_session(meta=meta, results=results)
    for px in (250.0, 248.0, 252.0):
        _save_cal(db, px)
    assert db.get_session_detail(sid)["session"]["sample_id"] == "STDSLIDE-001"
    assert len(db.list_sessions(limit=10)) == 1
    assert len(db.list_calibrations()) == 3