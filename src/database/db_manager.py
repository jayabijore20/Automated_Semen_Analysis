"""
src/database/db_manager.py
============================
Phase 5 · Local Database Manager

Provides a SQLite-backed persistent store for all semen analysis sessions.
Uses only the Python standard library (sqlite3) — no ORM dependency.

Schema
------
Table: sessions
    session_id          TEXT PRIMARY KEY   (UUID4)
    sample_id           TEXT
    animal_id           TEXT
    species             TEXT
    breed               TEXT
    sample_type         TEXT
    collection_date     TEXT
    operator            TEXT
    lab_name            TEXT
    chamber             TEXT
    microscope_type     TEXT
    magnification       TEXT
    staining_type       TEXT
    camera_resolution   TEXT
    camera_fps          TEXT
    session_type        TEXT               ('single' | 'monitoring')
    capture_time        TEXT               ISO timestamp
    interval_min        INTEGER            (monitoring only, else 0)
    duration_min        INTEGER            (monitoring only, else 0)
    n_captures          INTEGER
    created_at          TEXT               ISO timestamp (auto)
    notes               TEXT

Table: session_results
    result_id           TEXT PRIMARY KEY   (UUID4)
    session_id          TEXT REFERENCES sessions(session_id)
    capture_index       INTEGER            (0-based)
    elapsed_min         REAL
    analysis_json       TEXT               (full run_video result dict, JSON)
    report_pdf          TEXT               (absolute path or "")
    report_csv          TEXT
    report_json         TEXT
    annotated_video     TEXT
    trend_report_json   TEXT               (monitoring sessions only)
    trend_pdf           TEXT
    trend_csv           TEXT
    who_score           REAL
    who_category        TEXT
    progressive_pct     REAL
    viability_pct       REAL
    total_sperm         INTEGER
    created_at          TEXT

Public API
----------
    from src.database.db_manager import DatabaseManager

    db = DatabaseManager(config)

    # Save a single-analysis session
    session_id = db.save_single_session(
        meta=analysis_meta,
        results=pipeline_results,
        report_paths={"pdf":"...","csv":"...","json":"..."},
        annotated_video="...",
    )

    # Save a completed monitoring session
    session_id = db.save_monitoring_session(
        meta=analysis_meta,
        monitor_session=mon_session,
        trend_report=trend_report,
        monitoring_report_paths={"pdf":"...","csv":"...","json":"..."},
    )

    # Query
    rows = db.search_by_sample_id("BVS-001")
    rows = db.search_by_animal_id("BULL-HF-007")
    all  = db.list_sessions(limit=50, offset=0)
    det  = db.get_session_detail(session_id)
    comp = db.get_sessions_for_comparison([sid1, sid2, sid3])

    # Delete
    db.delete_session(session_id)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Schema DDL ────────────────────────────────────────────────
_DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    sample_id           TEXT NOT NULL DEFAULT '',
    animal_id           TEXT NOT NULL DEFAULT '',
    species             TEXT NOT NULL DEFAULT '',
    breed               TEXT NOT NULL DEFAULT '',
    sample_type         TEXT NOT NULL DEFAULT '',
    collection_date     TEXT NOT NULL DEFAULT '',
    operator            TEXT NOT NULL DEFAULT '',
    lab_name            TEXT NOT NULL DEFAULT '',
    chamber             TEXT NOT NULL DEFAULT '',
    microscope_type     TEXT NOT NULL DEFAULT '',
    magnification       TEXT NOT NULL DEFAULT '',
    staining_type       TEXT NOT NULL DEFAULT '',
    camera_resolution   TEXT NOT NULL DEFAULT '',
    camera_fps          TEXT NOT NULL DEFAULT '',
    session_type        TEXT NOT NULL DEFAULT 'single',
    capture_time        TEXT NOT NULL DEFAULT '',
    interval_min        INTEGER NOT NULL DEFAULT 0,
    duration_min        INTEGER NOT NULL DEFAULT 0,
    n_captures          INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT '',
    notes               TEXT NOT NULL DEFAULT ''
);
"""

_DDL_RESULTS = """
CREATE TABLE IF NOT EXISTS session_results (
    result_id           TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    capture_index       INTEGER NOT NULL DEFAULT 0,
    elapsed_min         REAL NOT NULL DEFAULT 0.0,
    analysis_json       TEXT NOT NULL DEFAULT '{}',
    report_pdf          TEXT NOT NULL DEFAULT '',
    report_csv          TEXT NOT NULL DEFAULT '',
    report_json         TEXT NOT NULL DEFAULT '',
    annotated_video     TEXT NOT NULL DEFAULT '',
    trend_report_json   TEXT NOT NULL DEFAULT '{}',
    trend_pdf           TEXT NOT NULL DEFAULT '',
    trend_csv           TEXT NOT NULL DEFAULT '',
    who_score           REAL NOT NULL DEFAULT 0.0,
    who_category        TEXT NOT NULL DEFAULT '',
    progressive_pct     REAL NOT NULL DEFAULT 0.0,
    viability_pct       REAL NOT NULL DEFAULT 0.0,
    total_sperm         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
"""

_DDL_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_sample_id  ON sessions(sample_id);",
    "CREATE INDEX IF NOT EXISTS idx_animal_id  ON sessions(animal_id);",
    "CREATE INDEX IF NOT EXISTS idx_species    ON sessions(species);",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON sessions(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_sid_fk     ON session_results(session_id);",
]

_DDL_CALIBRATIONS = """
CREATE TABLE IF NOT EXISTS calibrations (
    calibration_id      TEXT PRIMARY KEY,
    status              TEXT NOT NULL DEFAULT 'VALIDATED',
    um_per_pixel        REAL NOT NULL DEFAULT 0.0,
    mean_um_per_pixel   REAL NOT NULL DEFAULT 0.0,
    std_um_per_pixel    REAL NOT NULL DEFAULT 0.0,
    cv_pct              REAL NOT NULL DEFAULT 0.0,
    n_measurements      INTEGER NOT NULL DEFAULT 1,
    microscope          TEXT NOT NULL DEFAULT '',
    magnification       TEXT NOT NULL DEFAULT '',
    resolution          TEXT NOT NULL DEFAULT '',
    digital_zoom        TEXT NOT NULL DEFAULT 'None',
    stage_device        TEXT NOT NULL DEFAULT '',
    division_um         REAL NOT NULL DEFAULT 100.0,
    measurements_json   TEXT NOT NULL DEFAULT '[]',
    config_key          TEXT NOT NULL DEFAULT '',
    saved_at            TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT '',
    notes               TEXT NOT NULL DEFAULT ''
);
"""

_DDL_CALIBRATION_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_cal_config_key ON calibrations(config_key);",
    "CREATE INDEX IF NOT EXISTS idx_cal_status      ON calibrations(status);",
    "CREATE INDEX IF NOT EXISTS idx_cal_created_at  ON calibrations(created_at);",
]



# ══════════════════════════════════════════════════════════════
# DatabaseManager
# ══════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    SQLite-backed persistent store for all semen analysis sessions.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Auto-loaded if None.
    db_path : str | Path, optional
        Override database file path.  Defaults to ``data/semen_analysis.db``.
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 db_path: Optional[str | Path] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        if db_path is None:
            db_dir  = ensure_dir("data")
            db_path = db_dir / "semen_analysis.db"

        self.db_path = Path(db_path)
        self._init_db()
        logger.info("Database ready: {}", self.db_path)

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new connection with foreign-key enforcement and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """
        Create tables and indices if they do not yet exist.
        Also applies safe migrations for new tables added in later phases.
        The existing data/semen_analysis.db is never dropped or recreated.
        """
        with self._connect() as conn:
            conn.execute(_DDL_SESSIONS)
            conn.execute(_DDL_RESULTS)
            for idx_ddl in _DDL_INDICES:
                conn.execute(idx_ddl)
            # Phase 7: calibrations table (safe migration — IF NOT EXISTS)
            conn.execute(_DDL_CALIBRATIONS)
            for idx_ddl in _DDL_CALIBRATION_IDX:
                conn.execute(idx_ddl)
            conn.commit()
        logger.debug("Schema verified")

    # ----------------------------------------------------------
    # Save helpers
    # ----------------------------------------------------------

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _extract_scalars(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pull key scalar metrics from a pipeline results dict.
        Handles both flat (AutomatedSemenAnalysis) and nested (SemenAnalysisPipeline)
        return schemas.
        """
        def _g(*keys, default=0.0):
            obj = results
            for k in keys:
                if not isinstance(obj, dict):
                    return default
                obj = obj.get(k, default)
            return obj if obj is not None else default

        if "total_sperm" in results:           # flat schema
            return {
                "who_score":       float(_g("quality", "score")),
                "who_category":    str(results.get("quality", {}).get("category", "")),
                "progressive_pct": float(_g("progressive")),
                "viability_pct":   float(_g("predicted_viability")),
                "total_sperm":     int(_g("total_sperm")),
            }
        # nested schema
        qual  = results.get("quality",   {})
        mot   = results.get("motility",  {})
        return {
            "who_score":       float(qual.get("score",             0.0)),
            "who_category":    str(qual.get("category",           "")),
            "progressive_pct": float(mot.get("progressive_pct",   0.0)),
            "viability_pct":   float(results.get("viability_pct", 0.0)),
            "total_sperm":     int(results.get("detection", {}).get("count", 0)),
        }

    # ----------------------------------------------------------
    # Public: Save single-analysis session
    # ----------------------------------------------------------

    def save_single_session(self,
                             meta: Dict[str, Any],
                             results: Dict[str, Any],
                             report_paths: Optional[Dict[str, str]] = None,
                             annotated_video: str = "") -> str:
        """
        Persist one single-analysis session.

        Parameters
        ----------
        meta            : dict   intake form values from Streamlit
        results         : dict   return value of pipeline.run_video()
        report_paths    : dict   {"pdf","csv","json"} paths
        annotated_video : str    path to annotated video file

        Returns
        -------
        str  session_id (UUID)
        """
        session_id = self._new_id()
        result_id  = self._new_id()
        now        = self._now()
        rp         = report_paths or {}
        scalars    = self._extract_scalars(results)

        session_row = (
            session_id,
            meta.get("sample_id",       ""),
            meta.get("animal_id",       ""),
            meta.get("species",         ""),
            meta.get("breed",           ""),
            meta.get("sample_type",     ""),
            meta.get("collection_date", ""),
            meta.get("operator",        ""),
            meta.get("lab_name",        ""),
            meta.get("chamber",         ""),
            meta.get("microscope_type", ""),
            meta.get("magnification",   ""),
            meta.get("staining_type",   ""),
            meta.get("camera_resolution", ""),
            meta.get("camera_fps",      ""),
            "single",
            now,          # capture_time
            0, 0, 1,      # interval_min, duration_min, n_captures
            now,          # created_at
            "",           # notes
        )

        result_row = (
            result_id,
            session_id,
            0,            # capture_index
            0.0,          # elapsed_min
            json.dumps(results, default=str),
            rp.get("pdf",  ""),
            rp.get("csv",  ""),
            rp.get("json", ""),
            annotated_video,
            "{}",         # trend_report_json
            "", "",       # trend_pdf, trend_csv
            scalars["who_score"],
            scalars["who_category"],
            scalars["progressive_pct"],
            scalars["viability_pct"],
            scalars["total_sperm"],
            now,
        )

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                session_row,
            )
            conn.execute(
                "INSERT INTO session_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                result_row,
            )
            conn.commit()

        logger.success("Saved single session {} (sample={})",
                       session_id[:8], meta.get("sample_id", "—"))
        return session_id

    # ----------------------------------------------------------
    # Public: Save monitoring session
    # ----------------------------------------------------------

    def save_monitoring_session(self,
                                 meta: Dict[str, Any],
                                 monitor_session: Any,
                                 trend_report: Optional[Any] = None,
                                 monitoring_report_paths: Optional[Dict[str, str]] = None
                                 ) -> str:
        """
        Persist a completed monitoring session and all its captures.

        Parameters
        ----------
        meta                     : dict   intake form values
        monitor_session          : MonitorSession
        trend_report             : TrendReport | None
        monitoring_report_paths  : dict {"pdf","csv","json"} for the monitoring report

        Returns
        -------
        str  session_id (UUID)
        """
        session_id = self._new_id()
        now        = self._now()
        mrp        = monitoring_report_paths or {}

        session_row = (
            session_id,
            meta.get("sample_id",       ""),
            meta.get("animal_id",       ""),
            meta.get("species",         ""),
            meta.get("breed",           ""),
            meta.get("sample_type",     ""),
            meta.get("collection_date", ""),
            meta.get("operator",        ""),
            meta.get("lab_name",        ""),
            meta.get("chamber",         ""),
            meta.get("microscope_type", ""),
            meta.get("magnification",   ""),
            meta.get("staining_type",   ""),
            meta.get("camera_resolution", ""),
            meta.get("camera_fps",      ""),
            "monitoring",
            now,
            monitor_session.interval_min,
            monitor_session.duration_min,
            monitor_session.captures_done,
            now,
            "",
        )

        # Serialise trend report scalars
        trend_json = "{}"
        trend_pdf  = mrp.get("pdf",  "")
        trend_csv  = mrp.get("csv",  "")
        if trend_report is not None:
            try:
                m = trend_report.trend_metrics
                trend_json = json.dumps({
                    "progressive_change":     m.progressive_change,
                    "viability_change":       m.viability_change,
                    "score_change":           m.score_change,
                    "motility_rate_per_min":  m.motility_degradation_rate_per_min,
                    "viability_rate_per_min": m.viability_degradation_rate_per_min,
                    "overall":                trend_report.degradation_summary.get("overall",""),
                    "plot_paths":             trend_report.plot_paths,
                }, default=str)
            except Exception as exc:
                logger.warning("Could not serialise trend metrics: {}", exc)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                session_row,
            )

            for r in monitor_session.results:
                scalars = self._extract_scalars(r.analysis_results) if r.analysis_results else {
                    "who_score": 0.0, "who_category": "", "progressive_pct": 0.0,
                    "viability_pct": 0.0, "total_sperm": 0,
                }
                rp = getattr(r, "report_paths", {}) or {}
                result_row = (
                    self._new_id(),
                    session_id,
                    r.interval_index,
                    r.elapsed_min,
                    json.dumps(r.analysis_results, default=str),
                    rp.get("pdf",  ""),
                    rp.get("csv",  ""),
                    rp.get("json", ""),
                    "",           # annotated_video (per-capture not stored separately)
                    trend_json if r.interval_index == 0 else "{}",
                    trend_pdf  if r.interval_index == 0 else "",
                    trend_csv  if r.interval_index == 0 else "",
                    scalars["who_score"],
                    scalars["who_category"],
                    scalars["progressive_pct"],
                    scalars["viability_pct"],
                    scalars["total_sperm"],
                    now,
                )
                conn.execute(
                    "INSERT INTO session_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    result_row,
                )
            conn.commit()

        logger.success("Saved monitoring session {} ({} captures, sample={})",
                       session_id[:8], monitor_session.captures_done,
                       meta.get("sample_id", "—"))
        return session_id

    # ----------------------------------------------------------
    # Public: Query
    # ----------------------------------------------------------

    def list_sessions(self,
                      limit: int = 50,
                      offset: int = 0,
                      session_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a summary list of sessions ordered newest-first.

        Parameters
        ----------
        limit, offset : int   pagination
        session_type  : str | None   filter to 'single' or 'monitoring'

        Returns
        -------
        list of dict
        """
        where = ""
        params: List[Any] = []
        if session_type:
            where = "WHERE session_type = ?"
            params.append(session_type)

        sql = (
            f"SELECT session_id, sample_id, animal_id, species, breed, "
            f"session_type, capture_time, n_captures, interval_min, duration_min, "
            f"operator, created_at "
            f"FROM sessions {where} "
            f"ORDER BY created_at DESC "
            f"LIMIT ? OFFSET ?"
        )
        params += [limit, offset]

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count_sessions(self) -> int:
        """Return total number of stored sessions."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def search_by_sample_id(self, sample_id: str,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Full-text search on sample_id (LIKE, case-insensitive).

        Parameters
        ----------
        sample_id : str   search term (partial match supported)
        limit     : int

        Returns
        -------
        list of dict
        """
        sql = (
            "SELECT session_id, sample_id, animal_id, species, breed, "
            "session_type, capture_time, n_captures, created_at "
            "FROM sessions "
            "WHERE sample_id LIKE ? COLLATE NOCASE "
            "ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{sample_id}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def search_by_animal_id(self, animal_id: str,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Full-text search on animal_id (LIKE, case-insensitive).

        Parameters
        ----------
        animal_id : str   search term (partial match supported)
        limit     : int

        Returns
        -------
        list of dict
        """
        sql = (
            "SELECT session_id, sample_id, animal_id, species, breed, "
            "session_type, capture_time, n_captures, created_at "
            "FROM sessions "
            "WHERE animal_id LIKE ? COLLATE NOCASE "
            "ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{animal_id}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def get_session_detail(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Return full session record including all result rows.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        dict | None
            {
              "session": {all sessions columns},
              "results": [list of session_results rows with parsed analysis_json]
            }
        """
        with self._connect() as conn:
            sess_row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if sess_row is None:
                return None

            res_rows = conn.execute(
                "SELECT * FROM session_results WHERE session_id = ? "
                "ORDER BY capture_index ASC",
                (session_id,),
            ).fetchall()

        results_list = []
        for r in res_rows:
            d = dict(r)
            # Parse JSON blobs
            for json_field in ("analysis_json", "trend_report_json"):
                raw = d.get(json_field, "{}")
                try:
                    d[json_field] = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    d[json_field] = {}
            results_list.append(d)

        return {
            "session": dict(sess_row),
            "results": results_list,
        }

    def get_sessions_for_comparison(self,
                                     session_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Return compact summaries for a list of session IDs, suitable for
        side-by-side comparison tables.

        For each session, extracts the key scalar metrics from the first
        (or only) result row.

        Parameters
        ----------
        session_ids : list of str

        Returns
        -------
        list of dict   one entry per found session
        """
        out = []
        for sid in session_ids:
            detail = self.get_session_detail(sid)
            if detail is None:
                continue
            s = detail["session"]
            r = detail["results"][0] if detail["results"] else {}
            out.append({
                "session_id":     s["session_id"],
                "sample_id":      s["sample_id"],
                "animal_id":      s["animal_id"],
                "species":        s["species"],
                "breed":          s["breed"],
                "session_type":   s["session_type"],
                "capture_time":   s["capture_time"],
                "n_captures":     s["n_captures"],
                "who_score":      r.get("who_score",       0.0),
                "who_category":   r.get("who_category",   ""),
                "progressive_pct":r.get("progressive_pct", 0.0),
                "viability_pct":  r.get("viability_pct",   0.0),
                "total_sperm":    r.get("total_sperm",      0),
                "report_pdf":     r.get("report_pdf",      ""),
                "trend_pdf":      r.get("trend_pdf",       ""),
            })
        return out

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its results (CASCADE).

        Parameters
        ----------
        session_id : str

        Returns
        -------
        bool   True if a row was deleted
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted session {}", session_id[:8])
        return deleted

    def update_notes(self, session_id: str, notes: str) -> None:
        """Update the notes field for a session."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET notes = ? WHERE session_id = ?",
                (notes, session_id),
            )
            conn.commit()


    # ══════════════════════════════════════════════════════════
    # Calibration persistence (Phase 7)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _build_config_key(calibration: Dict[str, Any]) -> str:
        """
        Build a deterministic configuration key from the imaging parameters.

        The key is used to look up the correct calibration when analysing
        a video — it must match exactly the configuration used during capture.

        Format: ``<microscope>|<magnification>|<resolution>|<digital_zoom>``
        """
        parts = [
            str(calibration.get("microscope",   "")).strip().lower(),
            str(calibration.get("magnification","")).strip().lower(),
            str(calibration.get("resolution",   "")).strip().lower(),
            str(calibration.get("digital_zoom", "None")).strip().lower(),
        ]
        return "|".join(parts)

    def save_calibration(self, calibration: Dict[str, Any]) -> str:
        """
        Persist an optical calibration record.

        Supersedes any previous calibration with the same configuration key.

        Parameters
        ----------
        calibration : dict
            Must contain at minimum:
                um_per_pixel, mean, std, cv_pct, n_measurements,
                microscope, magnification, resolution, digital_zoom,
                stage_device, division_um, measurements (list),
                status, saved_at

        Returns
        -------
        str   calibration_id (UUID)
        """
        cal_id     = self._new_id()
        now        = self._now()
        config_key = self._build_config_key(calibration)

        # Supersede any active calibration with the same config key
        with self._connect() as conn:
            conn.execute(
                "UPDATE calibrations SET status = 'SUPERSEDED' "
                "WHERE config_key = ? AND status = 'VALIDATED'",
                (config_key,),
            )
            conn.execute(
                """INSERT INTO calibrations VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cal_id,
                    calibration.get("status",          "VALIDATED"),
                    float(calibration.get("um_per_pixel", 0.0)),
                    float(calibration.get("mean",         calibration.get("um_per_pixel", 0.0))),
                    float(calibration.get("std",          0.0)),
                    float(calibration.get("cv_pct",       0.0)),
                    int(calibration.get("n_measurements", 1)),
                    calibration.get("microscope",   ""),
                    calibration.get("magnification",""),
                    calibration.get("resolution",   ""),
                    calibration.get("digital_zoom", "None"),
                    calibration.get("stage_device", ""),
                    float(calibration.get("division_um", 100.0)),
                    json.dumps(calibration.get("measurements", []), default=str),
                    config_key,
                    calibration.get("saved_at",  now),
                    now,
                    calibration.get("notes", ""),
                ),
            )
            conn.commit()

        logger.success(
            "Calibration saved {} — {:.4f} µm/px config={}",
            cal_id[:8], calibration.get("um_per_pixel", 0.0), config_key,
        )
        return cal_id

    def get_calibration(self,
                         microscope:    str,
                         magnification: str,
                         resolution:    str,
                         digital_zoom:  str = "None",
                         status:        str = "VALIDATED") -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent calibration matching the exact imaging
        configuration.

        Parameters
        ----------
        microscope, magnification, resolution, digital_zoom : str
        status : str   default 'VALIDATED'

        Returns
        -------
        dict | None
            Calibration record, or None if no match found.
            Includes a parsed ``measurements`` list from JSON.
        """
        config_key = self._build_config_key({
            "microscope":    microscope,
            "magnification": magnification,
            "resolution":    resolution,
            "digital_zoom":  digital_zoom,
        })

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM calibrations "
                "WHERE config_key = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (config_key, status),
            ).fetchone()

        if row is None:
            return None

        d = dict(row)
        try:
            d["measurements"] = json.loads(d.get("measurements_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["measurements"] = []
        return d

    def list_calibrations(self,
                           limit: int = 50,
                           status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a summary list of calibration records, newest first.

        Parameters
        ----------
        limit  : int
        status : str | None   filter to 'VALIDATED', 'SUPERSEDED', etc.

        Returns
        -------
        list of dict
        """
        where  = "WHERE status = ?" if status else ""
        params: List[Any] = ([status] if status else []) + [limit]
        sql = (
            "SELECT calibration_id, status, um_per_pixel, cv_pct, "
            "n_measurements, microscope, magnification, resolution, "
            "digital_zoom, stage_device, config_key, saved_at, created_at "
            f"FROM calibrations {where} "
            "ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_calibration(self, calibration_id: str) -> bool:
        """Delete a calibration record by ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM calibrations WHERE calibration_id = ?",
                (calibration_id,),
            )
            conn.commit()
        return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return aggregate statistics across all stored sessions.

        Returns
        -------
        dict
            total_sessions, single_sessions, monitoring_sessions,
            avg_who_score, avg_progressive_pct, avg_viability_pct,
            species_counts  {species: count},
            recent_sessions  [last 5]
        """
        with self._connect() as conn:
            total      = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            single     = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_type='single'"
            ).fetchone()[0]
            monitoring = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_type='monitoring'"
            ).fetchone()[0]

            agg = conn.execute(
                "SELECT AVG(who_score), AVG(progressive_pct), AVG(viability_pct) "
                "FROM session_results"
            ).fetchone()

            species_rows = conn.execute(
                "SELECT species, COUNT(*) as cnt FROM sessions "
                "GROUP BY species ORDER BY cnt DESC"
            ).fetchall()

            recent = conn.execute(
                "SELECT session_id, sample_id, animal_id, species, "
                "session_type, created_at "
                "FROM sessions ORDER BY created_at DESC LIMIT 5"
            ).fetchall()

        return {
            "total_sessions":       total,
            "single_sessions":      single,
            "monitoring_sessions":  monitoring,
            "avg_who_score":        round(float(agg[0] or 0), 2),
            "avg_progressive_pct":  round(float(agg[1] or 0), 2),
            "avg_viability_pct":    round(float(agg[2] or 0), 2),
            "species_counts":       {r["species"]: r["cnt"] for r in species_rows},
            "recent_sessions":      [dict(r) for r in recent],
        }