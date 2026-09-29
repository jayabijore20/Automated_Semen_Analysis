"""
src/camera/monitor.py
======================
Time-Interval Semen Sample Monitor

Manages the automatic, timed capture-and-analyse loop that is the
core of Phase 3.  All camera I/O is delegated to
``USBMicroscope.record_clip``; all analysis is delegated to
``AutomatedSemenAnalysis.run_video``.  This module owns only the
scheduling logic and the result store.

Design overview
---------------
``MonitorSession`` is a plain dataclass that represents one monitoring
session.  It is **serialisable to / from a plain dict** so it can be
stored in Streamlit ``session_state`` and survives page reruns.

``IntervalMonitor`` holds a ``MonitorSession`` and drives the loop.
It is instantiated once, stored in ``session_state``, and its
``start()``/``tick()`` methods perform the actual (blocking) capture
and analysis work.

Why synchronous (not threaded) *inside this module*?
-------------------------------------------------------
Streamlit reruns are single-threaded, and background threads writing
directly to ``session_state`` produce race conditions that are hard to
debug.  This module therefore stays deliberately simple: ``start()``
and ``tick()`` are plain, synchronous, blocking calls with no internal
threading of their own — exactly what the docstring originally said.

BUGFIX NOTE (current app.py architecture): calling these blocking
methods directly from the Streamlit script thread turned out to have a
different problem — combined with ``streamlit-autorefresh`` requesting
a new rerun every ``_MONITOR_TICK_MS``, a rerun request could arrive
and interrupt the script *while it was still stuck inside a blocking
``tick()`` call*, corrupting in-progress captures. The fix for that
lives entirely in ``app.py``: a small ``_MonitorRunner`` wrapper now
calls ``start()``/``tick()``/``stop()`` from one dedicated background
thread instead of the Streamlit thread. This module's own API and
internal behaviour are unchanged by that fix — it is still exactly the
plain synchronous object the rest of this docstring describes;
*something else* now simply calls it from a different thread than the
Streamlit script thread.

  1. ``streamlit-autorefresh`` is used to trigger a rerun every
     ``_TICK_INTERVAL_MS`` (default 10 s), purely so the UI redraws.
  2. ``IntervalMonitor.tick()`` checks
     ``time.time() - last_capture_time`` whenever it is called.
  3. If the interval has elapsed it runs the capture + analysis
     synchronously (blocking the caller for ~clip_duration + pipeline_time).
  4. The result is appended to ``MonitorSession.results``.

Session persistence
-------------------
A ``MonitorSession`` survives Streamlit reruns because it lives in
``st.session_state["_monitor_session"]``.  It does NOT survive a
browser hard-refresh or server restart.  This is documented clearly in
the UI.

Concentration / MCC / SML support
------------------------------------
``MonitorSession`` carries an optional ``concentration_params`` dict
(built by ``app.py`` from the Section 3/4 intake-form fields — sample
volume, dilution factor, chamber, calibration, etc. — see
``src.analysis.concentration``). When present, it is forwarded to
``AutomatedSemenAnalysis.run_video(..., concentration_params=...)`` on
every capture, so each capture's ``analysis_results`` comes back with
concentration/MCC/progressive-concentration fields already computed
using that capture's REAL recorded-clip pixel dimensions (read
directly from the video file inside ``run_video()``) — not an
approximation based on the configured camera resolution string. When
``concentration_params`` is ``None`` (the default — preserves all
prior behaviour exactly), nothing related to concentration changes:
``run_video()`` is called exactly as before.

Sustained Motility Lifetime (SML) is NOT computed inside this module —
it only needs the ``(elapsed_min, progressive_pct)`` time series that
``MonitorSession.results`` already provides, so it is computed
directly in ``app.py`` from that existing data
(``src.analysis.concentration.calculate_sml``).

Public API used by app.py
--------------------------
    from src.camera.monitor import IntervalMonitor, MonitorSession

    # Start
    session = MonitorSession(interval_min=10, duration_min=60,
                             sample_id="BVS-001", device_index=0,
                             resolution="1280×720", fps="30 FPS",
                             clip_duration_s=12.0,
                             concentration_params=my_params)   # optional
    monitor = IntervalMonitor(session, pipeline)
    monitor.start()                       # records + analyses first capture now

    # On every rerun
    monitor.tick()                        # no-op if interval not elapsed yet

    # UI queries
    monitor.session.results               # list of CaptureResult
    monitor.session.is_running            # bool
    monitor.time_to_next_capture_s        # float seconds until next capture
    monitor.elapsed_total_s               # float seconds since session start
    monitor.session.is_complete           # bool (total duration elapsed)

    # Stop
    monitor.stop()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.camera.usb_microscope import USBMicroscope


# ══════════════════════════════════════════════════════════════
# CaptureResult
# ══════════════════════════════════════════════════════════════

@dataclass
class CaptureResult:
    """
    Stores the output of one capture-and-analyse cycle.

    Attributes
    ----------
    interval_index : int
        0-based index of this capture (0 = first, 1 = second, …).
    capture_time   : float
        ``time.time()`` at the moment recording started.
    analysis_results : dict
        Full return value of ``AutomatedSemenAnalysis.run_video``.
    error          : str | None
        Error message if the capture or analysis failed.
    clip_path      : str
        Path to the recorded clip (may have been deleted already).
    elapsed_min    : float
        Minutes elapsed since session start when this capture began.
    """

    interval_index:   int
    capture_time:     float
    analysis_results: Dict[str, Any]
    error:            Optional[str]
    clip_path:        str
    elapsed_min:      float

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (for JSON serialisation)."""
        return {
            "interval_index":   self.interval_index,
            "capture_time":     self.capture_time,
            "error":            self.error,
            "clip_path":        self.clip_path,
            "elapsed_min":      round(self.elapsed_min, 2),
            # Embed key motility / morphology / viability numbers only
            # (the full results dict may contain large numpy arrays)
            "summary": _extract_summary(self.analysis_results),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureResult":
        return cls(
            interval_index   = d["interval_index"],
            capture_time     = d["capture_time"],
            analysis_results = {},         # not round-tripped (too large)
            error            = d.get("error"),
            clip_path        = d.get("clip_path", ""),
            elapsed_min      = d.get("elapsed_min", 0.0),
        )


def _extract_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact scalar summary from ``run_video`` results.
    Safe to call with an empty dict on error.

    Includes concentration/MCC/progressive-concentration fields when
    present (i.e. when the capture was run with ``concentration_params``
    set) — read via ``.get(..., None)`` so a capture that predates this
    feature, or one where concentration could not be calculated, simply
    comes back with ``None`` for these keys rather than raising or
    fabricating a number (Feature 14/15 honesty requirement).
    """
    if not results:
        return {}

    def _g(*keys: str, default: float = 0.0) -> float:
        obj: Any = results
        for k in keys:
            if not isinstance(obj, dict):
                return default
            obj = obj.get(k, default)
        return float(obj) if obj is not None else default

    # Support both flat (AutomatedSemenAnalysis) and nested (SemenAnalysisPipeline) schemas
    if "total_sperm" in results:
        summary = {
            "total_sperm":       int(_g("total_sperm")),
            "progressive_pct":   round(_g("progressive"),     2),
            "nonprog_pct":       round(_g("non_progressive"), 2),
            "immotile_pct":      round(_g("immotile"),        2),
            "normal_pct":        round(_g("normal_pct"),      2),
            "viability_pct":     round(_g("predicted_viability"), 2),
            "who_score":         round(_g("quality", "score"), 2),
            "who_category":      results.get("quality", {}).get("category", "—"),
        }
    else:
        # nested schema
        det   = results.get("detection",  {})
        mot   = results.get("motility",   {})
        morph = results.get("morphology", {})
        qual  = results.get("quality",    {})
        summary = {
            "total_sperm":       int(det.get("count", 0)),
            "progressive_pct":   round(float(mot.get("progressive_pct",    0)), 2),
            "nonprog_pct":       round(float(mot.get("nonprogressive_pct", 0)), 2),
            "immotile_pct":      round(float(mot.get("immotile_pct",       0)), 2),
            "normal_pct":        round(float(morph.get("normal_pct",       0)), 2),
            "viability_pct":     round(float(results.get("viability_pct",  0)), 2),
            "who_score":         round(float(qual.get("score",             0)), 2),
            "who_category":      qual.get("category", "—"),
        }

    # Concentration/MCC (Feature 8/12) — never invented, always None
    # when not present/not calculable.
    summary["concentration_status"] = results.get("concentration_status")
    total_m = results.get("total_concentration_m_cells_per_ml")
    mcc_m   = results.get("motile_concentration_m_cells_per_ml")
    prog_m  = results.get("progressive_concentration_m_cells_per_ml")
    summary["total_concentration_m_ml"]      = round(total_m, 2) if total_m is not None else None
    summary["mcc_m_ml"]                      = round(mcc_m, 2)   if mcc_m   is not None else None
    summary["progressive_concentration_m_ml"] = round(prog_m, 2) if prog_m  is not None else None

    return summary


# ══════════════════════════════════════════════════════════════
# MonitorSession
# ══════════════════════════════════════════════════════════════

@dataclass
class MonitorSession:
    """
    Serialisable state for one time-interval monitoring session.

    All primitive fields survive Streamlit page reruns via
    ``session_state``.

    Parameters
    ----------
    interval_min   : int     minutes between captures
    duration_min   : int     total monitoring window in minutes
    sample_id      : str
    device_index   : int
    resolution     : str     e.g. "1280×720"
    fps            : str     e.g. "30 FPS"
    clip_duration_s: float   recording length per capture in seconds
    concentration_params : dict | None
        Optional sample-prep / calibration parameters (see
        ``src.analysis.concentration.compute_concentration_metrics_from_params``
        for the expected keys). When set, every capture in this
        session is analysed with concentration/MCC/progressive-
        concentration calculated automatically. ``None`` (the
        default) preserves the exact prior behaviour — no
        concentration fields are added to any capture's results.
    """

    interval_min:    int
    duration_min:    int
    sample_id:       str
    device_index:    int
    resolution:      str
    fps:             str
    clip_duration_s: float
    concentration_params: Optional[Dict[str, Any]] = field(default=None)

    # Set when start() is called
    start_time:      float                  = field(default=0.0)
    last_capture_time: float                = field(default=0.0)

    # Running state
    is_running:      bool                   = field(default=False)
    is_complete:     bool                   = field(default=False)

    # Results list — one CaptureResult per interval
    results:         List[CaptureResult]    = field(default_factory=list)

    # ── Derived helpers ────────────────────────────────────────

    @property
    def interval_s(self) -> float:
        """Interval in seconds."""
        return float(self.interval_min * 60)

    @property
    def duration_s(self) -> float:
        """Total duration in seconds."""
        return float(self.duration_min * 60)

    @property
    def expected_captures(self) -> int:
        """Total number of captures expected over the session duration."""
        return int(self.duration_min // self.interval_min) + 1

    @property
    def captures_done(self) -> int:
        return len(self.results)

    def elapsed_s(self, now: Optional[float] = None) -> float:
        """Seconds elapsed since session start."""
        if self.start_time == 0.0:
            return 0.0
        return (now or time.time()) - self.start_time

    def time_to_next_s(self, now: Optional[float] = None) -> float:
        """Seconds until the next scheduled capture."""
        now = now or time.time()
        if not self.is_running or self.last_capture_time == 0.0:
            return 0.0
        nxt = self.last_capture_time + self.interval_s
        return max(0.0, nxt - now)

    def next_capture_at_min(self) -> float:
        """Minutes from session start when the next capture is due."""
        if self.last_capture_time == 0.0 or self.start_time == 0.0:
            return 0.0
        return (self.last_capture_time - self.start_time + self.interval_s) / 60.0


# ══════════════════════════════════════════════════════════════
# IntervalMonitor
# ══════════════════════════════════════════════════════════════

class IntervalMonitor:
    """
    Drives the automatic capture-and-analyse loop.

    Stored in ``st.session_state["_monitor_obj"]``. In the current
    app.py architecture, its ``start()``/``tick()``/``stop()`` methods
    are called from a dedicated background thread (``_MonitorRunner``
    in app.py) rather than directly from the Streamlit script thread —
    see the "BUGFIX NOTE" in the module docstring. This class itself
    has no knowledge of that and requires no changes to support it: it
    remains a plain, synchronous object.

    Parameters
    ----------
    session  : MonitorSession
    pipeline : AutomatedSemenAnalysis instance (already loaded)
    """

    def __init__(self, session: MonitorSession, pipeline: Any) -> None:
        self.session  = session
        self._pipeline = pipeline

    # ----------------------------------------------------------

    def start(self) -> None:
        """
        Start the monitoring session.

        Records and analyses the very first capture immediately
        (t = 0 min) so the user sees a result before waiting for
        the first interval.
        """
        now = time.time()
        self.session.start_time       = now
        self.session.last_capture_time = now
        self.session.is_running       = True
        self.session.is_complete      = False
        self.session.results          = []

        # Capture interval 0 immediately
        self._capture_and_analyse()

    # ----------------------------------------------------------

    def tick(self) -> bool:
        """
        Called on every Streamlit rerun.  Triggers a capture if the
        interval has elapsed; checks if the session is complete.

        Returns
        -------
        bool
            True if a new capture was performed on this tick.
        """
        if not self.session.is_running or self.session.is_complete:
            return False

        now     = time.time()
        elapsed = now - self.session.start_time

        # Check total duration — mark complete when exceeded
        if elapsed >= self.session.duration_s:
            # Final check: run one last capture if we haven't hit the
            # expected number yet and interval is due
            if (now - self.session.last_capture_time) >= self.session.interval_s:
                self._capture_and_analyse()
            self.session.is_running  = False
            self.session.is_complete = True
            return True

        # Normal interval check
        if (now - self.session.last_capture_time) >= self.session.interval_s:
            self._capture_and_analyse()
            return True

        return False

    # ----------------------------------------------------------

    def stop(self) -> None:
        """Stop monitoring (user-initiated stop)."""
        self.session.is_running  = False
        self.session.is_complete = False   # not complete — stopped early

    # ----------------------------------------------------------

    def _capture_and_analyse(self) -> None:
        """
        Execute one full capture → record → analyse cycle.

        On success: appends a ``CaptureResult`` to ``session.results``.
        On failure: appends a ``CaptureResult`` with ``error`` set.
        """
        now          = time.time()
        idx          = len(self.session.results)
        elapsed_min  = (now - self.session.start_time) / 60.0
        clip_path_str = ""

        try:
            # ── Camera connect (reconnect each capture for robustness) ──
            mic = USBMicroscope(
                device_index = self.session.device_index,
                resolution   = self.session.resolution,
                fps          = self.session.fps,
            )
            if not mic.connect():
                raise RuntimeError(
                    f"Could not open camera at index {self.session.device_index}"
                )

            # ── Record clip ─────────────────────────────────────────────
            clip_path = mic.record_clip(duration_s=self.session.clip_duration_s)
            mic.disconnect()

            if clip_path is None:
                raise RuntimeError("record_clip returned None — recording failed")

            clip_path_str = str(clip_path)

            # ── Run AI pipeline ─────────────────────────────────────────
            # Concentration/MCC/progressive-concentration (Features
            # 1-29): forwarded only when the session was configured
            # with concentration_params (None preserves prior
            # behaviour exactly — see MonitorSession docstring).
            # run_video() reads this clip's REAL pixel dimensions
            # itself, so concentration for monitoring captures is
            # exactly as accurate as for a single analysis.
            analysis_results = self._pipeline.run_video(
                clip_path,
                concentration_params=self.session.concentration_params,
            )

            # Clean up temp clip
            clip_path.unlink(missing_ok=True)

            result = CaptureResult(
                interval_index   = idx,
                capture_time     = now,
                analysis_results = analysis_results,
                error            = None,
                clip_path        = clip_path_str,
                elapsed_min      = elapsed_min,
            )

        except Exception as exc:  # noqa: BLE001
            # Clean up any partial clip
            if clip_path_str:
                Path(clip_path_str).unlink(missing_ok=True)

            result = CaptureResult(
                interval_index   = idx,
                capture_time     = now,
                analysis_results = {},
                error            = str(exc),
                clip_path        = clip_path_str,
                elapsed_min      = elapsed_min,
            )

        self.session.results.append(result)
        self.session.last_capture_time = time.time()