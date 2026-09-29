"""
src/camera/analysis_progress.py
=================================
Phase 6 · Analysis Progress Tracker

Provides the professional pipeline-progress dashboard shown while
``AutomatedSemenAnalysis.run_video()`` is executing.

Because ``run_video`` is a blocking call, Streamlit cannot update the
UI mid-execution.  This module uses a ``threading.Thread`` to run the
pipeline and exposes a ``ProgressTracker`` that can be polled on each
Streamlit rerun via ``streamlit-autorefresh``.

Architecture
------------
1. Caller creates a ``ProgressTracker`` and calls ``.start(pipeline, video_path)``.
2. The tracker spawns a daemon thread that runs ``pipeline.run_video(video_path)``.
3. On every Streamlit rerun, the UI calls ``.get_snapshot()`` which returns a
   ``ProgressSnapshot`` with all fields needed to render the dashboard.
4. When the thread finishes, ``.is_done`` becomes True and ``.result`` holds
   the pipeline return dict (or ``.error`` holds the exception message).

Stage progression
-----------------
Because the pipeline is a black box from the UI perspective, stages are
estimated from wall-clock time compared to a per-stage time budget.
If ``total_frames`` is provided (read from the video before starting),
the "Detection" stage progress is the most accurate.

System metrics
--------------
CPU, RAM, GPU/VRAM are read via ``psutil`` (always available) and
``pynvml`` (NVIDIA GPU, optional).  Missing metrics show "—".

Public API
----------
    from src.camera.analysis_progress import ProgressTracker

    tracker = ProgressTracker()
    tracker.start(pipeline, video_path, total_frames=820)

    # on every rerun while tracker.is_running:
    snap = tracker.get_snapshot()
    render_progress_dashboard(snap)   # in app.py

    if tracker.is_done:
        results = tracker.result      # pipeline output dict
        error   = tracker.error       # str if failed, else ""
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Stage definitions
# ══════════════════════════════════════════════════════════════

class StageStatus(str, Enum):
    WAITING   = "waiting"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"


@dataclass
class PipelineStage:
    name:           str
    status:         StageStatus = StageStatus.WAITING
    progress_pct:   float       = 0.0     # 0-100 within this stage
    detail:         str         = ""      # e.g. "Frame 256 / 820"

    # Estimated fraction of total pipeline time this stage takes
    time_weight:    float       = 1.0


# Ordered pipeline stages with relative time weights
_STAGES: List[PipelineStage] = [
    PipelineStage("Video Loaded",         time_weight=0.03),
    PipelineStage("Frame Extraction",     time_weight=0.05),
    PipelineStage("Preprocessing",        time_weight=0.05),
    PipelineStage("YOLO Detection",       time_weight=0.30),
    PipelineStage("ByteTrack Tracking",   time_weight=0.15),
    PipelineStage("Motility Analysis",    time_weight=0.12),
    PipelineStage("Morphology",           time_weight=0.12),
    PipelineStage("Viability Prediction", time_weight=0.05),
    PipelineStage("WHO Scoring",          time_weight=0.03),
    PipelineStage("Report Generation",    time_weight=0.10),
]

# Cumulative weight prefix sums (for overall % calculation)
def _build_cumulative() -> List[float]:
    total = sum(s.time_weight for s in _STAGES)
    cumulative = []
    running = 0.0
    for s in _STAGES:
        cumulative.append(running / total * 100.0)
        running += s.time_weight
    cumulative.append(100.0)
    return cumulative

_CUM_PCT = _build_cumulative()
_TOTAL_WEIGHT = sum(s.time_weight for s in _STAGES)


# ══════════════════════════════════════════════════════════════
# System metrics
# ══════════════════════════════════════════════════════════════

@dataclass
class SystemMetrics:
    cpu_pct:     float = 0.0
    ram_pct:     float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb:float = 0.0
    gpu_pct:     float = -1.0    # -1 = not available
    vram_pct:    float = -1.0
    disk_free_gb:float = 0.0
    thread_count:int   = 0


def _collect_system_metrics() -> SystemMetrics:
    """Collect CPU / RAM / GPU metrics. Never raises."""
    m = SystemMetrics()
    try:
        import psutil
        m.cpu_pct      = psutil.cpu_percent(interval=None)
        ram             = psutil.virtual_memory()
        m.ram_pct      = ram.percent
        m.ram_used_gb  = round(ram.used  / 1e9, 2)
        m.ram_total_gb = round(ram.total / 1e9, 2)
        m.disk_free_gb = round(psutil.disk_usage("/").free / 1e9, 1)
        m.thread_count = threading.active_count()
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("psutil metrics failed: {}", exc)

    # NVIDIA GPU via pynvml (optional)
    try:
        import pynvml   # type: ignore[import]
        pynvml.nvmlInit()
        handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
        util     = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        m.gpu_pct  = float(util.gpu)
        m.vram_pct = round(mem_info.used / mem_info.total * 100.0, 1)
        pynvml.nvmlShutdown()
    except Exception:
        pass   # GPU not available — leave as -1

    return m


# ══════════════════════════════════════════════════════════════
# Progress snapshot (read by UI)
# ══════════════════════════════════════════════════════════════

@dataclass
class ProgressSnapshot:
    """
    Everything the UI needs to render the progress dashboard.

    Attributes
    ----------
    is_running         : bool
    is_done            : bool
    overall_pct        : float   0-100
    current_stage_name : str
    current_stage_idx  : int
    stages             : list of PipelineStage copies
    elapsed_s          : float
    eta_s              : float   -1 if unknown
    processed_frames   : int
    total_frames       : int
    detected_sperm     : int
    tracked_objects    : int
    inference_fps      : float
    system             : SystemMetrics
    error              : str     non-empty on failure
    """
    is_running:          bool            = False
    is_done:             bool            = False
    overall_pct:         float           = 0.0
    current_stage_name:  str             = "—"
    current_stage_idx:   int             = 0
    stages:              List[PipelineStage] = field(default_factory=list)
    elapsed_s:           float           = 0.0
    eta_s:               float           = -1.0
    processed_frames:    int             = 0
    total_frames:        int             = 0
    detected_sperm:      int             = 0
    tracked_objects:     int             = 0
    inference_fps:       float           = 0.0
    system:              SystemMetrics   = field(default_factory=SystemMetrics)
    error:               str             = ""

    @property
    def elapsed_mmss(self) -> str:
        s = max(0, int(self.elapsed_s))
        return f"{s // 60:02d}:{s % 60:02d}"

    @property
    def eta_mmss(self) -> str:
        if self.eta_s < 0:
            return "—"
        s = max(0, int(self.eta_s))
        return f"{s // 60:02d}:{s % 60:02d}"


# ══════════════════════════════════════════════════════════════
# ProgressTracker
# ══════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Runs the AI pipeline in a background thread and tracks progress.

    One instance per analysis run; store in session_state and discard
    when analysis completes.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._thread:       Optional[threading.Thread] = None

        # State written by thread, read by UI
        self._stages        = [PipelineStage(s.name, time_weight=s.time_weight)
                                for s in _STAGES]
        self._current_idx   = 0
        self._start_time:   Optional[float] = None
        self._is_running    = False
        self._is_done       = False
        self._result:       Optional[Dict[str, Any]] = None
        self._error         = ""
        self._total_frames  = 0
        self._concentration_params: Optional[Dict[str, Any]] = None
        self._metrics_cache = SystemMetrics()
        self._metrics_time  = 0.0

    # ----------------------------------------------------------

    def start(self,
              pipeline:      Any,
              video_path:    Path,
              total_frames:  int = 0,
              concentration_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Start the pipeline in a background thread.

        Parameters
        ----------
        pipeline    : AutomatedSemenAnalysis instance
        video_path  : Path
        total_frames: int   pre-read from the video (for accurate %)
        concentration_params : dict, optional
            Forwarded verbatim to ``pipeline.run_video()`` so the
            concentration/MCC/progressive-concentration calculation
            (see ``src.analysis.concentration``) runs as part of this
            background pipeline call. ``None`` preserves the exact
            prior behaviour (no concentration fields added).
        """
        if self._is_running:
            logger.warning("ProgressTracker.start() called while already running")
            return

        self._total_frames = total_frames
        self._concentration_params = concentration_params
        self._start_time   = time.time()
        self._is_running   = True
        self._is_done      = False
        self._result       = None
        self._error        = ""
        self._current_idx  = 0
        for s in self._stages:
            s.status       = StageStatus.WAITING
            s.progress_pct = 0.0
            s.detail       = ""

        self._thread = threading.Thread(
            target = self._run,
            args   = (pipeline, video_path),
            daemon = True,
            name   = "AnalysisPipeline",
        )
        self._thread.start()
        logger.info("ProgressTracker started — video: {}", video_path.name)

    # ----------------------------------------------------------

    def get_snapshot(self) -> ProgressSnapshot:
        """
        Return a consistent snapshot of current progress.
        Call on every Streamlit rerun.
        """
        with self._lock:
            now     = time.time()
            elapsed = (now - self._start_time) if self._start_time else 0.0

            # Overall % from current stage + time-weight interpolation
            idx     = self._current_idx
            stage   = self._stages[idx] if idx < len(self._stages) else self._stages[-1]
            base    = _CUM_PCT[idx]
            width   = _CUM_PCT[idx + 1] - base if idx + 1 < len(_CUM_PCT) else 0.0
            overall = min(99.9 if self._is_running else 100.0,
                          base + width * stage.progress_pct / 100.0)

            # ETA
            if overall > 1.0 and elapsed > 0:
                total_est = elapsed / (overall / 100.0)
                eta = max(0.0, total_est - elapsed)
            else:
                eta = -1.0

            # Inference FPS (rough estimate from frame progress)
            inf_fps = 0.0
            if elapsed > 0 and self._stages[3].progress_pct > 0:
                frames_done = (self._stages[3].progress_pct / 100.0) * self._total_frames
                inf_fps     = round(frames_done / elapsed, 1) if elapsed > 1 else 0.0

            # Collect system metrics (cached, updated every 2 s)
            if now - self._metrics_time > 2.0:
                self._metrics_cache = _collect_system_metrics()
                self._metrics_time  = now

            return ProgressSnapshot(
                is_running         = self._is_running,
                is_done            = self._is_done,
                overall_pct        = overall,
                current_stage_name = stage.name,
                current_stage_idx  = idx,
                stages             = [
                    PipelineStage(s.name, s.status, s.progress_pct, s.detail,
                                  s.time_weight)
                    for s in self._stages
                ],
                elapsed_s          = elapsed,
                eta_s              = eta,
                processed_frames   = int(
                    (self._stages[3].progress_pct / 100.0) * self._total_frames
                ) if self._total_frames > 0 else 0,
                total_frames       = self._total_frames,
                detected_sperm     = 0,   # pipeline does not expose mid-run counts
                tracked_objects    = 0,
                inference_fps      = inf_fps,
                system             = self._metrics_cache,
                error              = self._error,
            )

    # ----------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    @property
    def error(self) -> str:
        return self._error

    # ----------------------------------------------------------
    # Background thread
    # ----------------------------------------------------------

    def _run(self, pipeline: Any, video_path: Path) -> None:
        """
        Pipeline execution thread.

        Advances stage indices at estimated time boundaries.
        The final result is stored in self._result.
        """
        try:
            # Advance through early stages immediately
            self._advance_to(0, 100, "")   # Video Loaded
            self._advance_to(1, 100, "")   # Frame Extraction
            self._advance_to(2, 100, "")   # Preprocessing
            self._set_stage_running(3)     # YOLO Detection — main stage

            # Spawn a progress-simulation thread that advances the
            # detection stage based on elapsed time vs expected duration
            sim_stop = threading.Event()
            sim_thread = threading.Thread(
                target=self._simulate_detection_progress,
                args=(sim_stop,), daemon=True
            )
            sim_thread.start()

            # ── Blocking pipeline call ─────────────────────────
            results = pipeline.run_video(
                video_path,
                concentration_params=self._concentration_params,
            )

            # Stop simulation and advance remaining stages
            sim_stop.set()
            sim_thread.join(timeout=1.0)

            # Mark stages 3-9 complete
            for i in range(3, len(self._stages)):
                self._advance_to(i, 100, "")

            with self._lock:
                self._result    = results
                self._is_done   = True
                self._is_running = False

            logger.success("ProgressTracker: pipeline complete")

        except Exception as exc:
            with self._lock:
                self._error      = str(exc)
                self._is_done    = True
                self._is_running = False
                if self._current_idx < len(self._stages):
                    self._stages[self._current_idx].status = StageStatus.FAILED
            logger.error("ProgressTracker: pipeline failed: {}", exc)

    def _simulate_detection_progress(self, stop_event: threading.Event) -> None:
        """
        Advance the YOLO detection stage progress (0→95%) over the
        estimated detection time.  Remaining stages advance after
        the real pipeline call returns.
        """
        # Rough expected time budgets per stage (seconds, very conservative)
        stage_budgets = {
            3: 60,    # YOLO Detection
            4: 30,    # Tracking
            5: 20,    # Motility
            6: 20,    # Morphology
            7: 10,    # Viability
            8: 5,     # WHO Scoring
            9: 15,    # Report Generation
        }

        current_stage = 3
        stage_start   = time.time()

        while not stop_event.is_set():
            elapsed_in_stage = time.time() - stage_start
            budget           = stage_budgets.get(current_stage, 30)
            stage_pct        = min(95.0, elapsed_in_stage / budget * 100.0)

            self._set_stage_progress(current_stage, stage_pct)

            # Move to next stage when budget is ~consumed
            if stage_pct >= 93.0 and current_stage < 9:
                self._advance_to(current_stage, 100, "")
                current_stage += 1
                self._set_stage_running(current_stage)
                stage_start = time.time()

            time.sleep(0.5)

    # ----------------------------------------------------------
    # Stage state helpers (all acquire self._lock)
    # ----------------------------------------------------------

    def _advance_to(self, idx: int, pct: float, detail: str) -> None:
        with self._lock:
            if idx < len(self._stages):
                self._stages[idx].status       = StageStatus.COMPLETE if pct >= 100 else StageStatus.RUNNING
                self._stages[idx].progress_pct = pct
                self._stages[idx].detail        = detail
                # Mark all prior stages complete
                for i in range(idx):
                    self._stages[i].status       = StageStatus.COMPLETE
                    self._stages[i].progress_pct = 100.0
                self._current_idx = idx

    def _set_stage_running(self, idx: int) -> None:
        with self._lock:
            if idx < len(self._stages):
                self._stages[idx].status       = StageStatus.RUNNING
                self._stages[idx].progress_pct = 0.0
                self._current_idx              = idx

    def _set_stage_progress(self, idx: int, pct: float) -> None:
        with self._lock:
            if idx < len(self._stages):
                self._stages[idx].progress_pct = pct