"""
src/camera/recording_manager.py
=================================
Phase 6 · Recording Manager

Drives the capture loop and provides live recording statistics.
Works with ``CameraManager`` — does NOT open its own VideoCapture.

Responsibilities
----------------
* Record a fixed-duration clip using frames from CameraManager.grab_frame().
* Verify every frame (not None, correct size, count increasing).
* Detect and attempt recovery from camera stream interruptions.
* Track elapsed time, remaining time, frame count, FPS, dropped frames.
* Expose ``RecordingStatus`` dataclass for the live UI panel.
* Save the clip to a temporary MP4 file and return its Path.

Public API
----------
    from src.camera.recording_manager import RecordingManager, RecordingStatus

    rec = RecordingManager(camera_manager)
    rec.start(duration_s=12, resolution=(1280, 720), fps=30.0)

    # on each Streamlit rerun (called from _render_recording_panel):
    status = rec.get_status()

    # blocking call that returns the clip path when done:
    clip_path = rec.wait_for_completion()   # None on failure
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.camera.camera_manager import CameraManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
_MAX_RECOVERY_ATTEMPTS  = 5
_FRAME_TIMEOUT_S        = 3.0   # seconds without a frame before recovery
_RECOVERY_WAIT_S        = 0.5   # wait between recovery attempts


class RecordingState(str, Enum):
    IDLE       = "idle"
    RECORDING  = "recording"
    RECOVERING = "recovering"
    COMPLETE   = "complete"
    FAILED     = "failed"


@dataclass
class RecordingStatus:
    """
    Live snapshot of recording progress — used by the UI panel.

    Attributes
    ----------
    state               : RecordingState
    elapsed_s           : float
    remaining_s         : float
    duration_s          : float
    frames_recorded     : int
    frames_dropped      : int
    current_fps         : float
    progress_pct        : float   0-100
    quality             : str     "Excellent" / "Good" / "Poor"
    frame_loss_pct      : float
    error_title         : str     non-empty only on FAILED
    error_detail        : str
    suggestion          : str
    clip_path           : str     non-empty only on COMPLETE
    """
    state:           RecordingState = RecordingState.IDLE
    elapsed_s:       float          = 0.0
    remaining_s:     float          = 0.0
    duration_s:      float          = 12.0
    frames_recorded: int            = 0
    frames_dropped:  int            = 0
    current_fps:     float          = 0.0
    progress_pct:    float          = 0.0
    quality:         str            = "—"
    frame_loss_pct:  float          = 0.0
    error_title:     str            = ""
    error_detail:    str            = ""
    suggestion:      str            = ""
    clip_path:       str            = ""

    @property
    def elapsed_mmss(self) -> str:
        s = max(0, int(self.elapsed_s))
        return f"{s // 60:02d}:{s % 60:02d}"

    @property
    def remaining_mmss(self) -> str:
        s = max(0, int(self.remaining_s))
        return f"{s // 60:02d}:{s % 60:02d}"


def _quality_label(frame_loss_pct: float, actual_fps: float, target_fps: float) -> str:
    """Map frame-loss and FPS deviation to a quality string."""
    fps_ratio = actual_fps / max(target_fps, 1.0)
    if frame_loss_pct < 1.0 and fps_ratio >= 0.97:
        return "Excellent"
    if frame_loss_pct < 5.0 and fps_ratio >= 0.90:
        return "Good"
    if frame_loss_pct < 15.0 and fps_ratio >= 0.70:
        return "Acceptable"
    return "Poor"


class RecordingManager:
    """
    Professional recording workflow with live statistics and frame verification.

    Parameters
    ----------
    camera_manager : CameraManager
        The already-connected camera manager instance.
    """

    def __init__(self, camera_manager: CameraManager) -> None:
        self._cam    = camera_manager
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()

        # Shared state (written by thread, read by UI)
        self._status  = RecordingStatus()
        self._stop_event = threading.Event()

        self._is_recording = False

    # ----------------------------------------------------------

    def start(self,
              duration_s:  float = 12.0,
              resolution:  Tuple[int, int] = (1280, 720),
              fps:         float = 30.0) -> bool:
        """
        Begin recording in a background thread.

        Parameters
        ----------
        duration_s  : float
        resolution  : (width, height)
        fps         : float

        Returns
        -------
        bool   False if already recording or camera not connected.
        """
        if not self._cam.is_connected:
            logger.error("RecordingManager.start(): camera not connected")
            return False

        with self._lock:
            if self._status.state == RecordingState.RECORDING:
                logger.warning("Already recording")
                return False

            self._status      = RecordingStatus(state=RecordingState.RECORDING,
                                                 duration_s=duration_s)

            self._is_recording = True
            self._stop_event.clear()

        self._thread = threading.Thread(
            target    = self._record_loop,
            args      = (duration_s, resolution, fps),
            daemon    = True,
            name      = "RecordingLoop",
        )
        self._thread.start()
        logger.info("Recording started: {} s @ {}×{} {:.0f}fps",
                    duration_s, resolution[0], resolution[1], fps)
        return True

    # ----------------------------------------------------------

    def stop(self) -> None:
        """Signal the recording loop to stop early."""
        self._stop_event.set()

    # ----------------------------------------------------------

    def get_status(self) -> RecordingStatus:
        """Thread-safe snapshot of current recording status."""
        with self._lock:
            # Return a shallow copy so the UI sees a consistent snapshot
            s = self._status
            return RecordingStatus(
                state           = s.state,
                elapsed_s       = s.elapsed_s,
                remaining_s     = s.remaining_s,
                duration_s      = s.duration_s,
                frames_recorded = s.frames_recorded,
                frames_dropped  = s.frames_dropped,
                current_fps     = s.current_fps,
                progress_pct    = s.progress_pct,
                quality         = s.quality,
                frame_loss_pct  = s.frame_loss_pct,
                error_title     = s.error_title,
                error_detail    = s.error_detail,
                suggestion      = s.suggestion,
                clip_path       = s.clip_path,
            )

    # ----------------------------------------------------------

    def wait_for_completion(self, timeout: float = 300.0) -> Optional[Path]:
        """
        Block until the recording thread finishes.

        Parameters
        ----------
        timeout : float   maximum seconds to wait

        Returns
        -------
        Path | None   clip file path on success, None on failure/timeout.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

        status = self.get_status()
        if status.state == RecordingState.COMPLETE and status.clip_path:
            return Path(status.clip_path)
        return None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    # ----------------------------------------------------------
    # Background recording loop
    # ----------------------------------------------------------

    def _record_loop(self,
                      duration_s:  float,
                      resolution:  Tuple[int, int],
                      fps:         float) -> None:
        """
        Main recording loop (runs in a daemon thread).

        * Writes frames to a temp MP4 file.
        * Verifies each frame (size, not None).
        * Detects stream interruptions and attempts recovery.
        * Updates self._status every frame.
        """
        w, h          = resolution
        total_frames  = int(fps * duration_s)
        frame_delay   = 1.0 / fps

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix="microscope_capture_"
        )
        tmp_path = Path(tmp.name)
        tmp.close()

        writer = cv2.VideoWriter(str(tmp_path), _FOURCC, fps, (w, h))
        if not writer.isOpened():
            self._set_failed(
                "VideoWriter Failed",
                f"Could not create output file at {tmp_path}.",
                "Check disk space and write permissions in the temp directory.",
            )
            tmp_path.unlink(missing_ok=True)
            return

        # ── Recording variables ────────────────────────────────
        frames_written  = 0
        frames_dropped  = 0
        fps_samples:    list = []
        t_start         = time.perf_counter()
        t_last_frame    = time.perf_counter()
        t_last_fps_calc = time.perf_counter()
        recovery_count  = 0
        current_fps_val = 0.0

        try:
            while not self._stop_event.is_set():
                elapsed   = time.perf_counter() - t_start
                remaining = max(0.0, duration_s - elapsed)
                progress  = min(100.0, elapsed / duration_s * 100.0)

                if elapsed >= duration_s:
                    break

                t_frame_start = time.perf_counter()
                frame         = self._cam.grab_frame()

                if frame is None:
                    frames_dropped += 1
                    time_since_last = time.perf_counter() - t_last_frame

                    # Stream interruption detection
                    if time_since_last > _FRAME_TIMEOUT_S:
                        if recovery_count < _MAX_RECOVERY_ATTEMPTS:
                            self._set_recovering(recovery_count + 1)
                            time.sleep(_RECOVERY_WAIT_S)
                            recovery_count += 1
                            t_last_frame = time.perf_counter()
                            continue
                        else:
                            self._set_failed(
                                "Camera Stream Interrupted",
                                f"No frames received for {_FRAME_TIMEOUT_S:.0f}s "
                                f"after {_MAX_RECOVERY_ATTEMPTS} recovery attempts.",
                                "Check the USB connection or smartphone camera connection and reconnect the device.",
                            )
                            break
                    time.sleep(0.01)
                    continue

                # Valid frame received
                recovery_count = 0
                t_last_frame   = time.perf_counter()

                # Resize if needed
                fh, fw = frame.shape[:2]
                if (fw, fh) != (w, h):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                writer.write(frame)
                frames_written += 1

                # FPS calculation
                dt = time.perf_counter() - t_frame_start
                if 0 < dt < 1.0:
                    fps_samples.append(1.0 / dt)
                    if len(fps_samples) > 30:
                        fps_samples.pop(0)

                if time.perf_counter() - t_last_fps_calc >= 0.5:
                    current_fps_val = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
                    t_last_fps_calc = time.perf_counter()

                total_attempts = frames_written + frames_dropped
                loss_pct = (
                    frames_dropped / total_attempts * 100.0
                    if total_attempts > 0 else 0.0
                )

                # Update shared status
                with self._lock:
                    self._status.state           = RecordingState.RECORDING
                    self._status.elapsed_s       = elapsed
                    self._status.remaining_s     = remaining
                    self._status.frames_recorded = frames_written
                    self._status.frames_dropped  = frames_dropped
                    self._status.current_fps     = round(current_fps_val, 1)
                    self._status.progress_pct    = progress
                    self._status.frame_loss_pct  = round(loss_pct, 2)
                    self._status.quality         = _quality_label(
                        loss_pct, current_fps_val, fps
                    )

                # Throttle to target FPS
                elapsed_frame = time.perf_counter() - t_frame_start
                sleep_for = frame_delay - elapsed_frame
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            writer.release()
            self._is_recording = False

        # ── Finalise ───────────────────────────────────────────
        if self._stop_event.is_set() and frames_written == 0:
            self._is_recording = False
            tmp_path.unlink(missing_ok=True)
            with self._lock:
                self._status.state = RecordingState.IDLE
            return

        if frames_written == 0:
            self._is_recording = False
            tmp_path.unlink(missing_ok=True)
            self._set_failed(
                "No Frames Recorded",
                "The recording loop completed but no frames were written.",
                "Ensure the selected USB Microscope or 3D Smartphone Microscope is streaming correctly and try again.",
            )
            return

        logger.success(
            "Recording complete: {} frames written, {} dropped, {:.1f}s",
            frames_written, frames_dropped,
            time.perf_counter() - t_start,
        )

        self._is_recording = False
        with self._lock:
            self._status.state           = RecordingState.COMPLETE
            self._status.frames_recorded = frames_written
            self._status.progress_pct    = 100.0
            self._status.clip_path       = str(tmp_path)

    # ----------------------------------------------------------
    # Private status setters (called from the recording thread)
    # ----------------------------------------------------------

    def _set_failed(self, title: str, detail: str, suggestion: str) -> None:
        with self._lock:
            self._status.state        = RecordingState.FAILED
            self._status.error_title  = title
            self._status.error_detail = detail
            self._status.suggestion   = suggestion
        logger.error("Recording FAILED — {}: {}", title, detail)

    def _set_recovering(self, attempt: int) -> None:
        with self._lock:
            self._status.state = RecordingState.RECOVERING
        logger.warning("Recording: stream interrupted — recovery attempt {}/{}",
                       attempt, _MAX_RECOVERY_ATTEMPTS)