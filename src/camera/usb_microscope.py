"""
src/camera/usb_microscope.py
==============================
USB Digital Microscope Controller

Manages the full lifecycle of a USB camera attached to the host machine
and used as a digital microscope.  Designed to run inside a Streamlit
application via simple method calls; all OpenCV I/O is encapsulated here
so the frontend never touches cv2.VideoCapture directly.

Responsibilities
----------------
* Connect / disconnect to the camera device.
* Deliver single frames for live preview (``grab_frame``).
* Record a fixed-duration clip to a temporary MP4 file
  (``record_clip``).  The returned ``Path`` is passed directly into
  ``AutomatedSemenAnalysis.run_video`` — no other code changes are needed.
* Expose simple status properties used by the UI.

Thread-safety
-------------
``grab_frame`` and ``record_clip`` are called synchronously inside
Streamlit reruns.  Because Streamlit's script-level execution is
single-threaded this is safe.  A ``threading.Lock`` is used anyway
to guard the ``cv2.VideoCapture`` object so the module works correctly
if a calling application uses background threads.

Supported platforms
-------------------
Any OS that OpenCV can open via ``cv2.VideoCapture(device_index)``.
On Linux the default index is usually 0; on Windows it may be 0 or 1
depending on built-in webcams.  The ``discover_cameras`` helper scans
indices 0-9 and returns those that open successfully.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
# Resolution helper
# ══════════════════════════════════════════════════════════════

def _parse_resolution(res_str: str) -> Tuple[int, int]:
    """
    Convert a resolution string like "1280×720" or "1280x720"
    to an (width, height) integer tuple.

    Parameters
    ----------
    res_str : str

    Returns
    -------
    (width, height)
    """
    sep = "×" if "×" in res_str else "x"
    parts = res_str.split(sep)
    return int(parts[0].strip()), int(parts[1].strip())


def _parse_fps(fps_str: str) -> float:
    """
    Convert a FPS string like "30 FPS" to a float.

    Parameters
    ----------
    fps_str : str

    Returns
    -------
    float
    """
    return float(fps_str.upper().replace("FPS", "").strip())


# ══════════════════════════════════════════════════════════════
# Camera discovery
# ══════════════════════════════════════════════════════════════

def discover_cameras(max_index: int = 9) -> List[Dict]:
    """
    Scan device indices 0–max_index and return metadata for every
    camera that OpenCV can open.

    Parameters
    ----------
    max_index : int
        Highest device index to probe (inclusive).

    Returns
    -------
    list of dict
        Each dict has keys:
            index      int
            label      str   e.g. "Camera 0 (640×480 @ 30.0 FPS)"
            width      int
            height     int
            fps        float
    """
    found: List[Dict] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        found.append({
            "index": idx,
            "label": f"Camera {idx}  ({w}×{h} @ {fps:.0f} FPS)",
            "width": w,
            "height": h,
            "fps":   fps,
        })
    return found


# ══════════════════════════════════════════════════════════════
# USBMicroscope
# ══════════════════════════════════════════════════════════════

class USBMicroscope:
    """
    Stateful controller for a single USB digital microscope.

    Typical call sequence
    ---------------------
    ::

        mic = USBMicroscope(device_index=0,
                            resolution="1280×720",
                            fps="30 FPS")

        mic.connect()                   # opens cv2.VideoCapture
        frame = mic.grab_frame()        # returns BGR np.ndarray or None
        ...
        clip_path = mic.record_clip(duration_s=12)   # records ~12 s
        mic.disconnect()

        # Pass clip_path straight into the existing pipeline:
        results = pipeline.run_video(clip_path)

    Parameters
    ----------
    device_index : int
        OpenCV device index (0 = first USB camera).
    resolution : str
        Resolution string e.g. "1280×720" or "640×480".
    fps : str | float
        FPS string e.g. "30 FPS" or a plain float.
    """

    # Default recording duration when caller does not specify
    DEFAULT_DURATION_S: float = 12.0

    # Output codec for recorded clips
    _FOURCC: int = cv2.VideoWriter_fourcc(*"mp4v")
    _CLIP_SUFFIX: str = ".mp4"

    def __init__(
        self,
        device_index: int = 0,
        resolution: str = "1280×720",
        fps: str | float = "30 FPS",
    ) -> None:
        self._device_index = device_index
        self._res_w, self._res_h = _parse_resolution(resolution)
        self._fps = float(fps) if isinstance(fps, (int, float)) else _parse_fps(fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._connected = False

        # Path of the most recently recorded clip (cleared on connect)
        self._last_clip: Optional[Path] = None

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True when the camera device is open."""
        return self._connected

    @property
    def last_clip_path(self) -> Optional[Path]:
        """Path to the last recorded clip, or None."""
        return self._last_clip

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height) tuple."""
        return self._res_w, self._res_h

    @property
    def fps(self) -> float:
        return self._fps

    # ----------------------------------------------------------
    # Connection management
    # ----------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the camera device.

        Returns
        -------
        bool
            True if the camera opened successfully.
        """
        with self._lock:
            if self._connected:
                return True

            cap = cv2.VideoCapture(self._device_index)
            if not cap.isOpened():
                cap.release()
                self._connected = False
                return False

            # Request target resolution and FPS from the device.
            # (The device driver may silently cap these to its hardware limits.)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._res_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res_h)
            cap.set(cv2.CAP_PROP_FPS,          self._fps)

            # Read back what the driver actually accepted
            self._res_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or self._res_w)
            self._res_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._res_h)
            self._fps    = cap.get(cv2.CAP_PROP_FPS) or self._fps

            self._cap        = cap
            self._connected  = True
            self._last_clip  = None
            return True

    def disconnect(self) -> None:
        """Release the camera device and free all resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._connected = False

    # ----------------------------------------------------------
    # Frame grab
    # ----------------------------------------------------------

    def grab_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns
        -------
        np.ndarray (BGR, H×W×3) or None if the camera is not connected
        or the read fails.
        """
        with self._lock:
            if self._cap is None or not self._connected:
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            return frame

    # ----------------------------------------------------------
    # Clip recording
    # ----------------------------------------------------------

    def record_clip(
        self,
        duration_s: float = DEFAULT_DURATION_S,
        on_progress: Optional[callable] = None,
    ) -> Optional[Path]:
        """
        Record a fixed-duration video clip to a temporary MP4 file.

        The method blocks for approximately ``duration_s`` seconds
        while writing frames to disk.  After it returns, the path can
        be passed directly into ``AutomatedSemenAnalysis.run_video``.

        Parameters
        ----------
        duration_s : float
            Duration of the clip in seconds (default 12).
        on_progress : callable | None
            Optional callback invoked every second with signature
            ``on_progress(elapsed_s: float, total_s: float)``.
            Useful for updating a Streamlit progress bar.

        Returns
        -------
        Path | None
            Path to the saved MP4 file, or None on failure.
        """
        with self._lock:
            if self._cap is None or not self._connected:
                return None

            # Create a named temp file that persists after this method returns.
            # The caller is responsible for deleting it (via Path.unlink).
            tmp = tempfile.NamedTemporaryFile(
                suffix=self._CLIP_SUFFIX,
                delete=False,
                prefix="usb_capture_",
            )
            tmp_path = Path(tmp.name)
            tmp.close()

            writer = cv2.VideoWriter(
                str(tmp_path),
                self._FOURCC,
                self._fps,
                (self._res_w, self._res_h),
            )

            if not writer.isOpened():
                tmp_path.unlink(missing_ok=True)
                return None

            total_frames  = int(self._fps * duration_s)
            frame_delay   = 1.0 / self._fps
            written       = 0
            last_progress = 0.0
            t_start       = time.perf_counter()

            for _ in range(total_frames):
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    # Camera dropped a frame — write a black frame to keep sync
                    frame = np.zeros(
                        (self._res_h, self._res_w, 3), dtype=np.uint8
                    )

                writer.write(frame)
                written += 1

                now = time.perf_counter() - t_start
                if on_progress is not None and now - last_progress >= 1.0:
                    on_progress(now, duration_s)
                    last_progress = now

                # Throttle to the target FPS
                elapsed = time.perf_counter() - t_start
                expected = written * frame_delay
                sleep_for = expected - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            writer.release()
            self._last_clip = tmp_path
            return tmp_path

    # ----------------------------------------------------------
    # Context manager support
    # ----------------------------------------------------------

    def __enter__(self) -> "USBMicroscope":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ----------------------------------------------------------
    # Repr
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"USBMicroscope(index={self._device_index}, "
            f"{self._res_w}×{self._res_h} @ {self._fps:.0f}fps, {status})"
        )