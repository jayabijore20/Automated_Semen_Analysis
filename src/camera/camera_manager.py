"""
src/camera/camera_manager.py
==============================
Phase 6 · Camera Manager — Real Hardware Detection & Validation

Genuine hardware enumeration, platform-aware device naming, and
frame-level validation before any preview starts.

Responsibilities
----------------
* Enumerate every camera the host OS exposes (indices 0-15).
* Retrieve device names via platform APIs:
    Windows  -> DirectShow enumeration (pygrabber, if installed) with a
                WMI / PowerShell fallback.
    Linux    -> /sys/class/video4linux (name file + v4l2-ctl fallback).
    macOS    -> AVFoundation via system_profiler.
* Classify each device using a configurable keyword blocklist /
  allowlist, so external imaging devices (USB microscopes, digital
  microscopes, generic USB cameras) are correctly separated from the
  laptop's built-in webcam -- WITHOUT relying on device index, which
  is not stable across OS reboots / USB re-enumeration (especially on
  Windows).
* Validate the selected device (open + set resolution + read frame).
* Provide ``CameraDevice`` dataclass used throughout the application.
* Provide ``CameraManager`` -- the single object stored in session_state.

Why index-based selection is unsafe
------------------------------------
Windows in particular does not guarantee a stable mapping between
OpenCV device index and physical hardware across reboots or USB
re-plugs.  This module NEVER assumes "index 0 = webcam" or
"index 1 = microscope" -- classification is always based on the
device's reported name, matched against the keyword lists below.

Why a naive "usb" substring match is not enough
-------------------------------------------------
Many laptops' built-in webcams are internally wired as UVC/USB
devices and are reported by the OS with names such as
"USB2.0 HD UVC WebCam" or "Integrated Camera (USB)".  A classifier
that only checks for the substring "usb" will misclassify these as
external cameras and connect to the laptop webcam -- which is exactly
the bug this module fixes.  The blocklist below is therefore always
checked FIRST and always wins over the allowlist.

Public API used by app.py / Section 5 live UI
------------------------------------------------
    from src.camera.camera_manager import CameraManager, CameraDevice, DeviceClass

    mgr              = CameraManager()
    devices          = mgr.enumerate_devices()          # list[CameraDevice], ALL devices found
    imaging_devices  = mgr.get_imaging_devices(devices)  # list[CameraDevice], external only
    result           = mgr.connect(device)               # ConnectionResult
    frame            = mgr.grab_frame()                  # np.ndarray | None
    status           = mgr.get_status()                  # CameraStatus
    mgr.disconnect()
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════
# Device classification keyword lists (CONFIGURABLE)
# ══════════════════════════════════════════════════════════════
# All matching is done against the lower-cased device name.  Edit
# these lists to tune detection for your lab's specific hardware --
# no other code needs to change.
#
# Matching order (see _classify_device):
#   1. EXCLUDED_KEYWORDS   -> always INTEGRATED_WEBCAM, always hidden
#   2. MICROSCOPE_KEYWORDS -> USB_MICROSCOPE / DIGITAL_MICROSCOPE
#   3. USB_CAMERA_KEYWORDS -> USB_CAMERA
#   4. no match            -> UNKNOWN (hidden by default -- see
#                              get_imaging_devices())

# Built-in / integrated laptop webcams. ALWAYS excluded, even if the
# reported name also happens to contain the word "usb" -- many
# internal webcams are internally UVC/USB devices.
EXCLUDED_KEYWORDS: List[str] = [

    "integrated",
    "internal",
    "built-in",
    "builtin",
    "built in",

    "easycamera",
    "truevision",
    "facetime",
    "rgb camera",
    "rgb-ir",
    "ir camera",
    "infrared",
    "hello",
    "surface",
    "front camera",
    "user facing",

    "integrated camera",
    "integrated webcam",
    "internal webcam",

    "hp truevision",
    "lenovo easycamera",
    "dell webcam",
    "asus webcam",
    "acer webcam",
    "webcam hd",

    "usb2.0 hd uvc webcam",
    "usb hd webcam",
]

# USB / digital microscope hardware. Matching a keyword here always
# takes priority over the generic USB_CAMERA classification.
MICROSCOPE_KEYWORDS: List[str] = [
    "microscope",
    "usb microscope",
    "digital microscope",
    "microscope camera",
    "amscope",
    "celestron",
    "dino-lite",
    "dinolite",
    "dino lite",
    "andonstar",
    "omax",
    "hayear",
    "mustcam",
    "tomlov",
    "vms1010",
    "koolertron",
    "plugable usb microscope",
    "jiusion",
    "wireless microscope",
]

# Generic external USB cameras that are NOT microscopes but are still
# valid external imaging devices (e.g. a UVC capture camera pointed at
# a lab microscope's eyepiece via a C-mount adapter).
USB_CAMERA_KEYWORDS: List[str] = [
    "usb camera",
    "usb2.0 camera",
    "usb3.0 camera",
    "hd usb camera",
    "uvc camera",
    "uvc",
    "usb video",
    "external camera",
    "capture card",
    "hdmi capture",
    "video capture",
]


# ══════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════
class DeviceClass(str, Enum):
    USB_MICROSCOPE     = "USB Microscope"
    DIGITAL_MICROSCOPE = "Digital Microscope"
    USB_CAMERA          = "USB Camera"
    INTEGRATED_WEBCAM   = "Integrated Webcam"   # never shown in the UI
    UNKNOWN              = "Unknown Device"


# Device classes that are allowed to appear in the live-microscope UI.
# INTEGRATED_WEBCAM and UNKNOWN are always excluded -- an unnamed /
# unidentifiable device is treated conservatively as "not a microscope"
# rather than risking a silent fallback to the laptop webcam.
IMAGING_DEVICE_CLASSES: Tuple[DeviceClass, ...] = (
    DeviceClass.USB_MICROSCOPE,
    DeviceClass.DIGITAL_MICROSCOPE,
    DeviceClass.USB_CAMERA,
)


@dataclass
class CameraDevice:
    """
    Metadata for one physical camera device.

    Attributes
    ----------
    index          : int    OpenCV device index
    raw_name       : str    name from OS API (may be empty)
    display_name   : str    human-readable label shown in UI
    device_class   : DeviceClass
    backend        : str    OpenCV backend string
    supported_res  : list of (w, h) tuples   detected resolutions
    default_fps    : float
    path           : str    device path (/dev/video0 on Linux, "" elsewhere)
    is_available   : bool   True if the device opened without error
    is_usb         : bool   True if the device is confirmed / assumed USB-connected
    """
    index:         int
    raw_name:      str             = ""
    display_name:  str             = ""
    device_class:  DeviceClass     = DeviceClass.UNKNOWN
    backend:       str             = ""
    supported_res: List[Tuple[int, int]] = field(default_factory=list)
    default_fps:   float           = 30.0
    path:          str             = ""
    is_available:  bool            = False
    is_usb:        bool            = False

    def label(self) -> str:
        """Short label for the dropdown: display_name + device class + index."""
        return f"{self.display_name}  ·  {self.device_class.value}  (index {self.index})"

    @property
    def is_imaging_device(self) -> bool:
        """True if this device is an allowed external imaging device."""
        return self.device_class in IMAGING_DEVICE_CLASSES


@dataclass
class ConnectionResult:
    """Result of CameraManager.connect()."""
    success:      bool
    device:       Optional[CameraDevice] = None
    error_title:  str                    = ""
    error_detail: str                    = ""
    suggestion:   str                    = ""


@dataclass
class CameraStatus:
    """Live status snapshot -- used by the status panel in the UI."""
    connected:       bool   = False
    device_name:     str    = ""
    resolution:      str    = ""
    target_fps:      float  = 0.0
    actual_fps:      float  = 0.0
    frames_received: int    = 0
    frames_dropped:  int    = 0
    is_recording:    bool   = False
    uptime_s:        float  = 0.0


# ══════════════════════════════════════════════════════════════
# Platform device-name helpers
# ══════════════════════════════════════════════════════════════

def _get_device_names_linux() -> Dict[int, str]:
    """
    Read device names from /sys/class/video4linux on Linux.

    Returns
    -------
    dict  {index: name}
    """
    names: Dict[int, str] = {}
    video_dir = Path("/sys/class/video4linux")
    if not video_dir.exists():
        return names

    for dev_path in sorted(video_dir.iterdir()):
        name_file = dev_path / "name"
        index_str = dev_path.name.replace("video", "")
        if not index_str.isdigit():
            continue
        idx = int(index_str)
        if name_file.exists():
            try:
                names[idx] = name_file.read_text().strip()
            except OSError:
                pass
        else:
            # Try v4l2-ctl
            try:
                result = subprocess.run(
                    ["v4l2-ctl", f"--device=/dev/video{idx}", "--info"],
                    capture_output=True, text=True, timeout=2
                )
                for line in result.stdout.splitlines():
                    if "Card type" in line or "card" in line.lower():
                        names[idx] = line.split(":", 1)[-1].strip()
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    return names


def _is_usb_linux(idx: int) -> Optional[bool]:
    """
    Determine whether a Linux /dev/videoN device is attached via the
    USB bus (as opposed to the internal "platform" bus most built-in
    webcams use). This is a strong, reliable signal on Linux.

    Returns
    -------
    True/False if determinable, None if the sysfs path is unavailable.
    """
    dev_path = Path(f"/sys/class/video4linux/video{idx}/device")
    try:
        real = dev_path.resolve()
    except OSError:
        return None
    if not dev_path.exists():
        return None
    return "usb" in str(real).lower()


def _get_device_names_macos() -> Dict[int, str]:
    """
    Use system_profiler to get camera names on macOS.

    Returns
    -------
    dict  {index: name}   (index is best-effort; OS doesn't guarantee order)
    """
    names: Dict[int, str] = {}
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5
        )
        idx = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("Camera") and ":" not in line:
                names[idx] = line
                idx += 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return names


def _get_device_names_windows() -> Dict[int, str]:
    """
    Get Windows camera names using DirectShow / pygrabber.

    pygrabber provides the DirectShow device list that corresponds
    much more closely to the OpenCV CAP_DSHOW camera indices.
    """

    

    # -------------------------------------------------------
    # Preferred: pygrabber / DirectShow
    # -------------------------------------------------------
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        devices = list(graph.get_input_devices())

        names: Dict[int, str] = {}

        for idx, name in enumerate(devices):
            names[idx] = str(name)

        print("=" * 70)
        print("WINDOWS DIRECTSHOW DEVICES")
        print("=" * 70)

        for idx, name in names.items():
            print(f"Index {idx} -> {name}")

        print("=" * 70)
        logger.info(
            "Windows DirectShow devices detected: {}",
            names
        )

        return names

    except ImportError:
        logger.warning(
            "pygrabber is not installed. "
            "Install it with: pip install pygrabber"
        )

    except Exception as exc:
        logger.warning(
            "pygrabber enumeration failed: %s",
            exc
        )

    # -------------------------------------------------------
    # Fallback: WMI
    # -------------------------------------------------------
    try:
        import win32com.client

        wmi = win32com.client.GetObject("winmgmts:")
        cameras = wmi.InstancesOf("Win32_PnPEntity")

        names: Dict[int, str] = {}
        idx = 0

        for cam in cameras:
            name = getattr(cam, "Name", "") or ""
            pnp_class = getattr(cam, "PNPClass", "") or ""
            

            name_lower = name.lower()
            pnp_class_lower = pnp_class.lower()

            if (
                pnp_class_lower == "camera"
                or "camera" in name.lower()
                or "webcam" in name.lower()
                or "video" in name.lower()
                or "vms1010" in name_lower
                or "tomlov" in name_lower
            ):
                names[idx] = name
                idx += 1

            print("=" * 70)
            print("WINDOWS WMI DEVICES")
            print("=" * 70)

            for idx, name in names.items():
               print(f"Index {idx} -> {name}")

            print("=" * 70)

            return names

    except ImportError:
        logger.warning(
            "pywin32 is not installed."
        )

    except Exception as exc:
        logger.warning(
            "WMI camera enumeration failed: %s",
            exc
        )

    return {}


def _get_platform_device_names() -> Dict[int, str]:
    """Dispatch to the correct platform API."""
    os_name = platform.system().lower()
    try:
        if os_name == "linux":
            return _get_device_names_linux()
        elif os_name == "darwin":
            return _get_device_names_macos()
        elif os_name == "windows":
            return _get_device_names_windows()
    except Exception as exc:
        logger.debug("Platform device name query failed: {}", exc)
    return {}


def _matches_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def _classify_device(name: str, index: int) -> Tuple[DeviceClass, str]:
    """
    Classify a device by its reported name.

    The blocklist (``EXCLUDED_KEYWORDS``) is always checked FIRST and
    always wins -- this is what prevents a laptop webcam whose driver
    name happens to contain "USB" (e.g. "USB2.0 HD UVC WebCam") from
    being misclassified as an external microscope/camera.

    A device with no usable name at all is classified UNKNOWN and is
    hidden from the UI by default (see ``get_imaging_devices``) --
    professional lab software should never guess.

    Parameters
    ----------
    name  : str   raw OS-reported device name (may be empty)
    index : int   OpenCV device index (used only for the display label)

    Returns
    -------
    (DeviceClass, display_name)
    """
    has_real_name = bool(name and name.strip())
    display_name = name.strip() if has_real_name else f"Unidentified Camera {index}"
    lname = display_name.lower()

    if not has_real_name:
        # No OS-level name available -- cannot safely confirm this is
        # an external imaging device. Default to UNKNOWN (hidden).
        return DeviceClass.UNKNOWN, display_name

    if _matches_any(lname, EXCLUDED_KEYWORDS):
        return DeviceClass.INTEGRATED_WEBCAM, display_name

    if _matches_any(lname, MICROSCOPE_KEYWORDS):
        if "digital" in lname:
            return DeviceClass.DIGITAL_MICROSCOPE, display_name
        return DeviceClass.USB_MICROSCOPE, display_name

    if _matches_any(lname, USB_CAMERA_KEYWORDS):
        return DeviceClass.USB_CAMERA, display_name

    return DeviceClass.UNKNOWN, display_name


def _probe_resolutions(cap: cv2.VideoCapture) -> List[Tuple[int, int]]:
    """
    Probe a set of common resolutions against an open VideoCapture.

    Parameters
    ----------
    cap : cv2.VideoCapture   must already be opened

    Returns
    -------
    list of (width, height) that the device accepted
    """
    candidates = [
        (640,  480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ]
    supported: List[Tuple[int, int]] = []
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) == (w, h):
            supported.append((w, h))

    # Restore original
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  orig_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)

    if not supported:
        supported.append((orig_w, orig_h))

    return supported


# ══════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════
class CameraError(Exception):
    """Base class for all camera-related errors."""


class NoDeviceFoundError(CameraError):
    """Raised when no external imaging device is available."""


class DeviceBusyError(CameraError):
    """Raised when the requested device could not be opened (in use elsewhere)."""


class FrameReadError(CameraError):
    """Raised when a device opens but never returns a valid frame."""


class DeviceDisconnectedError(CameraError):
    """Raised when a previously-connected device stops responding."""


# ══════════════════════════════════════════════════════════════
# CameraManager
# ══════════════════════════════════════════════════════════════
class CameraManager:
    """
    Hardware-validated camera manager.

    One instance is stored in
    ``st.session_state["_camera_manager"]``
    and reused across all Streamlit reruns.
    """

    def __init__(self, max_index: int = 15) -> None:
        self._max_index = max_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._device: Optional[CameraDevice] = None
        self._connected = False

        # Live stats
        self._frames_received = 0
        self._frames_dropped = 0
        self._connect_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        self._fps_samples: List[float] = []

        # Background frame reader
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_thread: Optional[threading.Thread] = None
        self._frame_stop_event = threading.Event()
        self._frame_lock = threading.Lock()

    # ==========================================================
    # Backend / Camera Opening
    # ==========================================================

    def _open_camera(self, index):
        print("=" * 60)
        print("OPEN CAMERA")
        print("Opening OpenCV Index :", index)
        print("=" * 60)

        cap = None

        # ------------------------------------------------------
        # 1. Try DirectShow first
        # ------------------------------------------------------
        print("Trying DirectShow...")

        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        except Exception as exc:
            print("DirectShow exception:", exc)
            cap = None

        # ------------------------------------------------------
        # 2. Fallback to default OpenCV backend
        # ------------------------------------------------------
        if cap is None or not cap.isOpened():
            print("DirectShow failed.")

            if cap is not None:
                cap.release()

            print("Trying default OpenCV backend...")

            try:
                cap = cv2.VideoCapture(index)
            except Exception as exc:
                print("Default backend exception:", exc)
                cap = None

        # ------------------------------------------------------
        # 3. Could not open camera
        # ------------------------------------------------------
        if cap is None or not cap.isOpened():
            print("Camera could not be opened.")

            if cap is not None:
                cap.release()

            return None

        print("Camera opened successfully.")

        # ------------------------------------------------------
        # 4. Force MJPEG
        # ------------------------------------------------------
        try:
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG")
            )

            print("Requested FOURCC: MJPG")

        except Exception as exc:
            print("Could not set MJPG FOURCC:", exc)

        # ------------------------------------------------------
        # 5. Conservative initialization resolution
        # ------------------------------------------------------
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            print("Requested initial resolution: 1280x720")

        except Exception as exc:
            print("Could not set initial resolution:", exc)

        # ------------------------------------------------------
        # 6. Give USB camera time to initialize
        # ------------------------------------------------------
        time.sleep(2)

        # ------------------------------------------------------
        # 7. Flush initial frames
        # ------------------------------------------------------
        valid_frame = False
        frame = None

        for i in range(15):

            try:
                ret, frame = cap.read()

            except cv2.error as exc:
                print(
                    f"OpenCV error while reading "
                    f"initialization frame {i}:"
                )
                print(exc)

                time.sleep(0.2)
                continue

            except Exception as exc:
                print(
                    f"Camera read exception during "
                    f"initialization frame {i}:"
                )
                print(exc)

                time.sleep(0.2)
                continue

            print(
                f"Initialization frame {i}: "
                f"ret={ret}, "
                f"frame={'None' if frame is None else frame.shape}"
            )

            if (
                ret
                and frame is not None
                and isinstance(frame, np.ndarray)
                and frame.size > 0
                and frame.ndim >= 2
            ):
                valid_frame = True
                break

            time.sleep(0.2)

        # ------------------------------------------------------
        # 8. No valid frame
        # ------------------------------------------------------
        if not valid_frame:
            print("=" * 60)
            print("CAMERA OPENED BUT NO VALID FRAME RECEIVED")
            print("Index:", index)
            print("=" * 60)

            cap.release()
            return None

        print("=" * 60)
        print("VALID CAMERA FRAME RECEIVED")
        print("Index:", index)
        print("Frame shape:", frame.shape)
        print("=" * 60)

        return cap

    # ==========================================================
    # Enumeration
    # ==========================================================

    def enumerate_devices(self) -> List[CameraDevice]:
        """
        Scan every camera index and return CameraDevice objects.

        Every successfully opened camera is returned, including
        integrated webcams and unidentified cameras.
        """

        logger.info(
            "Enumerating camera devices (max index {})",
            self._max_index
        )

        # Get OS-level device names once
        os_names = _get_platform_device_names()

        devices: List[CameraDevice] = []
        system = platform.system()

        for idx in range(self._max_index + 1):

            print("\n" + "=" * 70)
            print("ENUMERATION TEST")
            print("Testing OpenCV Index :", idx)
            print("=" * 70)

            cap = self._open_camera(idx)

            if cap is None:
                print("RESULT: CAMERA COULD NOT BE OPENED")
                print("SKIPPING INDEX:", idx)
                print("=" * 70)
                continue

            print("RESULT: CAMERA OPENED SUCCESSFULLY")
            print("INDEX:", idx)
            print("=" * 70)

            if not cap.isOpened():
                print("RESULT: cap.isOpened() = FALSE")
                cap.release()
                continue

            # --------------------------------------------------
            # Camera properties
            # --------------------------------------------------
            w = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
            )

            h = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
            )

            fps = (
                cap.get(cv2.CAP_PROP_FPS)
                or 30.0
            )

            # --------------------------------------------------
            # Verify real frame
            # --------------------------------------------------
            ret = False
            frame = None

            for _ in range(10):

                try:
                    ret, frame = cap.read()

                except cv2.error as exc:
                    print("OpenCV frame error:", exc)
                    ret = False
                    frame = None

                except Exception as exc:
                    print("Camera frame exception:", exc)
                    ret = False
                    frame = None

                if (
                    ret
                    and frame is not None
                    and isinstance(frame, np.ndarray)
                    and frame.size > 0
                ):
                    break

                time.sleep(0.1)

            is_available = bool(
                ret
                and frame is not None
                and isinstance(frame, np.ndarray)
                and frame.size > 0
            )

            print("=" * 70)
            print("FRAME TEST")
            print("Index           :", idx)
            print("ret             :", ret)
            print(
                "Frame           :",
                "None" if frame is None else frame.shape
            )
            print("is_available    :", is_available)
            print("=" * 70)

            # --------------------------------------------------
            # Device name
            # --------------------------------------------------
            raw_name = os_names.get(idx, "")

            print(f"OpenCV Index = {idx}")
            print(f"OS Name      = {raw_name}")

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------
            dev_class, display_name = _classify_device(
                raw_name,
                idx
            )

            # --------------------------------------------------
            # Windows unknown-camera fallback
            # --------------------------------------------------
            if (
                system == "Windows"
                and is_available
                and dev_class == DeviceClass.UNKNOWN
            ):

                print("=" * 70)
                print("WINDOWS UNKNOWN CAMERA FALLBACK")
                print("OpenCV Index :", idx)
                print("Raw Name     :", raw_name)
                print("Frame works  :", is_available)
                print("Action       : Treating as USB Camera")
                print("=" * 70)

                dev_class = DeviceClass.USB_CAMERA

                if (
                    not display_name
                    or display_name == "Unknown Device"
                ):
                    display_name = (
                        raw_name
                        or f"USB Camera (Index {idx})"
                    )

            # --------------------------------------------------
            # Debug
            # --------------------------------------------------
            print("=" * 70)
            print("OpenCV Index   :", idx)
            print("OS Name        :", raw_name)
            print("Display Name   :", display_name)
            print("Classification :", dev_class.value)
            print("=" * 70)

            # --------------------------------------------------
            # Device path
            # --------------------------------------------------
            dev_path = (
                f"/dev/video{idx}"
                if system.lower() == "linux"
                else ""
            )

            # --------------------------------------------------
            # USB confirmation
            # --------------------------------------------------
            if system.lower() == "linux":

                usb_flag = _is_usb_linux(idx)

                is_usb = (
                    bool(usb_flag)
                    if usb_flag is not None
                    else (
                        dev_class
                        in IMAGING_DEVICE_CLASSES
                    )
                )

            else:

                is_usb = (
                    dev_class
                    in IMAGING_DEVICE_CLASSES
                )

            print("=" * 70)
            print("USB / DEVICE CLASS DEBUG")
            print("Index          :", idx)
            print(
                "Device Class   :",
                dev_class.value
            )
            print(
                "IMAGING CLASS? :",
                dev_class in IMAGING_DEVICE_CLASSES
            )
            print("is_usb         :", is_usb)
            print("=" * 70)

            # --------------------------------------------------
            # Probe supported resolutions
            # --------------------------------------------------
            if is_available:
                supported_res = _probe_resolutions(cap)
            else:
                supported_res = [(w, h)]

            cap.release()

            # --------------------------------------------------
            # Backend
            # --------------------------------------------------
            if system == "Windows":
                backend = "DirectShow"

            elif system == "Linux":
                backend = "V4L2"

            elif system == "Darwin":
                backend = "AVFoundation"

            else:
                backend = "Unknown"

            # --------------------------------------------------
            # Create CameraDevice
            # --------------------------------------------------
            device = CameraDevice(
                index=idx,
                raw_name=raw_name,
                display_name=display_name,
                device_class=dev_class,
                backend=backend,
                supported_res=supported_res,
                default_fps=fps,
                path=dev_path,
                is_available=is_available,
                is_usb=is_usb,
            )

            devices.append(device)

            print(
                "DEVICE STATUS -> "
                f"index={device.index}, "
                f"name={device.display_name}, "
                f"class={device.device_class.value}, "
                f"available={device.is_available}, "
                f"imaging={device.is_imaging_device}"
            )

            if (
                dev_class
                == DeviceClass.INTEGRATED_WEBCAM
            ):

                logger.info(
                    "Excluded device {} ({}) -- "
                    "classified as integrated webcam, "
                    "will never be shown or auto-selected.",
                    idx,
                    display_name,
                )

            else:

                logger.debug(
                    "Found device {}: {} [{}] "
                    "available={} usb={}",
                    idx,
                    display_name,
                    dev_class.value,
                    is_available,
                    is_usb,
                )

        logger.info(
            "Enumeration complete: {} device(s) found, "
            "{} imaging-capable",
            len(devices),
            sum(
                1
                for d in devices
                if d.is_available
                and d.is_imaging_device
            ),
        )

        return devices

    # ==========================================================
    # Imaging Devices
    # ==========================================================

    def get_imaging_devices(
        self,
        devices: Optional[List[CameraDevice]] = None
    ) -> List[CameraDevice]:

        """
        Return only available external imaging devices.
        """

        if devices is None:
            devices = self.enumerate_devices()

        print("=" * 70)
        print("GET IMAGING DEVICES - INPUT")
        print("=" * 70)

        for d in devices:

            print(
                f"Index={d.index} | "
                f"Name={d.display_name} | "
                f"Class={d.device_class.value} | "
                f"Available={d.is_available} | "
                f"USB={d.is_usb} | "
                f"Imaging={d.is_imaging_device}"
            )

        print("=" * 70)

        imaging_devices = [
            d
            for d in devices
            if d.is_available
            and d.is_imaging_device
        ]

        print("=" * 70)
        print("IMAGING DEVICES")
        print("=" * 70)

        for d in imaging_devices:

            print(
                f"INDEX={d.index} | "
                f"NAME={d.display_name} | "
                f"CLASS={d.device_class.value} | "
                f"AVAILABLE={d.is_available}"
            )

        print("=" * 70)

        return imaging_devices

    # ==========================================================
    # Connection
    # ==========================================================

    def connect(
        self,
        device: CameraDevice,
        resolution: str = "1280×720",
        fps: float = 30.0
    ) -> ConnectionResult:

        """
        Open and validate the selected camera device.
        """

        self.disconnect()

        # ------------------------------------------------------
        # Validate device
        # ------------------------------------------------------
        if not device.is_imaging_device:

            logger.error(
                "Refused to connect: device {} ({}) "
                "is classified as {}, "
                "not an external imaging device.",
                device.index,
                device.display_name,
                device.device_class.value,
            )

            return ConnectionResult(
                success=False,
                error_title="Not a Supported Microscope Device",
                error_detail=(
                    f"'{device.display_name}' is classified as "
                    f"{device.device_class.value} and cannot "
                    f"be used for analysis."
                ),
                suggestion=(
                    "Select a USB Microscope, Digital Microscope, "
                    "or USB Camera from the list."
                ),
            )

        # ------------------------------------------------------
        # Parse resolution
        # ------------------------------------------------------
        sep = "×" if "×" in resolution else "x"

        parts = resolution.split(sep)

        try:
            req_w = int(parts[0].strip())
            req_h = int(parts[1].strip())

        except (IndexError, ValueError):
            req_w = 1280
            req_h = 720

        logger.info(
            "Connecting to device {} ({})",
            device.index,
            device.display_name,
        )

        print("=" * 60)
        print("CONNECT CALLED")
        print("Selected Device Index :", device.index)
        print("Selected Device Name  :", device.display_name)
        print("=" * 60)

        # ------------------------------------------------------
        # Open exact selected device
        # ------------------------------------------------------
        with self._lock:
            cap = self._open_camera(device.index)

        print("=" * 60)
        print("CONNECTED TO")
        print("Requested index :", device.index)

        if cap is None:

            print("Opened          : False")
            print("Camera could not be opened.")
            print("=" * 60)

            return ConnectionResult(
                success=False,
                error_title="Device Could Not Be Opened",
                error_detail=(
                    f"OpenCV could not open camera "
                    f"index {device.index}."
                ),
                suggestion=(
                    "Close any application using the camera "
                    "and reconnect it."
                ),
            )

        print("Opened          :", cap.isOpened())
        print("=" * 60)

        if not cap.isOpened():

            cap.release()

            return ConnectionResult(
                success=False,
                error_title="Device Could Not Be Opened",
                error_detail=(
                    f"OpenCV could not open camera "
                    f"index {device.index}. "
                    f"The device may be in use by "
                    f"another application."
                ),
                suggestion=(
                    "Close any application using the camera "
                    "and reconnect it."
                ),
            )

        # ------------------------------------------------------
        # Set requested properties
        # ------------------------------------------------------
        try:
            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                req_w
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                req_h
            )

            cap.set(
                cv2.CAP_PROP_FPS,
                fps
            )

        except Exception as exc:

            print("Could not set camera properties:")
            print(exc)

        # ------------------------------------------------------
        # Read actual properties
        # ------------------------------------------------------
        actual_w = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_h = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        actual_fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or fps
        )

        print("=" * 60)
        print("Camera Properties")
        print("WIDTH =", actual_w)
        print("HEIGHT =", actual_h)
        print("FPS =", actual_fps)
        print("=" * 60)

        # ------------------------------------------------------
        # Validate frame
        # ------------------------------------------------------
        frame_received = False

        for i in range(20):

            try:
                ret, frame = cap.read()

                print(
                    f"Attempt {i} | "
                    f"ret={ret} | "
                    f"frame="
                    f"{'None' if frame is None else frame.shape}"
                )

            except cv2.error as exc:

                print(
                    f"OpenCV error while reading frame "
                    f"{i}:"
                )
                print(exc)

                ret = False
                frame = None

            except Exception as exc:

                print(
                    f"Camera read exception frame {i}:"
                )
                print(exc)

                ret = False
                frame = None

            if (
                ret
                and frame is not None
                and isinstance(frame, np.ndarray)
                and frame.size > 0
                and frame.ndim >= 2
            ):
                frame_received = True
                break

            time.sleep(0.3)

        # ------------------------------------------------------
        # Frame failed
        # ------------------------------------------------------
        if not frame_received:

            cap.release()

            return ConnectionResult(
                success=False,
                error_title="Frame Read Failed",
                error_detail=(
                    "OpenCV failed to read a frame "
                    "from the selected camera."
                ),
                suggestion=(
                    "Camera opened but no frames were received."
                ),
            )

        # ------------------------------------------------------
        # Save successful connection
        # ------------------------------------------------------
        device.supported_res = [
            (actual_w, actual_h)
        ]

        device.default_fps = actual_fps
        device.is_available = True

        self._cap = cap
        self._device = device
        self._connected = True

        self._frames_received = 0
        self._frames_dropped = 0

        self._connect_time = time.time()
        self._last_frame_time = time.time()

        self._fps_samples = []

        self._frame_stop_event.clear()

        with self._frame_lock:
              self._latest_frame = frame.copy()

        self._frame_thread = threading.Thread(
            target=self._frame_reader_loop,
            name="CameraFrameReader",
            daemon=True,
        )

        self._frame_thread.start() 

        logger.success(
            "Connected: {} @ {}x{} {:.0f}fps",
            device.display_name,
            actual_w,
            actual_h,
            actual_fps,
        )

        return ConnectionResult(
            success=True,
            device=device
        )

    # ==========================================================
    # Disconnect
    # ==========================================================

    def disconnect(self) -> None:
        """Release the camera device."""


         # Stop background frame reader
        self._frame_stop_event.set()

        if self._frame_thread is not None:
           if self._frame_thread.is_alive():
              self._frame_thread.join(timeout=2.0)

        self._frame_thread = None

        with self._lock:

            if self._cap is not None:

                try:
                    self._cap.release()

                except Exception:
                    pass

                self._cap = None

            self._connected = False
            self._device = None
            self._connect_time = None

            self._frames_received = 0
            self._frames_dropped = 0
            self._fps_samples = []

             # Clear the last frame
            with self._frame_lock:
                self._latest_frame = None

    # ==========================================================
    # Frame Grab
    # ==========================================================

    def grab_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame captured by the background reader."""

        print("========== grab_frame() ==========")

        if not self._connected:
            print("Camera not connected")
            return None

        frame = self.get_latest_frame()

        if frame is None:
            self._frames_dropped += 1
            print("grab_frame: No frame available")
            return None

        self._frames_received += 1
        self._last_frame_time = time.time()

        return frame


    def _frame_reader_loop(self) -> None:
        """Continuously read frames in the background."""

        print("BACKGROUND FRAME READER STARTED")

        while not self._frame_stop_event.is_set():

                with self._lock:
                    cap = self._cap
                    connected = self._connected

                if not connected or cap is None:
                    time.sleep(0.01)
                    continue

                try:
                    ret, frame = cap.read()

                    if (
                        ret
                        and frame is not None
                        and isinstance(frame, np.ndarray)
                        and frame.size > 0
                    ):

                        # Store ONLY the newest frame.
                        # Never build a frame queue.
                        with self._frame_lock:
                            self._latest_frame = frame.copy()



                        if self._frames_received % 30 == 0:
                            print(
                                "LIVE CAMERA:",
                                "frames =", self._frames_received,
                                "shape =", frame.shape,
                                "mean =", float(frame.mean()),
                            )    

                        self._frames_received += 1
                        self._last_frame_time = time.time()

                    else:

                        self._frames_dropped += 1
                        time.sleep(0.01)

                except Exception as exc:

                    logger.error("Background camera frame error: {}", exc)
                    self._frames_dropped += 1
                    time.sleep(0.01)

        print("BACKGROUND FRAME READER STOPPED")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the newest frame captured by the background reader."""

        with self._frame_lock:

            if self._latest_frame is None:
                return None

            return self._latest_frame.copy()




    # ==========================================================
    # Status
    # ==========================================================

    def get_status(self) -> CameraStatus:
        """Return a snapshot of live camera status."""

        with self._lock:

            dev = self._device
            cap = self._cap

            if (
                not self._connected
                or dev is None
                or cap is None
            ):
                return CameraStatus(
                    connected=False
                )

            w = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            h = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = (
                cap.get(cv2.CAP_PROP_FPS)
                or dev.default_fps
            )

            actual_fps = (
                sum(self._fps_samples)
                / len(self._fps_samples)
                if self._fps_samples
                else 0.0
            )

            uptime = (
                time.time() - self._connect_time
                if self._connect_time
                else 0.0
            )

            return CameraStatus(
                connected=True,
                device_name=dev.display_name,
                resolution=f"{w} × {h}",
                target_fps=fps,
                actual_fps=round(actual_fps, 1),
                frames_received=self._frames_received,
                frames_dropped=self._frames_dropped,
                is_recording=False,
                uptime_s=uptime,
            )

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device(
        self
    ) -> Optional[CameraDevice]:
        return self._device

    @property
    def cap(
        self
    ) -> Optional[cv2.VideoCapture]:
        """Direct access for RecordingManager."""

        return self._cap

    # ==========================================================
    # Context Manager
    # ==========================================================

    def __enter__(
        self
    ) -> "CameraManager":

        return self

    def __exit__(
        self,
        *_: object
    ) -> None:

        self.disconnect()

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self) -> str:

        status = (
            "connected"
            if self._connected
            else "disconnected"
        )

        name = (
            self._device.display_name
            if self._device
            else "—"
        )

        return (
            f"CameraManager("
            f"{name}, "
            f"{status}, "
            f"{self._frames_received} frames)"
        )