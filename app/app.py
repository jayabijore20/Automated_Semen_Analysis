"""
app/app.py
===========
Automated Semen Viability and Counting Analysis
Professional Laboratory Dashboard — Phase 3: Time-Interval Monitoring

Phase 3 additions over Phase 2
--------------------------------
* New §6 "Monitoring Configuration" form card.
* New BRANCH C — monitoring dashboard rendered when a monitoring session
  is active (``_monitor_session`` key is set in session_state).
* A background ``_MonitorRunner`` thread drives ``IntervalMonitor.start()``
  and ``.tick()`` (see BUGFIX note below) so a full capture + analysis
  cycle never blocks the Streamlit script thread.
* ``streamlit-autorefresh`` triggers reruns at ``_MONITOR_TICK_MS``
  (default 10 s) purely to redraw the current monitoring snapshot.
* Monitoring results table with sparkline-style trend column.
* "Stop Monitoring" button available at any time.
* All Phase 1 (intake form, upload, WHO results) and Phase 2
  (live USB microscope, 6 camera buttons) code is 100 % preserved.

BUGFIX note (background-thread architecture for long-running work)
--------------------------------------------------------------------
Both the single-sample AI analysis and interval monitoring involve
calls that can block for a long time (a full detection/tracking/
morphology/viability pipeline run, or a microscope capture+analysis
cycle). Earlier versions of this file called that blocking work
directly from the Streamlit script thread inside a
``while ...: time.sleep(...)`` loop, while simultaneously using
``streamlit-autorefresh`` to trigger periodic reruns. Because a new
rerun request can interrupt an in-flight script run for the same
session, this combination could kill a capture or analysis partway
through -- silently losing results, dropping monitoring captures, and
occasionally leaving a temp video file locked (Windows ``WinError 32``)
because the file was deleted while a still-running background thread
was reading it.

The fix used throughout this file is the same state-driven pattern in
every case: a background ``threading.Thread`` (``ProgressTracker`` for
single analysis, ``_MonitorRunner`` for monitoring, both already
thread-safe) performs the blocking work, while the Streamlit thread
only ever reads a snapshot of that thread's current state, renders it
once, and schedules the next rerun -- it never blocks and is therefore
never a target for a competing rerun to interrupt mid-operation. A temp
video file is only ever deleted once the owning background thread has
confirmed completion (``tracker.is_done``), never inside a ``finally``
tied to a script-run scope that might get interrupted.

Session-state persistence note
--------------------------------
Monitoring state lives in ``st.session_state`` and survives Streamlit
reruns.  It does NOT survive a hard browser refresh or server restart.
A warning banner is shown so users are aware.

Run
---
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ── Ensure project root is on sys.path ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config                               # noqa: E402
from src.camera.usb_microscope import USBMicroscope, discover_cameras  # noqa: E402
from src.camera.monitor import IntervalMonitor, MonitorSession         # noqa: E402
from src.camera.camera_manager import CameraManager, CameraDevice, ConnectionResult, DeviceClass
from src.camera.recording_manager import RecordingManager, RecordingState  # noqa: E402
from src.camera.analysis_progress import ProgressTracker, StageStatus  # noqa: E402
from src.analysis.trend_analysis import TrendAnalyser, TrendReport    # noqa: E402
from src.analysis import concentration as conc                         # noqa: E402
from src.calibration import optical_calibration as oc                  # noqa: E402
from src.reporting.monitoring_report import MonitoringReportGenerator  # noqa: E402
from src.database.db_manager import DatabaseManager                    # noqa: E402
from pages.database_page import render_database_page
from loguru import logger
# ══════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SpermVision Lab · Semen Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**SpermVision Lab — Automated Semen Viability and Counting Analysis**\n\n"
            "AI-powered andrology pipeline using YOLOv8, ByteTrack, "
            "EfficientNetB0, and Random Forest.\n\n"
            "⚠️ For research and screening use only.  "
            "Results must be confirmed by a certified andrologist."
        )
    },
)

# ══════════════════════════════════════════════════════════════
# Design tokens
# ══════════════════════════════════════════════════════════════

_CLR_PRIMARY   = "#1d4ed8"
_CLR_SECONDARY = "#0f172a"
_CLR_ACCENT    = "#06b6d4"
_CLR_SUCCESS   = "#10b981"
_CLR_WARNING   = "#f59e0b"
_CLR_DANGER    = "#ef4444"
_CLR_BG        = "#f1f5f9"
_CLR_CARD      = "#ffffff"
_CLR_BORDER    = "#e2e8f0"
_CLR_TEXT_PRI  = "#0f172a"
_CLR_TEXT_SEC  = "#64748b"
_CLR_SIDEBAR   = "#0f172a"
_CLR_TEXT      = _CLR_TEXT_PRI
_CLR_MONITOR   = "#7c3aed"   # violet — monitoring brand colour

# ══════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════

_CSS = f"""
<style>
* {{ box-sizing:border-box; }}
html,body {{ font-family:'Inter','Segoe UI',system-ui,sans-serif; }}

[data-testid="stAppViewContainer"] {{ background:{_CLR_BG}; }}
[data-testid="stHeader"]           {{ background:transparent; }}
[data-testid="stSidebar"] {{
    background:{_CLR_SIDEBAR};border-right:1px solid #1e293b;
}}
[data-testid="stSidebar"] *  {{ color:#cbd5e1 !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color:#f1f5f9 !important; }}
[data-testid="stSidebar"] hr {{ border-color:#334155; }}
[data-testid="stSidebarNav"] {{ display:none; }}

/* ── Typography ─────────────────────────────── */
.lab-logo-text {{
    font-size:1.35rem;font-weight:800;letter-spacing:-0.3px;
    color:{_CLR_PRIMARY};line-height:1.1;
}}
.lab-logo-sub {{
    font-size:0.72rem;color:{_CLR_TEXT_SEC};
    letter-spacing:0.12em;text-transform:uppercase;margin-top:1px;
}}
.page-title {{
    font-size:1.85rem;font-weight:800;color:{_CLR_SECONDARY};
    letter-spacing:-0.4px;line-height:1.2;margin:0;
}}
.page-subtitle {{ font-size:0.92rem;color:{_CLR_TEXT_SEC};margin-top:0.25rem; }}

/* ── Form card ──────────────────────────────── */
.form-card {{
    background:{_CLR_CARD};border:1px solid {_CLR_BORDER};
    border-radius:14px;padding:1.35rem 1.6rem 1.1rem 1.6rem;
    margin-bottom:1.1rem;
    box-shadow:0 1px 3px rgba(0,0,0,.05),0 1px 2px rgba(0,0,0,.04);
}}
.form-card-title {{
    font-size:0.8rem;font-weight:700;letter-spacing:0.08em;
    text-transform:uppercase;color:{_CLR_PRIMARY};
    border-bottom:2px solid {_CLR_PRIMARY}20;
    padding-bottom:0.5rem;margin-bottom:0.9rem;
    display:flex;align-items:center;gap:0.4rem;
}}

/* ── Metric card ────────────────────────────── */
.metric-card {{
    background:{_CLR_CARD};border:1px solid {_CLR_BORDER};
    border-radius:12px;padding:1rem 1.2rem 0.85rem 1.2rem;
    border-top:4px solid {_CLR_PRIMARY};
    box-shadow:0 1px 3px rgba(0,0,0,.05);margin-bottom:0.6rem;
}}
.metric-card-value {{
    font-size:1.85rem;font-weight:800;color:{_CLR_SECONDARY};
    line-height:1;margin-bottom:0.2rem;
}}
.metric-card-label {{
    font-size:0.72rem;color:{_CLR_TEXT_SEC};
    text-transform:uppercase;letter-spacing:0.07em;font-weight:600;
}}

/* ── WHO score ──────────────────────────────── */
.who-score-box {{
    border-radius:16px;padding:1.8rem 1.4rem;text-align:center;
    color:white;box-shadow:0 4px 24px rgba(0,0,0,.12);
}}
.who-score-number {{ font-size:4.5rem;font-weight:900;line-height:1; }}
.who-score-denom  {{ font-size:1.1rem;opacity:.75;margin-top:-2px; }}
.who-score-label  {{ font-size:1.4rem;font-weight:700;margin-top:0.6rem; }}
.cat-excellent {{ background:linear-gradient(135deg,#059669,#10b981); }}
.cat-good      {{ background:linear-gradient(135deg,{_CLR_PRIMARY},#3b82f6); }}
.cat-average   {{ background:linear-gradient(135deg,#d97706,{_CLR_WARNING}); }}
.cat-poor      {{ background:linear-gradient(135deg,#dc2626,{_CLR_DANGER}); }}

/* ── Alerts ─────────────────────────────────── */
.flag-box {{
    background:#fef2f2;border-left:4px solid {_CLR_DANGER};
    border-radius:8px;padding:0.6rem 1rem;
    margin:0.3rem 0;font-size:0.875rem;color:#991b1b;
}}
.rec-box {{
    background:#f0fdf4;border-left:4px solid {_CLR_SUCCESS};
    border-radius:8px;padding:0.6rem 1rem;
    margin:0.3rem 0;font-size:0.875rem;color:#166534;
}}

/* ── Section header ─────────────────────────── */
.section-header {{
    font-size:0.95rem;font-weight:700;color:{_CLR_SECONDARY};
    border-bottom:2px solid {_CLR_BORDER};
    padding-bottom:0.35rem;margin:1.1rem 0 0.65rem 0;
}}

/* ── Sidebar stage badge ─────────────────────── */
.stage-badge {{
    display:inline-flex;align-items:center;gap:0.3rem;
    background:#1e293b;border-radius:6px;
    padding:0.2rem 0.55rem;font-size:0.78rem;
    color:#94a3b8;margin:0.15rem 0;width:100%;
}}
.stage-badge-num {{
    background:{_CLR_PRIMARY};color:white;border-radius:4px;
    padding:0 0.3rem;font-size:0.7rem;font-weight:700;
}}

/* ── Status bar ─────────────────────────────── */
.status-bar {{
    background:{_CLR_SECONDARY};border-radius:10px;
    padding:0.55rem 1.1rem;display:flex;
    justify-content:space-between;align-items:center;
    margin-bottom:1rem;flex-wrap:wrap;gap:0.4rem;
}}
.status-bar-label {{
    color:#94a3b8;font-size:0.75rem;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;
}}
.status-bar-value {{ color:#f1f5f9;font-size:0.85rem;font-weight:700; }}

/* ── Camera panel ───────────────────────────── */
.cam-status-on  {{ color:#4ade80;font-weight:700;font-size:0.85rem; }}
.cam-status-off {{ color:#f87171;font-weight:700;font-size:0.85rem; }}
.cam-preview-label {{
    font-size:0.72rem;color:#94a3b8;text-transform:uppercase;
    letter-spacing:0.1em;font-weight:700;margin-bottom:4px;
}}
.capture-ready {{
    background:#f0fdf4;border:2px solid {_CLR_SUCCESS};
    border-radius:10px;padding:0.7rem 1rem;
    font-size:0.875rem;color:#166534;font-weight:600;
    display:flex;align-items:center;gap:0.5rem;margin:0.5rem 0;
}}

/* ── Monitoring dashboard ────────────────────── */
.monitor-header {{
    background:linear-gradient(135deg,{_CLR_MONITOR},{_CLR_PRIMARY});
    border-radius:16px;padding:1.4rem 1.8rem;color:white;
    margin-bottom:1.2rem;
    box-shadow:0 4px 24px {_CLR_MONITOR}30;
}}
.monitor-title {{ font-size:1.3rem;font-weight:800;line-height:1.2; }}
.monitor-sub   {{ font-size:0.85rem;opacity:0.82;margin-top:0.25rem; }}
.monitor-kpi-row {{
    display:flex;gap:1rem;margin-top:1rem;flex-wrap:wrap;
}}
.monitor-kpi {{
    background:rgba(255,255,255,.15);border-radius:10px;
    padding:0.55rem 1rem;flex:1;min-width:110px;
}}
.monitor-kpi-val  {{ font-size:1.6rem;font-weight:800;line-height:1; }}
.monitor-kpi-lbl  {{ font-size:0.7rem;opacity:0.78;text-transform:uppercase;
    letter-spacing:0.07em; }}
.countdown-box {{
    background:{_CLR_SECONDARY};border-radius:12px;padding:1rem 1.4rem;
    border-left:4px solid {_CLR_MONITOR};margin-bottom:1rem;
}}
.countdown-val {{
    font-size:2.4rem;font-weight:900;color:{_CLR_MONITOR};
    font-variant-numeric:tabular-nums;line-height:1;
}}
.countdown-lbl {{ font-size:0.8rem;color:#94a3b8;margin-top:2px; }}
.interval-row {{
    display:flex;gap:0.5rem;align-items:center;margin:0.2rem 0;
    flex-wrap:wrap;
}}
.interval-dot {{
    width:12px;height:12px;border-radius:50%;flex-shrink:0;
}}
.interval-dot-done    {{ background:{_CLR_SUCCESS}; }}
.interval-dot-active  {{ background:{_CLR_MONITOR};animation:pulse 1s infinite; }}
.interval-dot-pending {{ background:#334155; }}
.result-row-ok  {{ background:#f0fdf4;border-radius:6px;margin:2px 0; }}
.result-row-err {{ background:#fef2f2;border-radius:6px;margin:2px 0; }}

/* ── Buttons ────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {{
    background:linear-gradient(135deg,{_CLR_PRIMARY},#2563eb) !important;
    border:none !important;border-radius:10px !important;
    font-weight:700 !important;font-size:1rem !important;
    padding:0.65rem 2rem !important;color:white !important;
    letter-spacing:0.02em !important;
    box-shadow:0 4px 14px {_CLR_PRIMARY}40 !important;
    transition:opacity .2s,transform .15s !important;
}}
[data-testid="stButton"] > button[kind="primary"]:hover {{
    opacity:0.9 !important;transform:translateY(-1px) !important;
}}

/* ── Upload Video File button ──────────────── */
[data-testid="stButton"] > button {{
    background:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
    border-radius:10px !important;
    font-weight:700 !important;
    opacity:1 !important;
}}

[data-testid="stButton"] > button p {{
    color:#111827 !important;
}}

[data-testid="stButton"] > button span {{
    color:#111827 !important;
}}

[data-testid="stButton"] > button:hover {{
    background:#f1f5f9 !important;
    color:#111827 !important;
    border-color:#94a3b8 !important;
}}

[data-testid="stButton"] > button:hover p,
[data-testid="stButton"] > button:hover span {{
    color:#111827 !important;
}}

/* ── Progress bar ───────────────────────────── */
[data-testid="stProgressBar"] > div > div {{
    background:linear-gradient(90deg,{_CLR_PRIMARY},{_CLR_ACCENT}) !important;
    border-radius:4px !important;
}}

/* ── Upload zone ────────────────────────────── */
[data-testid="stFileUploader"] {{
    background:white !important;border-radius:12px !important;
    border:2px dashed #93c5fd !important;padding:0.8rem !important;
}}

/* ── File uploader button contrast ──────────── */
[data-testid="stFileUploader"] button {{
    background:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
    border-radius:8px !important;
    opacity:1 !important;
}}

[data-testid="stFileUploader"] button span {{
    color:#111827 !important;
}}

[data-testid="stFileUploader"] button svg {{
    color:#111827 !important;
    fill:#111827 !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background:#f1f5f9 !important;
    color:#111827 !important;
}}

[data-testid="stFileUploader"] button:hover span {{
    color:#111827 !important;
}}

[data-testid="stFileUploader"] button:hover svg {{
    color:#111827 !important;
    fill:#111827 !important;
}}

/* ── Text colour safety ─────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color:{_CLR_TEXT_PRI} !important;
}}

.stTable table,
.stTable td,
.stTable th {{
    color:{_CLR_TEXT_PRI} !important;
}}

[data-testid="stMetricValue"] {{
    color:{_CLR_SECONDARY} !important;
}}

[data-testid="stMetricLabel"] {{
    color:{_CLR_TEXT_SEC} !important;
}}


/* =========================================================
   STREAMLIT INPUT / DROPDOWN CONTRAST
   ========================================================= */

/* Selectbox container */
[data-testid="stSelectbox"] {{
    color:{_CLR_TEXT_PRI} !important;
}}

/* Selectbox visible field */
[data-testid="stSelectbox"] [data-baseweb="select"] {{
    background-color:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
    border-radius:8px !important;
}}

/* Selectbox inner elements */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
    background-color:#ffffff !important;
    color:#111827 !important;
}}

/* Selected value */
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {{
    color:#111827 !important;
}}

/* Dropdown popup */
[data-baseweb="popover"] {{
    background-color:#ffffff !important;
    color:#111827 !important;
}}

/* Dropdown menu */
[data-baseweb="menu"] {{
    background-color:#ffffff !important;
    color:#111827 !important;
}}

/* Every dropdown option */
[data-baseweb="menu"] [role="option"] {{
    background-color:#ffffff !important;
    color:#111827 !important;
}}

/* Option text */
[data-baseweb="menu"] [role="option"] *,
[data-baseweb="menu"] [role="option"] span {{
    color:#111827 !important;
}}

/* Hover option */
[data-baseweb="menu"] [role="option"]:hover {{
    background-color:#e5e7eb !important;
    color:#111827 !important;
}}

/* Hover option text */
[data-baseweb="menu"] [role="option"]:hover * {{
    color:#111827 !important;
}}

/* Selected option */
[data-baseweb="menu"] [role="option"][aria-selected="true"] {{
    background-color:#dbeafe !important;
    color:#111827 !important;
}}

/* Selected option text */
[data-baseweb="menu"] [role="option"][aria-selected="true"] * {{
    color:#111827 !important;
}}


/* =========================================================
   TEXT INPUTS
   ========================================================= */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background-color:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
}}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInput"] input::placeholder {{
    color:#6b7280 !important;
    opacity:1 !important;
}}

/* =========================================================
   INPUT LABELS
   ========================================================= */


[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background-color:#ffffff !important;
    color:#111827 !important;
    border:1px solid #cbd5e1 !important;
}}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stNumberInput"] input::placeholder {{
    color:#6b7280 !important;
    opacity:1 !important;
}}

/* =========================================================
   RADIO / CHECKBOX LABELS
   ========================================================= */

[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {{
    color:{_CLR_TEXT_PRI} !important;
}}


/* =========================================================
   SELECTBOX ARROW / ICON
   ========================================================= */

[data-testid="stSelectbox"] svg {{
    fill:#475569 !important;
    color:#475569 !important;
}}
/* ── Live / recording badge ─────────────────── */
.live-badge {{
    display:inline-block;background:{_CLR_DANGER};color:white;
    font-size:0.65rem;font-weight:800;letter-spacing:0.1em;
    border-radius:4px;padding:1px 6px;vertical-align:middle;
    margin-left:6px;animation:blink 1.4s infinite;
}}
.rec-badge {{
    display:inline-block;background:{_CLR_MONITOR};color:white;
    font-size:0.65rem;font-weight:800;letter-spacing:0.1em;
    border-radius:4px;padding:1px 6px;vertical-align:middle;
    margin-left:6px;animation:blink 0.9s infinite;
}}
@keyframes blink  {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.3; }} }}
@keyframes pulse  {{ 0%,100% {{ opacity:1;transform:scale(1); }}
                     50%      {{ opacity:.6;transform:scale(1.25); }} }}
hr {{ border-color:{_CLR_BORDER} !important;margin:0.8rem 0 !important; }}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Dropdown option lists  (identical to Phase 1 & 2)
# ══════════════════════════════════════════════════════════════

_SAMPLE_TYPES         = ["Fresh Semen","Frozen-Thawed","Chilled","Stored"]
_SPECIES              = ["Cattle","Buffalo","Goat","Sheep","Horse","Pig","Human"]
_BREEDS               = [
    "Holstein Friesian","Jersey","Gir","Sahiwal","Red Sindhi",
    "Tharparkar","Murrah","Jaffarabadi","Nili Ravi","Other",
]
_CHAMBERS             = [
    "Makler Chamber","Leja Chamber","Hemocytometer",
    "Standard Glass Slide","Other",
]
_MICROSCOPE_TYPES     = [
    "USB Digital Microscope","Optical Microscope + Camera",
    "Laboratory Microscope","Other",
]
_MAGNIFICATIONS       = ["10×","20×","40×","100×"]
_CAMERA_RESOLUTIONS   = ["640×480","1280×720","1920×1080"]
_CAMERA_FPS           = ["15 FPS","30 FPS","60 FPS"]
_STAINING_TYPES       = [
    "No Stain","Eosin Nigrosin","H&E","Diff Quik","Giemsa","Other",
]
_SUPPORTED_VIDEO_FORMATS = ["mp4","avi","mov","mkv"]

_CATEGORY_CSS         = {
    "Excellent":"cat-excellent","Good":"cat-good",
    "Average":"cat-average","Poor":"cat-poor",
}
_CATEGORY_EMOJI       = {
    "Excellent":"🏆","Good":"✅","Average":"⚠️","Poor":"❌",
}

_PROGRESS_STAGES = [
    ("Video Loaded",             5),
    ("Detection",               20),
    ("Tracking",                38),
    ("Motility Analysis",       55),
    ("Morphology",              70),
    ("Viability Prediction",    82),
    ("WHO Scoring",             90),
    ("Report Generation",       96),
]

_RECORD_DURATION_S   : float = 12.0
_PREVIEW_REFRESH_MS  : int   = 150
_MONITOR_TICK_MS     : int   = 10_000     # rerun every 10 s during monitoring

# Monitoring configuration options
_MONITOR_INTERVALS   = [5, 10, 20, 30]    # minutes
_MONITOR_DURATIONS   = [30, 60, 90, 120]  # minutes

# ══════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def _load_config() -> Dict[str, Any]:
    try:
        return load_config("configs/config.yaml")
    except FileNotFoundError:
        st.error(
            "⚠️  `configs/config.yaml` not found.  "
            "Start Streamlit from the project root directory."
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def _load_pipeline(config_hash: int) -> Any:    # noqa: ARG001
    from src.run_analysis import AutomatedSemenAnalysis   # type: ignore[import]
    cfg = _load_config()
    return AutomatedSemenAnalysis(config=cfg)


@st.cache_resource(show_spinner=False)
def _load_db(config_hash: int) -> DatabaseManager:   # noqa: ARG001
    """Instantiate and cache the DatabaseManager (opens the SQLite file once)."""
    cfg = _load_config()
    return DatabaseManager(cfg)


# ══════════════════════════════════════════════════════════════
# HTML helpers
# ══════════════════════════════════════════════════════════════


def _card_open(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="form-card">'
        f'<div class="form-card-title"><span>{icon}</span> {title}</div>',
        unsafe_allow_html=True,
    )


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, accent: str = _CLR_PRIMARY) -> None:
    st.markdown(
        f'<div class="metric-card" style="border-top-color:{accent};">'
        f'<div class="metric-card-value">{value}</div>'
        f'<div class="metric-card-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _status_row(items: List[Tuple[str, str]]) -> None:
    inner = "".join(
        f'<span style="margin-right:1.8rem;">'
        f'<span class="status-bar-label">{lbl}&nbsp;&nbsp;</span>'
        f'<span class="status-bar-value">{val}</span></span>'
        for lbl, val in items
    )
    st.markdown(f'<div class="status-bar">{inner}</div>', unsafe_allow_html=True)


def _fmt_mmss(seconds: float) -> str:
    """Format seconds as MM:SS string."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════


def _render_sidebar() -> None:
    logo_path = Path("app/assets/logo.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
    else:
        st.sidebar.markdown(
            '<div style="padding:0.6rem 0 0.3rem 0;">'
            '<div class="lab-logo-text">Everse.AI</div>'
            '<div class="lab-logo-sub">AI Andrology System v3.0</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<span style="font-size:0.75rem;font-weight:700;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#475569;">'
        'Analysis Pipeline</span>',
        unsafe_allow_html=True,
    )
    for num, name in [
        ("1", "Detection"),
        ("2", "Tracking"),
        ("3", "Motility"),
        ("4", "Morphology"),
        ("5", "Viability"),
        ("6", "Scoring"),
        ("7", "Report"),
    ]:
      st.sidebar.markdown(
            f'<div class="stage-badge">'
            f'<span class="stage-badge-num">{num}</span>'
            f'<span style="font-weight:600;color:#e2e8f0;">{name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    tbl = '<table style="width:100%;border-collapse:collapse;margin-top:0.4rem;">'
    for p, v in [
        ("Concentration","≥ 16 M/mL"),("Total motility","≥ 42%"),
        ("Progressive","≥ 30%"),("Normal morphology","≥ 4%"),("Vitality","≥ 54%"),
    ]:
        tbl += (
            f'<tr><td style="padding:3px 0;font-size:0.75rem;color:#94a3b8;">{p}</td>'
            f'<td style="padding:3px 0;font-size:0.75rem;color:#e2e8f0;'
            f'text-align:right;font-weight:700;">{v}</td></tr>'
        )
    st.sidebar.markdown(
        '<span style="font-size:0.75rem;font-weight:700;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#475569;">'
        'WHO 2021 Reference Limits</span>' + tbl + "</table>",
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    ss = st.session_state
    if ss.get("_form_submitted"):
        st.sidebar.markdown(
            '<span style="font-size:0.75rem;font-weight:700;'
            'letter-spacing:0.1em;text-transform:uppercase;color:#475569;">'
            'Current Sample</span>',
            unsafe_allow_html=True,
        )
       # ── Sample volume ───────────────────────────────────────
    sample_volume_ul = st.sidebar.number_input(
        "Sample Volume Used (µL)",
        min_value=0.1,
        max_value=1000.0,
        value=float(ss.get("_sample_volume_ul", 2.0)),
        step=0.1,
        help="Enter the volume of semen loaded onto the slide using a micropipette.",
    )

    # Save it in session state
    ss["_sample_volume_ul"] = sample_volume_ul


    # ── Existing sample information ─────────────────────────
    for k, v in [
        ("Sample ID", ss.get("_sample_id", "—")),
        ("Animal ID", ss.get("_animal_id", "—")),
        ("Species",   ss.get("_species", "—")),
        ("Breed",     ss.get("_breed", "—")),
        ("Sample",    ss.get("_sample_type", "—")),
        ("Staining",  ss.get("_staining", "—")),
        ("Volume",    f"{sample_volume_ul:.1f} µL"),
    ]:
        st.sidebar.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:2px 0;">'
            f'<span style="font-size:0.73rem;color:#64748b;">{k}</span>'
            f'<span style="font-size:0.73rem;color:#e2e8f0;font-weight:600;">'
            f'{str(v)[:18]}</span></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    # ── Monitoring status in sidebar ──────────────────────────
    mon_session: Optional[MonitorSession] = ss.get("_monitor_session")
    if mon_session is not None and mon_session.is_running:
        elapsed_min = mon_session.elapsed_s() / 60
        remaining_s = max(0.0, mon_session.duration_s - mon_session.elapsed_s())
        st.sidebar.markdown(
            f'<div style="background:#2d1b69;border-radius:10px;padding:0.7rem 0.9rem;">'
            f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#a78bfa;">⏱ Monitoring Active</div>'
            f'<div style="font-size:1.4rem;font-weight:900;color:#c4b5fd;'
            f'font-variant-numeric:tabular-nums;">{_fmt_mmss(remaining_s)}</div>'
            f'<div style="font-size:0.72rem;color:#7c3aed;">remaining</div>'
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.35rem;">'
            f'Captures: {mon_session.captures_done} / {mon_session.expected_captures}<br>'
            f'Interval: {mon_session.interval_min} min · '
            f'Duration: {mon_session.duration_min} min</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif mon_session is not None and mon_session.is_complete:
        st.sidebar.markdown(
            f'<div style="background:#064e3b;border-radius:10px;padding:0.7rem 0.9rem;">'
            f'<div style="font-size:0.72rem;font-weight:700;color:#34d399;">'
            f'✅ Monitoring Complete</div>'
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.25rem;">'
            f'{mon_session.captures_done} captures over '
            f'{mon_session.duration_min} min</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        '<span style="font-size:0.75rem;font-weight:700;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#475569;">'
        'Navigation</span>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("_db_return_to_database"):
        st.session_state["_db_page"] = "database"
        st.session_state.pop("_db_return_to_database", None)


    current_page = st.session_state.get("_db_page", "analysis")
    btn_analysis = st.sidebar.button(
        "🔬  Analysis",
        use_container_width=True,
        key="_nav_analysis",
        type="primary" if current_page == "analysis" else "secondary",
    )
    btn_database = st.sidebar.button(
        "🗄️  Session Database",
        use_container_width=True,
        key="_nav_database",
        type="primary" if current_page == "database" else "secondary",
    )
    btn_calibration = st.sidebar.button(
        "📐  Calibration Setup",
        use_container_width=True,
        key="_nav_calibration",
        type="primary" if current_page == "calibration" else "secondary",
        help="Admin/lab one-time microscope optical calibration — not part of routine analysis.",
    )
    if btn_analysis:
        st.session_state["_db_page"] = "analysis"
        st.rerun()
    if btn_database:
        st.session_state["_db_page"] = "database"
        st.rerun()
    if btn_calibration:
        st.session_state["_db_page"] = "calibration"
        st.rerun()

    st.sidebar.caption(
        "⚠️ Research & screening use only.  "
        "Confirm results with a certified andrologist."
    )


# ══════════════════════════════════════════════════════════════
# §1 – §4  Intake form sections  (identical to Phase 1 & 2)
# ══════════════════════════════════════════════════════════════


def _render_section_1() -> Dict[str, Any]:
    _card_open("🧪", "Section 1 · Sample Information")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sample_id = st.text_input(
            "Sample ID *", placeholder="e.g. BVS-2024-0041",
            key="_inp_sample_id",
        )
    with c2:
        animal_id = st.text_input(
            "Animal ID", placeholder="e.g. BULL-HF-007",
            key="_inp_animal_id",
        )
    with c3:
        collection_date = st.date_input(
            "Collection Date", value=date.today(), key="_inp_collection_date",
        )
    c4, c5 = st.columns([1, 1])
    with c4:
        operator = st.text_input("Operator Name", placeholder="e.g. Dr. A. Kumar",
                                  key="_inp_operator")
    with c5:
        lab_name = st.text_input("Laboratory Name",
                                  placeholder="e.g. National Andrology Centre",
                                  key="_inp_lab_name")
    _card_close()
    return {
        "sample_id": sample_id, "animal_id": animal_id,
        "collection_date": str(collection_date),
        "operator": operator, "lab_name": lab_name,
    }


def _render_section_2() -> Dict[str, Any]:
    _card_open("🐄", "Section 2 · Sample Configuration")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sample_type = st.selectbox("Sample Type *", _SAMPLE_TYPES, index=0,
                                    key="_inp_sample_type")
    with c2:
        species = st.selectbox("Species *", _SPECIES, index=0, key="_inp_species")
    with c3:
        breed = st.selectbox("Breed / Strain", _BREEDS, index=0, key="_inp_breed")
    _card_close()
    return {"sample_type": sample_type, "species": species, "breed": breed}


def _render_section_3() -> Dict[str, Any]:
    _card_open("🔬", "Section 3 · Laboratory Configuration")
    c1, c2 = st.columns(2)
    with c1:
        chamber = st.selectbox("Counting Chamber", _CHAMBERS, index=0,
                                key="_inp_chamber")
        magnification = st.selectbox("Microscope Magnification", _MAGNIFICATIONS,
                                      index=2, key="_inp_magnification")
        camera_fps = st.selectbox("Camera FPS", _CAMERA_FPS, index=1,
                                   key="_inp_camera_fps")
    with c2:
        microscope_type = st.selectbox("Microscope Type", _MICROSCOPE_TYPES,
                                        index=0, key="_inp_microscope_type")
        camera_resolution = st.selectbox("Camera Resolution", _CAMERA_RESOLUTIONS,
                                          index=1, key="_inp_camera_resolution")
    _card_close()
    return {
        "chamber": chamber, "microscope_type": microscope_type,
        "magnification": magnification,
        "camera_resolution": camera_resolution, "camera_fps": camera_fps,
    }


def _slide_mode_from_chamber(chamber: str) -> str:
    """
    Map the existing 'chamber' selection (Section 3) onto the
    normal-slide vs counting-chamber distinction used for sample
    volume validation and concentration calculation (Feature 2/20/21).
    """
    return "normal_slide" if chamber == "Standard Glass Slide" else "chamber"


def _get_camera_identifier() -> str:
    """
    A stable "camera" identity string used as part of the exact
    hardware-configuration key for calibration lookup/storage (see
    `src.calibration.optical_calibration.build_configuration_key`).

    Uses the connected `CameraManager`'s reported device name when a
    camera is actually connected; falls back to a fixed generic label
    otherwise so the SAME identifier is used consistently between the
    Calibration Setup admin page and the normal analysis lookup (they
    must agree exactly, or a saved calibration would never be found).
    """
    mgr = st.session_state.get("_camera_manager")
    if mgr is not None:
        # Primary source: the connected device's reported name, via the
        # same `get_status()` the live status panel uses. This MUST be
        # the same string the Calibration Setup page files a
        # calibration under, or normal analysis would never find it.
        try:
            if getattr(mgr, "is_connected", False):
                name = getattr(mgr.get_status(), "device_name", "")
                if name:
                    return str(name)
        except Exception:
            pass
        try:
            info = mgr.get_debug_info()
            name = info.get("device_name") or info.get("name")
            if name:
                return str(name)
        except Exception:
            pass
    return "Default USB Camera"


def _render_calibration_status_block(
    microscope_type: str, magnification: str, camera_resolution: str,
) -> Dict[str, Any]:
    """
    Read-only optical-calibration status for the NORMAL operator
    workflow — replaces manual µm/pixel entry entirely (per the
    one-time calibration workflow: normal users never calculate or
    enter this value). Looks up a saved calibration for the exact
    current hardware configuration; if none exists, falls back to the
    legacy static `CALIBRATION_TABLE` (kept for backward compatibility
    with pre-existing configured entries); otherwise shows the exact
    "Optical calibration required" message and directs the operator to
    Calibration Setup.

    An "Administrator / Calibration Override" expander (collapsed by
    default, clearly labelled as not for routine use) remains
    available underneath, per the explicit allowance for an
    admin-only fallback.

    Returns
    -------
    dict with keys "saved_calibration_um_per_pixel" and
    "manual_microns_per_pixel" (the latter only ever populated via the
    admin override expander).
    """
    db = _load_db(load_config())
    camera_id = _get_camera_identifier()
    saved = db.get_calibration(microscope_type, camera_id, magnification, camera_resolution)

    st.markdown(
        f'<span style="font-size:0.9rem;font-weight:700;color:{_CLR_TEXT};">'
        "Optical Calibration</span>",
        unsafe_allow_html=True,
    )

    saved_um_per_px: Optional[float] = None
    if saved is not None:
        saved_um_per_px = saved["microns_per_pixel"]
        st.success(
            f"✅  Optical Calibration: Available — {magnification} @ {camera_resolution}\n\n"
            f"**Physical Scale: {saved_um_per_px:.4f} µm/pixel**  "
            f"(measured {saved['created_at']}, {microscope_type} / {camera_id})"
        )
        # Repeatability context, if repeat measurements exist for this
        # configuration. Reported without any pass/fail judgement.
        try:
            rep = db.get_calibration_repeatability(
                microscope_type, camera_id, magnification, camera_resolution,
            )
            if rep["n_measurements"] >= 2:
                st.caption(
                    f"Based on {rep['n_measurements']} repeat measurements — "
                    f"mean {rep['mean_um_per_pixel']:.4f} µm/pixel, "
                    f"CV {rep['cv_percent']:.2f}%."
                )
            else:
                st.caption(
                    "Single measurement — repeatability not yet assessed."
                )
        except Exception:
            pass
        # Scale-change safety: warn if the calibration was measured on
        # an image of different pixel dimensions than what is being
        # analysed now (crop/resize/binning would invalidate the scale).
        st.session_state["_active_calibration_record"] = saved
    else:
        table_value = conc.CALIBRATION_TABLE.get((magnification, camera_resolution))
        if table_value is not None:
            st.info(
                f"ℹ️  Using legacy configured value (not a saved stage-micrometer "
                f"calibration): **{table_value:g} µm/pixel** for {magnification} "
                f"@ {camera_resolution}."
            )
        else:
            st.warning(
                f"⚠️ Optical Calibration: Required\n\n"
                f"Calibration required for this imaging configuration:\n\n"
                f"**{microscope_type} / {camera_id} — {magnification} @ {camera_resolution}**\n\n"
                f"Please complete microscope calibration (see the "
                f"📐 Calibration Setup page) before quantitative "
                f"concentration analysis."
            )

    manual_um_per_px = 0.0
    with st.expander("🔒 Administrator / Calibration Override", expanded=False):
        st.caption(
            "Not intended for routine users. Overrides the saved/legacy "
            "calibration above for THIS analysis run only — does not "
            "save a new calibration record."
        )
        manual_um_per_px = st.number_input(
            "Override µm/pixel", min_value=0.0, max_value=100.0,
            value=0.0, step=0.01, format="%.4f",
            key="_inp_manual_um_per_px",
            help="Leave at 0 to use the calibration status shown above.",
        )

    return {
        "saved_calibration_um_per_pixel": saved_um_per_px,
        "manual_microns_per_pixel": manual_um_per_px if manual_um_per_px > 0 else None,
    }


def _render_section_4(chamber: str, magnification: str,
                      camera_resolution: str, microscope_type: str) -> Dict[str, Any]:
    """
    Section 4 · Sample Preparation.

    Extended (additive, existing "Staining Type" field unchanged) to
    also capture the sample-prep / calibration parameters needed for
    concentration, MCC, and progressive-concentration calculation.

    When `chamber == "Standard Glass Slide"`, shows the dedicated
    "Standard Glass Slide – CASA-Like Controlled Analysis" protocol
    display (target volume / tolerance / coverslip, then read-only
    technical diagnostics: theoretical average depth, calibration
    status, µm/pixel, FOV, field count) instead of the generic
    chamber-mode sample-volume input. All concentration MATH still
    happens centrally in `src.analysis.concentration` — this function
    only collects inputs and displays what that module already
    computed; it never calculates a concentration or depth itself.

    Parameters
    ----------
    chamber : str
        The chamber selected in Section 3 -- determines which sample
        volume range applies (normal-slide vs chamber) and which
        chamber depth is looked up.
    magnification, camera_resolution : str
        Also from Section 3 -- used to look up the saved microscope
        calibration / legacy µm/pixel table entry.
    microscope_type : str
        Also from Section 3 -- part of the exact hardware-
        configuration key used for calibration lookup.
    """
    _card_open("🧬", "Section 4 · Sample Preparation")

    c1, c2 = st.columns([1, 2])
    with c1:
        staining = st.selectbox("Staining Type", _STAINING_TYPES, index=0,
                                 key="_inp_staining")
    with c2:
        st.markdown(
            f'<div style="margin-top:1.8rem;padding:0.65rem 1rem;'
            f'background:{_CLR_BG};border-radius:8px;'
            f'border:1px solid {_CLR_BORDER};font-size:0.82rem;'
            f'color:{_CLR_TEXT_SEC};">'
            "<b>Staining note:</b> Eosin-Nigrosin is recommended for viability "
            "assessment.  Diff Quik or H&amp;E for morphology evaluation.  "
            "No stain is suitable for motility-only analysis."
            "</div>",
            unsafe_allow_html=True,
        )

    slide_mode = _slide_mode_from_chamber(chamber)
    st.markdown("<br>", unsafe_allow_html=True)

    if slide_mode == "normal_slide":
        result = _render_standard_slide_protocol_ui(magnification, camera_resolution, microscope_type)
    else:
        result = _render_chamber_mode_concentration_ui(chamber, magnification, camera_resolution, microscope_type)

    _card_close()
    return {"staining_type": staining, "slide_mode": slide_mode, **result}


def _render_chamber_mode_concentration_ui(
    chamber: str, magnification: str, camera_resolution: str, microscope_type: str,
) -> Dict[str, Any]:
    """
    Physical-chamber (Makler / Leja / Hemocytometer / Other) sample
    volume, dilution, field count, and calibration inputs.
    """
    st.markdown(
        f'<span style="font-size:0.9rem;font-weight:700;color:{_CLR_TEXT};">'
        "Concentration &amp; Sample Volume</span>",
        unsafe_allow_html=True,
    )

    c3, c4, c5 = st.columns(3)
    with c3:
        vol_lo, vol_hi = conc.CHAMBER_MIN_TOTAL_VOLUME_UL, conc.CHAMBER_MAX_TOTAL_VOLUME_UL
        vol_default = round((vol_lo + vol_hi) / 2, 1)
        vol_help = f"Counting chamber ({chamber}): {vol_lo:g}–{vol_hi:g} µL total"
        sample_volume_ul = st.number_input(
            "Sample Volume (µL) *", min_value=0.1, max_value=1000.0,
            value=vol_default, step=0.1, format="%.1f",
            key="_inp_sample_volume_ul", help=vol_help,
        )
        vol_warning = conc.validate_sample_volume("chamber", sample_volume_ul)
        if vol_warning:
            st.warning(f"⚠️  {vol_warning}")
    with c4:
        dilution_factor = st.number_input(
            "Dilution Factor", min_value=0.1, max_value=1000.0,
            value=conc.DEFAULT_DILUTION_FACTOR, step=0.5, format="%.2f",
            key="_inp_dilution_factor",
            help="1.0 = undiluted. For a 1:10 dilution, enter 10.",
        )
    with c5:
        number_of_counting_fields = st.number_input(
            "Counting Fields", min_value=1, max_value=50,
            value=conc.DEFAULT_NUMBER_OF_COUNTING_FIELDS, step=1,
            key="_inp_counting_fields",
            help=(
                "The current video recording is treated as one "
                "continuously-observed counting field (ByteTrack "
                "already deduplicates cells across the whole "
                "recording). Increase this only if you are averaging "
                "counts from multiple separately-captured clips."
            ),
        )

    st.markdown("<br>", unsafe_allow_html=True)
    calib = _render_calibration_status_block(microscope_type, magnification, camera_resolution)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<span style="font-size:0.9rem;font-weight:700;color:{_CLR_TEXT};">'
        "Chamber Depth</span>",
        unsafe_allow_html=True,
    )
    chamber_spec = conc.CHAMBER_CONFIG.get(chamber, conc.CHAMBER_CONFIG["Other"])
    if chamber_spec.depth_um is not None:
        st.success(f"✅  Chamber depth: **{chamber_spec.depth_um:g} µm** ({chamber})")
        st.caption(chamber_spec.note)
        manual_depth_um = st.number_input(
            "Override chamber depth µm (optional)", min_value=0.0,
            max_value=10_000.0, value=0.0, step=1.0,
            key="_inp_manual_depth_um",
            help="Leave at 0 to use the standard depth above.",
        )
    else:
        st.warning(f"⚠️  {chamber_spec.note}")
        manual_depth_um = st.number_input(
            "Enter validated effective depth µm (if known)",
            min_value=0.0, max_value=10_000.0, value=0.0, step=1.0,
            key="_inp_manual_depth_um",
            help="Leave at 0 — absolute concentration will be withheld "
                 "and marked 'Calibration Required'.",
        )

    return {
        "sample_volume_ul": sample_volume_ul,
        "dilution_factor": dilution_factor,
        "number_of_counting_fields": int(number_of_counting_fields),
        "manual_microns_per_pixel": calib["manual_microns_per_pixel"],
        "saved_calibration_um_per_pixel": calib["saved_calibration_um_per_pixel"],
        "manual_chamber_depth_um": manual_depth_um if manual_depth_um > 0 else None,
        "manual_effective_depth_um": None,
        "depth_correction_factor": None,
    }


def _render_standard_slide_protocol_ui(
    magnification: str, camera_resolution: str, microscope_type: str,
) -> Dict[str, Any]:
    """
    Dedicated "Standard Glass Slide – CASA-Like Controlled Analysis"
    display: shows the fixed protocol target (never claims a measured
    volume), lets the operator record the ACTUAL volume they pipetted
    (for tolerance checking only), and shows every technical quantity
    (theoretical depth, calibration, FOV, field count) as a read-only
    diagnostic computed by `src.analysis.concentration` — never
    entered by hand and never computed here.
    """
    protocol = conc.STANDARD_SLIDE_PROTOCOL

    st.markdown(
        f'<div style="padding:0.85rem 1.1rem;background:{_CLR_BG};'
        f'border-radius:10px;border:1px solid {_CLR_BORDER};margin-bottom:0.75rem;">'
        f'<span style="font-size:1rem;font-weight:800;color:{_CLR_TEXT};">'
        "STANDARD GLASS SLIDE</span><br>"
        f'<span style="font-size:0.85rem;color:{_CLR_TEXT_SEC};">'
        "CASA-Like Controlled Analysis</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<span style="font-size:0.88rem;font-weight:700;color:{_CLR_TEXT};">'
        "Preparation Protocol</span>",
        unsafe_allow_html=True,
    )
    p1, p2, p3 = st.columns(3)
    with p1:
        _metric_card("Protocol Target Volume", f"{protocol.sample_volume_ul:g} µL", _CLR_PRIMARY)
    with p2:
        _metric_card("Allowed Tolerance", f"±{protocol.volume_tolerance_ul:g} µL", _CLR_PRIMARY)
    with p3:
        _metric_card(
            "Coverslip",
            f"{protocol.coverslip_length_mm:g} × {protocol.coverslip_width_mm:g} mm",
            _CLR_PRIMARY,
        )
    st.caption(
        f"Protocol version: {protocol.protocol_version}  ·  "
        f"Status: {protocol.validation_state.upper()} — "
        "these are initial values pending laboratory validation, not "
        "universally-correct constants."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    actual_volume_ul = st.number_input(
        "Actual volume pipetted (µL) — operator-entered, NOT measured by software",
        min_value=0.1, max_value=1000.0, value=protocol.sample_volume_ul,
        step=0.1, format="%.1f", key="_inp_sample_volume_ul",
        help=(
            "The software cannot measure the pipetted volume; this is "
            "your own record of what was actually used, checked "
            f"against the ±{protocol.volume_tolerance_ul:g} µL protocol tolerance."
        ),
    )
    if not protocol.in_volume_tolerance(actual_volume_ul):
        st.warning(
            f"⚠️  Actual volume ({actual_volume_ul:g} µL) is outside the "
            f"protocol tolerance of {protocol.sample_volume_ul:g} ± "
            f"{protocol.volume_tolerance_ul:g} µL."
        )
    vol_warning = conc.validate_sample_volume("normal_slide", actual_volume_ul)
    if vol_warning:
        st.warning(f"⚠️  {vol_warning}")

    dilution_factor = st.number_input(
        "Dilution Factor", min_value=0.1, max_value=1000.0,
        value=conc.DEFAULT_DILUTION_FACTOR, step=0.5, format="%.2f",
        key="_inp_dilution_factor",
        help="1.0 = undiluted. For a 1:10 dilution, enter 10.",
    )

    # ── Read-only technical diagnostics (never hand-entered) ────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Technical Diagnostics (read-only, auto-calculated)", expanded=True):
        coverslip_area = conc.calculate_coverslip_area_mm2(
            protocol.coverslip_length_mm, protocol.coverslip_width_mm,
        )
        theoretical_depth_um = conc.calculate_theoretical_depth_um(
            actual_volume_ul, coverslip_area,
        )

        d1, d2 = st.columns(2)
        with d1:
            st.metric("Theoretical Average Depth", f"{theoretical_depth_um:.2f} µm")
            st.caption(
                "NOT the actual/machined chamber depth — a standard "
                "glass slide is not a precision chamber. This is a "
                "geometric estimate from volume ÷ coverslip area."
            )
        with d2:
            st.metric("Coverslip Area", f"{coverslip_area:g} mm²")

        d3, d4 = st.columns(2)
        with d3:
            calib = _render_calibration_status_block(microscope_type, magnification, camera_resolution)
        with d4:
            st.metric("Fields (protocol default)", f"{protocol.number_of_fields}")
            st.caption(
                "The current single-video pipeline analyses the "
                "recording as one continuously-observed field — see "
                "'Valid Fields' in the results for how many of the "
                f"protocol's {protocol.number_of_fields} target fields "
                "were actually captured this run."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-size:0.85rem;font-weight:700;color:{_CLR_TEXT};">'
            "Effective Depth Validation</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Effective slide depth calibration not validated by "
            "default — the theoretical depth above is NOT "
            "automatically treated as the true observation depth "
            "(PART 10/15/16). Only enter a value below if it comes "
            "from an actual reference-method validation."
        )
        manual_effective_depth_um = st.number_input(
            "Validated effective depth µm (optional — leave at 0 unless "
            "empirically validated)",
            min_value=0.0, max_value=10_000.0, value=0.0, step=0.1,
            key="_inp_manual_effective_depth_um",
            help="Leave at 0 to keep concentration as 'Estimated' "
                 "pending validation.",
        )
        depth_correction_factor_input = st.number_input(
            "OR validated depth correction factor (optional — leave at 0 "
            "unless empirically validated)",
            min_value=0.0, max_value=10.0, value=0.0, step=0.01,
            format="%.3f", key="_inp_depth_correction_factor",
            help="effective_depth = theoretical_depth × factor. "
                 "Leave at 0 unless determined from reference data.",
        )

    return {
        "sample_volume_ul": actual_volume_ul,
        "dilution_factor": dilution_factor,
        "number_of_counting_fields": protocol.number_of_fields,
        "manual_microns_per_pixel": calib["manual_microns_per_pixel"],
        "saved_calibration_um_per_pixel": calib["saved_calibration_um_per_pixel"],
        "manual_chamber_depth_um": None,
        "manual_effective_depth_um": manual_effective_depth_um if manual_effective_depth_um > 0 else None,
        "depth_correction_factor": depth_correction_factor_input if depth_correction_factor_input > 0 else None,
    }


# ══════════════════════════════════════════════════════════════
# §5  Input Source  (identical to Phase 2)
# ══════════════════════════════════════════════════════════════


def _get_camera_manager() -> CameraManager:
    """Return (or create) the shared CameraManager stored in session_state."""
    if st.session_state.get("_camera_manager") is None:
        st.session_state["_camera_manager"] = CameraManager()
    return st.session_state["_camera_manager"]


def _get_recording_manager() -> RecordingManager:
    """Return (or create) the shared RecordingManager stored in session_state."""
    mgr = _get_camera_manager()
    if st.session_state.get("_recording_manager") is None:
        st.session_state["_recording_manager"] = RecordingManager(mgr)
    return st.session_state["_recording_manager"]


# ──────────────────────────────────────────────────────────────
# Camera Status Panel (Phase 6)
# ──────────────────────────────────────────────────────────────

def _render_camera_status_panel() -> None:
    """Live camera status panel shown while preview is active."""
    mgr = _get_camera_manager()
    if not mgr.is_connected:
        return
    s = mgr.get_status()
    st.markdown(
        f'''<div style="background:#0f172a;border-radius:12px;
padding:0.9rem 1.2rem;margin:0.6rem 0;border:1px solid #1e293b;">
<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
text-transform:uppercase;color:#475569;margin-bottom:0.5rem;">
📷 Camera Status</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;">
<div><div style="font-size:0.68rem;color:#64748b;">Status</div>
<div style="font-size:0.85rem;font-weight:700;color:#4ade80;">● Connected</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Resolution</div>
<div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;">{s.resolution}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">FPS</div>
<div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;">{s.actual_fps:.1f}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Frames Received</div>
<div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;">{s.frames_received:,}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Dropped Frames</div>
<div style="font-size:0.85rem;font-weight:700;
color:{"#ef4444" if s.frames_dropped > 0 else "#4ade80"};">{s.frames_dropped}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Device</div>
<div style="font-size:0.75rem;font-weight:600;color:#94a3b8;"
title="{s.device_name}">{s.device_name[:22]}</div></div>
</div></div>''',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Recording Experience Panel (Phase 6)
# ──────────────────────────────────────────────────────────────

def _render_recording_panel() -> None:
    """Animated recording progress panel."""
    rec = _get_recording_manager()
    s   = rec.get_status()

    if s.state == RecordingState.IDLE:
        return

    state_colour = {
        RecordingState.RECORDING:  "#ef4444",
        RecordingState.RECOVERING: "#f59e0b",
        RecordingState.COMPLETE:   "#10b981",
        RecordingState.FAILED:     "#ef4444",
    }.get(s.state, "#64748b")

    state_label = {
        RecordingState.RECORDING:  "● RECORDING",
        RecordingState.RECOVERING: "⚠ RECOVERING",
        RecordingState.COMPLETE:   "✓ COMPLETE",
        RecordingState.FAILED:     "✗ FAILED",
    }.get(s.state, s.state.value.upper())

    if s.state == RecordingState.COMPLETE:
        st.success(
            f"✅  **Recording Complete** — "
            f"{s.duration_s:.0f}s captured · "
            f"{s.frames_recorded} frames · "
            f"Saved successfully"
        )
        return

    if s.state == RecordingState.FAILED:
        st.error(
            f"❌  **{s.error_title}**  \n"
            f"{s.error_detail}  \n"
            f"💡 {s.suggestion}"
        )
        return

    # RECORDING / RECOVERING
    st.markdown(
        f'''<div style="background:#0f172a;border-radius:14px;
padding:1.1rem 1.4rem;margin:0.6rem 0;border:1px solid #1e293b;">
<div style="display:flex;justify-content:space-between;align-items:center;
margin-bottom:0.8rem;">
<div style="font-size:1rem;font-weight:800;color:white;">Recording</div>
<div style="background:{state_colour};color:white;border-radius:6px;
padding:2px 10px;font-size:0.75rem;font-weight:800;
letter-spacing:0.1em;">{state_label}</div></div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;
margin-bottom:0.8rem;">
<div><div style="font-size:0.68rem;color:#64748b;">Elapsed</div>
<div style="font-size:1.3rem;font-weight:800;color:#e2e8f0;
font-variant-numeric:tabular-nums;">{s.elapsed_mmss}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Remaining</div>
<div style="font-size:1.3rem;font-weight:800;color:#e2e8f0;
font-variant-numeric:tabular-nums;">{s.remaining_mmss}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">FPS</div>
<div style="font-size:1.3rem;font-weight:800;color:#e2e8f0;">{s.current_fps:.1f}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Frames</div>
<div style="font-size:0.9rem;font-weight:700;color:#e2e8f0;">{s.frames_recorded}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Quality</div>
<div style="font-size:0.9rem;font-weight:700;
color:{"#10b981" if s.quality=="Excellent" else "#f59e0b" if s.quality=="Good" else "#ef4444"};">
{s.quality}</div></div>
<div><div style="font-size:0.68rem;color:#64748b;">Frame Loss</div>
<div style="font-size:0.9rem;font-weight:700;
color:{"#10b981" if s.frame_loss_pct<1 else "#f59e0b" if s.frame_loss_pct<5 else "#ef4444"};">
{s.frame_loss_pct:.1f}%</div></div>
</div>
<div style="font-size:0.72rem;color:#64748b;margin-bottom:4px;">
Recording Progress</div></div>''',
        unsafe_allow_html=True,
    )
    st.progress(
        min(1.0, s.progress_pct / 100.0),
        text=f"{s.progress_pct:.0f}%  —  {s.frames_recorded} frames recorded",
    )


# ──────────────────────────────────────────────────────────────
# Upload UI (unchanged logic, Phase 6 compatible)
# ──────────────────────────────────────────────────────────────

def _render_upload_ui() -> Optional[Path]:
    uploaded_file = st.file_uploader(
        "Drop your microscope video here or click Browse",
        type=_SUPPORTED_VIDEO_FORMATS,
        help="Supported formats: MP4 · AVI · MOV · MKV",
        label_visibility="collapsed",
        key="_video_uploader",
    )
    if uploaded_file is None:
        return None

    st.success(f"✅  **{uploaded_file.name}**  ({uploaded_file.size / 1_048_576:.1f} MB)")

    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded_file.name).suffix, delete=False,
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    cap = cv2.VideoCapture(str(tmp_path))
    ret, frame   = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val      = cap.get(cv2.CAP_PROP_FPS) or 0.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    prev_c, info_c = st.columns([2, 1])
    with prev_c:
        if ret and frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption="First frame preview", width="stretch")
        else:
            st.warning("Unable to read first frame for preview.")
    with info_c:
        st.markdown("**Video Properties**")
        st.table({"Property": ["File","Size","Resolution","Frames","FPS"],
                  "Value": [uploaded_file.name,
                            f"{uploaded_file.size / 1_048_576:.1f} MB",
                            f"{vid_w}×{vid_h}", str(total_frames), f"{fps_val:.1f}"]})

    st.session_state["_video_stem"] = Path(uploaded_file.name).stem
    return tmp_path


# ──────────────────────────────────────────────────────────────
# Live Microscope UI  (Phase 6 — real hardware detection)
# ──────────────────────────────────────────────────────────────

def _render_live_ui() -> Optional[Path]:
    """
    Full live microscope control panel with real hardware detection.

    State machine
    -------------
    _cam_connected False  → device selector + Connect button
    _cam_connected True   → Status panel + 6 control buttons
    _cam_streaming True   → Live preview
    _cam_captured  True   → Clip preview + clip path returned

    Hardware safety
    ---------------
    Only devices classified as USB Microscope / Digital Microscope /
    USB Camera (see ``DeviceClass`` in ``camera_manager.py``) are ever
    listed or connectable here. The laptop's integrated webcam is
    always excluded, regardless of its OpenCV device index -- index
    is never used to guess device identity.
    """
    ss  = st.session_state
    mgr = _get_camera_manager()

    # ── Device enumeration ─────────────────────────────────────
    if not ss.get("_cam_connected", False):
        
        st.markdown(
            f'''<div style="font-size:0.82rem;font-weight:600;
color:{_CLR_TEXT_PRI};margin-bottom:0.5rem;">Microscope Device</div>''',
            unsafe_allow_html=True,
        )

        # Enumerate (cached across reruns until Refresh is clicked)
        if ss.get("_available_devices") is None:
            with st.spinner("Scanning for microscope hardware…"):
                ss["_available_devices"] = mgr.enumerate_devices()

        all_devices: List[CameraDevice] = ss["_available_devices"] or []
        avail: List[CameraDevice] = mgr.get_imaging_devices(all_devices)
        ss["_no_microscope"] = not avail

        if not avail:
            st.markdown(
                '''<div style="background:#fef2f2;border:2px solid #ef4444;
border-radius:10px;padding:0.85rem 1.1rem;font-size:0.9rem;
color:#991b1b;font-weight:700;">
❌&nbsp; No USB microscope detected.</div>''',
                unsafe_allow_html=True,
            )
            st.caption(
                "The laptop's built-in webcam is intentionally hidden and "
                "will never be used. Connect a USB Microscope, Digital "
                "Microscope, or USB Camera, then click **Refresh**."
            )
            if st.button("🔄 Refresh Camera List", key="_btn_refresh_cams"):
                ss["_available_devices"] = None
                st.rerun()
        else:
            labels = [d.label() for d in avail]

             # Preserve the previously selected camera across Streamlit reruns
            current_idx = ss.get("_cam_device_idx")

            if current_idx is not None:
                current_device = next(
                    (d for d in avail if d.index == current_idx),
                    None
            )
            else:
                 current_device = None

            if current_device is not None:
                default_index = labels.index(current_device.label())
            else:
                default_index = 0
            sel_label = st.selectbox(
                "Select camera",
                options=labels,
                index=default_index,
                key="_cam_label_select",
                label_visibility="collapsed",
            )
            sel_dev = next(d for d in avail if d.label() == sel_label)
            ss["_cam_device_idx"] = sel_dev.index

            print("\n=========== AVAILABLE CAMERAS ===========")

            for d in avail:
                print(
                      f"Index={d.index} | Name={d.display_name} | "
                      f"Class={d.device_class}"
                    )

                print("Selected Camera =", sel_dev.display_name)
                print("Selected Index  =", sel_dev.index)
                print("=========================================\n")

            st.write("Selected Camera Index:", sel_dev.index)
            st.write("Selected Camera:", sel_dev.display_name)

            badge_colour = {
                DeviceClass.USB_MICROSCOPE:     "#059669",
                DeviceClass.DIGITAL_MICROSCOPE: "#059669",
                DeviceClass.USB_CAMERA:         "#2563eb",
            }.get(sel_dev.device_class, "#475569")
            res = sel_dev.supported_res[-1] if sel_dev.supported_res else (0, 0)
            st.markdown(
                f'''<span style="background:{badge_colour};color:white;
border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">
{sel_dev.device_class.value}</span>
&nbsp;
<span style="color:{_CLR_TEXT_SEC};font-size:0.78rem;">
{sel_dev.display_name} · {res[0]}×{res[1]} · {sel_dev.default_fps:.0f} FPS
· {"USB" if sel_dev.is_usb else "Unconfirmed"}</span>''',
                unsafe_allow_html=True,
            )

            if st.button("🔄 Refresh Camera List", key="_btn_refresh_cams"):
                ss["_available_devices"] = None
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    connected = ss.get("_cam_connected",  False)
    streaming = ss.get("_cam_streaming",  False)
    captured  = ss.get("_cam_captured",   False)
    clip_str  = ss.get("_camera_clip_path", "")

    # ── Six control buttons ────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    no_microscope = ss.get("_no_microscope", False)

    with r1c1:
        if st.button("🔌  Connect Microscope", use_container_width=True,
                     disabled=(connected or no_microscope), key="_btn_connect"):
            all_devs   = ss.get("_available_devices") or []
            imaging_devs = mgr.get_imaging_devices(all_devs)
            sel_idx = ss.get("_cam_device_idx")

            if sel_idx is None:
                st.error("Please select the microscope before connecting.")
                return None

            print("=" * 70)
            print("CONNECT DEBUG")
            print("Selected index:", sel_idx)

            for d in imaging_devs:
               print(
                    f"Imaging device -> "
                    f"Index={d.index} | "
                    f"Name={d.display_name} | "
                    f"Class={d.device_class.value}"
                )

            print("=" * 70)
            target     = next((d for d in imaging_devs if d.index == sel_idx), None)

            if target is None:
                st.error(
                    "❌  No supported microscope device selected.  "
                    "Refresh the camera list first."
                )
            else:
                resolution = ss.get("_inp_camera_resolution", "1280×720")
                fps_str    = ss.get("_inp_camera_fps",        "30 FPS")
                fps_val_   = float(fps_str.upper().replace("FPS", "").strip())

                with st.spinner(f"Connecting to {target.display_name}…"):

                    print("=" * 70)
                    print("APP CONNECT DEBUG")
                    print("Target device index :", target.index)
                    print("Target device name  :", target.display_name)
                    print("Resolution          :", resolution)
                    print("FPS                 :", fps_val_)
                    print("=" * 70)
                    result: ConnectionResult = mgr.connect(
                        target, resolution=resolution, fps=fps_val_
                    )

                if result.success:
                    ss["_cam_connected"]  = True
                    ss["_cam_streaming"]  = False
                    ss["_cam_captured"]   = False
                    ss["_camera_clip_path"] = ""
                    ss["_cam_resolution"] = resolution
                    ss["_cam_fps"]        = fps_str
                    st.toast(f"✅ {target.display_name} connected", icon="🔬")
                    st.rerun()
                else:
                    st.error(
                        f"❌  **{result.error_title}**\n\n"  
                        f"{result.error_detail}\n\n"    
                        f"💡 {result.suggestion}"
                    )

    with r1c2:
        if st.button("🔴  Disconnect", use_container_width=True,
                     disabled=not connected, key="_btn_disconnect"):
            mgr.disconnect()
            ss["_cam_connected"] = False
            ss["_cam_streaming"] = False
            st.toast("Microscope disconnected", icon="📵")
            st.rerun()

    with r1c3:
        if st.button("▶  Start Camera", use_container_width=True,
                     disabled=(not connected or streaming or no_microscope), key="_btn_start"):
            ss["_cam_streaming"]    = True
            ss["_cam_captured"]     = False
            ss["_camera_clip_path"] = ""

             # Create a fresh live-preview placeholder
            if "_live_frame_placeholder" in ss:
                del ss["_live_frame_placeholder"]
                
            st.rerun()

    with r2c1:
        if st.button(
            "⏹  Stop Camera",
            width="stretch",
            disabled=not streaming,
            key="_btn_stop",
        ):
            ss["_cam_streaming"] = False

            if "_live_frame_placeholder" in ss:
                del ss["_live_frame_placeholder"]

            st.rerun()

    with r2c2:
        if st.button(
            "📸  Capture Sample",
            width="stretch",
            disabled=(not streaming or no_microscope),
            type="primary",
            key="_btn_capture",
        ):
            print("=" * 70)
            print("CAPTURE SAMPLE BUTTON PRESSED")
            print("=" * 70)

            # Call capture ONLY ONCE
            success = _do_capture_phase6()

            print("Capture result:", success)
            print("=" * 70)

            if success:
                ss["_cam_captured"] = True
                ss["_cam_streaming"] = False

            st.rerun()

    with r2c3:
        if st.button("🔁  Retake Sample", use_container_width=True,
                     disabled=not captured, key="_btn_retake"):
            old = ss.get("_camera_clip_path", "")
            if old:
                Path(old).unlink(missing_ok=True)
            ss["_cam_captured"]     = False
            ss["_cam_streaming"]    = True
            ss["_camera_clip_path"] = ""
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status indicators ──────────────────────────────────────
    if not connected:
        st.markdown(
            f'''<span style="color:#f87171;font-weight:700;">● Disconnected</span>
&nbsp;&nbsp;—&nbsp;&nbsp;<span style="color:{_CLR_TEXT_SEC};">
Connect your USB microscope to begin.</span>''',
            unsafe_allow_html=True,
        )
    elif streaming:
        # ============================================================
        # LIVE MICROSCOPE PREVIEW
        # ============================================================

        _render_camera_status_panel()

        st.markdown(
            f'''
            <div style="
                padding:0.5rem 0;
                font-size:0.85rem;
            ">
                <span style="
                    color:#16a34a;
                    font-weight:800;
                    font-size:0.95rem;
                ">
                    ● LIVE
                </span>
                &nbsp;&nbsp;—&nbsp;&nbsp;
                <span style="color:{_CLR_TEXT_SEC};">
                    Microscope camera is streaming continuously.
                    Click <b>Capture Sample</b> to record
                    {_RECORD_DURATION_S:.0f}s.
                </span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        st.markdown(
            '''
            <div style="
                font-size:0.72rem;
                color:#64748b;
                text-transform:uppercase;
                letter-spacing:0.1em;
                font-weight:800;
                margin:0.5rem 0 0.4rem 0;
            ">
                🔬 LIVE MICROSCOPE FEED
            </div>
            ''',
            unsafe_allow_html=True,
        )


        # Display refresh rate is INDEPENDENT of camera capture FPS:
        # CameraManager's background reader keeps capturing at the
        # camera's own rate regardless of what this value is. 0.25s is
        # used here (rather than 0.15s) because at 0.15s Streamlit was
        # re-rendering faster than it could reliably paint, which is
        # what produced the visible blink.
        @st.fragment(run_every=0.25)
        def _render_live_microscope_feed():
            """Continuously display the newest frame from the microscope."""
            mgr = _get_camera_manager()

            # BLINK FIX: do NOT create an st.empty() placeholder inside
            # the fragment. A fragment already clears and re-paints its
            # own container on every run; adding st.empty() on top of
            # that creates a brand-new empty slot first and fills it
            # immediately after, so each refresh rendered an empty box
            # for one paint before the image appeared -- seen as a
            # blink/flicker. Rendering st.image() directly means the
            # container goes straight from old frame to new frame.

            # ------------------------------------------------------------
            # Read the CURRENT frame from the connected CameraManager
            # (display only -- CameraManager's background thread remains
            # the single owner of continuous cap.read())
            # ------------------------------------------------------------

            frame = mgr.get_latest_frame()

            if frame is None:
                st.warning(
                    "⚠️ No live frame received from the microscope. "
                    "Check the camera connection."
                )
                return

            try:
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                st.image(
                    frame_rgb,
                    use_container_width=True,
                )

            except cv2.error as exc:
                st.error(
                    f"⚠️ Unable to display microscope frame: {exc}"
                )

        # The fragment must actually be invoked so Streamlit registers
        # and re-runs it on its own timer, independent of full-page
        # reruns (which is what keeps the rest of the page stable).
        _render_live_microscope_feed()
    elif captured and clip_str:
        st.markdown(
            '''<div style="background:#f0fdf4;border:2px solid #10b981;
border-radius:10px;padding:0.7rem 1rem;font-size:0.875rem;
color:#166534;font-weight:600;">
✅&nbsp; Sample captured — ready for analysis.</div>''',
            unsafe_allow_html=True,
        )
    elif connected:
        _render_camera_status_panel()
        st.markdown(
            f'''<span style="color:#4ade80;font-weight:700;">● Connected</span>
&nbsp;&nbsp;—&nbsp;&nbsp;<span style="color:{_CLR_TEXT_SEC};">
Click <b>Start Camera</b> to begin preview.</span>''',
            unsafe_allow_html=True,
        )

    # Recording panel (shown while capture is running)
    _render_recording_panel()

    # Captured clip preview
    if captured and clip_str:
        clip_p = Path(clip_str)
        if clip_p.exists():
            cap_ = cv2.VideoCapture(str(clip_p))
            ret_, frm_ = cap_.read()
            n_fr = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT))
            fpv  = cap_.get(cv2.CAP_PROP_FPS) or 0.0
            cap_.release()
            pc, ic = st.columns([2, 1])
            with pc:
                if ret_ and frm_ is not None:
                    st.image(cv2.cvtColor(frm_, cv2.COLOR_BGR2RGB),
                             caption="Captured sample — first frame",
                             use_container_width=True)
            with ic:
                sz = clip_p.stat().st_size / 1_048_576
                st.markdown("**Captured Clip**")
                st.table({"Property": ["Source","Duration","Frames","FPS","Size"],
                          "Value": ["USB Microscope",f"{_RECORD_DURATION_S:.0f}s",
                                    str(n_fr),f"{fpv:.1f}",f"{sz:.1f} MB"]})
            ss["_video_stem"] = f"usb_capture_{ss.get('_sample_id','unknown')}"
            return clip_p

    return None


def _do_capture_phase6() -> None:
    """Record a clip using RecordingManager (Phase 6 — with live stats)."""
    ss  = st.session_state
    mgr = _get_camera_manager()

    if not mgr.is_connected:
        st.error("❌  Camera not connected.")
        return

    resolution_str = ss.get("_inp_camera_resolution", "1280×720")
    fps_str        = ss.get("_inp_camera_fps",        "30 FPS")
    sep            = "×" if "×" in resolution_str else "x"
    try:
        w, h = [int(x.strip()) for x in resolution_str.split(sep)]
    except ValueError:
        w, h = 1280, 720
    fps_f = float(fps_str.upper().replace("FPS","").strip())

    rec = _get_recording_manager()
    if not rec.start(duration_s=_RECORD_DURATION_S, resolution=(w, h), fps=fps_f):
        st.error("❌  Could not start recording.")
        return

    # Block until recording completes (show live updates)
    placeholder = st.empty()
    while True:
        s = rec.get_status()
        with placeholder.container():
            _render_recording_panel()
        if s.state in (RecordingState.COMPLETE, RecordingState.FAILED):
            break
        time.sleep(0.5)

    placeholder.empty()
    final = rec.get_status()
    if final.state == RecordingState.COMPLETE and final.clip_path:
        ss["_camera_clip_path"] = final.clip_path

        clip_path = Path(final.clip_path)

        print("=" * 70)
        print("CAPTURED VIDEO VALIDATION")
        print("Path       :", clip_path)
        print("Exists     :", clip_path.exists())
        print("File size  :", clip_path.stat().st_size if clip_path.exists() else "MISSING")
        print("=" * 70)

        if not clip_path.exists():
            st.error(
                f"❌ Captured video was not found:\n\n{clip_path}"
            )
            return
        ss["_cam_captured"]     = True
        ss["_cam_streaming"]    = False
        st.toast("✅ Sample captured", icon="📹")
    else:
        st.error(
            f"❌  **{final.error_title}**\n\n"  
            f"{final.error_detail}\n\n"  
            f"💡 {final.suggestion}"
        )
def _render_section_5() -> Optional[Path]:
    _card_open("📹", "Section 5 · Input Source")
    ss = st.session_state
    c1, c2 = st.columns(2)
    with c1:
        upload_btn = st.button(
            "⬆  Upload Video File", use_container_width=True, key="_src_upload_btn",
            type="primary" if ss.get("_input_source","upload") == "upload" else "secondary",
        )
    with c2:
        live_btn = st.button(
            "📡  Live Microscope", use_container_width=True, key="_src_live_btn",
            type="primary" if ss.get("_input_source") == "live" else "secondary",
        )

    if upload_btn:
        # FIXED: Added the missing underscores to the session state keys
        mgr = ss.get("_camera_manager") 

        if mgr is not None and ss.get("_cam_connected"): 
            try:
               mgr.disconnect()
            except Exception:
                pass
        ss["_cam_connected"] = False
        ss["_cam_streaming"] = False
        ss["_cam_captured"] = False
        ss["_input_source"] = "upload"
        
    # FIXED: Added the missing logic to handle the Live button click
    elif live_btn:
        ss["_input_source"] = "live"

         # Reset capture/recording state when switching to Live Microscope
        ss["_cam_captured"] = False
        ss["_camera_clip_path"] = None
        ss["_cam_streaming"] = False

    st.markdown("<br>", unsafe_allow_html=True)
    tmp_path = _render_upload_ui() if ss["_input_source"] == "upload" else _render_live_ui()
    _card_close()
    return tmp_path        

# ══════════════════════════════════════════════════════════════
# §6  Monitoring Configuration  (NEW Phase 3)
# ══════════════════════════════════════════════════════════════


def _render_section_6() -> Dict[str, Any]:
    """
    Render §6 Time-Interval Monitoring Configuration.

    Returns
    -------
    dict
        {enabled: bool, interval_min: int, duration_min: int}
    """
    _card_open("⏱", "Section 6 · Time-Interval Monitoring")

    ss = st.session_state
    enable_monitoring = st.toggle(
        "Enable automatic time-interval monitoring",
        value=ss.get("_monitor_enabled", False),
        key="_inp_monitor_enabled",
        help=(
            "When enabled the system automatically captures and analyses the "
            "same sample at regular intervals without user interaction."
        ),
    )
    ss["_monitor_enabled"] = enable_monitoring

    if enable_monitoring:
        st.markdown(
            f'<div style="background:{_CLR_MONITOR}10;border:1px solid {_CLR_MONITOR}30;'
            f'border-radius:10px;padding:0.75rem 1rem;margin:0.5rem 0 0.8rem 0;'
            f'font-size:0.85rem;color:{_CLR_SECONDARY};">'
            f'<b>How it works:</b>&nbsp; After clicking <em>Run Full Analysis</em>, '
            f'the system immediately captures and analyses the first sample '
            f'(t = 0 min), then automatically repeats at every interval until '
            f'the total monitoring duration is reached.  '
            f'<b>You do not need to press any button again.</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            interval_min = st.selectbox(
                "Monitoring Interval",
                options=_MONITOR_INTERVALS,
                index=1,        # default 10 min
                format_func=lambda x: f"{x} minutes",
                key="_inp_monitor_interval",
                help="Time between automatic captures",
            )
        with c2:
            duration_min = st.selectbox(
                "Monitoring Duration",
                options=_MONITOR_DURATIONS,
                index=1,        # default 60 min
                format_func=lambda x: f"{x} minutes",
                key="_inp_monitor_duration",
                help="Total length of the monitoring session",
            )

        n_captures = int(duration_min // interval_min) + 1
        st.markdown(
            f'<div style="display:flex;gap:1.2rem;margin-top:0.5rem;flex-wrap:wrap;">'
            f'<span style="font-size:0.82rem;color:{_CLR_TEXT_SEC};">'
            f'📸 Captures planned: <b style="color:{_CLR_SECONDARY};">{n_captures}</b></span>'
            f'<span style="font-size:0.82rem;color:{_CLR_TEXT_SEC};">'
            f'⏱ First capture: <b style="color:{_CLR_SECONDARY};">immediate</b></span>'
            f'<span style="font-size:0.82rem;color:{_CLR_TEXT_SEC};">'
            f'📹 Clip duration: <b style="color:{_CLR_SECONDARY};">{_RECORD_DURATION_S:.0f}s each</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "⚠️  **Session persistence**: monitoring state is stored in the browser "
            "session and survives Streamlit reruns, but **not** a hard browser refresh "
            "or server restart.  Keep this tab open during the monitoring session.",
            icon=None,
        )
    else:
        interval_min = _MONITOR_INTERVALS[1]
        duration_min = _MONITOR_DURATIONS[1]

    _card_close()
    return {
        "enabled":      enable_monitoring,
        "interval_min": interval_min,
        "duration_min": duration_min,
    }


# ══════════════════════════════════════════════════════════════
# Monitor background runner  (BUGFIX — Problem 4)
# ══════════════════════════════════════════════════════════════
#
# ROOT CAUSE (see task write-up):
#   IntervalMonitor.start() and IntervalMonitor.tick() are both
#   blocking, synchronous calls -- start() performs the first
#   capture+analysis cycle in-line, and tick() performs the next
#   capture+analysis cycle in-line whenever an interval has elapsed.
#   The previous app.py called both directly from the Streamlit
#   script thread while ALSO running `streamlit_autorefresh` on a
#   10 s timer. Because a capture+analysis cycle routinely takes far
#   longer than 10 s, the autorefresh's next rerun request arrives
#   WHILE the script is still stuck inside the blocking tick() call.
#   Streamlit interrupts the in-flight script run when a new rerun
#   for the same session is requested, which can kill the capture or
#   analysis mid-flight -- explaining why monitoring appeared to
#   "stop", skip captures, or leave temp video files locked (WinError
#   32) when something else tried to touch them right after.
#
# FIX:
#   IntervalMonitor's own API is left completely untouched (we do not
#   have monitor.py to modify, and the task explicitly asks us not to
#   redesign it). Instead we move the *calling* of start()/tick() into
#   a dedicated background thread, so the blocking work never happens
#   on the Streamlit script thread. The Streamlit thread only ever
#   reads `monitor.session` state to render the dashboard and
#   schedules the next autorefresh -- it never blocks and is never a
#   moving target for autorefresh to interrupt.
# ══════════════════════════════════════════════════════════════


class _MonitorRunner:
    """
    Drives ``IntervalMonitor.start()`` / ``.tick()`` from a single
    background daemon thread so the Streamlit script thread never
    blocks on a capture+analysis cycle.

    One instance is created per monitoring session and stored in
    ``st.session_state["_monitor_runner"]`` so it survives Streamlit
    reruns (a new instance must NOT be created on every rerun).
    """

    # How often the background thread re-checks whether an interval
    # has elapsed. This is independent of the UI's autorefresh cadence.
    _TICK_POLL_S: float = 1.0

    def __init__(self, monitor: "IntervalMonitor") -> None:
        self._monitor = monitor
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._last_error: str = ""

    def launch(self) -> None:
        """Start the background loop (no-op if already launched)."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="IntervalMonitorRunner",
        )
        self._thread.start()
        logger.info("MonitorRunner: background thread launched")

    def stop(self) -> None:
        """Request the loop to stop and forward the stop to IntervalMonitor."""
        self._stop_event.set()
        try:
            self._monitor.stop()
        except Exception as exc:
            logger.warning("MonitorRunner: monitor.stop() raised: {}", exc)

    @property
    def last_error(self) -> str:
        return self._last_error

    def _run(self) -> None:
        """
        Background loop: perform the (blocking) initial capture via
        ``monitor.start()``, then repeatedly call the (blocking when
        due, near-instant otherwise) ``monitor.tick()`` until the
        session completes, is stopped, or an unrecoverable error
        occurs. All of this runs off the Streamlit script thread.
        """
        try:
            self._monitor.start()
        except Exception as exc:
            self._last_error = str(exc)
            logger.error("MonitorRunner: monitor.start() failed: {}", exc)
            return

        session = self._monitor.session
        try:
            while not self._stop_event.is_set():
                if getattr(session, "is_complete", False):
                    break
                if not getattr(session, "is_running", False):
                    break
                try:
                    self._monitor.tick()
                except Exception as exc:
                    self._last_error = str(exc)
                    logger.error("MonitorRunner: monitor.tick() failed: {}", exc)
                    break
                self._stop_event.wait(self._TICK_POLL_S)
        finally:
            logger.info("MonitorRunner: background loop exited")


# ══════════════════════════════════════════════════════════════
# Monitoring dashboard renderer  (NEW Phase 3)
# ══════════════════════════════════════════════════════════════


def _render_monitoring_dashboard(config: Dict[str, Any]) -> None:
    """
    Render the full monitoring dashboard (Phase 3 live view + Phase 4 trend analysis).

    Called from BRANCH C in main().

    While session is running
    ─────────────────────────
    • Live countdown to next capture
    • Overall session progress bar
    • Interval indicator dots
    • Results table (one row per completed capture)
    • Live Streamlit line charts
    • Latest capture expander
    • Stop Monitoring button

    When session is complete or stopped
    ─────────────────────────────────────
    • All of the above (frozen)
    • Phase 4 Trend Analysis section:
      - Generate / show TrendMetrics degradation table
      - Show all eight trend plots (dashboard, motility overview, 6 singles)
      - Show degradation findings and WHO recommendations
      - Generate and offer download of PDF / CSV / JSON monitoring report
    """
    ss      = st.session_state
    monitor: IntervalMonitor = ss["_monitor_obj"]
    session: MonitorSession  = monitor.session

    # ── Auto-refresh while running (BUGFIX — Problem 4) ────────
    # `monitor.tick()` is intentionally NOT called here anymore.
    # It is a blocking call (it performs the actual capture+analysis
    # cycle in-line whenever an interval has elapsed) and calling it
    # from the Streamlit script thread -- combined with autorefresh
    # requesting a new rerun every `_MONITOR_TICK_MS` -- is exactly
    # what caused monitoring to stall / drop captures / corrupt temp
    # files: a rerun request could arrive and interrupt this script
    # run while it was still stuck inside tick()'s blocking capture or
    # analysis call. The actual ticking now happens continuously in
    # the background via `_MonitorRunner` (launched once from
    # `main()`'s monitoring-start path). This function only ever
    # reads `monitor.session` to render the current state and
    # schedules the next redraw.
    if session.is_running and not session.is_complete:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=_MONITOR_TICK_MS, key="_monitor_refresh")
        except ImportError:
            st.warning(
                "⚠️  `streamlit-autorefresh` not installed — "
                "countdown will not update automatically.  "
                "`pip install streamlit-autorefresh`"
            )

        runner = ss.get("_monitor_runner")
        if runner is not None and runner.last_error:
            st.error(
                f"❌  The background monitoring loop stopped after a "
                f"capture/analysis error and will not continue "
                f"automatically: {runner.last_error}\n\n"
                f"Use **Stop Monitoring** and start a new session."
            )

    now            = time.time()
    elapsed_s      = session.elapsed_s(now)
    remaining_s    = max(0.0, session.duration_s - elapsed_s)
    next_in_s      = session.time_to_next_s(now)
    progress_total = min(1.0, elapsed_s / max(session.duration_s, 1.0))

    # ── Header banner ─────────────────────────────────────────
    status_word = (
        "COMPLETE" if session.is_complete else
        "ACTIVE"   if session.is_running  else
        "STOPPED"
    )
    st.markdown(
        f'<div class="monitor-header">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div>'
        f'<div class="monitor-title">⏱ Time-Interval Monitoring Session</div>'
        f'<div class="monitor-sub">'
        f'Sample:&nbsp;{session.sample_id}&nbsp;·&nbsp;'
        f'Interval:&nbsp;{session.interval_min}&nbsp;min&nbsp;·&nbsp;'
        f'Duration:&nbsp;{session.duration_min}&nbsp;min'
        f'</div></div>'
        f'<div style="background:rgba(255,255,255,.15);border-radius:8px;'
        f'padding:0.3rem 0.8rem;font-size:0.78rem;font-weight:800;'
        f'letter-spacing:0.1em;">{status_word}</div>'
        f'</div>'
        f'<div class="monitor-kpi-row">'
        f'<div class="monitor-kpi"><div class="monitor-kpi-val">{session.captures_done}</div>'
        f'<div class="monitor-kpi-lbl">Captures Done</div></div>'
        f'<div class="monitor-kpi"><div class="monitor-kpi-val">{session.expected_captures}</div>'
        f'<div class="monitor-kpi-lbl">Total Planned</div></div>'
        f'<div class="monitor-kpi"><div class="monitor-kpi-val">{_fmt_mmss(elapsed_s)}</div>'
        f'<div class="monitor-kpi-lbl">Elapsed</div></div>'
        f'<div class="monitor-kpi"><div class="monitor-kpi-val">{_fmt_mmss(remaining_s)}</div>'
        f'<div class="monitor-kpi-lbl">Remaining</div></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.progress(
        progress_total,
        text=f"Session progress: {progress_total * 100:.1f}%  "
             f"({session.captures_done}/{session.expected_captures} captures)",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column live layout ─────────────────────────────────
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        # Countdown
        if session.is_running and not session.is_complete:
            st.markdown(
                f'<div class="countdown-box">'
                f'<div style="font-size:0.72rem;color:#94a3b8;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                f'Next capture in</div>'
                f'<div class="countdown-val">{_fmt_mmss(next_in_s)}</div>'
                f'<div class="countdown-lbl">'
                f'Interval {session.captures_done} → {session.captures_done + 1}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        elif session.is_complete:
            st.success("✅ All captures complete!")
        else:
            st.warning("⏸ Monitoring stopped.")

        # Interval dot timeline
        st.markdown("**Capture Timeline**")
        dots_html = '<div style="margin:0.4rem 0;">'
        for i in range(session.expected_captures):
            if i < session.captures_done:
                dc, lc = "interval-dot interval-dot-done",    _CLR_SUCCESS
            elif i == session.captures_done and session.is_running:
                dc, lc = "interval-dot interval-dot-active",  _CLR_MONITOR
            else:
                dc, lc = "interval-dot interval-dot-pending", "#64748b"
            tick = "  ✓" if i < session.captures_done else ""
            dots_html += (
                f'<div class="interval-row">'
                f'<div class="{dc}"></div>'
                f'<span style="font-size:0.78rem;color:{lc};font-weight:600;">'
                f't = {i * session.interval_min} min{tick}</span></div>'
            )
        dots_html += "</div>"
        st.markdown(dots_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Control buttons
        if session.is_running and not session.is_complete:
            if st.button("⏹  Stop Monitoring", use_container_width=True,
                         key="_btn_stop_monitor"):
                # BUGFIX: stop the background _MonitorRunner (which
                # forwards to monitor.stop() itself) so the background
                # thread actually exits instead of continuing to tick.
                runner = ss.get("_monitor_runner")
                if runner is not None:
                    runner.stop()
                else:
                    monitor.stop()
                st.rerun()
        else:
            if st.button("🔄  New Analysis", use_container_width=True,
                         key="_btn_new_from_monitor"):
                _clear_analysis_state()
                st.rerun()

    with right_col:
        _section_header("📋 Capture Results")

        if not session.results:
            st.info("⏳  First capture is being recorded…")
        else:
            import pandas as pd

            # Results table
            table_rows: List[Dict[str, Any]] = []
            for r in session.results:
                summ = r.to_dict().get("summary", {})
                ts   = datetime.fromtimestamp(r.capture_time).strftime("%H:%M:%S")
                table_rows.append({
                    "#":           r.interval_index,
                    "Time":        ts,
                    "t (min)":     f"{r.elapsed_min:.1f}",
                    "Sperm":       summ.get("total_sperm",     "—"),
                    "Progressive%":f"{summ.get('progressive_pct', 0):.1f}",
                    "Normal%":     f"{summ.get('normal_pct',       0):.1f}",
                    "Viability%":  f"{summ.get('viability_pct',    0):.1f}",
                    "WHO":         f"{summ.get('who_score',         0):.0f}",
                    "Grade":       summ.get("who_category", "—"),
                    "Status":      "❌ " + (r.error or "")[:25] if r.error else "✅",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True,
                         hide_index=True)

            # Live line charts (visible while running AND after complete)
            if len(session.results) >= 2:
                st.markdown("<br>", unsafe_allow_html=True)
                _section_header("📈 Live Trends")
                ok_results = [r for r in session.results if not r.error]
                if ok_results:
                    td = {
                        "t (min)":       [r.elapsed_min for r in ok_results],
                        "Progressive %": [r.to_dict()["summary"].get("progressive_pct", 0)
                                           for r in ok_results],
                        "Normal %":      [r.to_dict()["summary"].get("normal_pct",       0)
                                           for r in ok_results],
                        "Viability %":   [r.to_dict()["summary"].get("viability_pct",    0)
                                           for r in ok_results],
                        "WHO Score":     [r.to_dict()["summary"].get("who_score",         0)
                                           for r in ok_results],
                    }
                    df_live = pd.DataFrame(td).set_index("t (min)")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.markdown("**Motility & Morphology**")
                        st.line_chart(df_live[["Progressive %", "Normal %"]],
                                      use_container_width=True)
                    with cc2:
                        st.markdown("**Viability & WHO Score**")
                        st.line_chart(df_live[["Viability %", "WHO Score"]],
                                      use_container_width=True)

            # Latest result expander
            latest = session.results[-1]
            if not latest.error and latest.analysis_results:
                with st.expander(
                    f"📊 Latest capture detail  (t = {latest.elapsed_min:.1f} min)",
                    expanded=False,
                ):
                    # Enrich with concentration/MCC — IntervalMonitor's
                    # internal pipeline calls don't know about this
                    # feature, so compute it here as a post-processing
                    # step (see _enrich_results_with_concentration docstring).
                    enriched_results = _enrich_results_with_concentration(
                        latest.analysis_results,
                        st.session_state.get("_analysis_meta"),
                    )
                    _render_results(
                        enriched_results,
                        video_stem=f"monitor_{session.sample_id}_{latest.interval_index}",
                        config=config,
                        sample_meta=None,
                    )

    # ══════════════════════════════════════════════════════════
    # PHASE 4 — Full trend analysis (shown only when session is
    # complete or the user has stopped it)
    # ══════════════════════════════════════════════════════════
    if (session.is_complete or not session.is_running) and session.captures_done >= 2:
        st.divider()
        _render_trend_analysis_section(session, config)


def _render_trend_analysis_section(session: MonitorSession,
                                    config: Dict[str, Any]) -> None:
    """
    Phase 4 — Full trend analysis and monitoring report UI.

    Runs TrendAnalyser once, caches the TrendReport in session_state,
    renders plots, degradation summary, and report download buttons.

    Parameters
    ----------
    session : MonitorSession   completed or stopped monitoring session
    config  : dict             project configuration
    """
    ss = st.session_state

    # ── Section header ─────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_CLR_MONITOR},{_CLR_PRIMARY});'
        f'border-radius:14px;padding:1.2rem 1.6rem;margin-bottom:1rem;">'
        f'<div style="font-size:1.2rem;font-weight:800;color:white;">'
        f'📈 Phase 4 — Trend Analysis &amp; Monitoring Report</div>'
        f'<div style="font-size:0.85rem;color:rgba(255,255,255,.75);margin-top:0.2rem;">'
        f'Automatic trend computation, degradation analysis, and professional PDF report'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Run TrendAnalyser (or load from cache) ─────────────────
    trend_report: Optional[TrendReport] = ss.get("_trend_report")

    if trend_report is None:
        with st.spinner("Running trend analysis and generating plots…"):
            try:
                analyser     = TrendAnalyser(config)
                sample_meta  = ss.get("_analysis_meta") or {}
                trend_report = analyser.analyse(session, sample_meta=sample_meta)
                ss["_trend_report"] = trend_report
            except Exception as exc:
                st.error(f"❌  Trend analysis failed: {exc}")
                return

    m   = trend_report.trend_metrics
    dsm = trend_report.degradation_summary

    # ── Degradation metric cards ───────────────────────────────
    _section_header("📊 Degradation Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        delta_p = m.progressive_change
        _metric_card(
            "Progressive Motility Change",
            f"{delta_p:+.1f}%",
            accent=_CLR_DANGER if delta_p < 0 else _CLR_SUCCESS,
        )
        _metric_card(
            "Motility Degradation Rate",
            f"{m.motility_degradation_rate_per_min:+.4f} %/min",
            accent=_CLR_DANGER if m.motility_degradation_rate_per_min > 0 else _CLR_SUCCESS,
        )
    with c2:
        delta_v = m.viability_change
        _metric_card(
            "Viability Change",
            f"{delta_v:+.1f}%",
            accent=_CLR_DANGER if delta_v < 0 else _CLR_SUCCESS,
        )
        _metric_card(
            "Viability Degradation Rate",
            f"{m.viability_degradation_rate_per_min:+.4f} %/min",
            accent=_CLR_DANGER if m.viability_degradation_rate_per_min > 0 else _CLR_SUCCESS,
        )
    with c3:
        delta_s = m.score_change
        _metric_card(
            "WHO Score Change",
            f"{delta_s:+.1f} pts",
            accent=_CLR_DANGER if delta_s < 0 else _CLR_SUCCESS,
        )
        _metric_card(
            "Score Decline Rate",
            f"{m.score_decline_rate_per_min:+.4f} pts/min",
            accent=_CLR_DANGER if m.score_decline_rate_per_min > 0 else _CLR_SUCCESS,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend metrics table ────────────────────────────────────
    _section_header("📋 Trend Summary Table")
    import pandas as pd
    rows = [
        {"Metric": "Progressive (%)",    "Start": m.progressive_start,
         "End": m.progressive_end,       "Change": f"{m.progressive_change:+.2f}",
         "% Change": f"{m.progressive_pct_change:+.1f}%",
         "Rate/min": f"{m.motility_degradation_rate_per_min:+.4f}"},
        {"Metric": "Non-Progressive (%)", "Start": m.nonprog_start,
         "End": m.nonprog_end,            "Change": f"{m.nonprog_change:+.2f}",
         "% Change": f"{m.nonprog_pct_change:+.1f}%",  "Rate/min": "—"},
        {"Metric": "Immotile (%)",        "Start": m.immotile_start,
         "End": m.immotile_end,           "Change": f"{m.immotile_change:+.2f}",
         "% Change": f"{m.immotile_pct_change:+.1f}%", "Rate/min": "—"},
        {"Metric": "Viability (%)",       "Start": m.viability_start,
         "End": m.viability_end,          "Change": f"{m.viability_change:+.2f}",
         "% Change": f"{m.viability_pct_change:+.1f}%",
         "Rate/min": f"{m.viability_degradation_rate_per_min:+.4f}"},
        {"Metric": "Total Count",         "Start": int(m.count_start),
         "End": int(m.count_end),         "Change": f"{m.count_change:+.0f}",
         "% Change": f"{m.count_pct_change:+.1f}%",    "Rate/min": "—"},
        {"Metric": "WHO Score",           "Start": m.score_start,
         "End": m.score_end,              "Change": f"{m.score_change:+.2f}",
         "% Change": f"{m.score_pct_change:+.1f}%",
         "Rate/min": f"{m.score_decline_rate_per_min:+.4f}"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Trend plots ────────────────────────────────────────────
    _section_header("📈 Trend Charts")

    # Dashboard (full width)
    dash_path = trend_report.plot_paths.get("dashboard", "")
    if dash_path and Path(dash_path).exists():
        st.markdown("**Summary Dashboard — All Metrics**")
        st.image(dash_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Motility overview
    mot_path = trend_report.plot_paths.get("motility_overview", "")
    if mot_path and Path(mot_path).exists():
        st.markdown("**Motility Overview (Line + Stacked Area)**")
        st.image(mot_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Six individual plots in 2-column grid
    individual_plots = [
        ("progressive_pct", "Progressive Motility vs Time"),
        ("nonprog_pct",     "Non-Progressive Motility vs Time"),
        ("immotile_pct",    "Immotile Percentage vs Time"),
        ("viability_pct",   "Viability vs Time"),
        ("total_sperm",     "Total Count vs Time"),
        ("who_score",       "Quality Score vs Time"),
    ]
    pairs = list(zip(individual_plots[::2], individual_plots[1::2]))
    if len(individual_plots) % 2:
        pairs.append((individual_plots[-1], None))

    for left_spec, right_spec in pairs:
        col_l, col_r = st.columns(2)
        for col, spec in [(col_l, left_spec), (col_r, right_spec)]:
            with col:
                if spec is None:
                    continue
                col_key, caption = spec
                p = trend_report.plot_paths.get(col_key, "")
                if p and Path(p).exists():
                    st.markdown(f"**{caption}**")
                    st.image(p, use_container_width=True)

    st.divider()

    # ── Degradation findings ───────────────────────────────────
    _section_header("🔍 Degradation Findings")
    for key, sentence in dsm.items():
        if key == "overall":
            continue
        icon = "🔴" if "decreased" in sentence or "declining" in sentence else "🟢"
        st.markdown(
            f'<div style="padding:0.4rem 0.8rem;margin:0.2rem 0;'
            f'font-size:0.88rem;color:{_CLR_TEXT_PRI};">'
            f'{icon}&nbsp;&nbsp;{sentence}</div>',
            unsafe_allow_html=True,
        )

    overall = dsm.get("overall", "")
    bg = ("#dc2626" if "significant" in overall.lower()
          else "#059669" if "acceptable" in overall.lower()
          else "#d97706")
    st.markdown(
        f'<div style="background:{bg};color:white;border-radius:10px;'
        f'padding:0.8rem 1.2rem;margin:0.6rem 0;font-weight:700;'
        f'font-size:0.95rem;">📌 {overall}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── FEATURE 6/10/11 — Sustained Motility Lifetime ───────────
    # Computed directly from the raw per-capture time series
    # (elapsed_min, progressive_pct), independent of TrendAnalyser —
    # this is "the existing time-interval functionality" the SML
    # feature is meant to build on. No value is invented: if there
    # are too few points, or progressive motility never crosses the
    # 50% threshold, `calculate_sml` honestly reports that.
    ok_for_sml = [r for r in session.results if not r.error and r.analysis_results]
    sml_times = [r.elapsed_min for r in ok_for_sml]
    sml_progressive = [
        r.analysis_results.get("progressive", 0.0) for r in ok_for_sml
    ]
    sml_result = conc.calculate_sml(sml_times, sml_progressive)
    _render_sml_section(
        sml=sml_result.to_dict(),
        time_series=sml_result.time_series,
    )
    ss["_sml_result"] = sml_result.to_dict()  # available for report/DB use below

    st.divider()

    # ── Generate and cache monitoring report ───────────────────
    _section_header("📥 Monitoring Report")

    report_paths: Optional[Dict[str, str]] = ss.get("_trend_report_paths")

    # BUGFIX (Problem 3 — database):
    # Report generation used to require a manual button click before
    # `_trend_report_paths` was ever populated. The monitoring-session
    # auto-save in `main()` fires as soon as the session completes,
    # so on an unattended/automated run the DB row could end up saved
    # with empty report paths forever (the save-flag latches to True
    # and blocks any later retry). Report generation is now automatic
    # the first time, exactly like `_trend_report` above -- the
    # "Regenerate Report" button remains for anyone who wants to
    # re-run it manually afterwards.
    if report_paths is None:
        with st.spinner("Generating PDF report — this may take 15–30 seconds…"):
            try:
                gen          = MonitoringReportGenerator(config)
                sample_meta  = ss.get("_analysis_meta") or {}
                sid          = sample_meta.get("sample_id", "session")
                stem         = f"monitor_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                report_paths = gen.generate(trend_report, output_stem=stem)
                ss["_trend_report_paths"] = report_paths
            except Exception as exc:
                st.error(f"❌  Report generation failed: {exc}")
                return

    st.success("✅  Monitoring report ready for download.")
    _render_monitoring_report_downloads(report_paths)
    # Offer regeneration
    if st.button("🔁  Regenerate Report", key="_btn_regen_report"):
        ss["_trend_report_paths"] = None
        st.rerun()


def _render_monitoring_report_downloads(report_paths: Dict[str, str]) -> None:
    """
    Render PDF / CSV / JSON download buttons for the monitoring report.

    Reuses the exact same pattern as ``_render_download_buttons`` for
    single-sample reports — no code is duplicated.

    Parameters
    ----------
    report_paths : dict  {"pdf": str, "csv": str, "json": str}
    """
    col_pdf, col_csv, col_json = st.columns(3)
    mime_map = {
        "pdf":  ("⬇  PDF Report",          "application/pdf"),
        "csv":  ("⬇  CSV Data",            "text/csv"),
        "json": ("⬇  JSON (Full Data)",    "application/json"),
    }
    columns = {"pdf": col_pdf, "csv": col_csv, "json": col_json}

    for fmt, (label, mime) in mime_map.items():
        path_str = report_paths.get(fmt, "")
        col      = columns[fmt]
        if path_str:
            p = Path(path_str)
            if p.exists():
                try:
                    col.download_button(
                        label=label, data=p.read_bytes(),
                        file_name=p.name, mime=mime,
                        use_container_width=True,
                    )
                except OSError as exc:
                    col.caption(f"{fmt.upper()} unreadable: {exc}")
            else:
                col.caption(f"{fmt.upper()}: file not found")
        else:
            col.caption(f"{fmt.upper()}: not generated")


# ══════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════


def _validate_form(sample_info: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not sample_info["sample_id"].strip():
        errors.append("Sample ID is required (§1 Sample Information).")
    return errors


# ══════════════════════════════════════════════════════════════
# Result extraction helpers  (identical to Phase 1 & 2 — UNCHANGED)
# ══════════════════════════════════════════════════════════════


def _get(results: Dict, *keys: str, default: Any = 0) -> Any:
    obj: Any = results
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj if obj is not None else default


def _extract_counts(results: Dict) -> Dict[str, Any]:
    if "total_sperm" in results:
        return {
            "total_sperm":   int(_get(results, "total_sperm")),
            "unique_tracks": int(_get(results, "total_sperm")),
            "video_frames":  int(_get(results, "video_frames")),
            "fps":           float(_get(results, "fps", default=0.0)),
        }
    det = results.get("detection", {})
    return {
        "total_sperm":   int(_get(det, "count",         default=0)),
        "unique_tracks": int(_get(det, "unique_tracks", default=0)),
        "video_frames":  int(_get(det, "n_frames",      default=0)),
        "fps":           0.0,
    }


def _extract_motility(results: Dict) -> Dict[str, float]:
    if "progressive" in results:
        return {
            "progressive_pct":    float(_get(results, "progressive")),
            "nonprogressive_pct": float(_get(results, "non_progressive")),
            "immotile_pct":       float(_get(results, "immotile")),
            "mean_VCL":           float(_get(results, "mean_VCL")),
            "mean_VSL":           float(_get(results, "mean_VSL")),
            "mean_VAP":           float(_get(results, "mean_VAP")),
        }
    mot = results.get("motility", {})
    return {
        "progressive_pct":    float(_get(mot, "progressive_pct")),
        "nonprogressive_pct": float(_get(mot, "nonprogressive_pct")),
        "immotile_pct":       float(_get(mot, "immotile_pct")),
        "mean_VCL":           float(_get(mot, "mean_VCL")),
        "mean_VSL":           float(_get(mot, "mean_VSL")),
        "mean_VAP":           float(_get(mot, "mean_VAP")),
    }


def _extract_morphology(results: Dict) -> Dict[str, Any]:
    if "normal_pct" in results:
        return {
            "normal_pct":     float(_get(results, "normal_pct")),
            "abnormal_pct":   float(_get(results, "abnormal_pct")),
            "normal_count":   int(_get(results, "normal_count")),
            "abnormal_count": int(_get(results, "abnormal_count")),
            "total":          int(_get(results, "normal_count"))
                              + int(_get(results, "abnormal_count")),
        }
    morph = results.get("morphology", {})

    
    return {
        "normal_pct":     float(_get(morph, "normal_pct")),
        "abnormal_pct":   float(_get(morph, "abnormal_pct")),
        "normal_count":   int(_get(morph, "normal_count")),
        "abnormal_count": int(_get(morph, "abnormal_count")),
        "total":          int(_get(morph, "total")),
    }

def _generate_morphology_plot(
    morph: Dict[str, Any],
    plot_dir: Path,
) -> Path:
    """Generate morphology distribution chart."""

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    normal_pct = float(morph.get("normal_pct", 0.0))
    abnormal_pct = float(morph.get("abnormal_pct", 0.0))

    fig, ax = plt.subplots(figsize=(7, 4))

    categories = ["Normal", "Abnormal"]
    values = [normal_pct, abnormal_pct]

    ax.bar(categories, values)

    ax.set_title("Morphology Distribution")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)

    for i, value in enumerate(values):
        ax.text(
            i,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()

    output_path = plot_dir / "morphology_distribution.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def _extract_viability(results: Dict) -> float:
    if "predicted_viability" in results:
        return float(_get(results, "predicted_viability"))
    return float(_get(results, "viability_pct", default=0.0))


def _extract_quality(results: Dict) -> Dict[str, Any]:
    return results.get(
        "quality",
        {"score": 0.0, "category": "Unknown", "subscores": {},
         "who_flags": [], "recommendations": []},
    )


def _extract_report_paths(results: Dict) -> Dict[str, str]:
    return results.get("report_paths", {})


# ══════════════════════════════════════════════════════════════
# UI component renderers  (identical to Phase 1 & 2 — UNCHANGED)
# ══════════════════════════════════════════════════════════════


def _render_who_score_banner(score: float, category: str) -> None:
    css_cat = _CATEGORY_CSS.get(category, "cat-poor")
    emoji   = _CATEGORY_EMOJI.get(category, "")
    st.markdown(
        f'<div class="who-score-box {css_cat}">'
        f'<div class="who-score-number">{score:.1f}</div>'
        f'<div class="who-score-denom">/ 100</div>'
        f'<div class="who-score-label">{emoji}&nbsp;{category}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_subscore_bars(subscores: Dict[str, float]) -> None:
    label_map  = {"count":"Count","motility":"Motility",
                  "morphology":"Morphology","viability":"Viability"}
    colour_map = {"count":_CLR_PRIMARY,"motility":_CLR_SUCCESS,
                  "morphology":"#8b5cf6","viability":_CLR_WARNING}
    for key, label in label_map.items():
        val    = float(subscores.get(key, 0.0))
        pct    = max(0, min(100, int(val)))
        colour = colour_map.get(key, _CLR_PRIMARY)
        cl, cr = st.columns([3, 1])
        with cl:
            st.markdown(
                f'<div style="font-size:0.82rem;font-weight:600;'
                f'color:{_CLR_TEXT_PRI};margin-bottom:3px;">{label}</div>'
                f'<div style="height:9px;background:{_CLR_BORDER};border-radius:5px;">'
                f'<div style="width:{pct}%;height:100%;background:{colour};'
                f'border-radius:5px;"></div></div>',
                unsafe_allow_html=True,
            )
        with cr:
            st.markdown(
                f'<div style="text-align:right;font-size:0.85rem;'
                f'font-weight:800;color:{colour};margin-top:4px;">'
                f'{val:.1f}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


def _render_optional_plot(path, title):
    """Render an analysis plot safely."""
    path = Path(path)

    _section_header(f"📊 {title}")

    if not path.exists():
        st.warning(
            f"{title} chart is unavailable.\n\n"
            f"Expected file: `{path}`"
        )
        return

    if not path.is_file():
        st.warning(f"{title}: path exists but is not a file: `{path}`")
        return

    try:
        file_size = path.stat().st_size

        if file_size == 0:
            st.error(f"{title}: PNG file is empty: `{path}`")
            return

        # Read the actual bytes instead of passing the file path.
        image_bytes = path.read_bytes()

        st.image(
            image_bytes,
            caption=title,
            width="stretch",
        )

    except Exception as exc:
        st.error(
            f"Unable to display {title}.\n\n"
            f"File: `{path}`\n\n"
            f"Error: `{exc}`"
        )

def _render_concentration_section(results: Dict[str, Any]) -> None:
    """
    Concentration & MCC results section.

    Reads the concentration/MCC/progressive-concentration fields that
    `run_video()` (via `compute_concentration_metrics_from_params`, or
    `_enrich_results_with_concentration` for monitoring captures)
    merges into the results dict. If the status is anything other
    than "ok", this shows the honest reason instead of a fabricated
    number — never silently falls back to a guess.

    Displays a different header block depending on whether this
    result came from the Standard Glass Slide CASA-like controlled
    pathway (`slide_category == "controlled_estimation"`) or a
    physical chamber (`"physical_chamber"`) — using
    `concentration_label`/`slide_category` exactly as computed
    centrally in `src.analysis.concentration`; this function never
    computes or re-derives a label itself.
    """
    _section_header("🧮 Concentration & MCC")

    status = results.get("concentration_status")
    if status is None:
        st.caption(
            "Concentration was not requested for this analysis "
            "(no sample-prep/calibration parameters were supplied)."
        )
        return

    is_standard_slide = results.get("slide_category") == "controlled_estimation"

    if is_standard_slide:
        st.markdown(
            f'<div style="padding:0.6rem 1rem;background:{_CLR_BG};'
            f'border-radius:8px;border:1px solid {_CLR_BORDER};margin-bottom:0.6rem;">'
            f'<b>Method:</b> Standard Glass Slide – CASA-Like Controlled Analysis'
            f'</div>',
            unsafe_allow_html=True,
        )
        protocol = conc.STANDARD_SLIDE_PROTOCOL
        pc1, pc2 = st.columns(2)
        with pc1:
            st.caption(
                f"Protocol: {protocol.sample_volume_ul:g} µL ±{protocol.volume_tolerance_ul:g} µL"
            )
        with pc2:
            st.caption(
                f"Coverslip: {protocol.coverslip_length_mm:g} × {protocol.coverslip_width_mm:g} mm"
            )
        theoretical_depth = results.get("theoretical_depth_um")
        if theoretical_depth is not None:
            st.caption(f"Theoretical Average Depth: {theoretical_depth:.2f} µm "
                      f"(geometric estimate — NOT a machined chamber depth)")

    if status != "ok":
        reason = results.get("concentration_status_reason") or "Calibration required."
        st.warning(f"⚠️  Total concentration: **N/A**\n\nReason: {reason}")
        _render_measurement_quality(results)
        return

    total_m       = results.get("total_concentration_m_cells_per_ml")
    motile_m      = results.get("motile_concentration_m_cells_per_ml")
    prog_m        = results.get("progressive_concentration_m_cells_per_ml")
    nonprog_m     = results.get("non_progressive_concentration_m_cells_per_ml")
    immotile_m    = results.get("immotile_concentration_m_cells_per_ml")
    motile_pct    = results.get("motile_percentage")
    prog_pct      = results.get("progressive_percentage")
    nonprog_pct   = results.get("non_progressive_percentage")
    immotile_pct  = results.get("immotile_percentage")
    quality       = results.get("concentration_counting_quality", "—")
    warnings_list = results.get("concentration_warnings") or []
    label         = results.get("concentration_label", "Concentration")

    c1, c2, c3 = st.columns(3)
    with c1:
        _metric_card(f"Total Concentration ({label})", f"{total_m:.1f} M/mL", _CLR_PRIMARY)
    with c2:
        _metric_card("Motile Concentration (MCC)", f"{motile_m:.1f} M/mL", _CLR_SUCCESS)
    with c3:
        _metric_card("Progressive Concentration", f"{prog_m:.1f} M/mL", "#8b5cf6")

    # IMPORTANT: for Standard Glass Slide this badge must never read
    # VALIDATED — `concentration_label` is centrally computed and is
    # only ever "Estimated Concentration" for this slide type unless
    # the protocol itself has been promoted after actual reference-lab
    # validation (not implemented yet — see audit). For a physical
    # chamber it reads "Measured Concentration".
    badge_color = _CLR_WARNING if "Estimated" in label or "Provisional" in label else _CLR_SUCCESS
    st.markdown(
        f'<span style="display:inline-block;padding:0.25rem 0.75rem;'
        f'border-radius:999px;background:{badge_color};color:white;'
        f'font-size:0.8rem;font-weight:700;">CONCENTRATION STATUS: '
        f'{label.upper()}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    field_stats_rows_labels: List[str] = []
    field_stats_rows_values: List[str] = []
    if is_standard_slide:
        n_valid = results.get("n_valid_fields")
        n_target = results.get("number_of_fields", results.get("number_of_counting_fields"))
        field_stats_rows_labels = [
            "Fields analyzed / target", "Mean field count", "Median field count",
            "Field standard deviation", "Field CV %",
        ]
        field_stats_rows_values = [
            f"{n_valid} / {n_target}",
            f"{results.get('field_mean_count', 0):.1f}",
            f"{results.get('field_median_count', 0):.1f}",
            f"{results.get('field_std_dev', 0):.2f}",
            f"{results.get('field_cv_percent', 0):.1f}%",
        ]

    st.table({
        "Measurement": [
            "Total concentration", "Motile concentration (MCC)",
            "Progressive concentration", "Non-progressive concentration",
            "Immotile concentration", "Motile %", "Progressive %",
            "Non-progressive %", "Immotile %", "Counting quality",
            "Observed volume (mL)",
            "Effective depth (µm)" if is_standard_slide else "Chamber depth (µm)",
            "Dilution factor",
        ] + field_stats_rows_labels,
        "Result": [
            f"{total_m:.1f} M/mL", f"{motile_m:.1f} M/mL",
            f"{prog_m:.1f} M/mL", f"{nonprog_m:.1f} M/mL",
            f"{immotile_m:.1f} M/mL",
            f"{motile_pct:.1f}%", f"{prog_pct:.1f}%",
            f"{nonprog_pct:.1f}%", f"{immotile_pct:.1f}%",
            str(quality),
            f"{results.get('observed_volume_ml', 0):.6f}",
            f"{results.get('chamber_depth_um', 0):g}",
            f"{results.get('dilution_factor', 1.0):g}",
        ] + field_stats_rows_values,
    })

    for w in warnings_list:
        st.warning(f"⚠️  {w}")

    field_warning = results.get("field_variability_warning")
    if field_warning:
        st.warning(f"⚠️  {field_warning}")

    _render_measurement_quality(results)


def _render_measurement_quality(results: Dict[str, Any]) -> None:
    """
    "Measurement Quality: XX / 100" with itemized reasons — displayed
    for BOTH the Standard Slide and physical-chamber pathways (Phase
    5/6), reading `measurement_quality_score`/`measurement_quality_reasons`
    exactly as computed centrally by
    `calculate_measurement_quality_score` — never recomputed here.
    """
    score = results.get("measurement_quality_score")
    if score is None:
        return
    reasons = results.get("measurement_quality_reasons") or []

    st.markdown("<br>", unsafe_allow_html=True)
    color = _CLR_SUCCESS if score >= 70 else (_CLR_WARNING if score >= 40 else "#ef4444")
    st.markdown(
        f'<div style="padding:0.75rem 1rem;background:{_CLR_BG};'
        f'border-radius:8px;border:1px solid {_CLR_BORDER};">'
        f'<span style="font-size:0.95rem;font-weight:700;color:{color};">'
        f'Measurement Quality: {score} / 100</span></div>',
        unsafe_allow_html=True,
    )
    for r in reasons:
        st.caption(r)


def _render_sml_section(sml: Optional[Dict[str, Any]] = None,
                        time_series: Optional[List[Tuple[float, float]]] = None) -> None:
    """
    FEATURE 10/11 — Sustained Motility Lifetime section.

    A single-timepoint analysis will always be "insufficient data" —
    SML fundamentally requires repeated measurements over time (this
    is the honest, expected result for a standalone single analysis;
    the real multi-point SML is shown in the monitoring trend section).
    """
    _section_header("⏳ Sustained Motility Lifetime")

    if sml is None or sml.get("sml_status") is None:
        st.caption(
            "Sustained Motility Lifetime requires repeated measurements "
            "over time (see Interval Monitoring)."
        )
        return

    if sml.get("sml_status") != "ok":
        st.info(
            f"ℹ️  {sml.get('sml_status_reason', 'SML unavailable — insufficient time-series measurements.')}"
        )
        if sml.get("sml_number_of_time_points"):
            st.caption(f"Time points available: {sml['sml_number_of_time_points']}")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Initial Progressive", f"{sml['initial_progressive_motility']:.1f}%", _CLR_PRIMARY)
    with c2:
        _metric_card("SML Threshold", f"{sml['sml_threshold_percentage']:.1f}%", _CLR_WARNING)
    with c3:
        _metric_card("SML", f"{sml['sml_minutes']:.1f} min", _CLR_SUCCESS)
    with c4:
        _metric_card("Time Points", str(sml.get("sml_number_of_time_points", 0)), "#6366f1")

    ts = time_series or [
        (row["minutes"], row["progressive_pct"]) for row in sml.get("sml_time_series", [])
    ]
    if len(ts) >= 2:
        import pandas as pd
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Progressive Motility (%) vs Time (minutes)**")
        df_sml = pd.DataFrame(ts, columns=["Time (min)", "Progressive %"]).set_index("Time (min)")
        st.line_chart(df_sml, use_container_width=True)


def _render_who_flags(flags: List[str]) -> None:
    if not flags:
        st.success("✅  All parameters are within WHO 2021 reference limits.")
        return
    for flag in flags:
        st.markdown(
            f'<div class="flag-box">⚠️&nbsp;&nbsp;{flag}</div>',
            unsafe_allow_html=True,
        )


def _render_recommendations(recs: List[str]) -> None:
    for rec in recs:
        st.markdown(
            f'<div class="rec-box">💡&nbsp;&nbsp;{rec}</div>',
            unsafe_allow_html=True,
        )


def _render_download_buttons(report_paths: Dict[str, str]) -> None:

    st.markdown(
        """
        <style>
        [data-testid="stDownloadButton"] > button {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }

        [data-testid="stDownloadButton"] > button:hover {
            background-color: #f8fafc !important;
            color: #111827 !important;
            border-color: #2563eb !important;
        }

        [data-testid="stDownloadButton"] > button *,
        [data-testid="stDownloadButton"] > button span {
            color: #111827 !important;
        }

        [data-testid="stDownloadButton"] > button svg {
            color: #111827 !important;
            fill: currentColor !important;
            stroke: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_pdf, col_csv, col_json = st.columns(3)
    mime_map = {
        "pdf":  ("⬇  PDF Report",  "application/pdf"),
        "csv":  ("⬇  CSV Summary", "text/csv"),
        "json": ("⬇  JSON Report", "application/json"),
    }
    columns = {"pdf": col_pdf, "csv": col_csv, "json": col_json}
    for fmt, (label, mime) in mime_map.items():
        path_str = report_paths.get(fmt, "")
        col      = columns[fmt]
        if path_str:
            p = Path(path_str)
            if p.exists():
                try:
                    col.download_button(label=label, data=p.read_bytes(),
                                        file_name=p.name, mime=mime,
                                        use_container_width=True)
                except OSError as exc:
                    col.caption(f"{fmt.upper()} unreadable: {exc}")
            else:
                col.caption(f"{fmt.upper()}: file not found")
        else:
            col.caption(f"{fmt.upper()}: not generated")


def _render_results(
    results: Dict[str, Any],
    video_stem: str,
    config: Dict[str, Any],
    sample_meta: Optional[Dict[str, Any]] = None,
) -> None:
    plot_dir     = Path(config["paths"]["outputs"]["plots"])
    tracking_dir = Path(config["paths"]["outputs"]["tracking"])

    counts   = _extract_counts(results)
    motility = _extract_motility(results)
    morph    = _extract_morphology(results)
    viab     = _extract_viability(results)
    quality  = _extract_quality(results)
    rp       = _extract_report_paths(results)

    _generate_morphology_plot(morph, plot_dir)

    score     = float(quality.get("score",    0.0))
    category  = str(quality.get("category",  "Unknown"))
    subscores = quality.get("subscores",     {})
    flags     = quality.get("who_flags",     [])
    recs      = quality.get("recommendations", [])

    if sample_meta:
        _status_row([
            ("Sample ID",   sample_meta.get("sample_id",  "—") or "—"),
            ("Animal ID",   sample_meta.get("animal_id",  "—") or "—"),
            ("Species",     st.session_state.get("_species",     "—")),
            ("Sample Type", st.session_state.get("_sample_type", "—")),
            ("Staining",    st.session_state.get("_staining",    "—")),
            ("Operator",    sample_meta.get("operator",   "—") or "—"),
        ])

    _section_header("📊 Summary Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, lbl, val, accent in [
        (c1,"Total Sperm",    str(counts["total_sperm"]),               _CLR_PRIMARY),
        (c2,"Frames",         str(counts["video_frames"]),               "#6366f1"),
        (c3,"Progressive",    f"{motility['progressive_pct']:.1f}%",    _CLR_SUCCESS),
        (c4,"Normal Morph.",  f"{morph['normal_pct']:.1f}%",            "#8b5cf6"),
        (c5,"Viability",      f"{viab:.1f}%",                            _CLR_WARNING),
        (c6,"WHO Score",      f"{score:.1f}",                            _CLR_DANGER),
    ]:
        with col:
            _metric_card(lbl, val, accent)

    st.divider()
    left_col, right_col = st.columns([1, 2], gap="large")
    with left_col:
        _section_header("🏆 WHO Quality Score")
        _render_who_score_banner(score, category)
    with right_col:
        _section_header("📈 Parameter Sub-scores")
        if subscores:
            _render_subscore_bars(subscores)
        else:
            st.caption("Sub-scores unavailable.")

    st.divider()
    tab_mot, tab_morph, tab_via = st.tabs(
        ["🏃 Motility Analysis", "🔬 Morphology", "💉 Viability"]
    )
    with tab_mot:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**CASA Motility Metrics**")
            st.table({"Parameter": [
                "Progressive","Non-Progressive","Immotile",
                "Mean VCL (µm/s)","Mean VSL (µm/s)","Mean VAP (µm/s)",
            ], "Value": [
                f"{motility['progressive_pct']:.1f}%",
                f"{motility['nonprogressive_pct']:.1f}%",
                f"{motility['immotile_pct']:.1f}%",
                f"{motility['mean_VCL']:.2f}",
                f"{motility['mean_VSL']:.2f}",
                f"{motility['mean_VAP']:.2f}",
            ]})
        with cb:
            _render_optional_plot(
                plot_dir / "motility_analysis.png",
                "Motility Distribution"
            )

            motility_path = plot_dir / "motility_analysis.png"

            st.write("DEBUG plot_dir:", str(plot_dir))
            st.write("DEBUG motility path:", str(motility_path))
            st.write("DEBUG exists:", motility_path.exists())

            if motility_path.exists():
                st.write(
                    "DEBUG size:",
                    motility_path.stat().st_size,
                    "bytes"
                )
    with tab_morph:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Morphology Classification**")
            st.table({"Parameter": [
                "Normal","Abnormal","Normal Count","Abnormal Count","Total Classified",
            ], "Value": [
                f"{morph['normal_pct']:.1f}%", f"{morph['abnormal_pct']:.1f}%",
                str(morph["normal_count"]), str(morph["abnormal_count"]),
                str(morph["total"]),
            ]})
        with cb:
            _render_optional_plot(plot_dir / "morphology_confusion_matrix.png",
                                   "Morphology Distribution")
    with tab_via:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Viability Estimation**")
            st.table({"Parameter": ["Predicted Viability","WHO Reference Min"],
                      "Value":     [f"{viab:.1f}%","54%"]})
            who_via = 54.0
            st.metric("Viability vs WHO threshold", f"{viab:.1f}%",
                      delta=f"{viab - who_via:+.1f}%",
                      delta_color="normal" if viab >= who_via else "inverse")
        with cb:
            _render_optional_plot(plot_dir / "viability_predictions.png",
                                  "Viability Model Predictions")

    st.divider()
    traj_path = tracking_dir / f"{video_stem}_trajectories.png"
    if traj_path.exists():
        _section_header("🛤️ Sperm Trajectory Map")
        st.image(str(traj_path), width="stretch")
        st.divider()

    _section_header("⚠️ WHO 2021 Threshold Check")
    _render_who_flags(flags)
    st.divider()

    _render_concentration_section(results)
    st.divider()
    _render_sml_section(sml=None)  # single analysis: honestly reports "insufficient data"
    st.divider()

    if recs:
        _section_header("💡 Clinical Recommendations")
        _render_recommendations(recs)
        st.divider()
    _section_header("📥 Download Reports")
    _render_download_buttons(rp)


# ══════════════════════════════════════════════════════════════
# Pipeline runner  (BUGFIX — Problems 1 & 2)
# ══════════════════════════════════════════════════════════════
#
# ROOT CAUSE:
#   The previous implementation started the ProgressTracker's
#   background thread correctly, but then blocked the Streamlit
#   script thread itself inside a
#       while tracker.is_running: ... time.sleep(0.5)
#   loop -- while *also* running `streamlit_autorefresh`. Two
#   failures followed from this:
#     1. Calling `_render_analysis_dashboard()` repeatedly inside a
#        single script execution doesn't "update" the dashboard in
#        place -- it appends a brand new copy of every element on
#        each iteration, and Streamlit never gets to actually
#        *rerun* the script (which is what autorefresh is supposed
#        to trigger), so the UI does not update the way a normal
#        Streamlit rerun would.
#     2. Because autorefresh keeps requesting new reruns on a timer
#        while the script is stuck in that loop, Streamlit can
#        interrupt/cancel the in-flight script run before it ever
#        reaches `return tracker.result or {}`. If that happens, the
#        real pipeline result is silently lost even though the
#        background thread finished successfully -- explaining why
#        the final dashboard sometimes never appeared. The `finally:
#        tmp_path.unlink(missing_ok=True)` around the old call site
#        could also fire on that same interruption, deleting the
#        video file while the (still-running, independent) background
#        thread was still reading it -- the WinError 32 symptom.
#
# FIX:
#   Split the old function into two non-blocking halves that follow
#   the state-driven pattern used everywhere else in this file (see
#   `_render_recording_panel`, which already gets this right):
#     * `_start_single_analysis()` creates the ProgressTracker, starts
#       its background thread, stashes it (plus the video path and
#       metadata needed once it finishes) in session_state, and
#       returns immediately -- it never blocks.
#     * `_poll_single_analysis()` is called once near the top of
#       `main()`, on every rerun. If a tracker is active it renders
#       exactly ONE snapshot, schedules the next rerun (autorefresh,
#       or a short sleep+rerun fallback), and returns -- handing
#       control back to Streamlit immediately either way. Only once
#       `tracker.is_done` does it extract the real result, store it
#       into `_analysis_results` etc., delete the temp video (now
#       guaranteed safe -- the background thread has already
#       returned from `pipeline.run_video()` by the time `is_done`
#       is set), persist to the database, and rerun once more into
#       BRANCH A.
# ══════════════════════════════════════════════════════════════


def _parse_resolution_string(res_str: str) -> Tuple[int, int]:
    """
    Parse a resolution string like "1280×720" or "1280x720" into
    (width_px, height_px). Returns (0, 0) if it cannot be parsed.
    """
    try:
        sep = "×" if "×" in res_str else "x"
        w_str, h_str = res_str.split(sep)
        return int(w_str.strip()), int(h_str.strip())
    except (ValueError, AttributeError):
        return 0, 0


def _enrich_results_with_concentration(
    results: Dict[str, Any],
    analysis_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    FALLBACK ONLY — as of the `src/camera/monitor.py` update,
    `MonitorSession.concentration_params` is forwarded directly into
    `IntervalMonitor`'s internal `pipeline.run_video()` call for every
    capture, so `latest.analysis_results` normally ALREADY contains
    concentration/MCC/progressive-concentration fields computed from
    that capture's real recorded-clip pixel dimensions — this function
    is a no-op in that case (see the early-return below).

    This remains here only to gracefully handle results that predate
    that change (e.g. a monitoring session already in progress when
    the app was updated) or any future caller that produces a results
    dict without concentration fields. When it DOES run, it uses the
    CONFIGURED camera resolution (from Section 3 / `analysis_meta`) as
    an approximation of the field-of-view width/height, since it has
    no access to the actual capture's real frame size after the fact.

    Idempotent / side-effect-free: returns a NEW dict, does not mutate
    the input, and is cheap enough to call on every render.
    """
    if not analysis_meta or not results:
        return results
    if results.get("concentration_status") is not None:
        return results  # already enriched (or computed inline) — don't redo

    params = _build_concentration_params(analysis_meta)
    width_px, height_px = _parse_resolution_string(
        analysis_meta.get("camera_resolution", "")
    )
    params["image_width_px"]  = width_px
    params["image_height_px"] = height_px

    try:
        conc_result = conc.compute_concentration_metrics_from_params(results, params)
        enriched = dict(results)
        enriched.update(conc_result.to_dict())
        return enriched
    except Exception as exc:
        logger.warning("Monitoring-capture concentration enrichment failed: {}", exc)
        return results


def _build_concentration_params(analysis_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the `concentration_params` dict expected by
    `src.analysis.concentration.compute_concentration_metrics_from_params`
    from the merged `analysis_meta` dict (sample_info + sample_cfg +
    lab_cfg + prep_cfg — see `main()`). Image width/height are filled
    in later by `run_video()` itself (it reads the actual video frame
    size), so they are omitted here.

    `manual_effective_depth_um` / `depth_correction_factor` are only
    meaningful when `slide_mode == "normal_slide"` (Standard Glass
    Slide) — `compute_concentration_metrics_from_params` routes to the
    Standard-Slide pathway internally when it sees that slide_mode and
    reads these; the physical-chamber pathway ignores them.
    """
    return {
        "slide_mode":                analysis_meta.get("slide_mode", "normal_slide"),
        "chamber_name":              analysis_meta.get("chamber", "Other"),
        "magnification":             analysis_meta.get("magnification", ""),
        "camera_resolution":         analysis_meta.get("camera_resolution", ""),
        "dilution_factor":           analysis_meta.get("dilution_factor", conc.DEFAULT_DILUTION_FACTOR),
        "number_of_counting_fields": analysis_meta.get("number_of_counting_fields", conc.DEFAULT_NUMBER_OF_COUNTING_FIELDS),
        "manual_microns_per_pixel":  analysis_meta.get("manual_microns_per_pixel"),
        "saved_calibration_um_per_pixel": analysis_meta.get("saved_calibration_um_per_pixel"),
        "manual_chamber_depth_um":   analysis_meta.get("manual_chamber_depth_um"),
        "manual_effective_depth_um": analysis_meta.get("manual_effective_depth_um"),
        "depth_correction_factor":   analysis_meta.get("depth_correction_factor"),
    }


def _start_single_analysis(pipeline: Any,
                           video_path: Path,
                           analysis_meta: Dict[str, Any]) -> None:
    """
    Kick off a single-sample analysis in the background and return
    immediately. Call `st.rerun()` right after this so control passes
    to `_poll_single_analysis()` on the next script run.
    """
    import cv2 as _cv2

    # Read total frames so progress % is accurate (cheap, released immediately)
    cap_ = _cv2.VideoCapture(str(video_path))
    total_frames = int(cap_.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap_.release()

    concentration_params = _build_concentration_params(analysis_meta)

    tracker = ProgressTracker()
    tracker.start(pipeline, video_path, total_frames=total_frames,
                  concentration_params=concentration_params)

    ss = st.session_state
    ss["_progress_tracker"]    = tracker
    ss["_progress_video_path"] = str(video_path)
    ss["_progress_meta"]       = analysis_meta
    ss["_progress_video_stem"] = ss.get("_video_stem")
    logger.info("Single-analysis ProgressTracker started for {}", video_path.name)


def _poll_single_analysis(db: DatabaseManager) -> bool:
    """
    Non-blocking poll of the active single-analysis ProgressTracker.

    Call once near the top of `main()`, before the BRANCH A/B/C
    routing. Returns True if it rendered something and the caller
    should stop (return) -- i.e. an analysis is in progress or just
    finished this run. Returns False if there is nothing to do
    (no tracker active), so `main()` should continue normally.
    """
    ss = st.session_state
    tracker: Optional[ProgressTracker] = ss.get("_progress_tracker")
    if tracker is None:
        return False

    snap = tracker.get_snapshot()

    if not snap.is_done:
        # Render exactly once per rerun -- never loop/sleep here.
        _render_analysis_dashboard(snap)
        try:
            from streamlit_autorefresh import st_autorefresh   # type: ignore
            st_autorefresh(interval=1000, key="_analysis_refresh")
        except ImportError:
            # Fallback if streamlit-autorefresh isn't installed: a short
            # sleep followed by an explicit rerun. Still only ONE sleep
            # per script run (not a loop), so Streamlit stays responsive
            # and this rerun cannot itself get interrupted mid-work.
            time.sleep(1.0)
            st.rerun()
        return True

    # ── Tracker is done: the background thread has fully returned from
    #    pipeline.run_video() by now, so it is safe to clean up the
    #    temp video file it owned. ─────────────────────────────────
    video_path_str = ss.get("_progress_video_path")
    meta            = ss.get("_progress_meta") or {}
    video_stem      = ss.get("_progress_video_stem") or ""

    ss["_progress_tracker"]    = None
    ss["_progress_video_path"] = None
    ss["_progress_meta"]       = None
    ss["_progress_video_stem"] = None

    if video_path_str:
        try:
            Path(video_path_str).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete temp video {}: {}", video_path_str, exc)

    _render_analysis_dashboard(snap)

    if snap.error:
        st.error(f"❌  Unexpected pipeline error: {snap.error}")
        return True

    results = tracker.result or {}
    if not results:
        st.warning("Analysis returned no results. Check logs for details.")
        return True

    elapsed = snap.elapsed_s

    # Store BEFORE any further st.* call, and BEFORE the DB save, so a
    # save failure never prevents the real results from being displayed.
    ss["_analysis_results"]    = results
    ss["_analysis_video_stem"] = video_stem
    ss["_analysis_elapsed"]    = elapsed
    ss["_analysis_meta"]       = meta

    # Auto-save to database -- results are guaranteed real and final here.
    try:
        db.save_single_session(
            meta            = meta,
            results         = results,
            report_paths    = _extract_report_paths(results),
            annotated_video = str(results.get("annotated_video_path", "")),
        )
        ss["_db_session_saved"] = True
        st.toast("✅ Session saved to database", icon="🗄️")
    except Exception as exc:
        logger.warning("DB auto-save (single) failed: {}", exc)

    st.rerun()   # clean transition into BRANCH A on the next run
    return True


def _render_analysis_dashboard(snap) -> None:
    """
    Render the professional AI analysis progress dashboard.

    Parameters
    ----------
    snap : ProgressSnapshot
    """
    # Header
    st.markdown(
        f'''<div style="background:linear-gradient(135deg,{_CLR_PRIMARY},#4f46e5);
border-radius:14px;padding:1rem 1.4rem;color:white;margin-bottom:0.8rem;">
<div style="font-size:1.1rem;font-weight:800;">🤖 AI Analysis Progress</div>
<div style="font-size:0.82rem;opacity:0.8;margin-top:2px;">
{snap.current_stage_name} &nbsp;·&nbsp; Elapsed: {snap.elapsed_mmss}
&nbsp;·&nbsp; ETA: {snap.eta_mmss}</div></div>''',
        unsafe_allow_html=True,
    )

        # Overall progress
   # =========================================================
# OVERALL PROGRESS
# =========================================================

    _section_header("📊 Overall Progress")

    st.progress(
        min(1.0, snap.overall_pct / 100.0),
        text=f"Overall Progress: {snap.overall_pct:.1f}%",
    )

    st.markdown("<br>", unsafe_allow_html=True)


    # =========================================================
    # PROGRESS STAGES
    # =========================================================

    _section_header("📋 Progress Stages")

    for stage_name, stage_percent in _PROGRESS_STAGES:

        # Completed
        if snap.overall_pct >= stage_percent:
            icon = "✅"
            text_colour = _CLR_SUCCESS

        # Currently running / close to this stage
        elif snap.overall_pct >= stage_percent - 15:
            icon = "🔄"
            text_colour = _CLR_PRIMARY

        # Waiting
        else:
            icon = "⏳"
            text_colour = _CLR_TEXT_SEC

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:0.5rem;
                margin:0.55rem 0;
                font-size:0.9rem;
                color:{text_colour};
            ">
                <span style="font-size:1rem;">{icon}</span>
                <span>{stage_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ══════════════════════════════════════════════════════════════
# Session-state registry
# ══════════════════════════════════════════════════════════════

_SS_DEFAULTS: Dict[str, Any] = {
    # Pipeline results (single analysis)
    "_analysis_results":    None,
    "_analysis_video_stem": None,
    "_analysis_elapsed":    0.0,
    "_analysis_meta":       None,
    # UI flow
    "_form_submitted":      False,
    "_input_source":        "upload",
    "_video_stem":          "",
    # Sidebar form cache
    "_sample_id":           "",
    "_animal_id":           "",
    "_species":             "",
    "_breed":               "",
    "_sample_type":         "",
    "_staining":            "",
    # Camera state
    "_cam_connected":       False,
    "_cam_streaming":       False,
    "_cam_captured":        False,
    "_camera_clip_path":    "",
    "_cam_device_idx":      None,
    "_cam_resolution":      "",
    "_cam_fps":             "",
    # Phase 6 — real hardware manager objects
    "_camera_manager":      None,    # CameraManager | None
    "_recording_manager":   None,    # RecordingManager | None
    "_progress_tracker":    None,    # ProgressTracker | None
    "_available_devices":   None,    # list[CameraDevice] | None
    "_no_microscope":       False,   # True when zero imaging devices are detected
    # Non-blocking single-analysis progress (BUGFIX — see _poll_single_analysis)
    "_progress_video_path": None,    # str | None -- temp video owned by the tracker thread
    "_progress_meta":       None,    # dict | None -- analysis_meta to attach once done
    "_progress_video_stem": None,    # str | None
    # Monitoring (Phase 3)
    "_monitor_enabled":     False,
    "_monitor_session":     None,    # MonitorSession | None
    "_monitor_obj":         None,    # IntervalMonitor | None
    "_monitor_runner":      None,    # _MonitorRunner | None (BUGFIX — background driver)
    "_db_session_saved":    False,
    # Trend analysis (Phase 4)
    "_trend_report":        None,    # TrendReport | None
    "_trend_report_paths":  None,    # dict {"pdf","csv","json"} | None
    # Database (Phase 5)
    "_db_page":             "analysis",  # "analysis" | "database"
    "_db_view_session_id":  None,
    "_db_compare_set":      set(),
    "_db_confirm_delete":   None,
}


def _init_session_state() -> None:
    for key, default in _SS_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _clear_analysis_state() -> None:
    """Reset all analysis, camera, monitoring and progress state."""
    clear_keys = [
        "_analysis_results","_analysis_video_stem",
        "_analysis_elapsed","_analysis_meta",
        "_form_submitted","_input_source","_video_stem",
        "_cam_connected","_cam_streaming","_cam_captured",
        "_camera_clip_path","_cam_device_idx","_cam_resolution","_cam_fps",
        "_monitor_enabled","_monitor_session","_monitor_obj",
        "_trend_report","_trend_report_paths",
        "_db_view_session_id","_db_confirm_delete",
        "_available_devices","_progress_tracker","_no_microscope",
        # BUGFIX: these were previously never cleared, which caused
        # (a) a stale "_db_session_saved=True" from a prior session to
        #     silently block the DB auto-save for the very next analysis,
        #     and (b) an orphaned progress/monitor-runner thread and its
        #     temp video path to linger in session_state.
        "_progress_video_path","_progress_meta","_progress_video_stem",
        "_db_session_saved","_monitor_runner",
    ]

    # Stop any background monitor runner thread BEFORE dropping the
    # reference below, so it doesn't keep calling into a torn-down
    # session after this function returns.
    runner = st.session_state.get("_monitor_runner")
    if runner is not None:
        try:
            runner.stop()
        except Exception:
            pass

    for key in clear_keys:
        st.session_state[key] = _SS_DEFAULTS.get(key)

    # Disconnect CameraManager if present
    mgr = st.session_state.get("_camera_manager")
    if mgr is not None:
        try:
            mgr.disconnect()
        except Exception:
            pass
    st.session_state["_camera_manager"]    = None
    st.session_state["_recording_manager"] = None
    st.session_state["_db_compare_set"]    = set()


# ══════════════════════════════════════════════════════════════
# Calibration Setup (admin/lab, one-time microscope optical
# calibration)
# ══════════════════════════════════════════════════════════════

def _render_calibration_admin_page(db: DatabaseManager) -> None:
    """
    ONE-TIME LAB/ADMIN microscope optical calibration workflow —
    stage micrometer -> pixels -> µm/pixel, tied to an EXACT hardware
    configuration and saved for the normal analysis workflow to look
    up automatically (see `_render_calibration_status_block`).

    This page never asks the operator to calculate anything: they
    identify two points a known distance apart on a captured
    calibration image, and every arithmetic step happens in
    `src.calibration.optical_calibration`.

    LIMITATION (documented, not hidden): this environment has no
    click-to-select image widget available, so the "guided point
    selection" step uses numeric pixel-coordinate inputs next to the
    displayed calibration image, rather than clicking directly on it.
    The operator still performs no calculation — they only read two
    approximate pixel positions off the displayed image — but a true
    click/drag selector would be a meaningfully better experience if
    this project adds a compatible Streamlit component later.
    """
    st.title("📐 Microscope Calibration")
    st.caption(
        "Admin / lab setup — one-time, per exact hardware configuration. "
        "Not part of the routine semen-analysis workflow."
    )

    # ISSUE 2 FIX — single camera ownership path, checked FIRST so the
    # camera identity used in Section 1 is the genuinely connected
    # device rather than a guess. Without this, a calibration could be
    # measured on (or filed against) the wrong camera.
    mgr = _get_camera_manager()
    if not mgr.is_connected:
        st.warning(
            "⚠️  Connect the microscope camera to begin calibration.\n\n"
            "Calibration deliberately uses the same camera you select in "
            "the normal camera workflow — it will not pick a device for "
            "you, so that a microscope calibration can never be recorded "
            "against the PC webcam by mistake."
        )
        st.info(
            "Go to **🔬 Analysis → Input Source → Live Microscope**, select "
            "your microscope and connect it, then return to this page."
        )
        return

    status = mgr.get_status()
    connected_camera_name = getattr(status, "device_name", "") or _get_camera_identifier()

    _section_header("1. Hardware Configuration")
    c1, c2 = st.columns(2)
    with c1:
        microscope = st.selectbox("Microscope", _MICROSCOPE_TYPES, key="_cal_microscope")
        magnification = st.selectbox("Magnification", _MAGNIFICATIONS, key="_cal_magnification")
    with c2:
        camera = st.text_input(
            "Camera identifier", value=connected_camera_name, key="_cal_camera",
            disabled=True,
            help="Read-only: taken from the currently connected imaging "
                 "device, so the saved calibration can never be filed "
                 "against a different camera than the one it was measured on.",
        )
        camera_resolution = st.selectbox("Resolution", _CAMERA_RESOLUTIONS, key="_cal_resolution")
    digital_zoom = st.text_input(
        "Digital zoom / crop state (leave blank if not applicable)",
        value="", key="_cal_digital_zoom",
    )

    existing = db.get_calibration(microscope, camera, magnification, camera_resolution, digital_zoom)
    if existing:
        st.info(
            f"ℹ️  A calibration already exists for this exact configuration "
            f"({existing['microns_per_pixel']:.4f} µm/pixel, saved "
            f"{existing['created_at']}). Completing the steps below will "
            f"replace it."
        )

    st.divider()
    _section_header("2. Microscope View")

    # Camera ownership was already validated at the top of this page:
    # `mgr` is the shared CameraManager and is confirmed connected.
    # This page never enumerates, auto-selects, constructs a
    # VideoCapture/USBMicroscope, or falls back to another camera.
    st.success(
        f"✅  Using connected imaging device: **{connected_camera_name}**  \n"
        f"Resolution: {getattr(status, 'resolution', camera_resolution)}  ·  "
        f"Frames received: {getattr(status, 'frames_received', 0):,}"
    )
    st.caption(
        "This is the same live microscope feed used for analysis — the "
        "stage micrometer will appear in this image."
    )

    if st.button("📸 Capture Calibration Image", key="_cal_capture", type="primary"):
        # Display-only read from the shared background reader. No new
        # capture device, no second reader thread.
        frame = mgr.get_latest_frame()
        if frame is None:
            st.error(
                "❌  The connected device did not provide a frame. Check the "
                "microscope connection and try again — calibration will not "
                "switch to another camera."
            )
        else:
            st.session_state["_cal_captured_frame"] = frame
            st.success("Calibration frame captured from the microscope.")

    captured = st.session_state.get("_cal_captured_frame")
    if captured is not None:
        rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Calibration image — read approximate pixel "
                              "coordinates of the two known-distance points below.",
                 use_container_width=True)
        img_h, img_w = captured.shape[:2]
        st.caption(f"Image size: {img_w} × {img_h} px")
    else:
        st.info("Place the stage micrometer under the microscope, focus it, "
               "then capture a calibration frame to continue.")

    st.divider()
    _section_header("3. Known Distance & Point Selection")
    known_distance_um = st.number_input(
        "Known stage-micrometer distance (µm)", min_value=0.0, max_value=100_000.0,
        value=0.0, step=1.0, key="_cal_known_distance",
        help="Read directly off the stage micrometer's markings — never invented.",
    )

    st.caption(
        "Guided point selection: no click-to-select widget is available in this "
        "environment, so identify the two endpoints by their approximate pixel "
        "coordinates in the image above. You are not calculating µm/pixel "
        "yourself — only reading two pixel positions."
    )
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Point 1**")
        x1 = st.number_input("x1 (px)", min_value=0.0, value=0.0, step=1.0, key="_cal_x1")
        y1 = st.number_input("y1 (px)", min_value=0.0, value=0.0, step=1.0, key="_cal_y1")
    with p2:
        st.markdown("**Point 2**")
        x2 = st.number_input("x2 (px)", min_value=0.0, value=0.0, step=1.0, key="_cal_x2")
        y2 = st.number_input("y2 (px)", min_value=0.0, value=0.0, step=1.0, key="_cal_y2")

    if st.button("📏 Measure & Validate", key="_cal_measure", type="primary"):
        pixel_distance = oc.measure_pixel_distance(x1, y1, x2, y2)
        problems = oc.validate_calibration_inputs(known_distance_um, pixel_distance)
        if problems:
            st.error("❌ Calibration failed.")
            for p in problems:
                st.warning(f"⚠️  {p}")
            st.session_state["_cal_pending_record"] = None
        else:
            captured_frame = st.session_state.get("_cal_captured_frame")
            cal_img_h, cal_img_w = (
                captured_frame.shape[:2] if captured_frame is not None else (0, 0)
            )
            record = oc.build_calibration_record(
                microscope=microscope, camera=camera, magnification=magnification,
                camera_resolution=camera_resolution, known_distance_um=known_distance_um,
                pixel_distance=pixel_distance, digital_zoom=digital_zoom,
                image_width_px=cal_img_w, image_height_px=cal_img_h,
                operator=st.session_state.get("_inp_operator", ""),
                software_version=globals().get("_APP_VERSION", ""),
            )
            st.session_state["_cal_pending_record"] = record
            st.success(
                f"✅  Measured {pixel_distance:.1f} px = {known_distance_um:g} µm "
                f"→ **{record.microns_per_pixel:.4f} µm/pixel**"
            )

    pending = st.session_state.get("_cal_pending_record")
    if pending is not None:
        st.divider()
        _section_header("4. Save Calibration")
        st.write(
            f"Configuration: **{pending.microscope} / {pending.camera} / "
            f"{pending.magnification} @ {pending.camera_resolution}** "
            f"(zoom: {pending.digital_zoom})"
        )
        st.write(f"Result: **{pending.microns_per_pixel:.4f} µm/pixel**")
        if st.button("💾 Save Calibration", key="_cal_save", type="primary"):
            from dataclasses import asdict
            cal_id = db.save_calibration(asdict(pending))
            st.session_state["_cal_pending_record"] = None
            st.session_state["_cal_captured_frame"] = None
            st.success(f"Calibration saved ({cal_id[:8]}).")
            st.rerun()

    st.divider()
    _section_header("Saved Calibrations")
    st.caption(
        "One entry per imaging configuration, showing the CURRENT "
        "(most recent) calibration. Every earlier measurement is "
        "preserved as history — nothing is overwritten."
    )
    current_calibrations = db.list_current_calibrations()
    if not current_calibrations:
        st.caption("No calibrations saved yet.")
    else:
        for cal in current_calibrations:
            with st.expander(
                f"{cal['microscope']} / {cal['camera']} — {cal['magnification']} "
                f"@ {cal['camera_resolution']}  ·  {cal['microns_per_pixel']:.4f} µm/pixel"
            ):
                st.markdown("**Current calibration**")
                st.json({
                    "calibration_id": cal["calibration_id"],
                    "digital_zoom": cal["digital_zoom"],
                    "known_distance_um": cal["known_distance_um"],
                    "measured_pixel_distance": cal["measured_pixel_distance"],
                    "microns_per_pixel": cal["microns_per_pixel"],
                    "image_width_px": cal.get("image_width_px"),
                    "image_height_px": cal.get("image_height_px"),
                    "operator": cal.get("operator"),
                    "calibration_status": cal["calibration_status"],
                    "created_at": cal["created_at"],
                })

                history = db.get_calibration_history(
                    cal["microscope"], cal["camera"], cal["magnification"],
                    cal["camera_resolution"], cal["digital_zoom"],
                )
                rep = db.get_calibration_repeatability(
                    cal["microscope"], cal["camera"], cal["magnification"],
                    cal["camera_resolution"], cal["digital_zoom"],
                )

                st.markdown(f"**Repeat measurements: {rep['n_measurements']}**")
                if rep["n_measurements"] >= 2:
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.metric("Mean", f"{rep['mean_um_per_pixel']:.4f} µm/px")
                    with rc2:
                        st.metric("Std Dev", f"{rep['std_dev_um_per_pixel']:.5f}")
                    with rc3:
                        st.metric("CV", f"{rep['cv_percent']:.2f}%")
                    st.caption(
                        "Reported as-is. This software applies no pass/fail "
                        "acceptance threshold to calibration repeatability — "
                        "judge these against your own laboratory protocol."
                    )
                else:
                    st.caption(
                        "Repeatability not yet assessed — a single measurement "
                        "cannot establish variation. Repeat the calibration to "
                        "obtain repeatability statistics."
                    )

                if len(history) > 1:
                    st.markdown("**Measurement history**")
                    st.table({
                        "Date": [h["created_at"] for h in history],
                        "µm/pixel": [f"{h['microns_per_pixel']:.4f}" for h in history],
                        "Known (µm)": [f"{h['known_distance_um']:g}" for h in history],
                        "Measured (px)": [f"{h['measured_pixel_distance']:.1f}" for h in history],
                        "Operator": [h.get("operator") or "—" for h in history],
                    })

                if st.button("🗑️ Delete current measurement",
                             key=f"_cal_del_{cal['calibration_id']}"):
                    db.delete_calibration(cal["calibration_id"])
                    st.rerun()


# ══════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """
    Entry point.

    Page routing
    ------------
    _db_page == "database"  → render_database_page (Phase 5)
    _db_page == "analysis"  → BRANCH C / A / B (Phases 1-4)

    Session-state branching (analysis page)
    ----------------------------------------
    BRANCH C  monitoring session active
    BRANCH A  single analysis results stored
    BRANCH B  intake form + run button

    Database auto-save
    ------------------
    Every completed analysis (single or monitoring) is automatically
    persisted to the local SQLite database.  No user action required.
    """
    _init_session_state()
    _render_sidebar()
    config = _load_config()
    db     = _load_db(id(config))

    # ── Page header ────────────────────────────────────────────
    hdr_col, badge_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(
            '<div class="page-title">🔬 Automated Semen Analysis System</div>'
            '<div class="page-subtitle">'
            "AI-powered andrology platform · YOLOv8 · ByteTrack · CASA · WHO 2021"
            "</div>",
            unsafe_allow_html=True,
        )
    with badge_col:
        st.markdown(
            f'<div style="margin-top:0.6rem;text-align:right;">'
            f'<span style="background:{_CLR_PRIMARY};color:white;'
            f'border-radius:6px;padding:3px 10px;font-size:0.72rem;'
            f'font-weight:700;letter-spacing:0.05em;">v5.0 RESEARCH</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    ss = st.session_state

    # ══════════════════════════════════════════════════════════
    # DATABASE PAGE (Phase 5)
    # ══════════════════════════════════════════════════════════
    if ss.get("_db_page") == "database":
        render_database_page(db)
        return

    # ══════════════════════════════════════════════════════════
    # CALIBRATION SETUP PAGE (admin/lab, one-time)
    # ══════════════════════════════════════════════════════════
    if ss.get("_db_page") == "calibration":
        _render_calibration_admin_page(db)
        return

    # ══════════════════════════════════════════════════════════
    # NON-BLOCKING SINGLE-ANALYSIS PROGRESS  (BUGFIX — Problems 1 & 2)
    # ══════════════════════════════════════════════════════════
    # Must be checked before BRANCH A/B/C: if an analysis is currently
    # running in the background, render its live snapshot and hand
    # control straight back to Streamlit (no blocking loop). See
    # `_poll_single_analysis` for the full explanation.
    if _poll_single_analysis(db):
        return

    # ══════════════════════════════════════════════════════════
    # BRANCH C — monitoring session active
    # ══════════════════════════════════════════════════════════
    if ss.get("_monitor_session") is not None:
        mon_session: MonitorSession = ss["_monitor_session"]

        top_l, top_r = st.columns([3, 1])
        with top_l:
            status = "Complete" if mon_session.is_complete else (
                "Active" if mon_session.is_running else "Stopped")
            st.markdown(
                f'<span style="font-size:1rem;font-weight:700;'
                f'color:{_CLR_MONITOR};">⏱ Monitoring Session · {status}</span>',
                unsafe_allow_html=True,
            )
        with top_r:
            if st.button("🔄  New Analysis", use_container_width=True,
                         key="_btn_new_from_c",
                         help="End monitoring and start fresh"):
                # BUGFIX: stop the background _MonitorRunner (not just
                # the IntervalMonitor object) so its thread actually
                # exits instead of continuing to tick in the background
                # after the session state it depends on is cleared.
                runner = ss.get("_monitor_runner")
                if runner is not None:
                    runner.stop()
                elif ss.get("_monitor_obj") is not None:
                    ss["_monitor_obj"].stop()
                _clear_analysis_state()
                st.rerun()

        st.divider()
        _render_monitoring_dashboard(config)

        # ── Auto-save completed monitoring session (BUGFIX — Problem 3) ──
        # ROOT CAUSE: the previous condition fired as soon as the
        # session was complete/stopped and had >=1 capture, regardless
        # of whether the monitoring report (PDF/CSV/JSON) had actually
        # been generated yet. `_trend_report_paths` was still None at
        # that point (report generation was a separate manual button
        # click), so the DB row got saved with an empty
        # `monitoring_report_paths` dict -- and because
        # `_db_session_saved` was immediately set True, the row was
        # NEVER updated later even after the user generated the report.
        # FIX: also require the report paths to exist before saving.
        # (`_render_trend_analysis_section` now auto-generates them the
        # same way it already auto-generates the trend report, so this
        # will populate itself without requiring a manual click.)
        if (mon_session.is_complete or not mon_session.is_running) \
                and mon_session.captures_done >= 1 \
                and ss.get("_trend_report_paths") \
                and not ss.get("_db_session_saved"):
            try:
                # FEATURE 9: enrich every capture's analysis_results with
                # concentration/MCC before persisting, so the stored
                # analysis_json for each capture carries the same
                # concentration fields the dashboard displays (in-place
                # dict update — safe regardless of the exact
                # MonitorSessionResult container type).
                analysis_meta_for_conc = ss.get("_analysis_meta") or {}
                sml_for_db = ss.get("_sml_result") or {}
                analysis_meta_for_conc = {
                    **analysis_meta_for_conc,
                    "sml_minutes":                 sml_for_db.get("sml_minutes"),
                    "sml_threshold_percentage":    sml_for_db.get("sml_threshold_percentage"),
                    "initial_progressive_motility":sml_for_db.get("initial_progressive_motility"),
                }
                for r in mon_session.results:
                    if not r.error and r.analysis_results:
                        enriched = _enrich_results_with_concentration(
                            r.analysis_results, analysis_meta_for_conc,
                        )
                        r.analysis_results.update(enriched)

                db.save_monitoring_session(
                    meta                    = analysis_meta_for_conc,
                    monitor_session         = mon_session,
                    trend_report            = ss.get("_trend_report"),
                    monitoring_report_paths = ss.get("_trend_report_paths") or {},
                )
                ss["_db_session_saved"] = True
                st.toast("✅ Session saved to database", icon="🗄️")
            except Exception as exc:
                logger.warning("DB auto-save (monitoring) failed: {}", exc)
        return

    # ══════════════════════════════════════════════════════════
    # BRANCH A — single analysis results stored
    # ══════════════════════════════════════════════════════════
    if ss["_analysis_results"] is not None:
        elapsed = ss["_analysis_elapsed"]
        results = ss["_analysis_results"]
        stem    = ss["_analysis_video_stem"] or ""
        meta    = ss.get("_analysis_meta")

        top_l, top_r = st.columns([3, 1])
        with top_l:
            st.success(f"✅  Analysis complete in **{elapsed:.1f}s**")
        with top_r:
            if st.button("🔄  New Analysis", use_container_width=True,
                         help="Clear results and start a new sample intake"):
                _clear_analysis_state()
                st.rerun()

        st.divider()
        _render_results(results, stem, config, sample_meta=meta)
        return

    # ══════════════════════════════════════════════════════════
    # BRANCH B — intake form
    # ══════════════════════════════════════════════════════════
    sample_info = _render_section_1()
    sample_cfg  = _render_section_2()
    lab_cfg     = _render_section_3()
    prep_cfg    = _render_section_4(
        chamber           = lab_cfg["chamber"],
        magnification     = lab_cfg["magnification"],
        camera_resolution = lab_cfg["camera_resolution"],
        microscope_type   = lab_cfg["microscope_type"],
    )
    tmp_path    = _render_section_5()
    monitor_cfg = _render_section_6()

    # Use the captured microscope video as the analysis input
    if (
        ss.get("_input_source") == "microscope"
        and ss.get("_camera_clip_path")
    ):
        tmp_path = Path(ss["_camera_clip_path"])

        print("=" * 70)
        print("MICROSCOPE ANALYSIS INPUT")
        print("Video path :", tmp_path)
        print("Exists     :", tmp_path.exists())
        print("Size       :", tmp_path.stat().st_size if tmp_path.exists() else "N/A")
        print("=" * 70)

    st.markdown("<br>", unsafe_allow_html=True)

    run_col, _ = st.columns([2, 3])
    with run_col:
        run_clicked = st.button(
            "▶  Run Full Analysis",
            type="primary",
            use_container_width=True,
            help="Validate form → run AI pipeline → display results",
            key="_run_btn",
        )

    if not run_clicked:
        if tmp_path and ss.get("_input_source") == "upload":
            tmp_path.unlink(missing_ok=True)
        return

    # ── Validation ──────────────────────────────────────────────
    errors = _validate_form(sample_info)
    if monitor_cfg["enabled"] and ss.get("_input_source") != "live":
        errors.append(
            "Monitoring requires Live Microscope input (§5).  "
            "Switch to 'Live Microscope' or disable monitoring."
        )
    elif tmp_path is None:
        errors.append("No video available (§5 Input Source).")

    if errors:
        for err in errors:
            st.error(f"❌  {err}")
        if tmp_path and ss.get("_input_source") == "upload":
            tmp_path.unlink(missing_ok=True)
        return

    # ── Cache form values ────────────────────────────────────────
    ss["_sample_id"]        = sample_info["sample_id"]
    ss["_animal_id"]        = sample_info["animal_id"]
    ss["_species"]          = sample_cfg["species"]
    ss["_breed"]            = sample_cfg["breed"]
    ss["_sample_type"]      = sample_cfg["sample_type"]
    ss["_staining"]         = prep_cfg["staining_type"]
    ss["_form_submitted"]   = True
    ss["_db_session_saved"] = False   # reset save flag for new session

    analysis_meta = {
        **sample_info, **sample_cfg, **lab_cfg, **prep_cfg,
        "video_stem":   ss["_video_stem"],
        "input_source": ss["_input_source"],
    }

    st.divider()

    # ── Load pipeline ────────────────────────────────────────────
    try:
        pipeline = _load_pipeline(id(config))
    except ImportError as exc:
        st.error(f"❌  Could not import backend: `{exc}`")
        if tmp_path and ss.get("_input_source") == "upload":
            tmp_path.unlink(missing_ok=True)
        return
    except Exception as exc:
        st.error(f"❌  Pipeline init failed: {exc}")
        if tmp_path and ss.get("_input_source") == "upload":
            tmp_path.unlink(missing_ok=True)
        return

    # ══════════════════════════════════════════════════════════
    # PATH M — monitoring session
    # ══════════════════════════════════════════════════════════
    if monitor_cfg["enabled"]:
        # BUGFIX (Problem 4 — device contention):
        # If the user connected the live microscope preview
        # (CameraManager.connect()) before enabling monitoring, its
        # background frame-reader thread still holds the physical
        # camera device open via DirectShow. IntervalMonitor opens its
        # own, independent capture handle to the SAME device index —
        # on Windows a camera generally cannot be opened by two
        # capture handles at once, so this silently prevented
        # monitoring from ever capturing anything. Release the live
        # preview's handle first so the device is free.
        live_mgr = ss.get("_camera_manager")
        if live_mgr is not None:
            try:
                live_mgr.disconnect()
            except Exception as exc:
                logger.warning("Could not disconnect live camera manager "
                               "before starting monitoring: {}", exc)
            ss["_camera_manager"] = None
            ss["_cam_connected"]  = False
            ss["_cam_streaming"]  = False

        mon_session = MonitorSession(
            interval_min    = monitor_cfg["interval_min"],
            duration_min    = monitor_cfg["duration_min"],
            sample_id       = sample_info["sample_id"] or "unknown",
            device_index    = int(ss.get("_cam_device_idx", 0)),
            resolution      = ss.get("_inp_camera_resolution", "1280×720"),
            fps             = ss.get("_inp_camera_fps", "30 FPS"),
            clip_duration_s = _RECORD_DURATION_S,
            # FEATURE 1-29: forwarded to every capture's run_video()
            # call inside IntervalMonitor, so each capture's
            # concentration/MCC/progressive-concentration is computed
            # using that capture's REAL recorded-clip pixel dimensions
            # (see src/camera/monitor.py). None-safe: if the operator
            # never configured calibration, this is still a valid
            # dict — compute_concentration_metrics simply returns
            # status="calibration_required"/"depth_required" for each
            # capture rather than fabricating a number.
            concentration_params = _build_concentration_params(analysis_meta),
        )
        monitor = IntervalMonitor(mon_session, pipeline)
        ss["_monitor_session"]  = mon_session
        ss["_monitor_obj"]      = monitor
        ss["_analysis_meta"]    = analysis_meta
        ss["_db_session_saved"] = False

        # BUGFIX (Problems 1/4 — blocking calls under autorefresh):
        # `monitor.start()` performs the first capture+analysis cycle
        # in-line and can take well over a minute. Calling it directly
        # here (even under st.spinner) blocked the whole script; once
        # the monitoring dashboard's autorefresh kicked in on later
        # reruns, an equivalent blocking call inside monitor.tick()
        # could get interrupted mid-capture by a competing rerun
        # request. `_MonitorRunner` now performs BOTH the initial
        # start() and every subsequent tick() from one dedicated
        # background thread, so the Streamlit thread never blocks and
        # is never a moving target for autorefresh to interrupt.
        runner = _MonitorRunner(monitor)
        runner.launch()
        ss["_monitor_runner"] = runner

        st.info(
            f"⏱  Starting monitoring session — first capture beginning now…\n"
            f"Interval: **{monitor_cfg['interval_min']} min** · "
            f"Duration: **{monitor_cfg['duration_min']} min** · "
            f"Planned captures: **{mon_session.expected_captures}**"
        )
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════
    # PATH S — single analysis
    # ══════════════════════════════════════════════════════════
    # BUGFIX (Problems 1 & 2): previously this called
    # `_run_pipeline_with_progress()` synchronously and deleted
    # `tmp_path` in a `finally` block tied to THIS script run's own
    # execution scope. If a competing autorefresh-triggered rerun
    # interrupted that scope before the background tracker thread had
    # actually finished, the `finally` could still fire and delete the
    # video out from under the still-running thread (WinError 32),
    # and the real result could be lost entirely.
    #
    # Now we only ever kick the tracker off and hand control straight
    # back to Streamlit. `_poll_single_analysis()` (checked near the
    # top of `main()`) owns rendering the live dashboard, detecting
    # completion, deleting the temp video (only once it's confirmed
    # safe to do so), storing the results, and saving to the database.
    try:
        _start_single_analysis(pipeline, tmp_path, analysis_meta)
    except Exception as exc:
        st.error(f"❌  Could not start analysis: {exc}")
        tmp_path.unlink(missing_ok=True)
        return

    st.rerun()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()