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
* ``IntervalMonitor.tick()`` is called on every rerun; if an interval
  has elapsed it executes a full capture + analysis cycle synchronously.
* ``streamlit-autorefresh`` triggers reruns at ``_MONITOR_TICK_MS``
  (default 10 s) so the countdown / progress always updates.
* Monitoring results table with sparkline-style trend column.
* "Stop Monitoring" button available at any time.
* All Phase 1 (intake form, upload, WHO results) and Phase 2
  (live USB microscope, 6 camera buttons) code is 100 % preserved.

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
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import base64
import cv2
import numpy as np
import streamlit as st


# ================================================================
# Stage Micrometer Click Component
# Streamlit 1.58+ native Components V2
# ================================================================

_CAL_IMAGE_CLICKER = st.components.v2.component(
    name="everse_stage_micrometer_clicker",
    
    html="""
    <div id="cal-click-container">
        <canvas id="cal-click-canvas"></canvas>
        <div id="cal-click-status">
            Click the first graduation line, then the second graduation line.
        </div>
    </div>
    """,

    css="""
    #cal-click-container {
        width: 100%;
        box-sizing: border-box;
    }

    #cal-click-canvas {
        display: block;
        width: 100%;
        height: auto;
        cursor: crosshair;
        border-radius: 8px;
        border: 2px solid rgba(100, 116, 139, 0.35);
    }

    #cal-click-status {
        margin-top: 8px;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 14px;
        color: var(--st-text-color);
        background: rgba(100, 116, 139, 0.08);
    }
    """,

    js="""
    export default function(component) {

        const {
            parentElement,
            data,
            setTriggerValue
        } = component;

        const canvas = parentElement.querySelector("#cal-click-canvas");
        const status = parentElement.querySelector("#cal-click-status");

        if (!canvas || !data || !data.image_base64) {
            return;
        }

        const ctx = canvas.getContext("2d");

        const image = new Image();

        image.onload = () => {

            const originalWidth = Number(data.original_width);
            const originalHeight = Number(data.original_height);

            const displayWidth = Number(data.display_width);

            const scale = displayWidth / originalWidth;
            const displayHeight = Math.round(originalHeight * scale);

            canvas.width = displayWidth;
            canvas.height = displayHeight;

            canvas.style.width = displayWidth + "px";
            canvas.style.height = displayHeight + "px";

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

            ctx.drawImage(
                image,
                0,
                0,
                displayWidth,
                displayHeight
            );

            // Draw existing selected points.
            const points = data.points || [];

            if (points.length >= 1) {

                const x = Number(points[0][0]) * scale;
                const y = Number(points[0][1]) * scale;

                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fillStyle = "#00dc50";
                ctx.fill();

                ctx.lineWidth = 3;
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();

                ctx.fillStyle = "#00dc50";
                ctx.font = "bold 18px sans-serif";
                ctx.fillText("A", x + 12, y - 12);
            }

            if (points.length >= 2) {

                const x1 = Number(points[0][0]) * scale;
                const y1 = Number(points[0][1]) * scale;

                const x2 = Number(points[1][0]) * scale;
                const y2 = Number(points[1][1]) * scale;

                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.lineWidth = 3;
                ctx.strokeStyle = "#00dc50";
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(x2, y2, 8, 0, Math.PI * 2);
                ctx.fillStyle = "#00dc50";
                ctx.fill();

                ctx.lineWidth = 3;
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();

                ctx.fillStyle = "#00dc50";
                ctx.font = "bold 18px sans-serif";
                ctx.fillText("B", x2 + 12, y2 - 12);

                status.textContent =
                    "Line A and Line B selected. Click again to start a new measurement.";
            }
            else if (points.length === 1) {

                status.textContent =
                    "Line A selected. Now click the second graduation line (Line B).";
            }
            else {

                status.textContent =
                    "Click the first graduation line, then the second graduation line.";
            }
        };

        image.src = "data:image/jpeg;base64," + data.image_base64;

        // Prevent multiple handlers from accumulating.
        if (canvas.dataset.clickHandlerAttached === "true") {
            return;
        }

        canvas.dataset.clickHandlerAttached = "true";

        canvas.addEventListener("click", (event) => {

            const rect = canvas.getBoundingClientRect();

            if (
                rect.width <= 0 ||
                rect.height <= 0
            ) {
                return;
            }

            const originalWidth = Number(data.original_width);
            const originalHeight = Number(data.original_height);

            // Convert browser/display coordinates
            // back to ORIGINAL image coordinates.
            const x = Math.round(
                ((event.clientX - rect.left) / rect.width)
                * originalWidth
            );

            const y = Math.round(
                ((event.clientY - rect.top) / rect.height)
                * originalHeight
            );

            const safeX = Math.max(
                0,
                Math.min(originalWidth - 1, x)
            );

            const safeY = Math.max(
                0,
                Math.min(originalHeight - 1, y)
            );

            setTriggerValue(
                "point_clicked",
                {
                    x: safeX,
                    y: safeY
                }
            );
        });

        return () => {
            // Streamlit handles component cleanup.
        };
    }
    """
)



# ── Ensure project root is on sys.path ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config                               # noqa: E402
from src.camera.usb_microscope import USBMicroscope, discover_cameras  # noqa: E402
from src.camera.monitor import IntervalMonitor, MonitorSession         # noqa: E402
from src.camera.camera_manager import CameraManager, CameraDevice      # noqa: E402
from src.camera.recording_manager import RecordingManager, RecordingState  # noqa: E402
from src.camera.analysis_progress import ProgressTracker, StageStatus  # noqa: E402
from src.analysis.trend_analysis import TrendAnalyser, TrendReport    # noqa: E402
from src.reporting.monitoring_report import MonitoringReportGenerator  # noqa: E402
from src.database.db_manager import DatabaseManager                    # noqa: E402
from pages.database_page import render_database_page               # noqa: E402
from loguru import logger
# ══════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Everse.ai Semen Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Everse.ai — Automated Semen Viability and Counting Analysis**\n\n"
            "AI-powered automated andrology analysis.\n\n"
            "⚠️ For research and screening use only.  "
            "Results must be confirmed by a certified andrologist."
        )
    },
)

# ══════════════════════════════════════════════════════════════
# Design tokens
# ══════════════════════════════════════════════════════════════

_CLR_PRIMARY   = "#0e4b7a"
_CLR_SECONDARY = "#0a2540"
_CLR_ACCENT    = "#0d9488"
_CLR_SUCCESS   = "#10b981"
_CLR_WARNING   = "#f59e0b"
_CLR_DANGER    = "#ef4444"
_CLR_BG        = "#f1f5f9"
_CLR_CARD      = "#ffffff"
_CLR_BORDER    = "#e2e8f0"
_CLR_TEXT_PRI  = "#0f172a"
_CLR_TEXT_SEC  = "#64748b"
_CLR_SIDEBAR   = "#0a2540"
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

/* ── Text colour safety ─────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{ color:{_CLR_TEXT_PRI} !important; }}
.stTable table,.stTable td,.stTable th {{ color:{_CLR_TEXT_PRI} !important; }}
[data-testid="stMetricValue"] {{ color:{_CLR_SECONDARY} !important; }}
[data-testid="stMetricLabel"] {{ color:{_CLR_TEXT_SEC}  !important; }}

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
    ("Initialising pipeline…",            5),
    ("Running detection…",               20),
    ("Cell tracking…",                    38),
    ("Computing motility metrics…",      55),
    ("Classifying morphology…",          70),
    ("Predicting viability…",            82),
    ("Calculating WHO quality score…",   90),
    ("Generating PDF / CSV / JSON…",     96),
    ("Analysis complete!",              100),
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
            '<div class="lab-logo-text">Everse.ai</div>'
            '<div class="lab-logo-sub">Semen Analyzer</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    # Reference card (no internal model names exposed)
    st.sidebar.markdown(
        '<span style="font-size:0.75rem;font-weight:700;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#475569;">'
        'Reference Values</span>',
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
        'Laboratory Reference Values</span>' + tbl + "</table>",
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
        for k, v in [
            ("Sample ID", ss.get("_sample_id","—")),
            ("Animal ID", ss.get("_animal_id","—")),
            ("Species",   ss.get("_species","—")),
            ("Breed",     ss.get("_breed","—")),
            ("Sample",    ss.get("_sample_type","—")),
            ("Staining",  ss.get("_staining","—")),
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
    current_page = st.session_state.get("_db_page", "analysis")
    btn_analysis = st.sidebar.button(
        "Analysis",
        use_container_width=True,
        key="_nav_analysis",
        type="primary" if current_page == "analysis" else "secondary",
    )
    btn_calibration = st.sidebar.button(
        "Calibration",
        use_container_width=True,
        key="_nav_calibration",
        type="primary" if current_page == "calibration" else "secondary",
    )
    btn_database = st.sidebar.button(
        "Sessions",
        use_container_width=True,
        key="_nav_database",
        type="primary" if current_page == "database" else "secondary",
    )
    if btn_analysis:
        st.session_state["_db_page"] = "analysis"
        st.rerun()
    if btn_calibration:
        st.session_state["_db_page"] = "calibration"
        st.rerun()
    if btn_database:
        st.session_state["_db_page"] = "database"
        st.rerun()

    st.sidebar.caption(
        "⚠️ Research & screening use only.  "
        "Confirm results with a certified andrologist."
    )


# ══════════════════════════════════════════════════════════════
# §1 – §4  Intake form sections  (identical to Phase 1 & 2)
# ══════════════════════════════════════════════════════════════


def _render_section_1() -> Dict[str, Any]:
    _card_open("", "Sample Information")
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
    _card_open("", "Sample")
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
    _card_open("", "Laboratory Configuration")
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


def _render_section_4() -> Dict[str, Any]:
    _card_open("", "Preparation")
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
    _card_close()
    return {"staining_type": staining}


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
                     caption="First frame preview", use_container_width=True)
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


@st.fragment(run_every=0.25)
def _render_live_microscope_feed():
    """Display the latest microscope frame without rerunning the whole app."""
    mgr = _get_camera_manager()

    if not mgr.is_connected:
        st.caption("Microscope is not connected.")
        return

    frame = mgr.get_latest_frame()

    if frame is None:
        st.caption("Waiting for image from microscope…")
        return

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(
            frame_rgb,
            use_container_width=True
        )
    except cv2.error as exc:
        st.error(f"Unable to display microscope image: {exc}")


# ──────────────────────────────────────────────────────────────
# Live USB Microscope UI  (Phase 6 — real hardware detection)
# ──────────────────────────────────────────────────────────────

def _render_live_ui() -> Optional[Path]:
    """
    Live microscope panel using the real CameraManager.

    Uses detect_cameras() which returns ONLY external cameras (is_external=True),
    so the built-in webcam is never shown to the user.
    No debug st.write() calls. No commented-out code.
    """
    ss  = st.session_state
    mgr = _get_camera_manager()

    connected = ss.get("_cam_connected",  False)
    streaming = ss.get("_cam_streaming",  False)
    captured  = ss.get("_cam_captured",   False)
    clip_str  = ss.get("_camera_clip_path", "")

    # ── Device selector (only when disconnected) ───────────────
    if not connected:
        if ss.get("_available_devices") is None:
            with st.spinner("Scanning for microscope…"):
                devices: List[CameraDevice] = mgr.enumerate_devices()
                ss["_available_devices"] = devices

        devices: List[CameraDevice] = ss.get("_available_devices") or []

        if not devices:
            st.warning(
                "No USB microscope detected.  "
                "Connect your microscope and click **Refresh**."
            )
            if st.button("Refresh", key="_btn_refresh_cams",
                         use_container_width=True):
                ss["_available_devices"] = None
                st.rerun()
            return None

        labels = [d.display_name for d in devices]
        sel_label = st.selectbox(
            "Select Microscope",
            options=labels,
            index=0,
            key="_cam_label_select",
            label_visibility="visible",
        )
        sel_dev = next(d for d in devices if d.display_name == sel_label)
        ss["_cam_device_idx"] = sel_dev.index
        ss["_cam_device_obj"] = sel_dev

        if st.button("Refresh", key="_btn_refresh_cams",
                     use_container_width=False):
            ss["_available_devices"] = None
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Six control buttons ────────────────────────────────────
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    with r1c1:
        if st.button("Connect", use_container_width=True,
                     disabled=connected, key="_btn_connect"):
            avail_devs = ss.get("_available_devices") or []
            sel_dev    = ss.get("_cam_device_obj")
            if sel_dev is None:
                idx = ss.get("_cam_device_idx")
                sel_dev = next((d for d in avail_devs if d.index == idx), None)

            if sel_dev is None:
                st.error("No microscope selected.  Refresh the list.")
            else:
                with st.spinner(f"Connecting to {sel_dev.display_name}…"):
                    ok = mgr.connect(sel_dev)
                if ok:
                    ss["_cam_connected"]    = True
                    ss["_cam_streaming"]    = False
                    ss["_cam_captured"]     = False
                    ss["_camera_clip_path"] = ""
                    ss["_cam_resolution"]   = ss.get("_inp_camera_resolution", "1280×720")
                    ss["_cam_fps"]          = ss.get("_inp_camera_fps", "30 FPS")
                    st.rerun()
                else:
                    st.error(
                        "Could not connect to the microscope.  "
                        "Check the USB connection and try again."
                    )

    with r1c2:
        if st.button("Disconnect", use_container_width=True,
                     disabled=not connected, key="_btn_disconnect"):
            mgr.disconnect()
            ss["_cam_connected"] = False
            ss["_cam_streaming"] = False
            st.rerun()

    with r1c3:
        if st.button("Start Preview", use_container_width=True,
                     disabled=(not connected or streaming), key="_btn_start"):
            ss["_cam_streaming"]    = True
            ss["_cam_captured"]     = False
            ss["_camera_clip_path"] = ""
            st.rerun()

    with r2c1:
        if st.button("Stop Preview", use_container_width=True,
                     disabled=not streaming, key="_btn_stop"):
            ss["_cam_streaming"] = False
            st.rerun()

    with r2c2:
        if st.button("Capture Sample", use_container_width=True,
                     disabled=not streaming, type="primary", key="_btn_capture"):
            _do_capture_phase6()
            st.rerun()

    with r2c3:
        if st.button("Retake", use_container_width=True,
                     disabled=not captured, key="_btn_retake"):
            old_clip = ss.get("_camera_clip_path", "")
            if old_clip:
                Path(old_clip).unlink(missing_ok=True)
            ss["_cam_captured"]     = False
            ss["_cam_streaming"]    = True
            ss["_camera_clip_path"] = ""
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status text ────────────────────────────────────────────
    if not connected:
        st.caption("Connect your microscope to begin.")
    elif streaming:
        _render_camera_status_panel()
        _render_live_microscope_feed()
    elif captured and clip_str:
        st.success("Sample recorded — ready for analysis.")
    elif connected:
        _render_camera_status_panel()
        st.caption("Click **Start Preview** to see the live feed.")

    # Recording panel
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
                st.markdown("**Captured Sample**")
                st.table({"Property": ["Source","Duration","Frames","FPS","Size"],
                          "Value": ["Microscope", f"{_RECORD_DURATION_S:.0f}s",
                                    str(n_fr), f"{fpv:.1f}", f"{sz:.1f} MB"]})
            ss["_video_stem"] = f"capture_{ss.get('_sample_id','unknown')}"
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
    _card_open("", "Input Source")
    ss = st.session_state
    c1, c2 = st.columns(2)
    with c1:
        upload_btn = st.button(
            "⬆  Upload Video File", use_container_width=True, key="_src_upload_btn",
            type="primary" if ss.get("_input_source","upload") == "upload" else "secondary",
        )
    with c2:
        live_btn = st.button(
            "📡  Live USB Microscope", use_container_width=True, key="_src_live_btn",
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

    st.markdown("<br>", unsafe_allow_html=True)
    tmp_path = _render_upload_ui() if ss["_input_source"] == "upload" else _render_live_ui()
    _card_close()
    return tmp_path        

# def _render_section_5() -> Optional[Path]:
#     _card_open("", "Input Source")
#     ss = st.session_state
#     c1, c2 = st.columns(2)
#     with c1:
#         upload_btn = st.button(
#             "⬆  Upload Video File", use_container_width=True, key="_src_upload_btn",
#             type="primary" if ss.get("_input_source","upload") == "upload" else "secondary",
#         )
#     with c2:
#         live_btn = st.button(
#             "📡  Live USB Microscope", use_container_width=True, key="_src_live_btn",
#             type="primary" if ss.get("_input_source") == "live" else "secondary",
#         )
#     if upload_btn:
#         mgr = ss.get("camera_manager")

#         if mgr is not None and ss.get("cam_connected"):
#             try:
#                mgr.disconnect()
#             except Exception:
#                 pass
#         ss["_cam_connected"] = False
#         ss["_cam_streaming"] = False
#         ss["_cam_captured"] = False
#         ss["_input_source"] = "upload"

#     st.markdown("<br>", unsafe_allow_html=True)
#     tmp_path = _render_upload_ui() if ss["_input_source"] == "upload" else _render_live_ui()
#     _card_close()
#     return tmp_path


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
    _card_open("", "Monitoring")

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

    # ── Auto-refresh while running ────────────────────────────
    if session.is_running and not session.is_complete:
        try:
            from streamlit_autorefresh import st_autorefresh   # type: ignore[import]
            st_autorefresh(interval=_MONITOR_TICK_MS, key="_monitor_refresh")
        except ImportError:
            st.warning(
                "⚠️  `streamlit-autorefresh` not installed — "
                "countdown will not update automatically.  "
                "`pip install streamlit-autorefresh`"
            )
        monitor.tick()

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
                monitor.stop()
                st.rerun()
        else:
            if st.button("New Analysis", use_container_width=True,
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
                    _render_results(
                        latest.analysis_results,
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

    # ── Generate and cache monitoring report ───────────────────
    _section_header("📥 Monitoring Report")

    report_paths: Optional[Dict[str, str]] = ss.get("_trend_report_paths")

    if report_paths is None:
        if st.button("📄  Generate Full Monitoring Report (PDF + CSV + JSON)",
                     type="primary", use_container_width=True,
                     key="_btn_gen_report"):
            with st.spinner("Generating PDF report — this may take 15–30 seconds…"):
                try:
                    gen          = MonitoringReportGenerator(config)
                    sample_meta  = ss.get("_analysis_meta") or {}
                    sid          = sample_meta.get("sample_id", "session")
                    stem         = f"monitor_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    report_paths = gen.generate(trend_report, output_stem=stem)
                    ss["_trend_report_paths"] = report_paths
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌  Report generation failed: {exc}")
    else:
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


def _render_optional_plot(plot_path: Path, caption: str) -> None:
    if plot_path.exists():
        st.image(str(plot_path), caption=caption, use_container_width=True)


def _render_who_flags(flags: List[str]) -> None:
    if not flags:
        st.success("✅  All parameters are within reference limits.")
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

    _section_header("Summary")
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
        _section_header("Quality Score")
        _render_who_score_banner(score, category)
    with right_col:
        _section_header("Parameter Scores")
        if subscores:
            _render_subscore_bars(subscores)
        else:
            st.caption("Sub-scores unavailable.")

    st.divider()
    tab_mot, tab_morph, tab_via = st.tabs(
        ["Motility", "Morphology", "Vitality"]
    )
    with tab_mot:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Motility Metrics**")
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
            _render_optional_plot(plot_dir / "motility_analysis.png",
                                  "Motility Distribution")
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
                                  "Morphology Confusion Matrix")
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
        _section_header("Cell Trajectories")
        st.image(str(traj_path), use_container_width=True)
        st.divider()

    _section_header("Reference Limits")
    _render_who_flags(flags)
    st.divider()
    if recs:
        _section_header("Recommendations")
        _render_recommendations(recs)
        st.divider()
    _section_header("📥 Download Reports")
    _render_download_buttons(rp)


# ══════════════════════════════════════════════════════════════
# Pipeline runner  (identical to Phase 2 — UNCHANGED)
# ══════════════════════════════════════════════════════════════


def _run_pipeline_with_progress(pipeline: Any,
                                video_path: Path) -> Dict[str, Any]:
    """
    Phase 6 — Professional analysis progress dashboard.

    Runs the pipeline in a ProgressTracker background thread and
    renders the live stage dashboard + system metrics panel on every
    Streamlit rerun (driven by streamlit-autorefresh at 2 s).

    Falls back to a simple spinner if streamlit-autorefresh is not
    installed.
    """
    import cv2 as _cv2

    # Read total frames so progress % is accurate
    cap_ = _cv2.VideoCapture(str(video_path))
    total_frames = int(cap_.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap_.release()

    tracker = ProgressTracker()
    tracker.start(pipeline, video_path, total_frames=total_frames)
    st.session_state["_progress_tracker"] = tracker

    progress_placeholder = st.empty()

    # Render the progress dashboard on every rerun
    while tracker.is_running:
        snap = tracker.get_snapshot()

        with progress_placeholder.container():
            _render_analysis_dashboard(snap)

        time.sleep(0.5)

        snap = tracker.get_snapshot()
        if not snap.is_running:
            break

    # Final render into the SAME placeholder
    snap = tracker.get_snapshot()

    with progress_placeholder.container():
        _render_analysis_dashboard(snap)

    if tracker.error:
        raise RuntimeError(tracker.error)

    return tracker.result or {}


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
    st.progress(
        min(1.0, snap.overall_pct / 100.0),
        text=f"Overall Progress: {snap.overall_pct:.1f}%",
    )

    st.markdown("<br>", unsafe_allow_html=True)


    
    _section_header("Stages")

    status_icons = {
            StageStatus.COMPLETE: "✓",
            StageStatus.RUNNING:  "▶",
            StageStatus.WAITING:  "⏳",
            StageStatus.FAILED:   "✕",
        }
    status_colours = {
        StageStatus.COMPLETE: _CLR_SUCCESS,
        StageStatus.RUNNING:  _CLR_PRIMARY,
        StageStatus.WAITING:  _CLR_TEXT_SEC,
        StageStatus.FAILED:   _CLR_DANGER,
    }
    for stage in snap.stages:
        icon   = status_icons.get(stage.status, "⏳")
        colour = status_colours.get(stage.status, _CLR_TEXT_SEC)
        detail = f"  — {stage.detail}" if stage.detail else ""
        st.markdown(
                f'''<div style="display:flex;align-items:center;gap:0.5rem;
padding:3px 0;font-size:0.875rem;">
<span style="font-size:0.9rem;">{icon}</span>
<span style="font-weight:{"700" if stage.status==StageStatus.RUNNING else "400"};
color:{colour};">{stage.name}</span>
<span style="font-size:0.75rem;color:{_CLR_TEXT_SEC};">{detail}</span>
</div>''',
                unsafe_allow_html=True,
            )
     

# ══════════════════════════════════════════════════════════════
# Optical Calibration Page (Phase 7)
# ══════════════════════════════════════════════════════════════

def _render_calibration_page() -> None:
    """
    Graphical optical calibration using a stage micrometer.

    The user:
      1. Connects a microscope.
      2. Captures a frame containing the stage micrometer.
      3. Clicks two graduation lines on the displayed image.
      4. The software calculates µm/pixel automatically.
      5. Repeated measurements build a repeatability assessment.
      6. User validates and saves the calibration.

    Coordinates come from st_canvas click events on the ORIGINAL
    camera frame — never from browser / CSS coordinates.
    """
    ss = st.session_state

    st.markdown(
        '<div class="page-title">Optical Calibration</div>'
        '<div class="page-subtitle">'
        "Set the pixel-to-micron scale for your current microscope configuration."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Configuration card ──────────────────────────────────────
    with st.expander("Microscope Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            cal_scope = st.selectbox(
                "Microscope", _MICROSCOPE_TYPES, index=0,
                key="_cal_scope",
            )
            cal_mag = st.selectbox(
                "Objective", _MAGNIFICATIONS, index=2,
                key="_cal_mag",
            )
        with c2:
            cal_res = st.selectbox(
                "Camera Resolution", _CAMERA_RESOLUTIONS, index=1,
                key="_cal_res",
            )
            cal_zoom = st.selectbox(
                "Digital Zoom", ["None", "1.5×", "2×", "4×"],
                index=0, key="_cal_zoom",
            )
        with c3:
            st.markdown("**Stage Micrometer**")
            cal_device = st.selectbox(
                "Device", ["Erma Tokyo SM-01", "Custom"],
                index=0, key="_cal_device",
            )
            if cal_device == "Erma Tokyo SM-01":
                division_um = 100
                st.caption("Division: 100 µm · Total: 10 mm")
            else:
                division_um = st.number_input(
                    "Division (µm)", min_value=1, max_value=10000,
                    value=100, key="_cal_div_um",
                )

    # ── Capture / upload micrometer image ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Step 1 — Obtain a Micrometer Image**")

    col_cap, col_up = st.columns(2)
    with col_cap:
        if st.button("Capture from Microscope", key="_cal_btn_capture"):
            mgr = ss.get("_camera_manager")
            if mgr is None or not mgr.is_connected:
                st.error(
                    "Microscope not connected.  "
                    "Go to Analysis → connect your microscope, then return here."
                )
            else:
                frame = mgr.get_latest_frame()
                if frame is not None:
                    ss["_cal_frame_bgr"]  = frame
                    ss["_cal_frame_h"]    = frame.shape[0]
                    ss["_cal_frame_w"]    = frame.shape[1]
                    ss["_cal_pts"]        = []
                    ss["_cal_meas_list"]  = ss.get("_cal_meas_list") or []
                    st.rerun()
                else:
                    st.error("No frame received from microscope.")
    with col_up:
        up = st.file_uploader(
            "Or upload an image", type=["png","jpg","jpeg","bmp","tif","tiff"],
            key="_cal_img_upload", label_visibility="collapsed",
        )
        if up is not None:
            import io
            from PIL import Image as PILImage
            pil = PILImage.open(io.BytesIO(up.read())).convert("RGB")
            frame_np = np.array(pil)
            ss["_cal_frame_bgr"] = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            ss["_cal_frame_h"]   = frame_np.shape[0]
            ss["_cal_frame_w"]   = frame_np.shape[1]
            ss["_cal_pts"]       = []
            ss["_cal_meas_list"] = ss.get("_cal_meas_list") or []
            st.rerun()

    # ── Point selection ─────────────────────────────────────────
    frame_bgr = ss.get("_cal_frame_bgr")
    if frame_bgr is None:
        st.info(
            "Capture or upload a stage-micrometer image above to begin."
        )
        return

    orig_h: int = ss.get("_cal_frame_h", 1)
    orig_w: int = ss.get("_cal_frame_w", 1)
    pts: List    = ss.get("_cal_pts") or []

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "**Step 2 — Select Two Graduation Lines**  \n"
        "Click the first line, then the second line on the image below.  "
        "The software calculates the pixel distance automatically."
    )

    # Display image with annotation overlay
    display_frame = frame_bgr.copy()
    if len(pts) >= 1:
        cv2.circle(display_frame, (pts[0][0], pts[0][1]), 8, (0, 220, 80), -1)
        cv2.putText(display_frame, "A", (pts[0][0]+10, pts[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,80), 2)
    if len(pts) >= 2:
        cv2.circle(display_frame, (pts[1][0], pts[1][1]), 8, (220, 80, 0), -1)
        cv2.putText(display_frame, "B", (pts[1][0]+10, pts[1][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,80,0), 2)
        cv2.line(display_frame, (pts[0][0],pts[0][1]),
                 (pts[1][0],pts[1][1]), (255,255,0), 2)

    # Try streamlit-drawable-canvas for click detection
    # ------------------------------------------------------------------
    # Step 2 — Click two graduation lines directly on the image
    # ------------------------------------------------------------------
  # ------------------------------------------------------------------
# Step 2 — Click Two Graduation Lines
# ------------------------------------------------------------------

    st.markdown(
        "**Click the first graduation line, then click the second "
        "graduation line on the image below.**"
    )

    DISPLAY_W = 800

    # Convert calibration image to JPEG/base64.
    ok, encoded_image = cv2.imencode(
        ".jpg",
        display_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )

    if not ok:
        st.error("Unable to prepare the calibration image.")
        return

    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    # Get currently selected points.
    pts = ss.get("_cal_pts") or []

    pts = [
        (int(p[0]), int(p[1]))
        for p in pts[:2]
    ]

    # Mount clickable image.
    click_result = _CAL_IMAGE_CLICKER(
        data={
            "image_base64": image_base64,
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "display_width": int(DISPLAY_W),
            "points": pts,
        },
        key="_cal_stage_micrometer_clicker",
        on_point_clicked_change=lambda: None,
    )

    # ---------------------------------------------------------------
    # Receive click from JavaScript
    # ---------------------------------------------------------------

    clicked_point = getattr(
        click_result,
        "point_clicked",
        None,
    )

    if clicked_point is not None:

        try:
            clicked_x = int(clicked_point["x"])
            clicked_y = int(clicked_point["y"])

            # First click = Line A
            if len(pts) == 0:

                ss["_cal_pts"] = [
                    (clicked_x, clicked_y)
                ]

            # Second click = Line B
            elif len(pts) == 1:

                ss["_cal_pts"] = [
                    pts[0],
                    (clicked_x, clicked_y)
                ]

            # Third click starts a new measurement
            else:

                ss["_cal_pts"] = [
                    (clicked_x, clicked_y)
                ]

            st.rerun()

        except (
            TypeError,
            KeyError,
            ValueError,
        ):
            pass

    # Refresh points after click.
    pts = ss.get("_cal_pts") or []

    # ---------------------------------------------------------------
    # Display selected-point information
    # ---------------------------------------------------------------

    if len(pts) == 1:

        st.info(
            f"Line A selected at "
            f"X = {pts[0][0]} px, "
            f"Y = {pts[0][1]} px. "
            f"Now click Line B."
        )

    elif len(pts) == 2:

        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]

        pixel_dist = float(
            np.sqrt(
                dx * dx +
                dy * dy
            )
        )

        st.success(
            f"Line A: ({pts[0][0]}, {pts[0][1]}) px  |  "
            f"Line B: ({pts[1][0]}, {pts[1][1]}) px  |  "
            f"Pixel distance: {pixel_dist:.2f} px"
        )

        if st.button(
            "Reset Selected Points",
            key="_cal_reset_clicked_points",
        ):
            ss["_cal_pts"] = []
            st.rerun()
    # ------------------------------------------------------------------
    # Show the currently selected points and line
    # ------------------------------------------------------------------
    if len(pts) == 2:
        preview = display_frame.copy()

        # Scale original coordinates to displayed image coordinates.
        DISPLAY_W = 800
        scale = DISPLAY_W / float(orig_w)
        display_h = int(orig_h * scale)

        p1_display = (
            int(pts[0][0] * scale),
            int(pts[0][1] * scale),
        )

        p2_display = (
            int(pts[1][0] * scale),
            int(pts[1][1] * scale),
        )

        cv2.circle(
            preview,
            p1_display,
            8,
            (0, 220, 80),
            -1,
        )

        cv2.circle(
            preview,
            p2_display,
            8,
            (0, 220, 80),
            -1,
        )

        cv2.line(
            preview,
            p1_display,
            p2_display,
            (0, 220, 80),
            3,
        )

        cv2.putText(
            preview,
            "A",
            (
                p1_display[0] + 10,
                p1_display[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview,
            "B",
            (
                p2_display[0] + 10,
                p2_display[1] - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 80),
            2,
            cv2.LINE_AA,
        )

        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption="Selected calibration points",
            width=DISPLAY_W,
        )

        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]

        pixel_dist_preview = float(
            np.sqrt(dx * dx + dy * dy)
        )

        st.caption(
            f"Selected pixel distance: "
            f"{pixel_dist_preview:.2f} px"
        )

    # ── Point controls ──────────────────────────────────────────
    btn_clear, btn_measure = st.columns(2)
    with btn_clear:
        if st.button("Clear Points", key="_cal_clear"):
            ss["_cal_pts"] = []
            st.rerun()

    # ── Measurement ─────────────────────────────────────────────
    if len(pts) == 2:
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        pixel_dist = float(np.sqrt(dx*dx + dy*dy))

        # Interval selection
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Step 3 — Set Number of Intervals**")
        n_intervals = st.number_input(
            "Number of divisions between selected lines",
            min_value=1, max_value=200, value=6,
            help=f"Each division = {division_um} µm",
            key="_cal_n_intervals",
        )
        ref_dist_um = n_intervals * division_um

        st.markdown(
            f"""
| Measurement | Value |
|---|---|
| Point A | ({pts[0][0]}, {pts[0][1]}) px |
| Point B | ({pts[1][0]}, {pts[1][1]}) px |
| Selected intervals | {n_intervals} |
| Reference distance | {ref_dist_um} µm |
| Measured pixel distance | {pixel_dist:.2f} px |
| **Calculated scale** | **{ref_dist_um / pixel_dist:.4f} µm/pixel** |
"""
        )

        with btn_measure:
            if st.button("Add Measurement", type="primary", key="_cal_add_meas"):
                meas = {
                    "n_intervals":  int(n_intervals),
                    "ref_dist_um":  float(ref_dist_um),
                    "pixel_dist":   round(pixel_dist, 4),
                    "um_per_pixel": round(ref_dist_um / pixel_dist, 6),
                    "pt_a":         pts[0],
                    "pt_b":         pts[1],
                    "frame_w":      orig_w,
                    "frame_h":      orig_h,
                }
                meas_list = ss.get("_cal_meas_list") or []
                meas_list.append(meas)
                ss["_cal_meas_list"] = meas_list
                ss["_cal_pts"] = []
                st.toast(f"Measurement added: {meas['um_per_pixel']:.4f} µm/px")
                st.rerun()

    # ── Repeatability table ─────────────────────────────────────
    meas_list = ss.get("_cal_meas_list") or []
    if meas_list:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Step 4 — Repeatability**")

        import pandas as pd
        df_rows = [
            {
                "#": i + 1,
                "Intervals": m["n_intervals"],
                "Ref (µm)":  m["ref_dist_um"],
                "Pixels":    m["pixel_dist"],
                "µm/pixel":  m["um_per_pixel"],
            }
            for i, m in enumerate(meas_list)
        ]
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True,
                     hide_index=True)

        vals = [m["um_per_pixel"] for m in meas_list]
        mean_val = float(np.mean(vals))
        std_val  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv_val   = (std_val / mean_val * 100) if mean_val > 0 else 0.0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Mean",  f"{mean_val:.4f} µm/px")
        r2.metric("SD",    f"{std_val:.4f}")
        r3.metric("CV",    f"{cv_val:.2f}%")
        r4.metric("N",     str(len(vals)))

        # Validation threshold
        if cv_val <= 5.0 and len(vals) >= 2:
            st.success(
                f"Repeatability is acceptable (CV = {cv_val:.2f}%).  "
                "You may validate and save this calibration."
            )
            cal_valid = True
        elif len(vals) < 2:
            st.info("Add at least one more measurement to assess repeatability.")
            cal_valid = False
        else:
            st.warning(
                f"Repeatability CV = {cv_val:.2f}% (>5%).  "
                "Consider repeating measurements on clearer graduation lines."
            )
            cal_valid = False

        # Clear all measurements
        if st.button("Clear All Measurements", key="_cal_clear_all"):
            ss["_cal_meas_list"] = []
            st.rerun()

        # Save calibration
        if cal_valid:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Step 5 — Validate & Save**")
            if st.button("Validate & Save Calibration", type="primary",
                         key="_cal_save"):
                cal_record = {
                    "status":           "VALIDATED",
                    "um_per_pixel":     round(mean_val, 6),
                    "mean":             round(mean_val, 6),
                    "std":              round(std_val, 6),
                    "cv_pct":           round(cv_val, 4),
                    "n_measurements":   len(vals),
                    "microscope":       ss.get("_cal_scope", ""),
                    "magnification":    ss.get("_cal_mag", ""),
                    "resolution":       ss.get("_cal_res", ""),
                    "digital_zoom":     ss.get("_cal_zoom", "None"),
                    "stage_device":     ss.get("_cal_device", ""),
                    "division_um":      division_um,
                    "measurements":     meas_list,
                    "saved_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                ss["_current_calibration"] = cal_record
                # Persist to database if available
                try:
                    db = _load_db(id(_load_config()))
                    if hasattr(db, "save_calibration"):
                        db.save_calibration(cal_record)
                except Exception:
                    pass
                st.success(f"Calibration saved: {mean_val:.4f} \u00b5m/pixel")
                st.balloons()

    # ── Active calibration summary ──────────────────────────────
    cal = ss.get("_current_calibration")
    if cal:
        st.divider()
        st.markdown("**Active Calibration**")
        cal_text = (
            f"{cal['um_per_pixel']:.4f} µm/pixel | "
            f"{cal['microscope']} · {cal['magnification']} · {cal['resolution']} | "
            f"Validated {cal['saved_at']} | "
            f"N={cal['n_measurements']}  CV={cal['cv_pct']:.2f}%  Status: {cal['status']}"
        )
        st.info(cal_text)


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
    # Monitoring (Phase 3)
    "_monitor_enabled":     False,
    "_monitor_session":     None,    # MonitorSession | None
    "_monitor_obj":         None,    # IntervalMonitor | None
    # Trend analysis (Phase 4)
    "_trend_report":        None,    # TrendReport | None
    "_trend_report_paths":  None,    # dict {"pdf","csv","json"} | None
    # Database (Phase 5)
    "_db_page":             "analysis",  # "analysis" | "database" | "calibration"
    "_db_view_session_id":  None,
    "_db_compare_set":      set(),
    "_db_confirm_delete":   None,
    # Calibration (Phase 7)
    "_current_calibration": None,
    "_cal_frame_bgr":       None,
    "_cal_frame_h":         1,
    "_cal_frame_w":         1,
    "_cal_pts":             [],
    "_cal_meas_list":       [],
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
        "_available_devices","_progress_tracker",
    ]
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
            '<div class="page-title">Everse.ai Semen Analyzer</div>'
            '<div class="page-subtitle">'
            "Automated semen analysis and laboratory insights"
            "</div>",
            unsafe_allow_html=True,
        )
    with badge_col:
        st.markdown(
            f'<div style="margin-top:0.6rem;text-align:right;">'
            f'<span style="background:{_CLR_PRIMARY};color:white;'
            f'border-radius:6px;padding:3px 10px;font-size:0.72rem;'
            f'font-weight:700;letter-spacing:0.05em;">everse.ai</span>'
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

    if ss.get("_db_page") == "calibration":
        _render_calibration_page()
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
            if st.button("New Analysis", use_container_width=True,
                         key="_btn_new_from_c",
                         help="End monitoring and start fresh"):
                if ss.get("_monitor_obj") is not None:
                    ss["_monitor_obj"].stop()
                _clear_analysis_state()
                st.rerun()

        st.divider()
        _render_monitoring_dashboard(config)

        # Auto-save completed monitoring session
        if (mon_session.is_complete or not mon_session.is_running) \
                and mon_session.captures_done >= 1 \
                and not ss.get("_db_session_saved"):
            try:
                db.save_monitoring_session(
                    meta                    = ss.get("_analysis_meta") or {},
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
            if st.button("New Analysis", use_container_width=True,
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
    prep_cfg    = _render_section_4()
    tmp_path    = _render_section_5()
    monitor_cfg = _render_section_6()

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
            "Monitoring requires Live USB Microscope input (§5).  "
            "Switch to 'Live USB Microscope' or disable monitoring."
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
        mon_session = MonitorSession(
            interval_min    = monitor_cfg["interval_min"],
            duration_min    = monitor_cfg["duration_min"],
            sample_id       = sample_info["sample_id"] or "unknown",
            device_index    = int(ss.get("_cam_device_idx", 0)),
            resolution      = ss.get("_inp_camera_resolution", "1280×720"),
            fps             = ss.get("_inp_camera_fps", "30 FPS"),
            clip_duration_s = _RECORD_DURATION_S,
        )
        monitor = IntervalMonitor(mon_session, pipeline)
        ss["_monitor_session"]  = mon_session
        ss["_monitor_obj"]      = monitor
        ss["_analysis_meta"]    = analysis_meta
        ss["_db_session_saved"] = False

        st.info(
            f"⏱  Starting monitoring session — first capture beginning now…\n"
            f"Interval: **{monitor_cfg['interval_min']} min** · "
            f"Duration: **{monitor_cfg['duration_min']} min** · "
            f"Planned captures: **{mon_session.expected_captures}**"
        )
        with st.spinner(f"Recording first sample clip ({_RECORD_DURATION_S:.0f}s)…"):
            monitor.start()
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════
    # PATH S — single analysis
    # ══════════════════════════════════════════════════════════
    t0: float = time.perf_counter()
    results: Optional[Dict[str, Any]] = None

    try:
        results = _run_pipeline_with_progress(pipeline, tmp_path)
    except FileNotFoundError as exc:
        st.error(f"❌  Video file error: {exc}")
    except ValueError as exc:
        st.error(f"❌  Validation error: {exc}")
    except Exception as exc:
        st.error(f"❌  Unexpected pipeline error: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if results is None:
        st.warning("Analysis returned no results.  Check logs for details.")
        return

    elapsed = time.perf_counter() - t0

    # Store BEFORE any st.* call
    ss["_analysis_results"]    = results
    ss["_analysis_video_stem"] = ss["_video_stem"]
    ss["_analysis_elapsed"]    = elapsed
    ss["_analysis_meta"]       = analysis_meta

    # Auto-save to database
    try:
        db.save_single_session(
            meta            = analysis_meta,
            results         = results,
            report_paths    = _extract_report_paths(results),
            annotated_video = str(results.get("annotated_video_path", "")),
        )
        st.toast("✅ Session saved to database", icon="🗄️")
    except Exception as exc:
        logger.warning("DB auto-save (single) failed: {}", exc)

    st.rerun()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()