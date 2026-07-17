"""
app/app.py
===========
Automated Semen Viability and Counting Analysis — Streamlit Dashboard

Professional medical-grade web interface that:
  1. Accepts a semen microscope video upload.
  2. Calls the existing backend pipeline exactly once via
     ``AutomatedSemenAnalysis.run_video``.
  3. Renders all results — counts, motility, morphology, viability,
     WHO quality score, flags, recommendations, and output plots.
  4. Provides one-click download buttons for PDF / CSV / JSON reports.

This file contains NO AI logic.  All computation is delegated to the
existing backend modules in ``src/``.

Run
---
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import streamlit as st

# ── Ensure project root is importable regardless of CWD ───────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Project utilities ──────────────────────────────────────────
from src.utils.helpers import load_config

# ══════════════════════════════════════════════════════════════
# Page configuration  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Semen Analysis AI · Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Automated Semen Viability and Counting Analysis**\n\n"
            "AI-powered andrology pipeline using YOLOv8, ByteTrack, "
            "EfficientNetB0, and Random Forest.\n\n"
            "For research and screening use only."
        )
    },
)

# ══════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════

st.markdown(
"""
<style>
/* ── Global ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"]          { background: #1a2332; }
[data-testid="stSidebar"] *        { color: #e2e8f0 !important; }

/* ── Headings ────────────────────────────────────── */
.page-title {
    font-size: 2.1rem; font-weight: 800;
    color: #1a2332; letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 1rem; color: #64748b; margin-bottom: 1.5rem;
}

/* ── Cards ───────────────────────────────────────── */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    margin-bottom: 0.8rem;
    border-left: 4px solid #3b82f6;
}
.metric-card-value {
    font-size: 2rem; font-weight: 700; color: #1e293b;
    line-height: 1;
}
.metric-card-label {
    font-size: 0.78rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* ── WHO Score banner ────────────────────────────── */
.who-score-box {
    border-radius: 16px;
    padding: 1.6rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}
.who-score-number { font-size: 4rem; font-weight: 800; line-height: 1; }
.who-score-label  { font-size: 1.3rem; font-weight: 600; margin-top: 0.4rem; }
.who-score-sub    { font-size: 0.85rem; opacity: 0.85; margin-top: 0.2rem; }

/* Category colours */
.cat-excellent { background: linear-gradient(135deg, #10b981, #059669); }
.cat-good      { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.cat-average   { background: linear-gradient(135deg, #f59e0b, #d97706); }
.cat-poor      { background: linear-gradient(135deg, #ef4444, #dc2626); }

/* ── Alert boxes ─────────────────────────────────── */
.flag-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.9rem;
    color: #991b1b;
}
.rec-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin: 0.35rem 0;
    font-size: 0.9rem;
    color: #166534;
}

/* ── Section divider ─────────────────────────────── */
.section-header {
    font-size: 1.05rem; font-weight: 700;
    color: #1e293b; margin: 1.2rem 0 0.5rem 0;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.3rem;
}

/* ── Upload zone ─────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: white !important;
    border-radius: 12px !important;
    border: 2px dashed #93c5fd !important;
    padding: 1rem !important;
}

/* ── Run button ──────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.6rem 2rem !important; color: white !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88 !important;
}

/* ── Progress bar ────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
}


/* ---------- Safe text styling ---------- */

/* Markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span
 {
    color: #1e293b !important;
}

/* Tables */
.stTable table,
.stTable td,
.stTable th {
    color: #1e293b !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #475569;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: inherit;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] a {
    color: #e2e8f0 !important;
}

/* Selected text in sidebar */
[data-testid="stSidebar"] ::selection {
    background: #2563eb;
    color: white !important;
}

[data-testid="stSidebar"] ::-moz-selection {
    background: #2563eb;
    color: white !important;
}
</style> 
""", 
  unsafe_allow_html=True,

)

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

_SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

_CATEGORY_CSS: Dict[str, str] = {
    "Excellent": "cat-excellent",
    "Good":      "cat-good",
    "Average":   "cat-average",
    "Poor":      "cat-poor",
}
_CATEGORY_EMOJI: Dict[str, str] = {
    "Excellent": "🏆",
    "Good":      "✅",
    "Average":   "⚠️",
    "Poor":      "❌",
}

# Pipeline progress stages shown in the UI (label, 0-100 pct)
_PROGRESS_STAGES = [
    ("Initialising pipeline…",             5),
    ("Running sperm detection…",          20),
    ("Tracking with ByteTrack…",          38),
    ("Computing CASA motility metrics…",  55),
    ("Classifying sperm morphology…",     70),
    ("Predicting viability…",             82),
    ("Calculating WHO quality score…",    90),
    ("Generating PDF / CSV / JSON…",      96),
    ("Analysis complete!",               100),
]

# ══════════════════════════════════════════════════════════════
# Cached resource loaders
# ══════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def _load_config() -> Dict[str, Any]:
    """Load and cache project configuration from configs/config.yaml."""
    try:
        return load_config("configs/config.yaml")
    except FileNotFoundError:
        st.error(
            "⚠️  `configs/config.yaml` not found.  "
            "Start Streamlit from the project root directory."
        )
        st.stop()


@st.cache_resource(show_spinner=False)
def _load_pipeline(config_hash: int) -> Any:  # noqa: ARG001
    """
    Instantiate and cache ``AutomatedSemenAnalysis``.

    ``config_hash`` is used as the cache key so the pipeline is
    re-created if the config changes.
    """
    from src.run_analysis import AutomatedSemenAnalysis  # type: ignore[import]
    cfg = _load_config()
    return AutomatedSemenAnalysis(config=cfg)


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════


def _render_sidebar() -> None:
    """Render the left-hand sidebar with project info and settings."""
    logo_path = Path("app/assets/logo.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)

    st.sidebar.markdown(
        "## 🔬 Semen Analysis AI\n*AI-Powered Andrology Laboratory*"
    )
    st.sidebar.divider()

    st.sidebar.markdown(
        "### Pipeline Stages\n"
        "1. **Detection** · YOLOv8\n"
        "2. **Tracking** · ByteTrack\n"
        "3. **Motility** · CASA metrics\n"
        "4. **Morphology** · EfficientNetB0\n"
        "5. **Viability** · Random Forest\n"
        "6. **Scoring** · WHO 2021\n"
        "7. **Report** · PDF · CSV · JSON"
    )
    st.sidebar.divider()

    st.sidebar.markdown(
        "### WHO 2021 Reference Limits\n"
        "| Parameter | Min |\n"
        "|-----------|-----|\n"
        "| Concentration | 16 M/mL |\n"
        "| Progressive | 30% |\n"
        "| Normal morphology | 4% |\n"
        "| Vitality | 54% |"
    )
    st.sidebar.divider()

    st.sidebar.caption(
        "⚠️ Research and screening use only.  "
        "Results must be confirmed by a certified andrologist."
    )
    st.sidebar.caption("v1.0 · CV Engineering Team")


# ══════════════════════════════════════════════════════════════
# Result extraction helpers
# (handle both AutomatedSemenAnalysis and SemenAnalysisPipeline schemas)
# ══════════════════════════════════════════════════════════════


def _get(results: Dict, *keys: str, default: Any = 0) -> Any:
    """
    Safe nested key access with a default fallback.

    Parameters
    ----------
    results : dict
    *keys   : str  — chain of nested keys to traverse
    default : Any

    Returns
    -------
    Any
    """
    obj: Any = results
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj if obj is not None else default


def _extract_counts(results: Dict) -> Dict[str, Any]:
    """
    Extract sperm count / track numbers from either result schema.

    Returns dict with keys: total_sperm, unique_tracks, video_frames, fps.
    """
    # AutomatedSemenAnalysis flat schema
    if "total_sperm" in results:
        return {
            "total_sperm":   int(_get(results, "total_sperm")),
            "unique_tracks": int(_get(results, "total_sperm")),  # same value
            "video_frames":  int(_get(results, "video_frames")),
            "fps":           float(_get(results, "fps", default=0.0)),
        }
    # SemenAnalysisPipeline nested schema
    det = results.get("detection", {})
    return {
        "total_sperm":   int(_get(det, "count", default=0)),
        "unique_tracks": int(_get(det, "unique_tracks", default=0)),
        "video_frames":  int(_get(det, "n_frames", default=0)),
        "fps":           0.0,
    }


def _extract_motility(results: Dict) -> Dict[str, float]:
    """Extract motility metrics from either result schema."""
    if "progressive" in results:  # flat schema
        return {
            "progressive_pct":    float(_get(results, "progressive")),
            "nonprogressive_pct": float(_get(results, "non_progressive")),
            "immotile_pct":       float(_get(results, "immotile")),
            "mean_VCL":           float(_get(results, "mean_VCL")),
            "mean_VSL":           float(_get(results, "mean_VSL")),
            "mean_VAP":           float(_get(results, "mean_VAP")),
        }
    mot = results.get("motility", {})  # nested schema
    return {
        "progressive_pct":    float(_get(mot, "progressive_pct")),
        "nonprogressive_pct": float(_get(mot, "nonprogressive_pct")),
        "immotile_pct":       float(_get(mot, "immotile_pct")),
        "mean_VCL":           float(_get(mot, "mean_VCL")),
        "mean_VSL":           float(_get(mot, "mean_VSL")),
        "mean_VAP":           float(_get(mot, "mean_VAP")),
    }


def _extract_morphology(results: Dict) -> Dict[str, Any]:
    """Extract morphology metrics from either result schema."""
    if "normal_pct" in results:  # flat schema
        return {
            "normal_pct":    float(_get(results, "normal_pct")),
            "abnormal_pct":  float(_get(results, "abnormal_pct")),
            "normal_count":  int(_get(results, "normal_count")),
            "abnormal_count":int(_get(results, "abnormal_count")),
            "total":         int(_get(results, "normal_count"))
                             + int(_get(results, "abnormal_count")),
        }
    morph = results.get("morphology", {})  # nested schema
    return {
        "normal_pct":    float(_get(morph, "normal_pct")),
        "abnormal_pct":  float(_get(morph, "abnormal_pct")),
        "normal_count":  int(_get(morph, "normal_count")),
        "abnormal_count":int(_get(morph, "abnormal_count")),
        "total":         int(_get(morph, "total")),
    }


def _extract_viability(results: Dict) -> float:
    """Extract predicted viability % from either result schema."""
    if "predicted_viability" in results:
        return float(_get(results, "predicted_viability"))
    return float(_get(results, "viability_pct", default=0.0))


def _extract_quality(results: Dict) -> Dict[str, Any]:
    """Extract WHO quality result dict."""
    return results.get(
        "quality",
        {"score": 0.0, "category": "Unknown", "subscores": {},
         "who_flags": [], "recommendations": []},
    )


def _extract_report_paths(results: Dict) -> Dict[str, str]:
    """Extract report file paths from results."""
    return results.get("report_paths", {})


# ══════════════════════════════════════════════════════════════
# UI component renderers
# ══════════════════════════════════════════════════════════════


def _render_metric_card(label: str, value: str, accent: str = "#3b82f6") -> None:
    """Render a white card with a coloured left border."""
    st.markdown(
        f'<div class="metric-card" style="border-left-color:{accent};">'
        f'<div class="metric-card-value">{value}</div>'
        f'<div class="metric-card-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_who_score_banner(score: float, category: str) -> None:
    """Render the large WHO score banner with colour-coded background."""
    css_cat = _CATEGORY_CSS.get(category, "cat-poor")
    emoji   = _CATEGORY_EMOJI.get(category, "")
    st.markdown(
        f'<div class="who-score-box {css_cat}">'
        f'<div class="who-score-number">{score:.1f}</div>'
        f'<div class="who-score-sub">/ 100</div>'
        f'<div class="who-score-label">{emoji} {category}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_subscore_bars(subscores: Dict[str, float]) -> None:
    """Render individual parameter sub-scores as progress bars."""
    label_map = {
        "count":      "Count",
        "motility":   "Motility",
        "morphology": "Morphology",
        "viability":  "Viability",
    }
    colour_map = {
        "count":      "#3b82f6",
        "motility":   "#10b981",
        "morphology": "#8b5cf6",
        "viability":  "#f59e0b",
    }
    for key, label in label_map.items():
        val = float(subscores.get(key, 0.0))
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown(
                f'<div style="font-size:0.85rem;font-weight:600;'
                f'color:#374151;margin-bottom:2px;">{label}</div>',
                unsafe_allow_html=True,
            )
            pct = max(0, min(100, int(val)))
            colour = colour_map.get(key, "#3b82f6")
            st.markdown(
                f'<div style="height:10px;background:#e5e7eb;border-radius:5px;">'
                f'<div style="width:{pct}%;height:100%;background:{colour};'
                f'border-radius:5px;"></div></div>',
                unsafe_allow_html=True,
            )
        with col_r:
            st.markdown(
                f'<div style="text-align:right;font-size:0.85rem;'
                f'font-weight:700;color:{colour};padding-top:2px;">'
                f'{val:.1f}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


def _render_optional_plot(plot_path: Path, caption: str) -> None:
    """Display a plot image if it exists; silently skip otherwise."""
    if plot_path.exists():
        st.image(str(plot_path), caption=caption, use_container_width=True)


def _render_who_flags(flags: list[str]) -> None:
    """Display WHO threshold violations in red alert boxes."""
    if not flags:
        st.success("✅  All parameters are within WHO 2021 reference limits.")
        return
    for flag in flags:
        st.markdown(
            f'<div class="flag-box">⚠️  {flag}</div>',
            unsafe_allow_html=True,
        )


def _render_recommendations(recs: list[str]) -> None:
    """Display clinical recommendations in green boxes."""
    if not recs:
        return
    for rec in recs:
        st.markdown(
            f'<div class="rec-box">💡  {rec}</div>',
            unsafe_allow_html=True,
        )


def _render_download_buttons(report_paths: Dict[str, str]) -> None:
    """
    Render PDF / CSV / JSON download buttons using pre-generated report files.

    Uses ``results["report_paths"]`` only.  Does NOT regenerate any reports.
    """
    col_pdf, col_csv, col_json = st.columns(3)

    mime_map = {
        "pdf":  ("PDF Report",  "application/pdf"),
        "csv":  ("CSV Summary", "text/csv"),
        "json": ("JSON Report", "application/json"),
    }

    columns = {"pdf": col_pdf, "csv": col_csv, "json": col_json}

    for fmt, (label, mime) in mime_map.items():
        path_str = report_paths.get(fmt, "")
        col      = columns[fmt]
        if path_str:
            path = Path(path_str)
            if path.exists():
                try:
                    data = path.read_bytes()
                    col.download_button(
                        label       = f"⬇  {label}",
                        data        = data,
                        file_name   = path.name,
                        mime        = mime,
                        use_container_width=True,
                    )
                except OSError as exc:
                    col.caption(f"{label} unreadable: {exc}")
            else:
                col.caption(f"{label}: file not found")
        else:
            col.caption(f"{label}: not generated")


# ══════════════════════════════════════════════════════════════
# Results page
# ══════════════════════════════════════════════════════════════


def _render_results(
    results: Dict[str, Any],
    video_stem: str,
    config: Dict[str, Any],
) -> None:
    """
    Render the full results dashboard from pipeline output.

    Parameters
    ----------
    results : dict
        Return value of ``AutomatedSemenAnalysis.run_video``.
    video_stem : str
        Stem of the uploaded video filename (used to locate trajectory plot).
    config : dict
        Loaded project configuration.
    """
    plot_dir    = Path(config["paths"]["outputs"]["plots"])
    tracking_dir= Path(config["paths"]["outputs"]["tracking"])

    counts   = _extract_counts(results)
    motility = _extract_motility(results)
    morph    = _extract_morphology(results)
    viab     = _extract_viability(results)
    quality  = _extract_quality(results)
    rp       = _extract_report_paths(results)

    st.write("DEBUG QUALITY:", quality)

    score    = float(quality.get("score",    0.0))
    category = str(quality.get("category",  "Unknown"))
    subscores= quality.get("subscores",     {})
    flags    = quality.get("who_flags",     [])
    recs     = quality.get("recommendations", [])

    # ── Top metric row ─────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Metrics</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _render_metric_card("Total Sperm", str(counts["total_sperm"]),   "#3b82f6")
    with c2:
        _render_metric_card("Frames",      str(counts["video_frames"]),  "#6366f1")
    with c3:
        _render_metric_card("Progressive", f"{motility['progressive_pct']:.1f}%", "#10b981")
    with c4:
        _render_metric_card("Normal Morph.", f"{morph['normal_pct']:.1f}%",       "#8b5cf6")
    with c5:
        _render_metric_card("Viability",   f"{viab:.1f}%",               "#f59e0b")
    with c6:
        _render_metric_card("WHO Score",   f"{score:.1f}",               "#ef4444")

    st.divider()

    # ── WHO Score + Sub-scores ─────────────────────────────────
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-header">🏆 WHO Quality Score</div>',
                    unsafe_allow_html=True)
        _render_who_score_banner(score, category)

    with right_col:
        st.markdown('<div class="section-header">📈 Parameter Sub-scores</div>',
                    unsafe_allow_html=True)
        if subscores:
            _render_subscore_bars(subscores)
        else:
            st.caption("Sub-scores unavailable.")

    st.divider()

    # ── Analysis tabs ──────────────────────────────────────────
    tab_mot, tab_morph, tab_via = st.tabs(
        ["🏃 Motility", "🔬 Morphology", "💉 Viability"]
    )

    # ─ Motility tab ────────────────────────────────────────────
    with tab_mot:
        c_a, c_b = st.columns(2)

        with c_a:
            st.markdown("**CASA Motility Metrics**")
            st.table(
                {
                    "Parameter": [
                        "Progressive",
                        "Non-Progressive",
                        "Immotile",
                        "Mean VCL (µm/s)",
                        "Mean VSL (µm/s)",
                        "Mean VAP (µm/s)",
                    ],
                    "Value": [
                        f"{motility['progressive_pct']:.1f}%",
                        f"{motility['nonprogressive_pct']:.1f}%",
                        f"{motility['immotile_pct']:.1f}%",
                        f"{motility['mean_VCL']:.2f}",
                        f"{motility['mean_VSL']:.2f}",
                        f"{motility['mean_VAP']:.2f}",
                    ],
                }
            )

        with c_b:
            _render_optional_plot(
                plot_dir / "motility_analysis.png",
                "Motility Distribution",
            )

    # ─ Morphology tab ──────────────────────────────────────────
    with tab_morph:
        c_a, c_b = st.columns(2)

        with c_a:
            st.markdown("**Morphology Classification**")
            st.table(
                {
                    "Parameter": [
                        "Normal",
                        "Abnormal",
                        "Normal Count",
                        "Abnormal Count",
                        "Total Classified",
                    ],
                    "Value": [
                        f"{morph['normal_pct']:.1f}%",
                        f"{morph['abnormal_pct']:.1f}%",
                        str(morph["normal_count"]),
                        str(morph["abnormal_count"]),
                        str(morph["total"]),
                    ],
                }
            )

        with c_b:
            _render_optional_plot(
                plot_dir / "morphology_confusion_matrix.png",
                "Morphology Confusion Matrix",
            )

    # ─ Viability tab ───────────────────────────────────────────
    with tab_via:
        c_a, c_b = st.columns(2)

        with c_a:
            st.markdown("**Viability Estimation**")
            st.table(
                {
                    "Parameter": ["Predicted Viability", "WHO Reference Min"],
                    "Value":     [f"{viab:.1f}%",        "54%"],
                }
            )
            who_via = 54.0
            delta_c = "normal" if viab >= who_via else "inverse"
            st.metric(
                "Viability vs WHO threshold",
                f"{viab:.1f}%",
                delta=f"{viab - who_via:+.1f}%",
                delta_color=delta_c,
            )

        with c_b:
            _render_optional_plot(
                plot_dir / "viability_predictions.png",
                "Viability Model Predictions",
            )

    st.divider()

    # ── Trajectory image ───────────────────────────────────────
    traj_path = tracking_dir / f"{video_stem}_trajectories.png"
    if traj_path.exists():
        st.markdown('<div class="section-header">🛤️ Sperm Trajectory Map</div>',
                    unsafe_allow_html=True)
        st.image(str(traj_path), use_container_width=True)
        st.divider()

    # ── WHO Flags ──────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ WHO Threshold Check</div>',
                unsafe_allow_html=True)
    _render_who_flags(flags)

    st.divider()

    # ── Recommendations ────────────────────────────────────────
    if recs:
        st.markdown('<div class="section-header">💡 Clinical Recommendations</div>',
                    unsafe_allow_html=True)
        _render_recommendations(recs)
        st.divider()

    # ── Downloads ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📥 Download Reports</div>',
                unsafe_allow_html=True)
    _render_download_buttons(rp)


# ══════════════════════════════════════════════════════════════
# Pipeline execution with staged progress bar
# ══════════════════════════════════════════════════════════════


def _run_pipeline_with_progress(
    pipeline: Any,
    video_path: Path,
) -> Dict[str, Any]:
    """
    Call ``pipeline.run_video`` once, animating a progress bar
    in the Streamlit UI while the backend runs.

    Because Streamlit cannot introspect backend progress, the stages
    advance on a timer before the blocking call, then jump to 100 on
    completion.

    Parameters
    ----------
    pipeline : AutomatedSemenAnalysis
    video_path : Path

    Returns
    -------
    dict  results from the pipeline
    """
    progress_bar = st.progress(0, text="Initialising…")
    status_box   = st.empty()

    # Advance a few early stages before the blocking call
    for label, pct in _PROGRESS_STAGES[:3]:
        progress_bar.progress(pct, text=label)
        status_box.info(f"⏳  {label}")
        time.sleep(0.25)

# ------------------------------------------
# Live tracking progress callback
# ------------------------------------------
    progress_bar = st.progress(0)
    status_box = st.empty()
    
    def update_tracking_progress(progress):
        percent = 5 + int(progress * 65)

        progress_bar.progress(
           percent,
           text=f"🔬 Tracking with ByteTrack... {percent}%"
        )

        status_box.info(
          f"⏳ Tracking sperm... {percent}%"
        )

    def update_stage_progress(percent, text):

         progress_bar.progress(
             percent,
             text=text,
        )

         status_box.info(text)   

    
    # ── Blocking pipeline call ─────────────────────────────────
    results = pipeline.run_video(
        video_path,
        progress_callback=update_tracking_progress, 
        progress_callback_stage=update_stage_progress,
    )

    # ── Advance remaining stages ───────────────────────────────
    for label, pct in _PROGRESS_STAGES[3:]:
        progress_bar.progress(pct, text=label)
        status_box.info(f"⏳  {label}")
        time.sleep(0.15)

    progress_bar.empty()
    status_box.empty()
    return results


# ══════════════════════════════════════════════════════════════
# Main application
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point — renders the complete Streamlit application."""

    # ── Sidebar ────────────────────────────────────────────────
    _render_sidebar()

    # ── Page header ────────────────────────────────────────────
    st.markdown(
        '<div class="page-title">🔬 Automated Semen Analysis</div>'
        '<div class="page-subtitle">'
        "Upload a microscope semen video to receive a full AI-powered "
        "quality assessment with WHO 2021 scoring."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Load config (cached) ───────────────────────────────────
    config = _load_config()

    # ── Video upload ───────────────────────────────────────────
    st.markdown('<div class="section-header">📤 Upload Video</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label        = "Drop or select a microscope semen video",
        type         = _SUPPORTED_VIDEO_FORMATS,
        help         = "Accepts MP4 · AVI · MOV · MKV",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info(
            "👆  Upload a semen microscope video to begin analysis.  \n"
            "Supported formats: **MP4 · AVI · MOV · MKV**"
        )
        return

    # ── Video preview ──────────────────────────────────────────
    file_size_mb = uploaded_file.size / 1_048_576
    video_stem   = Path(uploaded_file.name).stem

    st.success(
        f"✅  **{uploaded_file.name}** loaded "
        f"({file_size_mb:.1f} MB)"
    )

    prev_col, info_col = st.columns([2, 1])

    # Write to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded_file.name).suffix,
        delete=False,
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    with prev_col:
        cap = cv2.VideoCapture(str(tmp_path))
        ret, frame = cap.read()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val      = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if ret and frame is not None:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="First frame preview",
                use_container_width=True,
            )
        else:
            st.warning("Unable to read first frame for preview.")

    with info_col:
        st.markdown("**Video Information**")
        st.table(
            {
                "Property": ["File name", "Size", "Resolution", "Frames", "FPS"],
                "Value": [
                    uploaded_file.name,
                    f"{file_size_mb:.1f} MB",
                    f"{width} × {height}",
                    str(total_frames),
                    f"{fps_val:.1f}",
                ],
            }
        )

    st.divider()

    # ── Run button ─────────────────────────────────────────────
    run_clicked = st.button(
        "▶  Run Full Analysis",
        type="primary",
        use_container_width=True,
        help="Runs detection → tracking → motility → morphology → viability → WHO scoring → report",
    )

    if not run_clicked:
        tmp_path.unlink(missing_ok=True)
        return

    # ── Execute pipeline ───────────────────────────────────────
    st.divider()

    # Load pipeline (cached after first call)
    try:
        pipeline = _load_pipeline(id(config))
    except ImportError as exc:
        st.error(
            f"❌  Could not import backend pipeline: `{exc}`  \n"
            "Make sure you are running from the project root and all "
            "dependencies are installed."
        )
        tmp_path.unlink(missing_ok=True)
        return
    except Exception as exc:
        st.error(f"❌  Pipeline initialisation failed: {exc}")
        tmp_path.unlink(missing_ok=True)
        return

    t0 = time.perf_counter()
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
        # Always clean up the temporary file
        tmp_path.unlink(missing_ok=True)

    if results is None:
        st.warning("Analysis did not return results.  Check the logs for details.")
        return

    elapsed = time.perf_counter() - t0
    st.success(f"✅  Analysis complete in **{elapsed:.1f}s**")
    st.divider()

    # ── Render results dashboard ───────────────────────────────
    _render_results(results, video_stem, config)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()