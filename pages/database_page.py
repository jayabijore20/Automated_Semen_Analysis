"""
app/pages/database_page.py
============================
Phase 5 · Database Browser Page

Standalone Streamlit page that provides:
  1. Statistics dashboard — total sessions, averages, species breakdown
  2. Search panel — by Sample ID, by Animal ID, by species filter
  3. Session list — paginated table of all stored sessions
  4. Session detail view — full record with result metrics and report downloads
  5. Session comparison — side-by-side comparison of 2–6 selected sessions
  6. Notes editor — add/edit free-text notes per session
  7. Delete session — with confirmation guard

This file imports DatabaseManager only.
It has NO knowledge of the AI pipeline, camera, or monitoring modules.

Usage (Streamlit multi-page)
-----------------------------
    # Streamlit auto-discovers pages in app/pages/
    # URL: http://localhost:8501/Database

    # Or call render_database_page() directly from the main app
    from app.pages.database_page import render_database_page
    render_database_page(db)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.database.db_manager import DatabaseManager

# ══════════════════════════════════════════════════════════════
# Design tokens (match main app)
# ══════════════════════════════════════════════════════════════

_CLR_PRIMARY   = "#1d4ed8"
_CLR_SUCCESS   = "#10b981"
_CLR_WARNING   = "#f59e0b"
_CLR_DANGER    = "#ef4444"
_CLR_BG        = "#f1f5f9"
_CLR_CARD      = "#ffffff"
_CLR_BORDER    = "#e2e8f0"
_CLR_TEXT_PRI  = "#0f172a"
_CLR_TEXT_SEC  = "#64748b"
_CLR_MONITOR   = "#7c3aed"

_PAGE_SIZE = 15   # rows per page in the sessions list

# Category colour map
_CAT_CLR = {
    "Excellent": "#059669",
    "Good":      "#2563eb",
    "Average":   "#d97706",
    "Poor":      "#dc2626",
}

# ══════════════════════════════════════════════════════════════
# CSS injected once per page load
# ══════════════════════════════════════════════════════════════

_DB_CSS = f"""
<style>
.db-stat-card {{
    background:{_CLR_CARD};border:1px solid {_CLR_BORDER};
    border-radius:12px;padding:1rem 1.3rem;
    border-top:4px solid {_CLR_PRIMARY};
    box-shadow:0 1px 3px rgba(0,0,0,.05);
}}
.db-stat-val {{
    font-size:2rem;font-weight:800;color:{_CLR_TEXT_PRI};line-height:1;
}}
.db-stat-lbl {{
    font-size:0.72rem;color:{_CLR_TEXT_SEC};
    text-transform:uppercase;letter-spacing:0.07em;font-weight:600;
    margin-top:0.2rem;
}}
.db-section {{
    font-size:0.95rem;font-weight:700;color:{_CLR_TEXT_PRI};
    border-bottom:2px solid {_CLR_BORDER};
    padding-bottom:0.35rem;margin:1rem 0 0.6rem 0;
}}
.session-row {{
    background:{_CLR_CARD};border:1px solid {_CLR_BORDER};
    border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.5rem;
    display:flex;align-items:center;justify-content:space-between;
    flex-wrap:wrap;gap:0.5rem;
}}
.session-badge-single  {{background:#dbeafe;color:#1e40af;
    border-radius:4px;padding:1px 7px;font-size:0.72rem;font-weight:700;}}
.session-badge-monitor {{background:#ede9fe;color:#5b21b6;
    border-radius:4px;padding:1px 7px;font-size:0.72rem;font-weight:700;}}
.grade-badge {{
    border-radius:5px;padding:2px 8px;font-size:0.78rem;font-weight:700;
    color:white;
}}
.compare-card {{
    background:{_CLR_CARD};border:1px solid {_CLR_BORDER};
    border-radius:12px;padding:0.9rem 1.1rem;
    box-shadow:0 1px 3px rgba(0,0,0,.05);
}}
.compare-title {{
    font-size:0.85rem;font-weight:700;color:{_CLR_PRIMARY};
    border-bottom:1px solid {_CLR_BORDER};padding-bottom:0.3rem;
    margin-bottom:0.5rem;
}}
.metric-pill {{
    display:inline-block;background:{_CLR_BG};border:1px solid {_CLR_BORDER};
    border-radius:20px;padding:2px 10px;font-size:0.78rem;
    font-weight:600;color:{_CLR_TEXT_PRI};margin:2px;
}}
</style>
"""


def _inject_database_button_styles() -> None:
    """Fix database page button visibility and pagination controls."""

    st.markdown(
        """
        <style>

        /* =========================================================
           DATABASE PAGE - ALL NORMAL BUTTONS
           ========================================================= */

        [data-testid="stAppViewContainer"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        /* Button inner elements */
        [data-testid="stAppViewContainer"] button span,
        [data-testid="stAppViewContainer"] button p,
        [data-testid="stAppViewContainer"] button div {
            color: #111827 !important;
            opacity: 1 !important;
        }

        /* SVG / eye / delete icons */
        [data-testid="stAppViewContainer"] button svg {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
            opacity: 1 !important;
        }

        /* =========================================================
           VIEW / DELETE ICON BUTTONS
           ========================================================= */

        [data-testid="stAppViewContainer"]
        [data-testid="stButton"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            opacity: 1 !important;
        }

        [data-testid="stAppViewContainer"]
        [data-testid="stButton"] button:hover {
            background: #f1f5f9 !important;
            background-color: #f1f5f9 !important;
            color: #111827 !important;
            border-color: #2563eb !important;
        }

        [data-testid="stAppViewContainer"]
        [data-testid="stButton"] button svg {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* =========================================================
           PAGINATION BUTTONS
           ========================================================= */

        [data-testid="stAppViewContainer"] button[kind="secondary"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            opacity: 1 !important;
        }

        [data-testid="stAppViewContainer"] button[kind="secondary"]:hover {
            background: #f1f5f9 !important;
            background-color: #f1f5f9 !important;
            color: #111827 !important;
            border-color: #2563eb !important;
        }

        [data-testid="stAppViewContainer"] button[kind="secondary"] span,
        [data-testid="stAppViewContainer"] button[kind="secondary"] p {
            color: #111827 !important;
        }

        /* =========================================================
           DISABLED BUTTONS
           Keep them visible instead of black.
           ========================================================= */

        [data-testid="stAppViewContainer"] button:disabled {
            background: #f8fafc !important;
            background-color: #f8fafc !important;
            color: #94a3b8 !important;
            border: 1px solid #cbd5e1 !important;
            opacity: 0.65 !important;
            cursor: not-allowed !important;
        }

        [data-testid="stAppViewContainer"] button:disabled span,
        [data-testid="stAppViewContainer"] button:disabled p,
        [data-testid="stAppViewContainer"] button:disabled svg {
            color: #94a3b8 !important;
            fill: #94a3b8 !important;
            stroke: #94a3b8 !important;
        }

        /* =========================================================
           DOWNLOAD BUTTONS
           ========================================================= */

        [data-testid="stAppViewContainer"] [data-testid="stDownloadButton"] button {
            background: #ffffff !important;
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
        }

        [data-testid="stAppViewContainer"] [data-testid="stDownloadButton"] button svg {
            color: #111827 !important;
            fill: #111827 !important;
            stroke: #111827 !important;
        }

        /* =========================================================
           TOOLTIPS
           ========================================================= */

        [data-baseweb="tooltip"] {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }

        [data-baseweb="tooltip"] * {
            background: transparent !important;
            color: #111827 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# HTML helpers
# ══════════════════════════════════════════════════════════════

def _stat_card(label: str, value: str, accent: str = _CLR_PRIMARY) -> None:
    st.markdown(
        f'<div class="db-stat-card" style="border-top-color:{accent};">'
        f'<div class="db-stat-val">{value}</div>'
        f'<div class="db-stat-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _section(text: str) -> None:
    st.markdown(f'<div class="db-section">{text}</div>', unsafe_allow_html=True)


def _grade_badge(category: str) -> str:
    colour = _CAT_CLR.get(category, "#475569")
    return (
        f'<span class="grade-badge" style="background:{colour};">'
        f'{category}</span>'
    )


def _type_badge(session_type: str) -> str:
    if session_type == "monitoring":
        return '<span class="session-badge-monitor">⏱ MONITORING</span>'
    return '<span class="session-badge-single">📷 SINGLE</span>'


# ══════════════════════════════════════════════════════════════
# Sub-section renderers
# ══════════════════════════════════════════════════════════════

def _render_statistics(db: DatabaseManager) -> None:
    """Top statistics dashboard row."""
    stats = db.get_statistics()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: _stat_card("Total Sessions",    str(stats["total_sessions"]),      _CLR_PRIMARY)
    with c2: _stat_card("Single Analyses",   str(stats["single_sessions"]),     "#6366f1")
    with c3: _stat_card("Monitoring Sessions",str(stats["monitoring_sessions"]),_CLR_MONITOR)
    with c4: _stat_card("Avg WHO Score",     f"{stats['avg_who_score']:.1f}",   _CLR_SUCCESS)
    with c5: _stat_card("Avg Progressive %", f"{stats['avg_progressive_pct']:.1f}%", _CLR_WARNING)
    with c6: _stat_card("Avg Viability %",   f"{stats['avg_viability_pct']:.1f}%",  _CLR_DANGER)

    # Species breakdown
    if stats["species_counts"]:
        st.markdown("<br>", unsafe_allow_html=True)
        _section("🐄 Sessions by Species")
        sp_df = pd.DataFrame(
            [{"Species": k, "Sessions": v}
             for k, v in stats["species_counts"].items()]
        )
        st.bar_chart(sp_df.set_index("Species"), width="stretch", height=200)


def _render_search_panel(db: DatabaseManager) -> Optional[List[Dict[str, Any]]]:
    """
    Search controls.  Returns matching rows or None (show full list).
    """
    _section("🔍 Search Sessions")
    col_type, col_s, col_a, col_btn, col_clear = st.columns([1.5, 2, 2, 1, 1])

    with col_type:
        filter_type = st.selectbox(
            "Type", ["All", "Single", "Monitoring"],
            key="_db_filter_type", label_visibility="collapsed",
        )
    with col_s:
        sid_query = st.text_input(
            "Sample ID", placeholder="Search Sample ID…",
            key="_db_search_sample", label_visibility="collapsed",
        )
    with col_a:
        aid_query = st.text_input(
            "Animal ID", placeholder="Search Animal ID…",
            key="_db_search_animal", label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("🔍 Search", width="stretch",
                              key="_db_search_btn")
    with col_clear:
        do_clear = st.button("✕ Clear", width="stretch",
                             key="_db_clear_btn")

    if do_clear:
        for k in ("_db_search_sample", "_db_search_animal"):
            st.session_state.pop(k, None)
        st.rerun()

    if not (do_search or sid_query or aid_query):
        return None     # caller shows full list

    results: List[Dict] = []
    if sid_query.strip():
        results = db.search_by_sample_id(sid_query.strip())
    elif aid_query.strip():
        results = db.search_by_animal_id(aid_query.strip())
    else:
        st_map = {"All": None, "Single": "single", "Monitoring": "monitoring"}
        results = db.list_sessions(limit=200, session_type=st_map.get(filter_type))

    if filter_type != "All" and not (sid_query or aid_query):
        return results     # already filtered above

    if filter_type != "All":
        ft = filter_type.lower()
        results = [r for r in results if r.get("session_type") == ft]

    return results


def _render_session_table(rows: List[Dict[str, Any]],
                           db: DatabaseManager, table_id: str = "default",) -> None:
    """
    Render a paginated table of session rows with View / Compare checkboxes.

    Parameters
    ----------
    rows : list of dict   from list_sessions or search
    db   : DatabaseManager
    """
    if not rows:
        st.info("No sessions found.")
        return

    ss = st.session_state

    # ── Pagination ─────────────────────────────────────────────
    total    = len(rows)
    n_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    page_key = f"_db_page_{table_id}"

    if page_key not in ss:
        ss[page_key] = 0

    # Get current page
    page = ss[page_key]

    # Make sure pagination values are integers
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0

    try:
        n_pages = int(n_pages)
    except (TypeError, ValueError):
        n_pages = 1

    n_pages = max(1, n_pages)

    # Keep page within valid range
    page = max(0, min(page, n_pages - 1))

    # Save corrected value
    ss[page_key] = page

    # Get rows for current page
    page_rows = rows[
        page * _PAGE_SIZE:
        (page + 1) * _PAGE_SIZE
    ]
    # ── Comparison selection set ───────────────────────────────
    compare_key = "_db_compare_set"
    if compare_key not in ss:
        ss[compare_key] = set()

    # ── Table header ───────────────────────────────────────────
    header = st.columns([0.4, 1.6, 1.4, 1, 1, 1, 0.8, 0.8, 0.7, 0.7])
    for col, lbl in zip(header, [
        "Compare", "Sample ID", "Animal ID", "Species",
        "Type", "Date", "WHO Score", "Grade", "View", "Delete",
    ]):
        col.markdown(
            f'<span style="font-size:0.75rem;font-weight:700;'
            f'color:{_CLR_TEXT_SEC};text-transform:uppercase;'
            f'letter-spacing:0.06em;">{lbl}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<hr style="margin:0.3rem 0;border-color:{_CLR_BORDER};">',
        unsafe_allow_html=True,
    )

    # ── Rows ───────────────────────────────────────────────────
    for row_idx, row in enumerate(page_rows):

        sid = row["session_id"]

        # Guaranteed unique for this rendered table row
        unique_row_key = f"{table_id}_{page}_{row_idx}_{sid}"

        cols = st.columns(
            [0.4, 1.6, 1.4, 1, 1, 1, 0.8, 0.8, 0.7, 0.7]
        )

        # ── Compare checkbox ──────────────────────────────────
        with cols[0]:
            in_set = sid in ss[compare_key]

            cmp_key = f"_cmp_{unique_row_key}"

            chk = st.checkbox(
                "",
                value=in_set,
                key=cmp_key,
                label_visibility="collapsed",
            )

            if chk:
                ss[compare_key].add(sid)
            else:
                ss[compare_key].discard(sid)

        # ── Sample ID ─────────────────────────────────────────
        with cols[1]:
            st.markdown(
                f'<span style="font-size:0.88rem;font-weight:600;'
                f'color:{_CLR_TEXT_PRI};">'
                f'{row.get("sample_id", "—")}</span>',
                unsafe_allow_html=True,
            )

        # ── Animal ID ─────────────────────────────────────────
        with cols[2]:
            st.caption(row.get("animal_id", "—"))

        # ── Species ────────────────────────────────────────────
        with cols[3]:
            st.caption(row.get("species", "—"))

        # ── Session type ──────────────────────────────────────
        with cols[4]:
            st.markdown(
                _type_badge(row.get("session_type", "single")),
                unsafe_allow_html=True,
            )

        # ── Created ────────────────────────────────────────────
        with cols[5]:
            st.caption(row.get("created_at", "")[:16])

        # ── WHO score ──────────────────────────────────────────
        with cols[6]:
            detail = db.get_session_detail(sid)

            score = 0.0
            cat = "—"

            if detail and detail["results"]:
                score = detail["results"][0].get("who_score", 0.0)
                cat = detail["results"][0].get("who_category", "—")

            st.markdown(
                f'<span style="font-size:0.88rem;font-weight:700;">'
                f'{score:.1f}</span>',
                unsafe_allow_html=True,
            )

        # ── Grade ──────────────────────────────────────────────
        with cols[7]:
            st.markdown(
                _grade_badge(cat),
                unsafe_allow_html=True,
            )

        # ── View ───────────────────────────────────────────────
   # ── View ───────────────────────────────────────────────
            # ── View ───────────────────────────────────────────────
        with cols[8]:
            if st.button(
                "👁",
                key=f"_view_{unique_row_key}",
                help="View session detail",
            ):
                ss["_db_view_session_id"] = sid
                st.rerun()

        # ── Delete ─────────────────────────────────────────────
        with cols[9]:
            if st.button(
                "🗑",
                key=f"_del_{unique_row_key}",
                help="Delete session",
            ):
                print("=" * 60)
                print("DELETE BUTTON CLICKED")
                print("SID:", repr(sid))
                print("SID TYPE:", type(sid))
                print("ROW KEY:", repr(unique_row_key))
                print("=" * 60)

                ss["_db_confirm_delete"] = sid
                ss["_db_return_to_database"] = True
                st.rerun()

    # ── Pagination controls ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    p_l, p_info, p_r = st.columns([1, 3, 1])

    with p_l:
        if st.button(
            "◀ Prev",
            disabled=(page == 0),
            width="stretch",
            key=f"_db_prev_{table_id}",
        ):
            ss[page_key] = max(0, page - 1)
            st.rerun()

    with p_info:
        st.markdown(
            f'<div style="text-align:center;font-size:0.82rem;'
            f'color:{_CLR_TEXT_SEC};">'
            f'Page {page + 1} of {n_pages} '
            f'&nbsp;·&nbsp; {total} sessions'
            f'</div>',
            unsafe_allow_html=True,
        )

    with p_r:
        if st.button(
            "Next ▶",
            disabled=(page >= n_pages - 1),
            width="stretch",
            key=f"_db_next_{table_id}",
        ):
            ss[page_key] = min(n_pages - 1, page + 1)
            st.rerun()

def _render_session_detail(session_id: str, db: DatabaseManager) -> None:
    """
    Full detail view for a single session.

    Parameters
    ----------
    session_id : str
    db : DatabaseManager
    """
    detail = db.get_session_detail(session_id)
    if detail is None:
        st.error(f"Session {session_id[:8]} not found in database.")
        return

    s  = detail["session"]
    rs = detail["results"]

    # ── Back button ────────────────────────────────────────────
    if st.button("← Back to Session List", key="_db_back_btn"):
        st.session_state.pop("_db_view_session_id", None)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Session header ─────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_CLR_PRIMARY},#4f46e5);'
        f'border-radius:14px;padding:1.2rem 1.6rem;color:white;margin-bottom:1rem;">'
        f'<div style="font-size:1.25rem;font-weight:800;">'
        f'Session Detail</div>'
        f'<div style="font-size:0.88rem;opacity:0.82;margin-top:0.2rem;">'
        f'ID: {s["session_id"][:8]}…  ·  '
        f'Type: {s["session_type"].title()}  ·  '
        f'Created: {s["created_at"][:16]}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Two-column info grid ───────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        _section("🧪 Sample Information")
        st.table({
            "Field":  ["Sample ID","Animal ID","Species","Breed",
                       "Sample Type","Collection Date","Operator","Laboratory"],
            "Value":  [s.get("sample_id","—"), s.get("animal_id","—"),
                       s.get("species","—"),   s.get("breed","—"),
                       s.get("sample_type","—"), s.get("collection_date","—"),
                       s.get("operator","—"),  s.get("lab_name","—")],
        })
    with col_r:
        _section("🔬 Laboratory Configuration")
        st.table({
            "Field":  ["Chamber","Microscope Type","Magnification",
                       "Staining","Camera Resolution","Camera FPS",
                       "Interval (min)","Duration (min)"],
            "Value":  [s.get("chamber","—"), s.get("microscope_type","—"),
                       s.get("magnification","—"), s.get("staining_type","—"),
                       s.get("camera_resolution","—"), s.get("camera_fps","—"),
                       str(s.get("interval_min",0)), str(s.get("duration_min",0))],
        })

    # ── Notes editor ───────────────────────────────────────────
    _section("📝 Notes")
    current_notes = s.get("notes", "")
    new_notes = st.text_area(
        "Session notes", value=current_notes, height=80,
        key=f"_notes_{session_id}", label_visibility="collapsed",
    )
    if st.button("💾 Save Notes", key=f"_save_notes_{session_id}"):
        db.update_notes(session_id, new_notes)
        st.success("Notes saved.")

    # ── Per-capture results ────────────────────────────────────
    if rs:
        _section(f"📋 Capture Results ({len(rs)} captures)")

        summary_rows = []
        for r in rs:
            summary_rows.append({
                "#":            r["capture_index"],
                "t (min)":      f"{r['elapsed_min']:.1f}",
                "Sperm":        r["total_sperm"],
                "Progressive%": f"{r['progressive_pct']:.1f}",
                "Viability%":   f"{r['viability_pct']:.1f}",
                "WHO Score":    f"{r['who_score']:.1f}",
                "Grade":        r["who_category"],
                "Created":      r["created_at"][:16],
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch",
                     hide_index=True)

        # ── Report downloads ───────────────────────────────────
        _section("📥 Report Downloads")

        # Single-capture reports (first result row)
        first = rs[0]
        _render_file_downloads({
            "PDF Report":  first.get("report_pdf",  ""),
            "CSV Summary": first.get("report_csv",  ""),
            "JSON Data":   first.get("report_json", ""),
        })

        # Monitoring trend reports (if any)
        if s["session_type"] == "monitoring":
            trend_pdf = first.get("trend_pdf", "")
            trend_csv = first.get("trend_csv", "")
            if trend_pdf or trend_csv:
                _section("📈 Monitoring Trend Report")
                _render_file_downloads({
                    "Trend PDF": trend_pdf,
                    "Trend CSV": trend_csv,
                })

        # ── Annotated video ────────────────────────────────────
        av = first.get("annotated_video", "")
        if av and Path(av).exists():
            _section("🎬 Annotated Video")
            with open(av, "rb") as vf:
                st.download_button(
                    "⬇  Download Annotated Video",
                    data=vf.read(),
                    file_name=Path(av).name,
                    mime="video/mp4",
                )

        # ── Trend plots (monitoring) ───────────────────────────
        if s["session_type"] == "monitoring":
            trend_data = first.get("trend_report_json", {})
            if isinstance(trend_data, dict):
                plot_paths = trend_data.get("plot_paths", {})
                if plot_paths:
                    _section("📊 Trend Plots")
                    dash = plot_paths.get("dashboard", "")
                    if dash and Path(dash).exists():
                        st.image(dash, caption="Summary Dashboard",
                                 width="stretch")
                    mot = plot_paths.get("motility_overview", "")
                    if mot and Path(mot).exists():
                        st.image(mot, caption="Motility Overview",
                                 width="stretch")


def _render_file_downloads(files: Dict[str, str]) -> None:
    """
    Render one download button per file.  Silently skips missing paths.

    Parameters
    ----------
    files : dict  {label: path_str}
    """
    mime_map = {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "json": "application/json",
        "video": "video/mp4",
    }
    cols = st.columns(len(files))
    for col, (label, path_str) in zip(cols, files.items()):
        if not path_str:
            col.caption(f"{label}: not available")
            continue
        p = Path(path_str)
        if not p.exists():
            col.caption(f"{label}: file not found")
            continue
        ext  = p.suffix.lstrip(".").lower()
        mime = mime_map.get(ext, "application/octet-stream")
        try:
            col.download_button(
                label=f"⬇  {label}",
                data=p.read_bytes(),
                file_name=p.name,
                mime=mime,
                width="stretch",
            )
        except OSError:
            col.caption(f"{label}: read error")


def _render_comparison(db: DatabaseManager) -> None:
    """
    Side-by-side comparison for sessions selected via the Compare checkboxes.

    Parameters
    ----------
    db : DatabaseManager
    """
    ss          = st.session_state
    compare_set = ss.get("_db_compare_set", set())

    if not compare_set:
        return

    _section(f"⚖️  Comparing {len(compare_set)} Session(s)")

    if len(compare_set) < 2:
        st.info("Select at least 2 sessions via the Compare checkboxes to compare.")
        return

    if len(compare_set) > 6:
        st.warning("Maximum 6 sessions can be compared at once.")
        return

    sessions = db.get_sessions_for_comparison(list(compare_set))
    if not sessions:
        st.warning("No data found for selected sessions.")
        return

    # ── Summary comparison table ───────────────────────────────
    comp_rows = []
    for s in sessions:
        comp_rows.append({
            "Sample ID":      s["sample_id"],
            "Animal ID":      s["animal_id"],
            "Species":        s["species"],
            "Type":           s["session_type"].title(),
            "Date":           s.get("capture_time", "")[:16],
            "Sperm":          s["total_sperm"],
            "Progressive%":   f"{s['progressive_pct']:.1f}",
            "Viability%":     f"{s['viability_pct']:.1f}",
            "WHO Score":      f"{s['who_score']:.1f}",
            "Grade":          s["who_category"],
        })

    st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

    # ── Bar charts ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    labels = [f"{s['sample_id']} ({s['session_id'][:5]})" for s in sessions]

    chart_data = pd.DataFrame({
        "Sample":        labels,
        "Progressive %": [s["progressive_pct"] for s in sessions],
        "Viability %":   [s["viability_pct"]   for s in sessions],
        "WHO Score":     [s["who_score"]        for s in sessions],
    }).set_index("Sample")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Progressive Motility (%)**")
        st.bar_chart(chart_data[["Progressive %"]], width="stretch", height=220)
    with c2:
        st.markdown("**Viability (%)**")
        st.bar_chart(chart_data[["Viability %"]], width="stretch", height=220)
    with c3:
        st.markdown("**WHO Score**")
        st.bar_chart(chart_data[["WHO Score"]], width="stretch", height=220)

    # ── Clear comparison ───────────────────────────────────────
    if st.button("✕ Clear Comparison", key="_db_clear_compare"):
        ss["_db_compare_set"] = set()
        st.rerun()


def _render_delete_confirm(db: DatabaseManager) -> None:
    """Render delete confirmation and permanently delete the selected session."""

    ss = st.session_state
    sid = ss.get("_db_confirm_delete")

    if not sid:
        return

    st.warning(
        f"⚠️ Are you sure you want to permanently delete "
        f"session `{sid}`?",
        icon=None,
    )

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button(
            "🗑 Yes, Delete",
            type="primary",
            width="stretch",
            key="_db_confirm_yes",
        ):
            try:
                sid = str(sid).strip()

                print("=" * 60)
                print("DATABASE DELETE")
                print("Session ID:", repr(sid))
                print("Session ID type:", type(sid))
                print("=" * 60)

                # Check that the session exists before deleting
                existing = db.get_session_detail(sid)

                if existing is None:
                    st.error(
                        f"Session `{sid}` was not found in the database."
                    )
                    ss.pop("_db_confirm_delete", None)
                    return

                # Perform deletion
                result = db.delete_session(sid)

                print("DELETE RESULT:", repr(result))

                # Clear session state
                ss.pop("_db_confirm_delete", None)
                ss.pop("_db_view_session_id", None)

                # Remove deleted session from comparison
                compare_set = ss.get("_db_compare_set", set())

                if isinstance(compare_set, set):
                    compare_set.discard(sid)

                ss["_db_compare_set"] = compare_set

                ss["_db_return_to_database"] = True

                st.success(
                    f"Session `{sid[:8]}...` deleted successfully."
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    f"❌ Failed to delete session `{sid}`: {exc}"
                )

                print("=" * 60)
                print("DELETE ERROR")
                print("Session ID:", repr(sid))
                print("Error:", repr(exc))
                print("=" * 60)

    with col_no:
        if st.button(
            "Cancel",
            width="stretch",
            key="_db_confirm_no",
        ):
            ss.pop("_db_confirm_delete", None)
            st.rerun()
# ══════════════════════════════════════════════════════════════
# Main page entry point
# ══════════════════════════════════════════════════════════════

def render_database_page(db: DatabaseManager) -> None:
    """
    Render the complete database browser page.

    This is called directly from the main app's sidebar navigation.
    It is also the page entry point when used as a Streamlit multi-page app.

    Parameters
    ----------
    db : DatabaseManager
    """

    # Fix button, icon, tooltip and download-button visibility
    _inject_database_button_styles()

    # Inject database page CSS
    st.markdown(_DB_CSS, unsafe_allow_html=True)

    ss = st.session_state

    # ── Page title ─────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:1.7rem;font-weight:800;color:{_CLR_TEXT_PRI};'
        f'margin-bottom:0.2rem;">🗄️  Session Database</div>'
        f'<div style="font-size:0.9rem;color:{_CLR_TEXT_SEC};'
        f'margin-bottom:1rem;">Browse, search, and compare all stored semen '
        f'analysis sessions.</div>',
        unsafe_allow_html=True,
    )

    # ── Delete confirmation (takes priority) ───────────────────
    _render_delete_confirm(db)

    # ── Statistics ─────────────────────────────────────────────
    _render_statistics(db)
    st.divider()

    # ── Detail view (if a session has been selected) ───────────
    view_id = ss.get("_db_view_session_id")
    if view_id:
        _render_session_detail(view_id, db)
        return

    # ── Comparison panel ───────────────────────────────────────
    if ss.get("_db_compare_set"):
        _render_comparison(db)
        st.divider()

    # ── Search + list ──────────────────────────────────────────
    search_results = _render_search_panel(db)

    if search_results is not None:
        _section(f"🔎 Search Results ({len(search_results)} found)")
        _render_session_table(search_results, db, table_id="search")
    else:
        total = db.count_sessions()
        _section(f"📋 All Sessions ({total} total)")

        # Type filter tabs
        tab_all, tab_single, tab_monitor = st.tabs(
            ["All", "Single Analysis", "Monitoring"]
        )
        with tab_all:
            rows = db.list_sessions(limit=200)
            _render_session_table(rows, db, table_id="all")

        with tab_single:
            rows = db.list_sessions(limit=200, session_type="single")
            _render_session_table(rows, db, table_id="single")

        with tab_monitor:
            rows = db.list_sessions(limit=200, session_type="monitoring")
            _render_session_table(rows, db, table_id="monitoring")

# ══════════════════════════════════════════════════════════════
# Streamlit multi-page entry (when file is run directly)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.utils.helpers import load_config  # noqa: E402

    st.set_page_config(
        page_title="Session Database · SpermVision Lab",
        page_icon="🗄️",
        layout="wide",
    )
    _cfg = load_config("configs/config.yaml")
    _db  = DatabaseManager(_cfg)
    render_database_page(_db)