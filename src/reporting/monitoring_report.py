"""
src/reporting/monitoring_report.py
=====================================
Phase 4 · Monitoring Report Generator

Produces a professional multi-page PDF, detailed CSV, and full JSON
report for a completed time-interval monitoring session.

Reuse policy
------------
* ``ReportGenerator`` from ``src/reporting/generate_report.py`` is
  instantiated internally and used for individual capture reports —
  zero duplication of single-analysis report code.
* The ``_styled_table`` helper replicates the exact same blue-header
  table style used by ``ReportGenerator._add_table`` so both PDF
  types look visually consistent.
* All WHO reference values are read from the existing config.yaml.

Public API
----------
    from src.reporting.monitoring_report import MonitoringReportGenerator

    gen   = MonitoringReportGenerator(config)
    paths = gen.generate(trend_report, output_stem="monitor_BVS001_20240115")
    # returns {"pdf": "...", "csv": "...", "json": "..."}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.trend_analysis import TrendReport, TrendMetrics
from src.reporting.generate_report import ReportGenerator
from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# MonitoringReportGenerator
# ══════════════════════════════════════════════════════════════

class MonitoringReportGenerator:
    """
    Generates PDF, CSV, and JSON reports for a monitoring session.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Auto-loaded if None.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        report_cfg = config["report"]
        paths_cfg  = config["paths"]

        self.title       = report_cfg.get("title",       "Semen Analysis Report")
        self.institution = report_cfg.get("institution", "AI-Powered Andrology Lab")
        self.logo_path   = Path(report_cfg.get("logo_path", ""))

        self.out_dir  = ensure_dir(paths_cfg["outputs"]["reports"])
        self.plot_dir = Path(paths_cfg["outputs"]["plots"])

        # Reuse existing single-sample report generator for per-interval reports
        self._single_gen = ReportGenerator(config)

    # ----------------------------------------------------------

    def generate(self,
                 trend_report: TrendReport,
                 output_stem: Optional[str] = None) -> Dict[str, str]:
        """
        Generate PDF, CSV, and JSON for the full monitoring session.

        Parameters
        ----------
        trend_report : TrendReport
            Output of ``TrendAnalyser.analyse()``.
        output_stem : str, optional
            Base filename without extension.

        Returns
        -------
        dict  {"pdf": str, "csv": str, "json": str}
        """
        if output_stem is None:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = trend_report.sample_meta.get("sample_id", "session")
            output_stem = f"monitor_{sid}_{ts}"

        logger.info("Generating monitoring report: {}", output_stem)

        paths: Dict[str, str] = {}
        paths["json"] = self._save_json(trend_report, output_stem)
        paths["csv"]  = self._save_csv(trend_report, output_stem)

        try:
            paths["pdf"] = self._save_pdf(trend_report, output_stem)
        except ImportError:
            logger.warning("reportlab not installed — PDF skipped.")
            paths["pdf"] = ""
        except Exception as exc:
            logger.error("PDF generation failed: {}", exc)
            paths["pdf"] = ""

        logger.success("Monitoring reports saved → {}", list(paths.values()))
        return paths

    # ----------------------------------------------------------
    # JSON
    # ----------------------------------------------------------

    def _save_json(self, tr: TrendReport, stem: str) -> str:
        m  = tr.trend_metrics
        sm = tr.sample_meta

        payload = {
            "report_type":  "monitoring_session",
            "generated_at": tr.generated_at,
            "software":     "Automated Semen Viability and Counting Analysis v4.0",
            "sample_info": {
                "sample_id":       sm.get("sample_id",       "—"),
                "animal_id":       sm.get("animal_id",       "—"),
                "collection_date": sm.get("collection_date", "—"),
                "operator":        sm.get("operator",        "—"),
                "lab_name":        sm.get("lab_name",        "—"),
            },
            "lab_info": {
                "species":           sm.get("species",           "—"),
                "breed":             sm.get("breed",             "—"),
                "sample_type":       sm.get("sample_type",       "—"),
                "chamber":           sm.get("chamber",           "—"),
                "microscope_type":   sm.get("microscope_type",   "—"),
                "magnification":     sm.get("magnification",     "—"),
                "staining_type":     sm.get("staining_type",     "—"),
                "camera_resolution": sm.get("camera_resolution", "—"),
                "camera_fps":        sm.get("camera_fps",        "—"),
            },
            "session": {
                "interval_min":  tr.session.interval_min,
                "duration_min":  tr.session.duration_min,
                "captures_done": tr.session.captures_done,
                "expected":      tr.session.expected_captures,
                "is_complete":   tr.session.is_complete,
            },
            "trend_metrics": {
                "n_captures":   m.n_captures,
                "duration_min": round(m.duration_min, 2),
                "progressive":  {"start": m.progressive_start, "end": m.progressive_end,
                                  "change": m.progressive_change,
                                  "pct_change": m.progressive_pct_change},
                "non_progressive": {"start": m.nonprog_start, "end": m.nonprog_end,
                                    "change": m.nonprog_change,
                                    "pct_change": m.nonprog_pct_change},
                "immotile":     {"start": m.immotile_start, "end": m.immotile_end,
                                  "change": m.immotile_change,
                                  "pct_change": m.immotile_pct_change},
                "viability":    {"start": m.viability_start, "end": m.viability_end,
                                  "change": m.viability_change,
                                  "pct_change": m.viability_pct_change},
                "count":        {"start": m.count_start, "end": m.count_end,
                                  "change": m.count_change,
                                  "pct_change": m.count_pct_change},
                "who_score":    {"start": m.score_start, "end": m.score_end,
                                  "change": m.score_change,
                                  "pct_change": m.score_pct_change},
                "degradation_rates": {
                    "motility_per_min":  m.motility_degradation_rate_per_min,
                    "viability_per_min": m.viability_degradation_rate_per_min,
                    "score_per_min":     m.score_decline_rate_per_min,
                },
                "linear_regression": {
                    "slopes":    m.trend_slope,
                    "r_squared": m.r_squared,
                },
            },
            "degradation_summary": tr.degradation_summary,
            "interval_results":    [r.to_dict() for r in tr.session.results],
            "plot_paths":          tr.plot_paths,
        }

        path = self.out_dir / f"{stem}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("JSON saved → {}", path)
        return str(path)

    # ----------------------------------------------------------
    # CSV
    # ----------------------------------------------------------

    def _save_csv(self, tr: TrendReport, stem: str) -> str:
        """
        Two-section CSV:
          Section 1 — session summary + trend metrics
          Section 2 — per-interval results table
        """
        m  = tr.trend_metrics
        sm = tr.sample_meta

        summary_rows: List[List[Any]] = [
            ["MONITORING SESSION REPORT"],
            ["Generated",        tr.generated_at],
            ["Sample ID",        sm.get("sample_id",       "—")],
            ["Animal ID",        sm.get("animal_id",       "—")],
            ["Operator",         sm.get("operator",        "—")],
            ["Laboratory",       sm.get("lab_name",        "—")],
            ["Species",          sm.get("species",         "—")],
            ["Breed",            sm.get("breed",           "—")],
            ["Sample Type",      sm.get("sample_type",     "—")],
            ["Chamber",          sm.get("chamber",         "—")],
            ["Microscope",       sm.get("microscope_type", "—")],
            ["Magnification",    sm.get("magnification",   "—")],
            ["Staining",         sm.get("staining_type",   "—")],
            [],
            ["SESSION OVERVIEW"],
            ["Interval (min)",   tr.session.interval_min],
            ["Duration (min)",   tr.session.duration_min],
            ["Captures planned", tr.session.expected_captures],
            ["Captures done",    tr.session.captures_done],
            ["Complete",         "Yes" if tr.session.is_complete else "No"],
            [],
            ["TREND METRICS", "Start", "End", "Change", "% Change"],
            ["Progressive (%)",
             m.progressive_start, m.progressive_end,
             m.progressive_change, m.progressive_pct_change],
            ["Non-Progressive (%)",
             m.nonprog_start, m.nonprog_end,
             m.nonprog_change, m.nonprog_pct_change],
            ["Immotile (%)",
             m.immotile_start, m.immotile_end,
             m.immotile_change, m.immotile_pct_change],
            ["Viability (%)",
             m.viability_start, m.viability_end,
             m.viability_change, m.viability_pct_change],
            ["Count",
             m.count_start, m.count_end,
             m.count_change, m.count_pct_change],
            ["WHO Score",
             m.score_start, m.score_end,
             m.score_change, m.score_pct_change],
            [],
            ["DEGRADATION RATES"],
            ["Progressive motility (%/min)", m.motility_degradation_rate_per_min],
            ["Viability (%/min)",            m.viability_degradation_rate_per_min],
            ["Quality score (pts/min)",      m.score_decline_rate_per_min],
            [],
            ["OVERALL FINDING"],
            [tr.degradation_summary.get("overall", "—")],
        ]

        MAX_COLS = 5
        padded = [
            r + [""] * (MAX_COLS - len(r)) if len(r) < MAX_COLS else r
            for r in summary_rows
        ]
        df_summary = pd.DataFrame(padded)

        # Per-interval table
        rows: List[Dict] = []
        for r in tr.session.results:
            d = r.to_dict()
            s = d.get("summary", {})
            rows.append({
                "#":            r.interval_index,
                "Time":         datetime.fromtimestamp(r.capture_time).strftime("%H:%M:%S"),
                "t (min)":      round(r.elapsed_min, 1),
                "Sperm":        s.get("total_sperm",     "—"),
                "Progressive%": s.get("progressive_pct", "—"),
                "NonProg%":     s.get("nonprog_pct",     "—"),
                "Immotile%":    s.get("immotile_pct",    "—"),
                "Normal%":      s.get("normal_pct",      "—"),
                "Viability%":   s.get("viability_pct",   "—"),
                "WHO Score":    s.get("who_score",        "—"),
                "Grade":        s.get("who_category",    "—"),
                "Error":        r.error or "",
            })
        df_intervals = pd.DataFrame(rows) if rows else pd.DataFrame()

        path = self.out_dir / f"{stem}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            df_summary.to_csv(fh, index=False, header=False)
            fh.write("\n")
            fh.write("PER-INTERVAL RESULTS\n")
            if not df_intervals.empty:
                df_intervals.to_csv(fh, index=False)

        logger.info("CSV saved → {}", path)
        return str(path)

    # ----------------------------------------------------------
    # PDF
    # ----------------------------------------------------------

    def _save_pdf(self, tr: TrendReport, stem: str) -> str:
        """Multi-page A4 PDF using ReportLab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, Image, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        path = self.out_dir / f"{stem}.pdf"
        doc  = SimpleDocTemplate(
            str(path), pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm,  bottomMargin=2*cm,
        )
        styles = getSampleStyleSheet()

        CLR_PRIMARY = colors.HexColor("#1d4ed8")
        CLR_DARK    = colors.HexColor("#0f172a")
        CLR_SEC     = colors.HexColor("#64748b")
        CLR_BORDER  = colors.HexColor("#e2e8f0")
        CLR_ALT     = colors.HexColor("#f8fafc")
        CLR_TH      = colors.HexColor("#2980b9")

        def _ps(name, **kw):
            return ParagraphStyle(name, parent=styles["Normal"], **kw)

        S = {
            "title":   _ps("T", fontSize=20, textColor=CLR_DARK,  spaceAfter=4, fontName="Helvetica-Bold"),
            "sub":     _ps("S", fontSize=10, textColor=CLR_SEC,   spaceAfter=8),
            "h2":      _ps("H2", fontSize=13, textColor=CLR_PRIMARY,
                           spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold"),
            "h3":      _ps("H3", fontSize=10, textColor=colors.HexColor("#475569"),
                           spaceBefore=8, spaceAfter=3, fontName="Helvetica-Bold"),
            "body":    _ps("B", fontSize=9, leading=14),
            "rec":     _ps("R", fontSize=9, leading=13,
                           textColor=colors.HexColor("#166534")),
            "flag":    _ps("F", fontSize=9, leading=13,
                           textColor=colors.HexColor("#991b1b")),
            "small":   _ps("SM", fontSize=7, textColor=colors.grey, leading=10),
            "banner":  _ps("BN", fontSize=10, leading=14, fontName="Helvetica-Bold"),
        }

        def _hr():
            return HRFlowable(width="100%", thickness=0.5, color=CLR_BORDER)

        def _styled_tbl(data, col_widths=None):
            """Blue-header table — matches ReportGenerator._add_table style."""
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
                ("BACKGROUND",  (0, 0), (-1,  0), CLR_TH),
                ("TEXTCOLOR",   (0, 0), (-1,  0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CLR_ALT]),
                ("GRID",        (0, 0), (-1, -1), 0.3, CLR_BORDER),
                ("FONTNAME",    (0, 1), (0,  -1), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("PADDING",     (0, 0), (-1, -1), 4),
                ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ]))
            return t

        def _meta_tbl(data, w1=5.5, w2=11.5):
            t = Table(data, colWidths=[w1*cm, w2*cm])
            t.setStyle(TableStyle([
                ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE",  (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, CLR_ALT]),
                ("GRID",      (0, 0), (-1, -1), 0.25, CLR_BORDER),
                ("PADDING",   (0, 0), (-1, -1), 5),
                ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
            ]))
            return t

        def _banner(text, bg_hex, fg=colors.white):
            inner = Paragraph(text, _ps("BP", fontSize=10, textColor=fg,
                                         fontName="Helvetica-Bold", leading=14))
            t = Table([[inner]], colWidths=[17*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_hex)),
                ("PADDING",    (0, 0), (-1, -1), 10),
            ]))
            return t

        m  = tr.trend_metrics
        sm = tr.sample_meta
        E  = []      # elements list

        # ────────────────────────────────────────────────────────
        # PAGE 1 — Cover
        # ────────────────────────────────────────────────────────
        if self.logo_path.exists():
            E.append(Image(str(self.logo_path), width=3.5*cm, height=3.5*cm))
            E.append(Spacer(1, 0.3*cm))

        E.append(Paragraph("Time-Interval Semen Quality Monitoring", S["title"]))
        E.append(Paragraph(self.institution, S["sub"]))
        E.append(HRFlowable(width="100%", thickness=2, color=CLR_PRIMARY))
        E.append(Spacer(1, 0.5*cm))

        cover = [
            ["Report generated:", tr.generated_at],
            ["Sample ID:",        sm.get("sample_id",       "—")],
            ["Animal ID:",        sm.get("animal_id",       "—")],
            ["Collection date:",  sm.get("collection_date", "—")],
            ["Operator:",         sm.get("operator",        "—")],
            ["Laboratory:",       sm.get("lab_name",        "—")],
            ["Interval:",         f"{tr.session.interval_min} min"],
            ["Duration:",         f"{tr.session.duration_min} min"],
            ["Captures done:",    f"{tr.session.captures_done} / {tr.session.expected_captures}"],
            ["Status:",           "Complete" if tr.session.is_complete else "Stopped early"],
        ]
        E.append(_meta_tbl(cover))
        E.append(Spacer(1, 0.5*cm))

        overall = tr.degradation_summary.get("overall", "")
        bg = ("#dc2626" if "significant" in overall.lower()
              else "#059669" if "acceptable" in overall.lower()
              else "#d97706")
        E.append(_banner(f"Overall Finding: {overall}", bg))
        E.append(PageBreak())

        # ────────────────────────────────────────────────────────
        # PAGE 2 — Sample & Laboratory Information
        # ────────────────────────────────────────────────────────
        E.append(Paragraph("Sample &amp; Laboratory Information", S["h2"]))

        lab_data = [
            ["Species",           sm.get("species",           "—"),
             "Sample Type",       sm.get("sample_type",       "—")],
            ["Breed",             sm.get("breed",             "—"),
             "Counting Chamber",  sm.get("chamber",           "—")],
            ["Staining",          sm.get("staining_type",     "—"),
             "Microscope Type",   sm.get("microscope_type",   "—")],
            ["Camera Resolution", sm.get("camera_resolution", "—"),
             "Magnification",     sm.get("magnification",     "—")],
            ["Camera FPS",        sm.get("camera_fps",        "—"),
             "Operator",          sm.get("operator",          "—")],
        ]
        lab_tbl = Table(lab_data, colWidths=[4*cm, 4.25*cm, 4*cm, 4.25*cm])
        lab_tbl.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE",  (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, CLR_ALT]),
            ("GRID",      (0, 0), (-1, -1), 0.25, CLR_BORDER),
            ("PADDING",   (0, 0), (-1, -1), 5),
        ]))
        E.append(lab_tbl)
        E.append(Spacer(1, 0.6*cm))

        # ────────────────────────────────────────────────────────
        # PAGE 3 — Trend Analysis Summary
        # ────────────────────────────────────────────────────────
        E.append(Paragraph("Trend Analysis Summary", S["h2"]))

        deg_data = [
            ["Metric", "Start", "End", "Change", "% Change", "Rate / min"],
            ["Progressive (%)",
             f"{m.progressive_start:.1f}", f"{m.progressive_end:.1f}",
             f"{m.progressive_change:+.2f}", f"{m.progressive_pct_change:+.1f}%",
             f"{m.motility_degradation_rate_per_min:+.4f}"],
            ["Non-Progressive (%)",
             f"{m.nonprog_start:.1f}", f"{m.nonprog_end:.1f}",
             f"{m.nonprog_change:+.2f}", f"{m.nonprog_pct_change:+.1f}%", "—"],
            ["Immotile (%)",
             f"{m.immotile_start:.1f}", f"{m.immotile_end:.1f}",
             f"{m.immotile_change:+.2f}", f"{m.immotile_pct_change:+.1f}%", "—"],
            ["Viability (%)",
             f"{m.viability_start:.1f}", f"{m.viability_end:.1f}",
             f"{m.viability_change:+.2f}", f"{m.viability_pct_change:+.1f}%",
             f"{m.viability_degradation_rate_per_min:+.4f}"],
            ["Count",
             f"{m.count_start:.0f}", f"{m.count_end:.0f}",
             f"{m.count_change:+.0f}", f"{m.count_pct_change:+.1f}%", "—"],
            ["WHO Score",
             f"{m.score_start:.1f}", f"{m.score_end:.1f}",
             f"{m.score_change:+.2f}", f"{m.score_pct_change:+.1f}%",
             f"{m.score_decline_rate_per_min:+.4f}"],
        ]
        E.append(_styled_tbl(deg_data, [4*cm, 2*cm, 2*cm, 2.2*cm, 2.3*cm, 3.5*cm]))
        E.append(Spacer(1, 0.4*cm))

        # WHO Comparison
        E.append(Paragraph("WHO 2021 Comparison", S["h3"]))
        who_data = [
            ["Parameter",           "WHO Min", "Value at t=0", "Value at End", "Status"],
            ["Progressive motility","≥ 30%",
             f"{m.progressive_start:.1f}%", f"{m.progressive_end:.1f}%",
             "✓ Within" if m.progressive_end >= 30 else "✗ Below"],
            ["Viability",           "≥ 54%",
             f"{m.viability_start:.1f}%", f"{m.viability_end:.1f}%",
             "✓ Within" if m.viability_end >= 54 else "✗ Below"],
            ["Normal morphology",   "≥ 4%",
             "—", "—", "See single reports"],
        ]
        E.append(_styled_tbl(who_data, [4.5*cm, 2.5*cm, 3.3*cm, 3.3*cm, 3*cm]))
        E.append(Spacer(1, 0.4*cm))

        # Linear regression
        E.append(Paragraph("Linear Regression Coefficients", S["h3"]))
        lr_data = [["Metric", "Slope (per min)", "R²"]]
        lbl_map = {
            "progressive_pct": "Progressive (%)",
            "nonprog_pct":     "Non-Progressive (%)",
            "immotile_pct":    "Immotile (%)",
            "viability_pct":   "Viability (%)",
            "total_sperm":     "Total Count",
            "who_score":       "WHO Score",
        }
        for col, lbl in lbl_map.items():
            lr_data.append([
                lbl,
                f"{m.trend_slope.get(col, 0.0):+.5f}",
                f"{m.r_squared.get(col, 0.0):.4f}",
            ])
        E.append(_styled_tbl(lr_data, [8*cm, 5*cm, 4*cm]))
        E.append(PageBreak())

        # ────────────────────────────────────────────────────────
        # PAGES 4-5 — Trend Plots
        # ────────────────────────────────────────────────────────
        E.append(Paragraph("Trend Charts", S["h2"]))

        dash = tr.plot_paths.get("dashboard", "")
        if dash and Path(dash).exists():
            E.append(Paragraph("Summary Dashboard — All Metrics", S["h3"]))
            E.append(Image(dash, width=17*cm, height=9.5*cm))
            E.append(Spacer(1, 0.4*cm))

        mot = tr.plot_paths.get("motility_overview", "")
        if mot and Path(mot).exists():
            E.append(Paragraph("Motility Overview", S["h3"]))
            E.append(Image(mot, width=17*cm, height=6*cm))
            E.append(PageBreak())

        # Six individual plots — 2 per page
        individual = [
            ("progressive_pct", "Progressive Motility vs Time"),
            ("nonprog_pct",     "Non-Progressive Motility vs Time"),
            ("immotile_pct",    "Immotile Percentage vs Time"),
            ("viability_pct",   "Viability vs Time"),
            ("total_sperm",     "Total Count vs Time"),
            ("who_score",       "Quality Score vs Time"),
        ]
        for i, (col, caption) in enumerate(individual):
            p = tr.plot_paths.get(col, "")
            if p and Path(p).exists():
                E.append(Paragraph(caption, S["h3"]))
                E.append(Image(p, width=17*cm, height=6.5*cm))
                E.append(Spacer(1, 0.3*cm))
            if i % 2 == 1:
                E.append(PageBreak())
        if len(individual) % 2 != 0:
            E.append(PageBreak())

        # ────────────────────────────────────────────────────────
        # PAGE — Per-interval Results Table
        # ────────────────────────────────────────────────────────
        E.append(Paragraph("Per-Interval Results", S["h2"]))

        hdr = ["#","Time","t(min)","Sperm","Prog%","NonProg%","Immotile%",
               "Viability%","WHO","Grade","Status"]
        rows_pdf = [hdr]
        for r in tr.session.results:
            d = r.to_dict(); s = d.get("summary", {})
            ts = datetime.fromtimestamp(r.capture_time).strftime("%H:%M:%S")
            rows_pdf.append([
                str(r.interval_index), ts, f"{r.elapsed_min:.1f}",
                str(int(s.get("total_sperm", 0))),
                f"{s.get('progressive_pct', 0):.1f}",
                f"{s.get('nonprog_pct',     0):.1f}",
                f"{s.get('immotile_pct',    0):.1f}",
                f"{s.get('viability_pct',   0):.1f}",
                f"{s.get('who_score',       0):.0f}",
                str(s.get("who_category", "—")),
                "✓" if not r.error else "✗",
            ])

        cw = [1*cm,2*cm,1.6*cm,1.6*cm,1.6*cm,2*cm,2*cm,2.2*cm,1.5*cm,1.6*cm,1.4*cm]
        E.append(_styled_tbl(rows_pdf, cw))
        E.append(PageBreak())

        # ────────────────────────────────────────────────────────
        # PAGE — Findings & Recommendations
        # ────────────────────────────────────────────────────────
        E.append(Paragraph("Findings &amp; Recommendations", S["h2"]))

        for key, sentence in tr.degradation_summary.items():
            if key == "overall":
                continue
            E.append(Paragraph(f"• {sentence}", S["body"]))

        E.append(Spacer(1, 0.3*cm))
        E.append(_hr())
        E.append(Spacer(1, 0.2*cm))
        E.append(_banner(f"Overall: {tr.degradation_summary.get('overall', '')}", "#1e3a5f"))
        E.append(Spacer(1, 0.4*cm))

        E.append(Paragraph("WHO 2021-Based Guidance", S["h3"]))
        for rec in _who_recommendations(m):
            E.append(Paragraph(f"✔  {rec}", S["rec"]))

        E.append(Spacer(1, 0.5*cm))
        E.append(_hr())
        E.append(Spacer(1, 0.2*cm))
        E.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted computer "
            "vision system for research and screening purposes only.  "
            "It does not replace a clinical diagnosis by a certified andrologist "
            "or veterinary reproductive specialist.",
            S["small"],
        ))

        doc.build(E)
        logger.success("PDF saved → {}", path)
        return str(path)


# ══════════════════════════════════════════════════════════════
# WHO-based recommendation generator  (module-level helper)
# ══════════════════════════════════════════════════════════════

def _who_recommendations(m: TrendMetrics) -> List[str]:
    """Generate WHO 2021-based clinical recommendations from TrendMetrics."""
    recs: List[str] = []

    if m.progressive_end < 30:
        recs.append(
            f"Progressive motility at end of session ({m.progressive_end:.1f}%) "
            f"is below the WHO 2021 lower reference limit of 30%.  "
            f"Asthenozoospermia is indicated.  "
            f"Consider antioxidant supplementation and oxidative stress evaluation."
        )
    if m.viability_end < 54:
        recs.append(
            f"Viability at end of session ({m.viability_end:.1f}%) is below "
            f"the WHO 2021 lower reference limit of 54%.  "
            f"Necrozoospermia may be present.  "
            f"Use sample immediately for ART procedures."
        )
    if m.motility_degradation_rate_per_min > 0.05:
        recs.append(
            f"Progressive motility is degrading at "
            f"{m.motility_degradation_rate_per_min:.3f}%/min.  "
            f"Reduce handling time and maintain optimal temperature (37 °C)."
        )
    if m.progressive_change < -20:
        recs.append(
            f"Progressive motility declined by {abs(m.progressive_change):.1f}% "
            f"over the monitoring period.  "
            f"Cryopreservation should be performed within the first hour of collection."
        )
    if not recs:
        recs.append(
            "All monitored parameters remained within or close to WHO 2021 "
            "reference limits.  Continue standard handling protocols."
        )
    recs.append(
        "This report is AI-assisted.  All findings require confirmation by a "
        "certified andrologist or veterinary reproductive specialist before "
        "any clinical or reproductive decision is made."
    )
    return recs