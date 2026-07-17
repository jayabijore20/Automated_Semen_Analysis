"""
src/reporting/generate_report.py
==================================
Phase 8 · Report Generation — PDF + CSV + JSON

Generates a professional semen analysis report using ReportLab.

The report includes:
  - Patient / session metadata
  - Detection summary
  - Motility breakdown (table + pie reference)
  - Morphology breakdown
  - Viability prediction
  - WHO quality score and grade
  - WHO threshold flags
  - Clinical recommendations
  - Embedded output plots (if they exist)

Usage
-----
    from src.reporting.generate_report import ReportGenerator

    gen = ReportGenerator(config)
    paths = gen.generate(
        video_name="sample.mp4",
        detection_summary={"count": 45, ...},
        motility_summary={...},
        morphology_summary={...},
        viability_pct=71.0,
        quality_result={"score": 82.5, "category": "Excellent", ...},
        output_stem="sample_report"
    )
    print(paths)  # {"pdf": "...", "csv": "...", "json": "..."}
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Report Generator
# ══════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Generates PDF, CSV, and JSON semen analysis reports.

    Parameters
    ----------
    config : dict, optional
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        report_cfg = config["report"]
        paths_cfg  = config["paths"]

        self.title       = report_cfg["title"]
        self.institution = report_cfg["institution"]
        self.logo_path   = Path(report_cfg.get("logo_path", ""))

        self.out_dir  = ensure_dir(paths_cfg["outputs"]["reports"])
        self.plot_dir = Path(paths_cfg["outputs"]["plots"])

    # ----------------------------------------------------------

    def generate(self,
                 video_name: str,
                 detection_summary: Dict,
                 motility_summary: Dict,
                 morphology_summary: Dict,
                 viability_pct: float,
                 quality_result: Dict,
                 output_stem: Optional[str] = None) -> Dict[str, str]:
        """
        Generate all report formats and return file paths.

        Parameters
        ----------
        video_name : str
            Source video filename.
        detection_summary : dict
            Keys: count (int), n_frames (int), unique_tracks (int)
        motility_summary : dict
            Keys from MotilityAnalyser._build_summary():
            progressive_pct, nonprogressive_pct, immotile_pct,
            mean_VCL, mean_VSL, mean_VAP, total_tracks
        morphology_summary : dict
            Keys from MorphologyClassifier.batch_summary():
            normal_pct, abnormal_pct, total (int)
        viability_pct : float
        quality_result : dict
            From QualityScorer.score():
            score, category, subscores, who_flags, recommendations
        output_stem : str, optional
            Base filename for output files.

        Returns
        -------
        dict
            {"pdf": str, "csv": str, "json": str}
        """
        if output_stem is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_stem = f"semen_report_{ts}"

        # Build the unified data payload
        payload = self._build_payload(
            video_name, detection_summary, motility_summary,
            morphology_summary, viability_pct, quality_result
        )

        paths: Dict[str, str] = {}

        # ── JSON ───────────────────────────────────────────────
        json_path = self._save_json(payload, output_stem)
        paths["json"] = json_path

        # ── CSV ────────────────────────────────────────────────
        csv_path = self._save_csv(payload, output_stem)
        paths["csv"] = csv_path

        # ── PDF ────────────────────────────────────────────────
        try:
            pdf_path = self._save_pdf(payload, output_stem)
            paths["pdf"] = pdf_path
        except ImportError:
            logger.warning("reportlab not installed — PDF skipped. Install with: pip install reportlab")
            paths["pdf"] = ""
        except Exception as exc:
            logger.error("PDF generation failed: {}", exc)
            paths["pdf"] = ""

        logger.success("Reports saved → {}", list(paths.values()))
        return paths

    # ----------------------------------------------------------

    def _build_payload(self,
                       video_name: str,
                       detection_summary: Dict,
                       motility_summary: Dict,
                       morphology_summary: Dict,
                       viability_pct: float,
                       quality_result: Dict) -> Dict[str, Any]:
        """Combine all results into a single serialisable dict."""
        return {
            "metadata": {
                "report_title":  self.title,
                "institution":   self.institution,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "video_source":  video_name,
                "software":      "Automated Semen Viability and Counting Analysis v1.0",
            },
            "detection": {
                "total_sperm_count": detection_summary.get("count", 0),
                "frames_processed":  detection_summary.get("n_frames", 0),
                "unique_tracks":     detection_summary.get("unique_tracks", 0),
            },
            "motility": {
                "progressive_pct":      round(motility_summary.get("progressive_pct",    0.0), 2),
                "nonprogressive_pct":   round(motility_summary.get("nonprogressive_pct", 0.0), 2),
                "immotile_pct":         round(motility_summary.get("immotile_pct",       0.0), 2),
                "mean_VCL_um_s":        round(motility_summary.get("mean_VCL",          0.0), 2),
                "mean_VSL_um_s":        round(motility_summary.get("mean_VSL",          0.0), 2),
                "mean_VAP_um_s":        round(motility_summary.get("mean_VAP",          0.0), 2),
                "total_tracks":         motility_summary.get("total_tracks", 0),
            },
            "morphology": {
                "normal_pct":    round(morphology_summary.get("normal_pct",   0.0), 2),
                "abnormal_pct":  round(morphology_summary.get("abnormal_pct", 0.0), 2),
                "total_classified": morphology_summary.get("total", 0),
            },
            "viability": {
                "predicted_pct": round(viability_pct, 2),
            },
            "quality": {
                "composite_score": quality_result.get("score", 0.0),
                "category":        quality_result.get("category", "Unknown"),
                "subscores":       quality_result.get("subscores", {}),
                "who_flags":       quality_result.get("who_flags", []),
                "recommendations": quality_result.get("recommendations", []),
            },
        }

    # ----------------------------------------------------------

    def _save_json(self, payload: Dict, stem: str) -> str:
        """Save payload as JSON."""
        path = self.out_dir / f"{stem}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("JSON report saved → {}", path)
        return str(path)

    # ----------------------------------------------------------

    def _save_csv(self, payload: Dict, stem: str) -> str:
        """Save flat key-value summary as CSV."""
        rows = [
            # Metadata
            ("Report Title",        payload["metadata"]["report_title"]),
            ("Institution",         payload["metadata"]["institution"]),
            ("Analysis Date",       payload["metadata"]["analysis_date"]),
            ("Video Source",        payload["metadata"]["video_source"]),
            # Detection
            ("Total Sperm Count",   payload["detection"]["total_sperm_count"]),
            ("Frames Processed",    payload["detection"]["frames_processed"]),
            ("Unique Tracks",       payload["detection"]["unique_tracks"]),
            # Motility
            ("Progressive (%)",     payload["motility"]["progressive_pct"]),
            ("Non-Progressive (%)", payload["motility"]["nonprogressive_pct"]),
            ("Immotile (%)",        payload["motility"]["immotile_pct"]),
            ("Mean VCL (µm/s)",     payload["motility"]["mean_VCL_um_s"]),
            ("Mean VSL (µm/s)",     payload["motility"]["mean_VSL_um_s"]),
            ("Mean VAP (µm/s)",     payload["motility"]["mean_VAP_um_s"]),
            # Morphology
            ("Normal Morphology (%)",   payload["morphology"]["normal_pct"]),
            ("Abnormal Morphology (%)", payload["morphology"]["abnormal_pct"]),
            # Viability
            ("Predicted Viability (%)", payload["viability"]["predicted_pct"]),
            # Quality
            ("WHO Quality Score",   payload["quality"]["composite_score"]),
            ("WHO Category",        payload["quality"]["category"]),
            ("WHO Flags",           "; ".join(payload["quality"]["who_flags"])),
            ("Recommendations",     " | ".join(payload["quality"]["recommendations"])),
        ]
        df = pd.DataFrame(rows, columns=["Parameter", "Value"])
        path = self.out_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        logger.info("CSV report saved → {}", path)
        return str(path)

    # ----------------------------------------------------------

    def _save_pdf(self, payload: Dict, stem: str) -> str:
        """
        Generate a multi-page PDF report using ReportLab.

        Parameters
        ----------
        payload : dict
        stem : str

        Returns
        -------
        str
            Path to saved PDF.

        Raises
        ------
        ImportError
            If reportlab is not installed.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image, PageBreak, Paragraph, SimpleDocTemplate,
            Spacer, Table, TableStyle,
        )

        path = self.out_dir / f"{stem}.pdf"
        doc  = SimpleDocTemplate(str(path), pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "TitleStyle", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#2c3e50"),
            spaceAfter=6,
        )
        heading_style = ParagraphStyle(
            "Heading", parent=styles["Heading2"],
            fontSize=13, textColor=colors.HexColor("#2980b9"),
            spaceBefore=12, spaceAfter=4,
            borderPad=(0, 0, 2, 0),
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, leading=14,
        )
        flag_style = ParagraphStyle(
            "Flag", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#c0392b"), leading=14,
        )
        rec_style = ParagraphStyle(
            "Rec", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#27ae60"), leading=14,
        )

        # Colour for score category
        cat_colors = {
            "Excellent": "#27ae60",
            "Good":      "#2ecc71",
            "Average":   "#f39c12",
            "Poor":      "#e74c3c",
        }

        elements = []

        # ── Cover / Header ─────────────────────────────────────
        elements.append(Paragraph(self.title, title_style))
        elements.append(Paragraph(self.institution, styles["Heading3"]))
        elements.append(Spacer(1, 0.3*cm))

        # Metadata table
        meta = payload["metadata"]
        det  = payload["detection"]
        meta_data = [
            ["Analysis Date:", meta["analysis_date"]],
            ["Video Source:",  meta["video_source"]],
            ["Software:",      meta["software"]],
            ["Frames:",        str(det["frames_processed"])],
            ["Unique Tracks:", str(det["unique_tracks"])],
        ]
        meta_tbl = Table(meta_data, colWidths=[5*cm, 12*cm])
        meta_tbl.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
            ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#ecf0f1")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.white, colors.HexColor("#ecf0f1")]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#bdc3c7")),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))
        elements.append(meta_tbl)
        elements.append(Spacer(1, 0.5*cm))

        # ── Quality Score Banner ───────────────────────────────
        quality = payload["quality"]
        cat_color = cat_colors.get(quality["category"], "#7f8c8d")
        elements.append(Paragraph("WHO Quality Score", heading_style))
        score_data = [[
            f"Score: {quality['composite_score']:.1f} / 100",
            f"Category: {quality['category']}"
        ]]
        score_tbl = Table(score_data, colWidths=[8.5*cm, 8.5*cm])
        score_tbl.setStyle(TableStyle([
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 14),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("BACKGROUND", (0,0), (0,0),   colors.HexColor("#2c3e50")),
            ("BACKGROUND", (1,0), (1,0),   colors.HexColor(cat_color)),
            ("TEXTCOLOR",  (0,0), (-1,-1), colors.white),
            ("PADDING",    (0,0), (-1,-1), 10),
        ]))
        elements.append(score_tbl)
        elements.append(Spacer(1, 0.4*cm))

        # ── Sub-scores ─────────────────────────────────────────
        sub = quality.get("subscores", {})
        sub_data = [["Parameter", "Sub-Score (0-100)", "Weight"]] + [
            ["Count",      f"{sub.get('count',      0):.1f}", "25%"],
            ["Motility",   f"{sub.get('motility',   0):.1f}", "35%"],
            ["Morphology", f"{sub.get('morphology', 0):.1f}", "20%"],
            ["Viability",  f"{sub.get('viability',  0):.1f}", "20%"],
        ]
        self._add_table(elements, sub_data, heading_style, "Quality Sub-Scores")

        # ── Detection ──────────────────────────────────────────
        det_data = [
            ["Parameter", "Value"],
            ["Total Sperm Count", str(det.get("total_sperm_count", 0))],
            ["Frames Processed",  str(det.get("frames_processed", 0))],
            ["Unique Tracks",     str(det.get("unique_tracks", 0))],
        ]
        self._add_table(elements, det_data, heading_style, "Detection Summary")

        # ── Motility ───────────────────────────────────────────
        mot = payload["motility"]
        mot_data = [
            ["Parameter", "Value"],
            ["Progressive",          f"{mot['progressive_pct']:.1f}%"],
            ["Non-Progressive",       f"{mot['nonprogressive_pct']:.1f}%"],
            ["Immotile",              f"{mot['immotile_pct']:.1f}%"],
            ["Mean VCL (µm/s)",       f"{mot['mean_VCL_um_s']:.2f}"],
            ["Mean VSL (µm/s)",       f"{mot['mean_VSL_um_s']:.2f}"],
            ["Mean VAP (µm/s)",       f"{mot['mean_VAP_um_s']:.2f}"],
            ["Total Analysed Tracks", str(mot["total_tracks"])],
        ]
        self._add_table(elements, mot_data, heading_style, "Motility Analysis (CASA)")

        # ── Morphology ─────────────────────────────────────────
        morph = payload["morphology"]
        morph_data = [
            ["Parameter", "Value"],
            ["Normal Morphology",   f"{morph['normal_pct']:.1f}%"],
            ["Abnormal Morphology", f"{morph['abnormal_pct']:.1f}%"],
            ["Total Classified",    str(morph["total_classified"])],
        ]
        self._add_table(elements, morph_data, heading_style, "Morphology Classification")

        # ── Viability ──────────────────────────────────────────
        via = payload["viability"]
        via_data = [
            ["Parameter", "Value"],
            ["Predicted Viability", f"{via['predicted_pct']:.1f}%"],
            ["WHO Reference Min",   "54%"],
        ]
        self._add_table(elements, via_data, heading_style, "Viability Estimation")

        # ── Embedded plots ─────────────────────────────────────
        plot_files = [
            ("motility_analysis.png",  "Motility Analysis Chart"),
            ("morphology_confusion_matrix.png", "Morphology Confusion Matrix"),
            ("viability_predictions.png", "Viability Model Predictions"),
        ]
        for fname, caption in plot_files:
            plot_path = self.plot_dir / fname
            if plot_path.exists():
                elements.append(Paragraph(caption, heading_style))
                elements.append(Image(str(plot_path), width=15*cm, height=8*cm))
                elements.append(Spacer(1, 0.3*cm))

        # ── WHO Flags ──────────────────────────────────────────
        flags = quality.get("who_flags", [])
        if flags:
            elements.append(Paragraph("WHO Threshold Violations", heading_style))
            for flag in flags:
                elements.append(Paragraph(f"⚠  {flag}", flag_style))
            elements.append(Spacer(1, 0.3*cm))

        # ── Recommendations ────────────────────────────────────
        recs = quality.get("recommendations", [])
        if recs:
            elements.append(Paragraph("Clinical Recommendations", heading_style))
            for rec in recs:
                elements.append(Paragraph(f"•  {rec}", rec_style))
            elements.append(Spacer(1, 0.3*cm))

        # ── Disclaimer ─────────────────────────────────────────
        elements.append(Spacer(1, 0.5*cm))
        elements.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted computer vision system "
            "and is intended for research and screening purposes only. "
            "It does not replace a clinical diagnosis by a certified andrologist.",
            ParagraphStyle("Disclaimer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey, leading=12)
        ))

        # ── Build PDF ──────────────────────────────────────────
        doc.build(elements)
        logger.success("PDF report saved → {}", path)
        return str(path)

    # ----------------------------------------------------------

    def _add_table(self, elements: list, data: list,
                   heading_style, title: str) -> None:
        """Helper: append a titled two-column table to the element list."""
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

        elements.append(Paragraph(title, heading_style))
        n_cols = len(data[0]) if data else 2
        col_w  = 17.0 / n_cols
        tbl = Table(data, colWidths=[col_w * cm] * n_cols)
        tbl.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#2980b9")),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#eaf4fb")]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#bdc3c7")),
            ("FONTNAME",    (0,1), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("PADDING",     (0,0), (-1,-1), 5),
            ("ALIGN",       (1,1), (-1,-1), "RIGHT"),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 0.3*cm))