"""
src/analysis/trend_analysis.py
================================
Phase 4 · Trend Analysis Engine

Consumes a completed ``MonitorSession`` (list of ``CaptureResult`` objects)
and produces:

1. ``TrendMetrics`` — all degradation / decline figures computed from the
   time-series data.
2. Six publication-quality PNG trend plots saved to
   ``outputs/plots/monitor_<stem>/``.
3. A ``TrendReport`` dataclass that bundles everything needed by the
   monitoring report generator.

No Streamlit, no camera, no AI pipeline code lives here.
This module is purely data → metrics → charts.

Public API
----------
    from src.analysis.trend_analysis import TrendAnalyser

    analyser = TrendAnalyser(config)
    report   = analyser.analyse(session, sample_meta)

    report.trend_metrics          # TrendMetrics dataclass
    report.plot_paths             # dict  {metric_name: Path}
    report.interval_df            # pd.DataFrame — one row per capture
    report.degradation_summary    # dict  of readable strings for report
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.camera.monitor import CaptureResult, MonitorSession, _extract_summary
from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# WHO 2021 reference lower limits used for horizontal reference lines
_WHO_REFS: Dict[str, float] = {
    "progressive_pct": 30.0,
    "normal_pct":      4.0,
    "viability_pct":   54.0,
    "who_score":       0.0,     # no fixed WHO floor — shown as informational
}

# Colour palette for the six trend plots
_COLOURS: Dict[str, str] = {
    "progressive_pct":  "#10b981",   # emerald
    "nonprog_pct":      "#f59e0b",   # amber
    "immotile_pct":     "#ef4444",   # red
    "viability_pct":    "#3b82f6",   # blue
    "total_sperm":      "#8b5cf6",   # violet
    "who_score":        "#06b6d4",   # cyan
}

_LABELS: Dict[str, str] = {
    "progressive_pct":  "Progressive Motility (%)",
    "nonprog_pct":      "Non-Progressive Motility (%)",
    "immotile_pct":     "Immotile (%)",
    "viability_pct":    "Viability (%)",
    "total_sperm":      "Total Sperm Count",
    "who_score":        "WHO Quality Score",
}


# ══════════════════════════════════════════════════════════════
# TrendMetrics
# ══════════════════════════════════════════════════════════════

@dataclass
class TrendMetrics:
    """
    Scalar summary of degradation / change over the monitoring session.

    All ``_change`` fields are absolute (end - start).
    All ``_pct_change`` fields are percentage relative to the first value.
    Negative values indicate decline; positive values indicate improvement.

    Attributes
    ----------
    n_captures : int
    duration_min : float
    progressive_start, progressive_end, progressive_change, progressive_pct_change
    nonprog_start, nonprog_end, nonprog_change, nonprog_pct_change
    immotile_start, immotile_end, immotile_change, immotile_pct_change
    viability_start, viability_end, viability_change, viability_pct_change
    count_start, count_end, count_change, count_pct_change
    score_start, score_end, score_change, score_pct_change
    motility_degradation_rate_per_min  : float   (% progressive lost per minute)
    viability_degradation_rate_per_min : float   (% viability lost per minute)
    score_decline_rate_per_min         : float   (score points lost per minute)
    trend_slope                        : dict    {metric: slope (per minute)}
    r_squared                          : dict    {metric: R²}
    """
    n_captures:   int   = 0
    duration_min: float = 0.0

    # Progressive motility
    progressive_start:      float = 0.0
    progressive_end:        float = 0.0
    progressive_change:     float = 0.0
    progressive_pct_change: float = 0.0

    # Non-progressive motility
    nonprog_start:      float = 0.0
    nonprog_end:        float = 0.0
    nonprog_change:     float = 0.0
    nonprog_pct_change: float = 0.0

    # Immotile
    immotile_start:      float = 0.0
    immotile_end:        float = 0.0
    immotile_change:     float = 0.0
    immotile_pct_change: float = 0.0

    # Viability
    viability_start:      float = 0.0
    viability_end:        float = 0.0
    viability_change:     float = 0.0
    viability_pct_change: float = 0.0

    # Total sperm count
    count_start:      float = 0.0
    count_end:        float = 0.0
    count_change:     float = 0.0
    count_pct_change: float = 0.0

    # WHO score
    score_start:      float = 0.0
    score_end:        float = 0.0
    score_change:     float = 0.0
    score_pct_change: float = 0.0

    # Rates (per minute)
    motility_degradation_rate_per_min:  float = 0.0
    viability_degradation_rate_per_min: float = 0.0
    score_decline_rate_per_min:         float = 0.0

    # Linear regression coefficients
    trend_slope: Dict[str, float] = field(default_factory=dict)
    r_squared:   Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# TrendReport
# ══════════════════════════════════════════════════════════════

@dataclass
class TrendReport:
    """
    Everything produced by ``TrendAnalyser.analyse()``.

    Attributes
    ----------
    session        : MonitorSession
    sample_meta    : dict   form intake values
    interval_df    : pd.DataFrame   one row per successful capture
    trend_metrics  : TrendMetrics
    plot_paths     : dict   {metric_name: str(Path)}
    degradation_summary : dict  human-readable strings
    generated_at   : str    ISO timestamp
    """
    session:             MonitorSession
    sample_meta:         Dict[str, Any]
    interval_df:         pd.DataFrame
    trend_metrics:       TrendMetrics
    plot_paths:          Dict[str, str]
    degradation_summary: Dict[str, str]
    generated_at:        str


# ══════════════════════════════════════════════════════════════
# TrendAnalyser
# ══════════════════════════════════════════════════════════════

class TrendAnalyser:
    """
    Computes trend metrics and generates plots from a ``MonitorSession``.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Auto-loaded if None.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config    = config
        self._plot_dir = ensure_dir(
            Path(config["paths"]["outputs"]["plots"])
        )

    # ----------------------------------------------------------

    def analyse(self,
                session: MonitorSession,
                sample_meta: Optional[Dict[str, Any]] = None) -> TrendReport:
        """
        Full trend analysis pipeline.

        Parameters
        ----------
        session     : MonitorSession   completed (or stopped) session
        sample_meta : dict | None      intake form values from Streamlit

        Returns
        -------
        TrendReport
        """
        logger.info(
            "Starting trend analysis: {} captures over {} min",
            session.captures_done, session.duration_min
        )

        if sample_meta is None:
            sample_meta = {}

        # ── Build interval dataframe ───────────────────────────
        interval_df = self._build_interval_df(session)

        # ── Compute metrics ────────────────────────────────────
        metrics = self._compute_metrics(interval_df, session)

        # ── Generate plots ─────────────────────────────────────
        stem       = f"monitor_{sample_meta.get('sample_id', 'session')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        plot_subdir = ensure_dir(self._plot_dir / stem)
        plot_paths  = self._generate_plots(interval_df, metrics, plot_subdir, sample_meta)

        # ── Human-readable degradation summary ────────────────
        degradation_summary = self._build_degradation_summary(metrics)

        report = TrendReport(
            session             = session,
            sample_meta         = sample_meta,
            interval_df         = interval_df,
            trend_metrics       = metrics,
            plot_paths          = plot_paths,
            degradation_summary = degradation_summary,
            generated_at        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        logger.success("Trend analysis complete — {} plots generated", len(plot_paths))
        return report

    # ----------------------------------------------------------
    # Internal: build dataframe
    # ----------------------------------------------------------

    def _build_interval_df(self, session: MonitorSession) -> pd.DataFrame:
        """
        Convert session.results into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: interval_index, elapsed_min, capture_time_str,
                     total_sperm, progressive_pct, nonprog_pct,
                     immotile_pct, normal_pct, viability_pct,
                     who_score, who_category, error
        """
        rows = []
        for r in session.results:
            summ = _extract_summary(r.analysis_results)
            rows.append({
                "interval_index":  r.interval_index,
                "elapsed_min":     round(r.elapsed_min, 2),
                "capture_time":    datetime.fromtimestamp(r.capture_time).strftime("%H:%M:%S"),
                "total_sperm":     summ.get("total_sperm",      0),
                "progressive_pct": summ.get("progressive_pct",  0.0),
                "nonprog_pct":     summ.get("nonprog_pct",      0.0),
                "immotile_pct":    summ.get("immotile_pct",     0.0),
                "normal_pct":      summ.get("normal_pct",       0.0),
                "viability_pct":   summ.get("viability_pct",    0.0),
                "who_score":       summ.get("who_score",         0.0),
                "who_category":    summ.get("who_category",      "—"),
                "error":           r.error or "",
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Separate good and error rows — keep all but flag errors
        df["ok"] = df["error"] == ""
        return df

    # ----------------------------------------------------------
    # Internal: compute metrics
    # ----------------------------------------------------------

    def _compute_metrics(self,
                          df: pd.DataFrame,
                          session: MonitorSession) -> TrendMetrics:
        """
        Compute TrendMetrics from the interval DataFrame.

        Parameters
        ----------
        df      : pd.DataFrame from _build_interval_df
        session : MonitorSession

        Returns
        -------
        TrendMetrics
        """
        m = TrendMetrics(
            n_captures   = session.captures_done,
            duration_min = session.elapsed_s() / 60.0,
        )

        ok = df[df["ok"]] if "ok" in df.columns else df
        if ok.empty or len(ok) < 2:
            return m

        metrics_cols = [
            "progressive_pct", "nonprog_pct", "immotile_pct",
            "viability_pct",   "total_sperm", "who_score",
        ]
        t = ok["elapsed_min"].values.astype(float)

        # ── Start / end / absolute change / % change ──────────
        def _se(col: str) -> Tuple[float, float, float, float]:
            vals = ok[col].values.astype(float)
            s, e = float(vals[0]), float(vals[-1])
            delta = e - s
            pct   = (delta / s * 100) if abs(s) > 1e-9 else 0.0
            return s, e, round(delta, 3), round(pct, 2)

        m.progressive_start, m.progressive_end, m.progressive_change, m.progressive_pct_change = _se("progressive_pct")
        m.nonprog_start,     m.nonprog_end,     m.nonprog_change,     m.nonprog_pct_change     = _se("nonprog_pct")
        m.immotile_start,    m.immotile_end,    m.immotile_change,    m.immotile_pct_change    = _se("immotile_pct")
        m.viability_start,   m.viability_end,   m.viability_change,   m.viability_pct_change   = _se("viability_pct")
        m.count_start,       m.count_end,       m.count_change,       m.count_pct_change       = _se("total_sperm")
        m.score_start,       m.score_end,       m.score_change,       m.score_pct_change       = _se("who_score")

        # ── Linear regression per metric ──────────────────────
        for col in metrics_cols:
            vals = ok[col].values.astype(float)
            if len(t) >= 2 and np.std(t) > 0:
                slope, intercept = np.polyfit(t, vals, 1)
                ss_res = np.sum((vals - (slope * t + intercept)) ** 2)
                ss_tot = np.sum((vals - np.mean(vals)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
                m.trend_slope[col] = round(float(slope), 5)
                m.r_squared[col]   = round(max(0.0, float(r2)), 4)
            else:
                m.trend_slope[col] = 0.0
                m.r_squared[col]   = 0.0

        # ── Degradation rates (per minute) ────────────────────
        dur = max(session.elapsed_s() / 60.0, 1e-9)
        # Progressive motility lost per minute (positive = degradation)
        m.motility_degradation_rate_per_min  = round(
            -m.progressive_change / dur, 4
        )
        m.viability_degradation_rate_per_min = round(
            -m.viability_change / dur, 4
        )
        m.score_decline_rate_per_min = round(
            -m.score_change / dur, 4
        )

        return m

    # ----------------------------------------------------------
    # Internal: generate plots
    # ----------------------------------------------------------

    def _generate_plots(self,
                         df: pd.DataFrame,
                         metrics: TrendMetrics,
                         plot_dir: Path,
                         sample_meta: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate six individual trend plots + one summary overview.

        Parameters
        ----------
        df          : pd.DataFrame
        metrics     : TrendMetrics
        plot_dir    : Path   output directory
        sample_meta : dict

        Returns
        -------
        dict  {metric_name: str(path)}
        """
        ok  = df[df["ok"]] if "ok" in df.columns else df
        paths: Dict[str, str] = {}

        if ok.empty:
            logger.warning("No successful captures — skipping plots")
            return paths

        t = ok["elapsed_min"].values

        sample_id = sample_meta.get("sample_id", "—")
        species   = sample_meta.get("species",   "—")

        # ── 1-6: Individual metric plots ──────────────────────
        single_metrics = [
            ("progressive_pct", "Progressive Motility vs Time",       "%"),
            ("nonprog_pct",     "Non-Progressive Motility vs Time",   "%"),
            ("immotile_pct",    "Immotile Percentage vs Time",        "%"),
            ("viability_pct",   "Viability vs Time",                  "%"),
            ("total_sperm",     "Total Sperm Count vs Time",          "cells"),
            ("who_score",       "WHO Quality Score vs Time",          "/ 100"),
        ]

        for col, title, unit in single_metrics:
            path = self._plot_single_metric(
                t, ok[col].values.astype(float),
                title=title, ylabel=f"{_LABELS[col]}", unit=unit,
                colour=_COLOURS[col],
                who_ref=_WHO_REFS.get(col),
                slope=metrics.trend_slope.get(col, 0.0),
                r2=metrics.r_squared.get(col, 0.0),
                sample_id=sample_id, species=species,
                save_path=plot_dir / f"trend_{col}.png",
            )
            paths[col] = str(path)

        # ── 7: Combined motility overview ─────────────────────
        path = self._plot_motility_overview(t, ok, metrics, sample_id, species,
                                             plot_dir / "trend_motility_overview.png")
        paths["motility_overview"] = str(path)

        # ── 8: Dashboard summary (2×3 grid) ───────────────────
        path = self._plot_dashboard(t, ok, metrics, sample_id, species,
                                     plot_dir / "trend_dashboard.png")
        paths["dashboard"] = str(path)

        logger.info("Trend plots saved to {}", plot_dir)
        return paths

    # ----------------------------------------------------------

    def _plot_single_metric(self,
                             t: np.ndarray,
                             y: np.ndarray,
                             title: str,
                             ylabel: str,
                             unit: str,
                             colour: str,
                             who_ref: Optional[float],
                             slope: float,
                             r2: float,
                             sample_id: str,
                             species: str,
                             save_path: Path) -> Path:
        """
        Render one metric vs time with trend line, WHO reference, and annotations.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        # ── Data points + line ────────────────────────────────
        ax.plot(t, y, color=colour, linewidth=2.5, marker="o",
                markersize=6, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=2, zorder=3)
        ax.fill_between(t, y, alpha=0.12, color=colour)

        # ── Trend line ────────────────────────────────────────
        if len(t) >= 2 and np.std(t) > 0:
            t_fit = np.linspace(t[0], t[-1], 200)
            intercept = float(np.mean(y) - slope * np.mean(t))
            y_fit     = slope * t_fit + intercept
            ax.plot(t_fit, y_fit, "--", color="white", linewidth=1.2,
                    alpha=0.5, label=f"Trend  (slope {slope:+.3f}/{chr(0x6d)}in, R²={r2:.3f})")

        # ── WHO reference line ────────────────────────────────
        if who_ref is not None and who_ref > 0:
            ax.axhline(who_ref, color="#fbbf24", linewidth=1.0,
                       linestyle=":", alpha=0.9, label=f"WHO min: {who_ref}%")

        # ── Labels ────────────────────────────────────────────
        ax.set_xlabel("Time (minutes)", color="#94a3b8", fontsize=11)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=11)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(True, color="#1e293b", linewidth=0.7, alpha=0.6)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        if ax.get_legend_handles_labels()[0]:
            leg = ax.legend(fontsize=8, facecolor="#1e293b",
                            labelcolor="white", edgecolor="#334155")

        # ── Sample info annotation ────────────────────────────
        ax.text(0.99, 0.98, f"Sample: {sample_id} | {species}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#64748b")

        # ── Change annotation ─────────────────────────────────
        if len(y) >= 2:
            delta = float(y[-1]) - float(y[0])
            arrow_c = "#ef4444" if delta < 0 else "#10b981"
            arrow   = "▼" if delta < 0 else "▲"
            ax.text(0.01, 0.98,
                    f"{arrow} {abs(delta):.1f} {unit}  ({delta / max(abs(y[0]), 1e-9) * 100:+.1f}%)",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=arrow_c, fontweight="bold")

        plt.tight_layout(pad=0.8)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ----------------------------------------------------------

    def _plot_motility_overview(self,
                                 t: np.ndarray,
                                 df: pd.DataFrame,
                                 metrics: TrendMetrics,
                                 sample_id: str,
                                 species: str,
                                 save_path: Path) -> Path:
        """
        Three-line chart: Progressive, Non-progressive, Immotile vs Time.
        Stacked-area variant so the three always sum to 100 %.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0f172a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f172a")

        prog   = df["progressive_pct"].values.astype(float)
        nonprog= df["nonprog_pct"].values.astype(float)
        immot  = df["immotile_pct"].values.astype(float)

        # Left: line chart
        for y, col, lbl in [
            (prog,    "#10b981", "Progressive"),
            (nonprog, "#f59e0b", "Non-Progressive"),
            (immot,   "#ef4444", "Immotile"),
        ]:
            ax1.plot(t, y, linewidth=2, marker="o", markersize=5,
                     color=col, label=lbl)
            ax1.fill_between(t, y, alpha=0.08, color=col)

        ax1.axhline(30, color="#fbbf24", linestyle=":", linewidth=1, alpha=0.7,
                    label="WHO Progressive min 30%")
        ax1.set_title("Motility Classification vs Time",
                      color="white", fontweight="bold", fontsize=12)
        ax1.set_xlabel("Time (min)", color="#94a3b8")
        ax1.set_ylabel("Percentage (%)", color="#94a3b8")
        ax1.tick_params(colors="#64748b")
        for sp in ax1.spines.values(): sp.set_color("#1e293b")
        ax1.grid(True, color="#1e293b", linewidth=0.7, alpha=0.6)
        ax1.legend(fontsize=8, facecolor="#1e293b",
                   labelcolor="white", edgecolor="#334155")

        # Right: stacked area
        ax2.stackplot(t, prog, nonprog, immot,
                      labels=["Progressive","Non-Progressive","Immotile"],
                      colors=["#10b981","#f59e0b","#ef4444"],
                      alpha=0.75)
        ax2.set_title("Motility Stack vs Time",
                      color="white", fontweight="bold", fontsize=12)
        ax2.set_xlabel("Time (min)", color="#94a3b8")
        ax2.set_ylabel("Percentage (%)", color="#94a3b8")
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="#64748b")
        for sp in ax2.spines.values(): sp.set_color("#1e293b")
        ax2.grid(True, color="#1e293b", linewidth=0.7, alpha=0.3)
        ax2.legend(fontsize=8, facecolor="#1e293b",
                   labelcolor="white", edgecolor="#334155", loc="upper right")

        ax2.text(0.99, 0.98, f"Sample: {sample_id} | {species}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=7, color="#64748b")

        plt.tight_layout(pad=0.8)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ----------------------------------------------------------

    def _plot_dashboard(self,
                         t: np.ndarray,
                         df: pd.DataFrame,
                         metrics: TrendMetrics,
                         sample_id: str,
                         species: str,
                         save_path: Path) -> Path:
        """
        2×3 summary dashboard: all six metrics on one canvas.
        """
        cols_info = [
            ("progressive_pct",  "Progressive (%)",    "#10b981", 30.0),
            ("nonprog_pct",      "Non-Progressive (%)", "#f59e0b", None),
            ("immotile_pct",     "Immotile (%)",        "#ef4444", None),
            ("viability_pct",    "Viability (%)",       "#3b82f6", 54.0),
            ("total_sperm",      "Total Count",         "#8b5cf6", None),
            ("who_score",        "WHO Score",           "#06b6d4", None),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor("#0f172a")
        fig.suptitle(
            f"Semen Quality Trend Dashboard  ·  Sample: {sample_id}  ·  {species}",
            color="white", fontsize=13, fontweight="bold", y=1.01
        )

        for ax, (col, lbl, colour, who_ref) in zip(axes.flatten(), cols_info):
            ax.set_facecolor("#0f172a")
            y = df[col].values.astype(float)

            ax.plot(t, y, color=colour, linewidth=2.2, marker="o",
                    markersize=5, markerfacecolor="white",
                    markeredgecolor=colour, markeredgewidth=1.5)
            ax.fill_between(t, y, alpha=0.10, color=colour)

            # Trend line
            slope = metrics.trend_slope.get(col, 0.0)
            if len(t) >= 2 and abs(slope) > 1e-9:
                intercept = float(np.mean(y) - slope * np.mean(t))
                t_fit = np.linspace(t[0], t[-1], 100)
                ax.plot(t_fit, slope * t_fit + intercept,
                        "--", color="white", linewidth=0.9, alpha=0.45)

            # WHO reference
            if who_ref:
                ax.axhline(who_ref, color="#fbbf24", linewidth=0.8,
                           linestyle=":", alpha=0.8)

            ax.set_title(lbl, color=colour, fontsize=10, fontweight="bold")
            ax.set_xlabel("min", color="#64748b", fontsize=8)
            ax.tick_params(colors="#475569", labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#1e293b")
            ax.grid(True, color="#1e293b", linewidth=0.6, alpha=0.5)

            # Mini annotation
            if len(y) >= 2:
                delta = float(y[-1]) - float(y[0])
                c = "#ef4444" if delta < 0 else "#4ade80"
                ax.text(0.97, 0.97, f"{delta:+.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        color=c, fontsize=9, fontweight="bold")

        plt.tight_layout(pad=1.2)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # ----------------------------------------------------------
    # Internal: human-readable degradation summary
    # ----------------------------------------------------------

    def _build_degradation_summary(self, m: TrendMetrics) -> Dict[str, str]:
        """
        Build a dict of human-readable strings describing key findings.

        Returns
        -------
        dict  {finding_key: sentence}
        """

        def _pct_str(change: float, name: str) -> str:
            if abs(change) < 0.1:
                return f"{name} remained stable throughout the monitoring period."
            direction = "decreased" if change < 0 else "increased"
            return (
                f"{name} {direction} by {abs(change):.1f} percentage points "
                f"over the monitoring period."
            )

        def _rate_str(rate: float, name: str, unit: str = "%/min") -> str:
            if abs(rate) < 0.001:
                return f"{name} degradation rate was negligible."
            return (
                f"{name} degradation rate: {rate:.4f} {unit} "
                f"({'declining' if rate > 0 else 'improving'})."
            )

        return {
            "progressive_motility": _pct_str(m.progressive_change, "Progressive motility"),
            "non_progressive":      _pct_str(m.nonprog_change,     "Non-progressive motility"),
            "immotile":             _pct_str(m.immotile_change,     "Immotile fraction"),
            "viability":            _pct_str(m.viability_change,    "Viability"),
            "count":
                f"Total sperm count changed by {m.count_change:+.0f} "
                f"({m.count_pct_change:+.1f}%) from start to end.",
            "who_score":
                f"WHO quality score moved from {m.score_start:.1f} to {m.score_end:.1f} "
                f"({m.score_change:+.1f} points, {m.score_pct_change:+.1f}%).",
            "motility_rate":  _rate_str(m.motility_degradation_rate_per_min,  "Motility"),
            "viability_rate": _rate_str(m.viability_degradation_rate_per_min, "Viability"),
            "score_rate":     _rate_str(m.score_decline_rate_per_min,         "Quality score",
                                        "pts/min"),
            "overall":
                (
                    "Sample quality showed significant deterioration. "
                    "ART procedures should be performed as soon as possible."
                ) if m.progressive_change < -10 or m.viability_change < -15
                else (
                    "Sample quality remained within acceptable limits throughout monitoring."
                ) if abs(m.progressive_change) < 5 and abs(m.viability_change) < 10
                else (
                    "Moderate quality decline observed. "
                    "Monitor closely and use sample promptly."
                ),
        }