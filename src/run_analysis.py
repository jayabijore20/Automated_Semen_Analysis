"""
src/run_analysis.py
====================
Automated Semen Viability and Counting Analysis — Master Orchestrator

Integrates every pipeline stage into a single ``AutomatedSemenAnalysis``
class.  All AI work is delegated to existing modules; this file only
orchestrates, saves outputs, and returns results.

Pipeline
--------
Detection (SpermDetector.detect_frame)
    ↓
Tracking  (SpermTracker.track_video  → tracks_df)
    ↓
Motility  (MotilityAnalyser.analyse  → per_track_df, motility_summary)
    ↓
Morphology(MorphologyClassifier.classify_crops_from_frame / batch_summary)
    ↓
Viability (ViabilityPredictor.predict)
    ↓
Outputs   (annotated video, motility.csv, morphology.csv, summary.json,
           analysis_report.csv)

Usage
-----
    python -m src.run_analysis --input path/to/video.mp4

CLI flags
---------
    --input      path to video (required)
    --output     output root directory (default: outputs/)
    --save-video write annotated mp4        (default: True)
    --save-json  write summary.json         (default: True)
    --save-csv   write CSV reports          (default: True)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ── Project utilities ──────────────────────────────────────────
from src.utils.helpers import ensure_dir, generate_color_palette, load_config, timer
from src.utils.logger import get_logger

# ── Pipeline modules ───────────────────────────────────────────
from src.detection.inference import DetectionResult, SpermDetector
from src.tracking.track_sperm import SpermTracker
from src.motility.motility_analysis import MotilityAnalyser
from src.morphology.inference import MorphologyClassifier
from src.viability.predict_viability import ViabilityPredictor
from src.scoring.quality_score import QualityScorer
from src.reporting.generate_report import ReportGenerator
from src.analysis import concentration as _concentration
logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
)

# Colour used for the HUD overlay background (BGRA-compatible black)
_HUD_BG_COLOUR: Tuple[int, int, int] = (0, 0, 0)
_HUD_FG_COLOUR: Tuple[int, int, int] = (0, 255, 120)


# ══════════════════════════════════════════════════════════════
# AutomatedSemenAnalysis
# ══════════════════════════════════════════════════════════════

class AutomatedSemenAnalysis:
    """
    End-to-end semen analysis orchestrator.

    Loads all AI modules once in ``__init__`` and exposes two public
    entry points:

    * :meth:`run_video`  — full pipeline on a microscope video.
    * :meth:`run_image`  — detection + morphology on a single image.

    Parameters
    ----------
    config : dict, optional
        Loaded project config.  Auto-loaded from
        ``configs/config.yaml`` when *None*.
    output_root : str | Path, optional
        Root directory for all saved outputs.
        Defaults to the ``outputs/`` directory defined in config.
    """

    # ----------------------------------------------------------
    # Construction
    # ----------------------------------------------------------

    def __init__(self,
                 config: Optional[Dict] = None,
                 output_root: Optional[str | Path] = None) -> None:
        self.config = config or load_config()
        self._setup_output_dirs(output_root)
        self._initialize_modules()
        self.progress_callback = None

    # ----------------------------------------------------------

    def _setup_output_dirs(self, output_root: Optional[str | Path]) -> None:
        """
        Create all output subdirectories, optionally under a custom root.

        Parameters
        ----------
        output_root : str | Path | None
            When supplied every output path is re-rooted here.
        """
        if output_root is not None:
            root = Path(output_root)
            self.out_videos  = ensure_dir(root / "videos")
            self.out_csv     = ensure_dir(root / "csv")
            self.out_json    = ensure_dir(root / "json")
            self.out_reports = ensure_dir(root / "reports")
            self.out_plots   = ensure_dir(root / "plots")
            # Patch config so downstream modules respect the override
            self.config["paths"]["outputs"]["tracking"] = str(root / "tracking")
            self.config["paths"]["outputs"]["plots"]    = str(self.out_plots)
            self.config["paths"]["outputs"]["reports"]  = str(self.out_reports)
        else:
            paths = self.config["paths"]["outputs"]
            self.out_videos  = ensure_dir("outputs/videos")
            self.out_csv     = ensure_dir("outputs/csv")
            self.out_json    = ensure_dir("outputs/json")
            self.out_reports = ensure_dir(paths["reports"])
            self.out_plots   = ensure_dir(paths["plots"])

        ensure_dir("outputs")
        logger.info("Output directories ready")

    # ----------------------------------------------------------

    def _initialize_modules(self) -> None:
        """
        Instantiate all pipeline components exactly once.

        Models inside SpermDetector, MorphologyClassifier, and
        ViabilityPredictor are lazy-loaded on first inference call.
        """
        logger.info("Initialising SpermDetector…")
        try:
            self._detector = SpermDetector(self.config)
        except Exception as exc:
            logger.error("SpermDetector init failed: {}", exc)
            raise

        logger.info("Initialising SpermTracker…")
        try:
            self._tracker = SpermTracker(self.config)
        except Exception as exc:
            logger.error("SpermTracker init failed: {}", exc)
            raise

        logger.info("Initialising MorphologyClassifier…")
        try:
            self._morph_clf = MorphologyClassifier(self.config)
        except Exception as exc:
            logger.error("MorphologyClassifier init failed: {}", exc)
            raise

        logger.info("Initialising ViabilityPredictor…")
        try:
            self._via_pred = ViabilityPredictor(self.config)
        except Exception as exc:
            logger.error("ViabilityPredictor init failed: {}", exc)
            raise
        logger.info("Initialising QualityScorer…")
        try:
            self._quality_scorer = QualityScorer(self.config)
        except Exception as exc:
            logger.error("QualityScorer init failed: {}", exc)
            raise
        logger.success("All pipeline modules initialised.")
        logger.info("Initialising ReportGenerator...")
        try:
             self._report_generator = ReportGenerator(self.config)
        except Exception as exc:
            logger.error("ReportGenerator init failed: {}", exc)
            raise    
        
    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    @timer
    def run_video(self,
                  video_path: str | Path,
                  save_video: bool = True,
                  save_json:  bool = True,
                  save_csv:   bool = True, progress_callback=None, progress_callback_stage=None,
                  concentration_params: Optional[Dict[str, Any]] = None,
                  ) -> Dict[str, Any]:
        """
        Run the complete semen analysis pipeline on a video file.

        Parameters
        ----------
        video_path : str | Path
            Path to the microscope semen video.
        save_video : bool
            Write an annotated output video.
        save_json : bool
            Write ``summary.json``.
        save_csv : bool
            Write ``motility.csv``, ``morphology.csv``,
            and ``analysis_report.csv``.
        concentration_params : dict, optional
            Sample-prep / calibration parameters used to compute
            concentration, MCC, and progressive concentration (see
            ``src.analysis.concentration.compute_concentration_metrics_from_params``
            for the expected keys). When ``None`` (the default —
            preserves all existing callers' behaviour unchanged), no
            concentration fields are added to the result dict.

        Returns
        -------
        dict
            Final results dictionary (see module docstring for schema).
        """
        video_path = Path(video_path)
        self.progress_callback = progress_callback
        self.progress_callback_stage = progress_callback_stage

        pipeline_start = time.perf_counter()
        

        logger.info("═" * 60)
        logger.info("Starting analysis: {}", video_path.name)
        logger.info("═" * 60)

        # ── Validate input ─────────────────────────────────────
        self._validate_video(video_path)

        # ── Stage 1: Tracking (includes frame-level detection) ─
        logger.info("[1/5] Running detection + tracking…")
        tracking_summary = self._run_tracking(video_path)

        if self.progress_callback_stage:
           self.progress_callback_stage(70, "✅ Tracking Complete")

        n_frames      = tracking_summary.get("n_frames",      0)
        unique_tracks = tracking_summary.get("unique_tracks", 0)
        tracks_df     = tracking_summary.get("tracks_df",     None)

        # FPS from the video itself (SpermTracker already read it)
        fps = self._read_fps(video_path)

        logger.info("Tracking complete: {} frames | {} tracks", n_frames, unique_tracks)

        # ── Stage 2: Motility ──────────────────────────────────
        if self.progress_callback_stage:
           self.progress_callback_stage(75, "📈 Analysing Motility...")
        
        logger.info("[2/5] Running motility analysis…")
        per_track_df, motility_summary = self._run_motility(tracks_df, video_path)

        # ── Stage 3: Morphology ────────────────────────────────
        if self.progress_callback_stage:
           self.progress_callback_stage(85, "🧬 Classifying Morphology...")
        
        logger.info("[3/5] Running morphology classification…")
        morph_rows, morphology_summary = self._run_morphology(video_path, tracks_df)

        # ── Stage 4: Viability ─────────────────────────────────
        if self.progress_callback_stage:
           self.progress_callback_stage(93, "❤️ Predicting Viability...")
        
        logger.info("[4/5] Predicting viability…")
        viability_pct = self._run_viability(motility_summary, morphology_summary)
        
        print("\n===== VALUES PASSED TO QUALITY SCORER =====")
        print("Count:", unique_tracks)
        print("Progressive:", motility_summary["progressive_pct"])
        print("Normal:", morphology_summary["normal_pct"])
        print("Viability:", viability_pct)
        print("===========================================\n")


        quality_result = self._quality_scorer.score(
            count=unique_tracks,
            progressive_pct=motility_summary["progressive_pct"],
            normal_pct=morphology_summary["normal_pct"],
            viability_pct=viability_pct,
        )
        print("\n================ QUALITY RESULT ================")
        from pprint import pprint
        pprint(quality_result)
        print("===============================================\n")
        # ── Stage 5: Annotated video ───────────────────────────
        annotated_video_path: Optional[str] = None
        if save_video:
            logger.info("[5/5] Writing annotated video…")
            annotated_video_path = self._write_annotated_video(
                video_path, tracks_df, morph_rows,
                motility_summary, morphology_summary, viability_pct
            )
        else:
            logger.info("[5/5] Annotated video skipped (--save-video not set)")

        # ── Compile results ────────────────────────────────────
        elapsed = round(time.perf_counter() - pipeline_start, 2)
        results = self._generate_summary(
            video_path      = video_path,
            n_frames        = n_frames,
            fps             = fps,
            unique_tracks   = unique_tracks,
            motility_summary= motility_summary,
            morphology_summary= morphology_summary,
            viability_pct   = viability_pct,
            elapsed         = elapsed,
        )
        results["quality_score"] = quality_result
        results["quality"] = quality_result

        # ── Concentration / MCC / progressive concentration ────
        # Additive only: when concentration_params is None (the
        # default, and the only value ever passed by callers that
        # existed before this feature), nothing below runs and the
        # results dict is identical to before. See
        # src/analysis/concentration.py for the full scientific
        # rationale and honesty guarantees (never fabricates a
        # concentration when calibration/depth is missing).
        if concentration_params is not None:
            try:
                frame_w, frame_h = self._read_frame_dimensions(video_path)
                params_with_frame = {
                    **concentration_params,
                    "image_width_px":  frame_w,
                    "image_height_px": frame_h,
                }
                conc_result = _concentration.compute_concentration_metrics_from_params(
                    results, params_with_frame,
                )
                results.update(conc_result.to_dict())
                if conc_result.status != "ok":
                    logger.info(
                        "Concentration not calculated ({}): {}",
                        conc_result.status, conc_result.reason,
                    )
                else:
                    logger.info(
                        "Concentration: {:.2f} M/mL  MCC: {:.2f} M/mL  "
                        "Progressive: {:.2f} M/mL",
                        conc_result.total_concentration_m_per_ml,
                        conc_result.motile_concentration_m_per_ml,
                        conc_result.progressive_concentration_m_per_ml,
                    )
            except Exception as exc:
                logger.warning("Concentration calculation failed: {}", exc)
                results.update({
                    "concentration_status": "error",
                    "concentration_status_reason": str(exc),
                })

        if self.progress_callback_stage:
           self.progress_callback_stage(98, "📄 Generating Reports...")

        report_paths = self._report_generator.generate(
             video_name=video_path.name,
             detection_summary={
                 "count": unique_tracks,
                 "n_frames": n_frames,
                "unique_tracks": unique_tracks,
            },
             motility_summary=motility_summary,
             morphology_summary=morphology_summary,
             viability_pct=viability_pct,
             quality_result=quality_result,
             output_stem=video_path.stem,
        ) 

        print("\n================ REPORT PATHS ================")
        print(report_paths)
        print("==============================================\n")


        results["reports"] = report_paths
        results["report_paths"] = report_paths

        # ── Save outputs ───────────────────────────────────────
        stem = video_path.stem
        if save_csv:
            self._write_motility_csv(per_track_df, stem)
            self._write_morphology_csv(morph_rows, stem)
            self._write_report_csv(results, stem)

        if save_json:
            self._write_json(results, video_path, annotated_video_path, stem)

        
        logger.success(
            "Analysis complete in {:.1f}s — viability={:.1f}%  score={}/100",
            elapsed, viability_pct, results.get("total_sperm", 0)
        )

        print("\nQuality Score")
        print("---------------------------")
        print(f"Score     : {quality_result['score']:.2f}/100")
        print(f"Category  : {quality_result['category']}")
        print("\nGenerated Reports:")
        for fmt, path in report_paths.items():
            if path:
               print(f" {fmt.upper()} : {path}")
        if quality_result["who_flags"]:
           print("\nWHO Flags:")
           for flag in quality_result["who_flags"]:
               print(f" - {flag}")
               
        if self.progress_callback_stage:
           self.progress_callback_stage(100, "✅ Analysis Complete")

        return results

    # ----------------------------------------------------------

    def run_image(self,
                  image_path: str | Path) -> Dict[str, Any]:
        """
        Run detection and morphology classification on a single image.

        Parameters
        ----------
        image_path : str | Path
            Path to a microscope image.

        Returns
        -------
        dict
            {
              "count":           int,
              "boxes":           list,
              "confidences":     list,
              "morphology":      list of per-cell dicts,
              "morphology_summary": dict,
              "annotated_img":   np.ndarray (BGR),
            }
        """
        image_path = Path(image_path)
        logger.info("run_image: {}", image_path.name)

        if not image_path.exists():
            logger.error("Image not found: {}", image_path)
            return {}

        # Detection
        det: DetectionResult = self._detector.detect_image(image_path=image_path)
        logger.info("Detected {} sperm", det.count)

        # Morphology on detected crops
        morph_results: List[Dict] = []
        if det.count > 0 and det.annotated_img is not None:
            try:
                img_bgr = cv2.imread(str(image_path))
                if img_bgr is not None:
                    morph_results = self._morph_clf.classify_crops_from_frame(
                        img_bgr, det.boxes
                    )
            except Exception as exc:
                logger.warning("Morphology on image failed: {}", exc)

        morph_summary = self._morph_clf.batch_summary(morph_results) if morph_results else {
            "total": 0, "normal_count": 0, "abnormal_count": 0,
            "normal_pct": 0.0, "abnormal_pct": 0.0,
        }

        return {
            "count":              det.count,
            "boxes":              det.boxes,
            "confidences":        det.confidences,
            "morphology":         morph_results,
            "morphology_summary": morph_summary,
            "annotated_img":      det.annotated_img,
        }

    # ══════════════════════════════════════════════════════════
    # Private stage runners
    # ══════════════════════════════════════════════════════════

    def _validate_video(self, video_path: Path) -> None:
        """
        Raise descriptive errors for missing or unreadable videos.

        Parameters
        ----------
        video_path : Path

        Raises
        ------
        FileNotFoundError
        ValueError
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path.resolve()}")
        if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{video_path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )
        cap = cv2.VideoCapture(str(video_path))
        readable = cap.isOpened()
        cap.release()
        if not readable:
            raise ValueError(f"OpenCV cannot open video: {video_path}")
        logger.debug("Video validated: {}", video_path.name)

    # ----------------------------------------------------------

    def _read_fps(self, video_path: Path) -> float:
        """Return the FPS of *video_path*, defaulting to 30.0."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)

    # ----------------------------------------------------------

    def _read_frame_dimensions(self, video_path: Path) -> Tuple[int, int]:
        """
        Return (width_px, height_px) of *video_path*, used for the
        concentration/field-of-view calculation. Defaults to
        (0, 0) if the video cannot be opened; callers must treat
        (0, 0) as "unavailable" rather than a real dimension.
        """
        cap = cv2.VideoCapture(str(video_path))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return width, height

    # ----------------------------------------------------------

    def _run_tracking(self, video_path: Path) -> Dict:
        """
        Delegate to SpermTracker.track_video.

        Returns
        -------
        dict  {n_frames, unique_tracks, tracks_df}
        """
        try:
            result = self._tracker.track_video(video_path,
                                               progress_callback=self.progress_callback
                                               )
            if not result:
                logger.warning("SpermTracker returned empty result")
                return {"n_frames": 0, "unique_tracks": 0, "tracks_df": None}
            return result
        except Exception as exc:
            logger.error("Tracking failed: {}", exc)
            return {"n_frames": 0, "unique_tracks": 0, "tracks_df": None}

    # ----------------------------------------------------------

    def _run_motility(
        self,
        tracks_df: Optional[pd.DataFrame],
        video_path: Path,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Build a MotilityAnalyser, feed it tracks_df or the saved CSV,
        and return (per_track_df, summary).

        Parameters
        ----------
        tracks_df : pd.DataFrame | None
        video_path : Path

        Returns
        -------
        (pd.DataFrame, dict)
        """
        _empty_summary: Dict = {
            "total_tracks": 0,
            "progressive_pct": 0.0, "nonprogressive_pct": 0.0, "immotile_pct": 0.0,
            "mean_VCL": 0.0, "mean_VSL": 0.0, "mean_VAP": 0.0,
        }

        try:
            analyser = MotilityAnalyser(self.config, tracks_df=tracks_df)

            # Fall back to the CSV file SpermTracker already saved
            if tracks_df is None or tracks_df.empty:
                csv_path = (
                    Path(self.config["paths"]["outputs"]["tracking"])
                    / f"{video_path.stem}_tracks.csv"
                )
                if csv_path.exists():
                    logger.info("Loading tracks from saved CSV: {}", csv_path)
                    analyser.load_tracks(csv_path)
                else:
                    logger.warning("No tracks data available — motility skipped")
                    return pd.DataFrame(), _empty_summary

            per_track_df, summary = analyser.analyse()

            if per_track_df.empty:
                logger.warning("Motility analysis returned no tracks")
                return pd.DataFrame(), _empty_summary

            return per_track_df, summary

        except Exception as exc:
            logger.error("Motility analysis failed: {}", exc)
            return pd.DataFrame(), _empty_summary

    # ----------------------------------------------------------

    def _run_morphology(
        self,
        video_path: Path,
        tracks_df: Optional[pd.DataFrame],
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample frames from *video_path*, extract detected sperm crops
        from *tracks_df*, and classify each via MorphologyClassifier.

        Parameters
        ----------
        video_path : Path
        tracks_df : pd.DataFrame | None

        Returns
        -------
        (morph_rows, morphology_summary)
            morph_rows : list of dicts  {track_id, frame_id, class, confidence}
            morphology_summary : dict from MorphologyClassifier.batch_summary
        """
        _empty_summary: Dict = {
            "total": 0, "normal_count": 0, "abnormal_count": 0,
            "normal_pct": 0.0, "abnormal_pct": 0.0,
        }

        if tracks_df is None or tracks_df.empty:
            logger.warning("No tracks — morphology classification skipped")
            return [], _empty_summary

        # Determine which frames to sample (evenly spaced, max 15)
        available_frames = sorted(tracks_df["frame_id"].unique().tolist())
        n_sample = min(15, len(available_frames))
        step = max(1, len(available_frames) // n_sample)
        sample_fids = available_frames[::step][:n_sample]

        cap = cv2.VideoCapture(str(video_path))
        all_clf_results: List[Dict] = []
        morph_rows: List[Dict] = []

        for fid in sample_fids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Skipping corrupt frame {}", fid)
                continue

            frame_rows = tracks_df[tracks_df["frame_id"] == fid]
            if frame_rows.empty:
                continue

            boxes = frame_rows[["x1", "y1", "x2", "y2"]].values.tolist()

            try:
                clf_results = self._morph_clf.classify_crops_from_frame(frame, boxes)
            except Exception as exc:
                logger.warning("Morphology failed on frame {}: {}", fid, exc)
                continue

            for tid, res in zip(frame_rows["track_id"].tolist(), clf_results):
                morph_rows.append({
                    "track_id":   int(tid),
                    "frame_id":   int(fid),
                    "Morphology": res.get("class",      "Unknown"),
                    "Confidence": res.get("confidence", 0.0),
                })
                all_clf_results.append(res)

        cap.release()

        if not all_clf_results:
            logger.warning("No sperm crops classified — returning empty morphology")
            return morph_rows, _empty_summary

        summary = self._morph_clf.batch_summary(all_clf_results)
        logger.info(
            "Morphology: Normal={:.1f}%  Abnormal={:.1f}%  (n={})",
            summary["normal_pct"], summary["abnormal_pct"], summary["total"]
        )
        return morph_rows, summary

    # ----------------------------------------------------------

    def _run_viability(
        self,
        motility_summary: Dict,
        morphology_summary: Dict,
    ) -> float:
        """
        Delegate to ViabilityPredictor.predict with REAL pipeline outputs.

        Parameters
        ----------
        motility_summary : dict   (from MotilityAnalyser._build_summary)
        morphology_summary : dict (from MorphologyClassifier.batch_summary)

        Returns
        -------
        float  predicted viability %
        """
        try:
            viability = self._via_pred.predict(
                motility_summary   = motility_summary,
                morphology_summary = morphology_summary,
            )
            logger.info("Predicted viability: {:.2f}%", viability)
            return viability
        except Exception as exc:
            logger.error("Viability prediction failed: {}", exc)
            return 0.0

    # ══════════════════════════════════════════════════════════
    # Annotated video writer
    # ══════════════════════════════════════════════════════════

    def _write_annotated_video(
        self,
        video_path: Path,
        tracks_df: Optional[pd.DataFrame],
        morph_rows: List[Dict],
        motility_summary: Dict,
        morphology_summary: Dict,
        viability_pct: float,
    ) -> Optional[str]:
        """
        Re-read the source video frame-by-frame and write an annotated
        output that overlays:
          • bounding boxes with track ID and morphology label
          • HUD overlay: total sperm, motility %, morphology %, viability

        Parameters
        ----------
        video_path : Path
        tracks_df : pd.DataFrame | None
        morph_rows : list of dicts  (track_id, frame_id, Morphology, Confidence)
        motility_summary, morphology_summary : dict
        viability_pct : float

        Returns
        -------
        str | None   path to the saved video, or None on failure
        """
        out_path = self.out_videos / f"{video_path.stem}_annotated.mp4"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot re-open video for annotation: {}", video_path)
            return None

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        # Build fast lookups from tracks and morphology data
        frame_boxes:  Dict[int, List]        = {}   # frame_id → [[x1,y1,x2,y2,tid], ...]
        morph_lookup: Dict[Tuple[int,int], Dict] = {}  # (frame_id, track_id) → morph dict

        if tracks_df is not None and not tracks_df.empty:
            for _, row in tracks_df.iterrows():
                fid = int(row["frame_id"])
                frame_boxes.setdefault(fid, []).append([
                    row["x1"], row["y1"], row["x2"], row["y2"], int(row["track_id"])
                ])

        for mrow in morph_rows:
            key = (int(mrow["frame_id"]), int(mrow["track_id"]))
            morph_lookup[key] = mrow

        palette = generate_color_palette(500)

        # Pre-compute HUD text lines (constant across all frames)
        prog_pct  = motility_summary.get("progressive_pct",    0.0)
        nprog_pct = motility_summary.get("nonprogressive_pct", 0.0)
        imm_pct   = motility_summary.get("immotile_pct",       0.0)
        norm_pct  = morphology_summary.get("normal_pct",        0.0)
        total_sp  = tracks_df["track_id"].nunique() if (
            tracks_df is not None and not tracks_df.empty) else 0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            # ── Draw bounding boxes for this frame ─────────────
            for entry in frame_boxes.get(frame_idx, []):
                x1, y1, x2, y2, tid = entry
                color = palette[int(tid) % len(palette)]
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)), (int(x2), int(y2)),
                    color, 2
                )

                # Morphology label
                mkey = (frame_idx, int(tid))
                if mkey in morph_lookup:
                    mrow = morph_lookup[mkey]
                    label = (
                        f"#{tid} {mrow['Morphology']} "
                        f"{mrow['Confidence']:.2f}"
                    )
                else:
                    label = f"#{tid}"

                cv2.putText(
                    annotated, label,
                    (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                )

            # ── HUD overlay ────────────────────────────────────
            annotated = self._draw_hud(
                annotated, frame_idx, total, total_sp,
                prog_pct, nprog_pct, imm_pct, norm_pct, viability_pct
            )

            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.debug(
                    "Annotating frame {}/{} ({:.0f}%)",
                    frame_idx, total, 100 * frame_idx / max(total, 1)
                )

        cap.release()
        writer.release()
        logger.success("Annotated video saved → {}", out_path)
        return str(out_path)

    # ----------------------------------------------------------

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        total_sperm: int,
        prog_pct: float,
        nprog_pct: float,
        imm_pct: float,
        norm_pct: float,
        viability_pct: float,
    ) -> np.ndarray:
        """
        Draw a semi-transparent HUD in the top-left corner of *frame*.

        Parameters
        ----------
        frame : np.ndarray (BGR)
        frame_idx : int
        total_frames : int
        total_sperm, prog_pct, nprog_pct, imm_pct, norm_pct, viability_pct : numeric

        Returns
        -------
        np.ndarray (BGR)
        """
        lines = [
            f"Frame    : {frame_idx}/{total_frames}",
            f"Sperm    : {total_sperm}",
            f"Progr.   : {prog_pct:.1f}%",
            f"Non-prog.: {nprog_pct:.1f}%",
            f"Immotile : {imm_pct:.1f}%",
            f"Normal   : {norm_pct:.1f}%",
            f"Viability: {viability_pct:.1f}%",
        ]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness  = 1
        line_h     = 20
        pad        = 6
        box_w      = 230
        box_h      = len(lines) * line_h + pad * 2

        # Semi-transparent black background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), _HUD_BG_COLOUR, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h
            cv2.putText(
                frame, line, (pad, y),
                font, font_scale, _HUD_FG_COLOUR, thickness, cv2.LINE_AA
            )

        return frame

    # ══════════════════════════════════════════════════════════
    # Summary builder
    # ══════════════════════════════════════════════════════════

    def _generate_summary(
        self,
        video_path: Path,
        n_frames: int,
        fps: float,
        unique_tracks: int,
        motility_summary: Dict,
        morphology_summary: Dict,
        viability_pct: float,
        elapsed: float,
    ) -> Dict[str, Any]:
        """
        Compile all pipeline outputs into the canonical result dictionary.

        Returns
        -------
        dict
        """
        return {
            # Counts
            "total_sperm":    unique_tracks,
            # Motility
            "progressive":    motility_summary.get("progressive_pct",    0.0),
            "non_progressive":motility_summary.get("nonprogressive_pct", 0.0),
            "immotile":       motility_summary.get("immotile_pct",       0.0),
            # Velocity
            "mean_VCL":       motility_summary.get("mean_VCL",  0.0),
            "mean_VSL":       motility_summary.get("mean_VSL",  0.0),
            "mean_VAP":       motility_summary.get("mean_VAP",  0.0),
            # Morphology
            "normal_count":   morphology_summary.get("normal_count",   0),
            "abnormal_count": morphology_summary.get("abnormal_count", 0),
            "normal_pct":     morphology_summary.get("normal_pct",     0.0),
            "abnormal_pct":   morphology_summary.get("abnormal_pct",   0.0),
            # Viability
            "predicted_viability": viability_pct,
            # Meta
            "processing_time": elapsed,
            "fps":             fps,
            "video_frames":    n_frames,
            "video_name":      video_path.name,
        }

    # ══════════════════════════════════════════════════════════
    # Output writers
    # ══════════════════════════════════════════════════════════

    def _write_motility_csv(
        self,
        per_track_df: pd.DataFrame,
        stem: str,
    ) -> None:
        """
        Save per-track CASA kinematics to motility.csv.

        Columns (from MotilityAnalyser._compute_kinematics +
                 _build_summary): track_id, VCL, VSL, VAP, LIN, STR,
                 WOB, ALH, total_distance (Distance), displacement
                 (Velocity proxy), motility_class (Classification).

        Parameters
        ----------
        per_track_df : pd.DataFrame
        stem : str
        """
        if per_track_df is None or per_track_df.empty:
            logger.warning("No motility data — motility.csv not written")
            return

        # Rename to match the required CSV column headers
        col_map = {
            "total_distance": "Distance",
            "displacement":   "Velocity",
            "motility_class": "Classification",
        }
        out = per_track_df.rename(columns=col_map)

        # Ensure the mandatory columns exist (fill with 0 if missing)
        required = ["track_id", "VCL", "VSL", "VAP", "LIN", "STR",
                    "WOB", "ALH", "Distance", "Velocity", "Classification"]
        for col in required:
            if col not in out.columns:
                out[col] = 0

        out = out[required]
        path = self.out_csv / f"{stem}_motility.csv"
        out.to_csv(path, index=False)
        logger.info("Motility CSV saved → {}", path)

    # ----------------------------------------------------------

    def _write_morphology_csv(
        self,
        morph_rows: List[Dict],
        stem: str,
    ) -> None:
        """
        Save per-detection morphology labels to morphology.csv.

        Columns: track_id, Morphology, Confidence.

        Parameters
        ----------
        morph_rows : list of dicts
        stem : str
        """
        if not morph_rows:
            logger.warning("No morphology data — morphology.csv not written")
            return

        df = pd.DataFrame(morph_rows)[["track_id", "Morphology", "Confidence"]]
        path = self.out_csv / f"{stem}_morphology.csv"
        df.to_csv(path, index=False)
        logger.info("Morphology CSV saved → {}", path)

    # ----------------------------------------------------------

    def _write_report_csv(
        self,
        results: Dict[str, Any],
        stem: str,
    ) -> None:
        """
        Save the overall analysis summary as a flat CSV.

        Parameters
        ----------
        results : dict
        stem : str
        """
        rows = [{"Parameter": k, "Value": v} for k, v in results.items()]
        df   = pd.DataFrame(rows)
        path = self.out_reports / f"{stem}_analysis_report.csv"
        df.to_csv(path, index=False)
        logger.info("Analysis report CSV saved → {}", path)

    # ----------------------------------------------------------

    def _write_json(
        self,
        results: Dict[str, Any],
        video_path: Path,
        annotated_video_path: Optional[str],
        stem: str,
    ) -> None:
        """
        Write summary.json with video metadata plus all analysis results.

        Parameters
        ----------
        results : dict
        video_path : Path
        annotated_video_path : str | None
        stem : str
        """
        payload = {
            "metadata": {
                "analysis_date":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "software":           "Automated Semen Viability and Counting Analysis v1.0",
                "video_source":       str(video_path.resolve()),
                "annotated_video":    annotated_video_path or "",
                "video_frames":       results.get("video_frames", 0),
                "fps":                results.get("fps", 0.0),
                "processing_time_s":  results.get("processing_time", 0.0),
            },
            "counts": {
                "total_sperm":    results.get("total_sperm", 0),
                "normal_count":   results.get("normal_count", 0),
                "abnormal_count": results.get("abnormal_count", 0),
            },
            "motility_summary": {
                "progressive_pct":    results.get("progressive",     0.0),
                "nonprogressive_pct": results.get("non_progressive", 0.0),
                "immotile_pct":       results.get("immotile",        0.0),
                "mean_VCL_um_s":      results.get("mean_VCL",        0.0),
                "mean_VSL_um_s":      results.get("mean_VSL",        0.0),
                "mean_VAP_um_s":      results.get("mean_VAP",        0.0),
            },
            "morphology_summary": {
                "normal_pct":   results.get("normal_pct",   0.0),
                "abnormal_pct": results.get("abnormal_pct", 0.0),
            },
            "viability": {
                "predicted_pct": results.get("predicted_viability", 0.0),
            },
        }

        path = self.out_json / f"{stem}_summary.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Summary JSON saved → {}", path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    """Return configured argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m src.run_analysis",
        description=(
            "Automated Semen Viability and Counting Analysis — "
            "end-to-end pipeline orchestrator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        metavar="VIDEO",
        help="Path to the microscope semen video file.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for output files.  "
            "When omitted, paths from configs/config.yaml are used."
        ),
    )
    p.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an annotated output video.",
    )
    p.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write summary.json.",
    )
    p.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write motility.csv, morphology.csv, and analysis_report.csv.",
    )
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        metavar="PATH",
        help="Path to the project YAML configuration file.",
    )
    return p


def main() -> None:
    """
    CLI entry point.

    Loads config, instantiates AutomatedSemenAnalysis, runs the
    pipeline, and exits with a non-zero code on failure.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error("Config not found: {}", exc)
        sys.exit(1)

    # Instantiate and run
    try:
        pipeline = AutomatedSemenAnalysis(
            config      = config,
            output_root = args.output,
        )
        results = pipeline.run_video(
            video_path  = args.input,
            save_video  = args.save_video,
            save_json   = args.save_json,
            save_csv    = args.save_csv,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("{}", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error: {}", exc)
        sys.exit(1)

    # Print final summary to stdout
    _print_console_summary(results)


def _print_console_summary(results: Dict[str, Any]) -> None:
    """Print a formatted console summary of *results*."""
    div = "=" * 52
    print(f"\n{div}")
    print("       SEMEN ANALYSIS COMPLETE")
    print(div)
    print(f"  {'Video':<26}: {results.get('video_name', '')}")
    print(f"  {'Frames':<26}: {results.get('video_frames', 0)}")
    print(f"  {'FPS':<26}: {results.get('fps', 0.0):.1f}")
    print(div)
    print(f"  {'Total Sperm (tracks)':<26}: {results.get('total_sperm', 0)}")
    print(f"  {'Progressive':<26}: {results.get('progressive', 0.0):.1f}%")
    print(f"  {'Non-Progressive':<26}: {results.get('non_progressive', 0.0):.1f}%")
    print(f"  {'Immotile':<26}: {results.get('immotile', 0.0):.1f}%")
    print(f"  {'Mean VCL (µm/s)':<26}: {results.get('mean_VCL', 0.0):.2f}")
    print(f"  {'Mean VSL (µm/s)':<26}: {results.get('mean_VSL', 0.0):.2f}")
    print(f"  {'Mean VAP (µm/s)':<26}: {results.get('mean_VAP', 0.0):.2f}")
    print(div)
    print(f"  {'Normal Morphology':<26}: {results.get('normal_pct', 0.0):.1f}%")
    print(f"  {'Abnormal Morphology':<26}: {results.get('abnormal_pct', 0.0):.1f}%")
    print(div)
    print(f"  {'Predicted Viability':<26}: {results.get('predicted_viability', 0.0):.1f}%")
    print(div)
    quality = results.get("quality_score", {})

    print(div)
    print(f"  {'Quality Score':<26}: {quality.get('score', 0):.2f}/100")
    print(f"  {'Category':<26}: {quality.get('category', 'N/A')}")

    if quality.get("who_flags"):
       print("  WHO Flags:")
       for flag in quality["who_flags"]:
           print(f"    - {flag}")

    if quality.get("recommendations"):
        print(div)
        print("  Recommendations:")
        for rec in quality["recommendations"]:
            print(f"   • {rec}")
    print(f"  {'Processing Time':<26}: {results.get('processing_time', 0.0):.1f}s")
    print(f"{div}\n")


if __name__ == "__main__":
    main()