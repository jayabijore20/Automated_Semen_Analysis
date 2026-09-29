"""
src/tracking/track_sperm.py
============================
Phase 3 · Sperm Tracking — YOLOv8 + ByteTrack

Uses Ultralytics' built-in ByteTrack integration to:
  1. Detect sperm in every video frame.
  2. Assign and maintain unique track IDs across frames.
  3. Accumulate frame-by-frame centroid trajectories.
  4. Draw colour-coded trajectories on output video.

Output
------
    outputs/tracking/<video_stem>_tracked.mp4
    outputs/tracking/<video_stem>_trajectories.png
    outputs/tracking/<video_stem>_tracks.csv

    Each row in tracks.csv:
        frame_id | track_id | x1 | y1 | x2 | y2 | cx | cy | conf

How to Run
----------
    python -m src.tracking.track_sperm --video path/to/video.avi
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, generate_color_palette, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Track dataclass
# ══════════════════════════════════════════════════════════════

class Track:
    """
    Holds the centroid history for a single tracked sperm.

    Attributes
    ----------
    track_id : int
    centroids : list of (cx, cy, frame_id)
    color : (B, G, R)
    """
    __slots__ = ("track_id", "centroids", "color")

    def __init__(self, track_id: int, color: Tuple[int, int, int]) -> None:
        self.track_id  = track_id
        self.color     = color
        self.centroids: List[Tuple[float, float, int]] = []

    def add_centroid(self, cx: float, cy: float, frame_id: int) -> None:
        self.centroids.append((cx, cy, frame_id))

    @property
    def positions(self) -> List[Tuple[float, float]]:
        """Return list of (cx, cy) positions."""
        return [(c[0], c[1]) for c in self.centroids]


# ══════════════════════════════════════════════════════════════
# Sperm Tracker
# ══════════════════════════════════════════════════════════════

class SpermTracker:
    """
    Runs YOLO detection + ByteTrack on a video and records trajectories.

    Parameters
    ----------
    config : dict, optional
    weights_path : str, optional
        Override for the detection model checkpoint.
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 weights_path: Optional[str] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        det_cfg     = config["detection"]
        track_cfg   = config["tracking"]
        paths_cfg   = config["paths"]

        self.conf       = track_cfg["conf_threshold"]
        self.iou        = track_cfg["iou_threshold"]
        self.traj_len   = track_cfg["trajectory_len"]
        self.img_size   = det_cfg["image_size"]
        self.device     = det_cfg["device"]

        if weights_path is None:
            weights_path = paths_cfg["models"]["detection"]
        self.weights_path = Path(weights_path)

        self.out_dir = ensure_dir(paths_cfg["outputs"]["tracking"])

        # Track registry: track_id → Track
        self._tracks: Dict[int, Track] = {}
        # Deterministic colour palette for track IDs
        self._palette = generate_color_palette(500)

    # ----------------------------------------------------------
    import traceback

    print("=" * 80)
    print("TRACK_VIDEO CALLED")
    traceback.print_stack(limit=8)
    print("=" * 80)
    
    def track_video(
        self,
        video_path: str | Path,
        progress_callback=None,
    ) -> Dict:
        
        """
        Process a full video: detect → track → annotate → save.

        Parameters
        ----------
        video_path : str | Path

        Returns
        -------
        dict
            Summary with keys: n_frames, unique_tracks, tracks_df (DataFrame)
        """
        video_path = Path(video_path)
        logger.info("Tracking video: {}", video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: {}", video_path)
            return {}

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_video_path = self.out_dir / f"{video_path.stem}_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

        # Ultralytics YOLO tracker
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required: pip install ultralytics")

        model = self._load_model()

        # All raw detection rows for CSV output
        all_rows: List[Dict] = []
        frame_id = 0

        logger.info("Processing {} frames at {:.1f} FPS", total, fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            # ── ByteTrack via Ultralytics ──────────────────────
            track_results = model.track(
                source  = frame,
                conf    = self.conf,
                iou     = self.iou,
                imgsz   = self.img_size,
                device  = self.device,
                tracker = "bytetrack.yaml",
                persist = True,          # keep IDs across frames
                verbose = False,
            )

            # ── Parse track results ────────────────────────────
            annotated = frame.copy()

            if track_results and track_results[0].boxes is not None:
                boxes_xyxy = track_results[0].boxes.xyxy.cpu().numpy()
                confs      = track_results[0].boxes.conf.cpu().numpy()

                # Track IDs may not always be present (depends on ultralytics version)
                if track_results[0].boxes.id is not None:
                    ids = track_results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.arange(len(boxes_xyxy))

                for tid, box, conf in zip(ids, boxes_xyxy, confs):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # Register / update track
                    self._update_track(int(tid), cx, cy, frame_id)

                    # Row for CSV
                    all_rows.append({
                        "frame_id": frame_id,
                        "track_id": int(tid),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "cx": round(cx, 2), "cy": round(cy, 2),
                        "conf": round(float(conf), 4),
                    })

                    # Draw box + ID
                    color = self._get_color(int(tid))
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                  color, 2)
                    cv2.putText(annotated, f"#{tid}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # ── Draw trajectories ──────────────────────────────
            annotated = self._draw_trajectories(annotated, frame_id)

            # ── Frame counter overlay ──────────────────────────
            info = f"Frame: {frame_id}/{total}  Tracks: {len(self._tracks)}"
            cv2.rectangle(annotated, (0, 0), (350, 32), (0, 0, 0), -1)
            cv2.putText(annotated, info, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA)

            writer.write(annotated)
            
            # ----------------------------
            # Update Streamlit progress
            # ----------------------------#
            if progress_callback and total > 0:
                progress_callback(frame_id / total)

            if frame_id % 50 == 0:
                logger.info(
                    "Processed {}/{} frames ({:.1f}%)",
                    frame_id,
                    total,
                    (frame_id / total) * 100
                )
             
        cap.release()
        writer.release()
        logger.success("Tracked video saved → {}", out_video_path)

        # ── Save tracks CSV ────────────────────────────────────
        tracks_df = pd.DataFrame(all_rows)
        csv_path  = self.out_dir / f"{video_path.stem}_tracks.csv"
        tracks_df.to_csv(csv_path, index=False)
        logger.info("Tracks CSV saved → {}", csv_path)

        # ── Trajectory plot ────────────────────────────────────
        self._plot_trajectories(video_path.stem, w, h)

        return {
            "n_frames":      frame_id,
            "unique_tracks": len(self._tracks),
            "tracks_df":     tracks_df,
        }

    # ----------------------------------------------------------

    def _load_model(self):
        """Load YOLO model, falling back to pretrained if needed."""
        from ultralytics import YOLO
        if self.weights_path.exists():
            return YOLO(str(self.weights_path))
        logger.warning("Weights not found; using yolov8n.pt baseline")
        return YOLO("yolov8n.pt")

    # ----------------------------------------------------------

    def _update_track(self, tid: int, cx: float, cy: float, frame_id: int) -> None:
        """Register a centroid into the track registry."""
        if tid not in self._tracks:
            self._tracks[tid] = Track(tid, self._get_color(tid))
        self._tracks[tid].add_centroid(cx, cy, frame_id)

    # ----------------------------------------------------------

    def _get_color(self, tid: int) -> Tuple[int, int, int]:
        """Return deterministic colour for a track ID."""
        return self._palette[tid % len(self._palette)]

    # ----------------------------------------------------------

    def _draw_trajectories(self, frame: np.ndarray, current_frame: int) -> np.ndarray:
        """
        Draw tail lines for all active tracks up to *self.traj_len* points.

        Parameters
        ----------
        frame : np.ndarray  (BGR)
        current_frame : int

        Returns
        -------
        np.ndarray
        """
        for track in self._tracks.values():
            pts = track.positions
            # Use only the last traj_len positions
            pts = pts[-self.traj_len:]
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                # Fade colour with age
                alpha = i / len(pts)
                c = track.color
                faded = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]),     int(pts[i][1]))
                cv2.line(frame, p1, p2, faded, 2, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------

    def _plot_trajectories(self, stem: str, frame_w: int, frame_h: int) -> None:
        """
        Save a static trajectory map (all tracks overlaid on black canvas).

        Parameters
        ----------
        stem : str
            Base name for the output file.
        frame_w, frame_h : int
            Video frame dimensions (for axis limits).
        """
        fig, ax = plt.subplots(figsize=(12, 8), facecolor="black")
        ax.set_facecolor("black")
        ax.set_xlim(0, frame_w)
        ax.set_ylim(frame_h, 0)   # invert y so (0,0) is top-left like cv2
        ax.set_title("Sperm Trajectory Map", color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        for track in self._tracks.values():
            pts = track.positions
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Normalise colour to [0,1] for matplotlib
            c_norm = (track.color[2] / 255, track.color[1] / 255, track.color[0] / 255)
            ax.plot(xs, ys, color=c_norm, linewidth=1.0, alpha=0.8)
            # Mark start
            ax.scatter(xs[0],  ys[0],  color="lime",   s=8,  zorder=3)
            # Mark end
            ax.scatter(xs[-1], ys[-1], color="red",    s=8,  zorder=3)

        ax.text(0.02, 0.98, f"Total tracks: {len(self._tracks)}",
                color="white", transform=ax.transAxes, va="top", fontsize=10)

        plt.tight_layout()
        save_path = self.out_dir / f"{stem}_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        plt.close()
        logger.info("Trajectory plot saved → {}", save_path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track sperm in a video using ByteTrack")
    parser.add_argument("--video",   required=True,  help="Path to input video")
    parser.add_argument("--weights", default=None,   help="Path to YOLO weights (best.pt)")
    args = parser.parse_args()

    cfg     = load_config()
    tracker = SpermTracker(cfg, weights_path=args.weights)
    summary = tracker.track_video(args.video)

    print(f"\nTracking complete!")
    print(f"  Frames processed : {summary.get('n_frames', 0)}")
    print(f"  Unique tracks    : {summary.get('unique_tracks', 0)}")