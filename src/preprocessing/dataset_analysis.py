"""
src/preprocessing/dataset_analysis.py
=======================================
Phase 1 · Data Preprocessing — Dataset Statistics & Visualisation

Analyses the YOLO-formatted dataset and produces:
  - Per-split sample counts
  - Bounding-box size distribution histogram
  - Box count per image histogram
  - Sample grid of annotated images
  - Summary CSV

Output files → outputs/plots/

How to Run
----------
    python -m src.preprocessing.dataset_analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.helpers import ensure_dir, load_config, read_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataset Analyser
# ══════════════════════════════════════════════════════════════

class DatasetAnalyser:
    """
    Generates statistics and visualisations for the YOLO dataset.

    Parameters
    ----------
    config : dict
        Loaded project configuration.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        self.yolo_root  = Path(config["paths"]["datasets"]["yolo_ready"])
        self.plot_dir   = ensure_dir(config["paths"]["outputs"]["plots"])
        self.class_names = config["dataset"]["class_names"]

    # ----------------------------------------------------------

    def run(self) -> Dict:
        """
        Full analysis pipeline.

        Returns
        -------
        dict
            Summary statistics for all splits.
        """
        logger.info("Starting dataset analysis")
        all_stats = {}

        for split in ("train", "val", "test"):
            split_dir = self.yolo_root / split
            if not split_dir.exists():
                logger.warning("Split not found: {}", split_dir)
                continue
            logger.info("Analysing split: {}", split)
            stats = self._analyse_split(split, split_dir)
            all_stats[split] = stats

        # Combined visualisations
        self._plot_split_distribution(all_stats)
        self._plot_bbox_stats(all_stats)
        self._save_summary_csv(all_stats)

        # Sample grid from training set
        train_img_dir = self.yolo_root / "train" / "images"
        train_lbl_dir = self.yolo_root / "train" / "labels"
        if train_img_dir.exists():
            self._plot_sample_grid(train_img_dir, train_lbl_dir)

        logger.success("Dataset analysis complete. Plots saved to {}", self.plot_dir)
        return all_stats

    # ----------------------------------------------------------

    def _analyse_split(self, split: str, split_dir: Path) -> Dict:
        """
        Collect per-split statistics.

        Parameters
        ----------
        split : str
        split_dir : Path

        Returns
        -------
        dict with keys:
            n_images, n_boxes, boxes_per_img (list),
            box_widths (list, normalised), box_heights (list, normalised)
        """
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        img_paths = sorted(img_dir.glob("*")) if img_dir.exists() else []
        img_paths = [p for p in img_paths
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        boxes_per_img: List[int] = []
        box_widths:    List[float] = []
        box_heights:   List[float] = []

        for img_path in tqdm(img_paths, desc=f"  {split}", leave=False):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                boxes_per_img.append(0)
                continue

            lines = lbl_path.read_text().strip().splitlines()
            valid = []
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    _, cx, cy, w, h = parts
                    box_widths.append(float(w))
                    box_heights.append(float(h))
                    valid.append(line)
            boxes_per_img.append(len(valid))

        return {
            "n_images":      len(img_paths),
            "n_boxes":       sum(boxes_per_img),
            "boxes_per_img": boxes_per_img,
            "box_widths":    box_widths,
            "box_heights":   box_heights,
        }

    # ----------------------------------------------------------

    def _plot_split_distribution(self, stats: Dict) -> None:
        """Bar chart: sample count per split."""
        splits = list(stats.keys())
        counts = [stats[s]["n_images"] for s in splits]
        box_counts = [stats[s]["n_boxes"] for s in splits]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Dataset Split Distribution", fontsize=14, fontweight="bold")

        # Images per split
        axes[0].bar(splits, counts, color=["#3498db", "#2ecc71", "#e74c3c"])
        axes[0].set_title("Images per Split")
        axes[0].set_ylabel("Count")
        for i, (s, c) in enumerate(zip(splits, counts)):
            axes[0].text(i, c + 5, str(c), ha="center", fontweight="bold")

        # Boxes per split
        axes[1].bar(splits, box_counts, color=["#9b59b6", "#f39c12", "#1abc9c"])
        axes[1].set_title("Bounding Boxes per Split")
        axes[1].set_ylabel("Count")
        for i, (s, c) in enumerate(zip(splits, box_counts)):
            axes[1].text(i, c + 5, str(c), ha="center", fontweight="bold")

        plt.tight_layout()
        save_path = self.plot_dir / "split_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → {}", save_path)

    # ----------------------------------------------------------

    def _plot_bbox_stats(self, stats: Dict) -> None:
        """Histogram of bbox width, height, and boxes-per-image."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Bounding Box Statistics (Training Set)", fontsize=14, fontweight="bold")

        train = stats.get("train", {})

        widths   = train.get("box_widths",    [])
        heights  = train.get("box_heights",   [])
        per_img  = train.get("boxes_per_img", [])

        def hist(ax, data, title, xlabel, color):
            if data:
                ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
                ax.axvline(np.mean(data), color="red", linestyle="--",
                           label=f"Mean: {np.mean(data):.3f}")
                ax.legend(fontsize=9)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frequency")

        hist(axes[0], widths,  "Box Width Distribution",    "Normalised Width",  "#3498db")
        hist(axes[1], heights, "Box Height Distribution",   "Normalised Height", "#e74c3c")
        hist(axes[2], per_img, "Boxes per Image",           "Box Count",          "#2ecc71")

        plt.tight_layout()
        save_path = self.plot_dir / "bbox_statistics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → {}", save_path)

    # ----------------------------------------------------------

    def _plot_sample_grid(self, img_dir: Path, lbl_dir: Path,
                          n_cols: int = 4, n_rows: int = 3) -> None:
        """
        Draw a grid of annotated sample images.

        Parameters
        ----------
        img_dir : Path
        lbl_dir : Path
        n_cols, n_rows : int
            Grid dimensions.
        """
        img_paths = sorted(img_dir.glob("*.jpg"))[:n_cols * n_rows]
        if not img_paths:
            img_paths = sorted(img_dir.glob("*.png"))[:n_cols * n_rows]
        if not img_paths:
            logger.warning("No images found for sample grid")
            return

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        fig.suptitle("Annotated Sample Images (Training Set)", fontsize=14, fontweight="bold")
        axes = axes.flatten()

        for ax, img_path in zip(axes, img_paths):
            img_bgr = read_image(img_path)
            if img_bgr is None:
                ax.axis("off")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            ax.imshow(img_rgb)
            ax.set_title(img_path.stem[:20], fontsize=7)
            ax.axis("off")

            # Draw ground-truth boxes
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = [float(v) for v in parts]
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    rect = patches.Rectangle(
                        (x1, y1), bw * w, bh * h,
                        linewidth=1, edgecolor="#00ff00", facecolor="none"
                    )
                    ax.add_patch(rect)

        # Turn off any unused axes
        for ax in axes[len(img_paths):]:
            ax.axis("off")

        plt.tight_layout()
        save_path = self.plot_dir / "sample_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved → {}", save_path)

    # ----------------------------------------------------------

    def _save_summary_csv(self, stats: Dict) -> None:
        """Save per-split summary to CSV."""
        rows = []
        for split, s in stats.items():
            bpi = s.get("boxes_per_img", [])
            rows.append({
                "split":             split,
                "n_images":          s.get("n_images", 0),
                "n_boxes":           s.get("n_boxes",  0),
                "avg_boxes_per_img": round(float(np.mean(bpi)), 2) if bpi else 0,
                "max_boxes_per_img": max(bpi) if bpi else 0,
            })
        df = pd.DataFrame(rows)
        out_path = self.plot_dir / "dataset_summary.csv"
        df.to_csv(out_path, index=False)
        logger.info("Summary CSV → {}", out_path)
        logger.info("\n{}", df.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    analyser = DatasetAnalyser(cfg)
    analyser.run()