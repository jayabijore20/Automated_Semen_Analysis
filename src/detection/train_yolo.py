"""
src/detection/train_yolo.py
============================
Phase 2 · Sperm Detection — YOLOv8 Training Pipeline

Trains a YOLOv8 model on the EVISAN + Roboflow sperm dataset.
Saves the best checkpoint to models/detection/best.pt

How to Run
----------
    python -m src.detection.train_yolo

Key outputs
-----------
    models/detection/best.pt        ← best checkpoint
    runs/detect/sperm_yolov8/       ← Ultralytics training artefacts
    outputs/plots/training_curves.png
"""

import shutil
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.utils.helpers import ensure_dir, load_config, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# YOLOv8 Trainer
# ══════════════════════════════════════════════════════════════

class SpermDetectionTrainer:
    """
    Wraps Ultralytics YOLOv8 training for the sperm detection task.

    Parameters
    ----------
    config : dict
        Loaded project configuration.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        det_cfg   = config["detection"]
        paths_cfg = config["paths"]

        self.model_variant = det_cfg["model_variant"]       # e.g. "yolov8n.pt"
        self.epochs        = det_cfg["epochs"]
        self.batch_size    = det_cfg["batch_size"]
        self.img_size      = det_cfg["image_size"]
        self.device        = det_cfg["device"]
        self.patience      = det_cfg["patience"]
        self.lr0           = det_cfg["lr0"]
        self.workers       = det_cfg["workers"]

        self.dataset_yaml  = Path(paths_cfg["datasets"]["yolo_ready"]) / "dataset.yaml"
        self.model_out_dir = ensure_dir(Path(paths_cfg["models"]["detection"]).parent)
        self.plot_dir      = ensure_dir(paths_cfg["outputs"]["plots"])

        self.project_name  = "sperm_yolov8"

    # ----------------------------------------------------------

    @timer
    def train(self) -> str:
        """
        Launch YOLOv8 training.

        Returns
        -------
        str
            Path to the saved best.pt checkpoint.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

        logger.info("Initialising YOLOv8 model: {}", self.model_variant)
        model = YOLO(self.model_variant)

        # Verify dataset YAML exists
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.dataset_yaml}\n"
                "Run xml_to_yolo.py and split_dataset.py first."
            )

        logger.info("Starting training for {} epochs on device={}", self.epochs, self.device)
        logger.info("Dataset: {}", self.dataset_yaml)

        # ── Train ──────────────────────────────────────────────
        results = model.train(
            data       = str(self.dataset_yaml),
            epochs     = self.epochs,
            batch      = self.batch_size,
            imgsz      = self.img_size,
            device     = self.device,
            patience   = self.patience,
            lr0        = self.lr0,
            workers    = self.workers,
            project    = "runs/detect",
            name       = self.project_name,
            exist_ok   = True,
            verbose    = True,
            save       = True,
            plots      = True,
        )

        # ── Copy best checkpoint ───────────────────────────────
        # ── Copy best checkpoint ───────────────────────────────
        run_dir = next(Path("runs").rglob(self.project_name))

        best_src = run_dir / "weights" / "best.pt"
        best_dst = self.model_out_dir / "best.pt"
       

        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            logger.success("Best checkpoint saved → {}", best_dst)
        else:
            logger.warning("best.pt not found at expected path: {}", best_src)
            # Try to locate it
            for pt in Path(f"runs/detect/{self.project_name}").rglob("best.pt"):
                shutil.copy2(pt, best_dst)
                logger.success("Found and copied best.pt → {}", best_dst)
                break

        # ── Plot training curves ───────────────────────────────
        self._plot_training_curves(results)

        return str(best_dst)

    # ----------------------------------------------------------

    def validate(self, weights_path: Optional[str] = None) -> Dict:
        """
        Run validation on the test split and return metrics.

        Parameters
        ----------
        weights_path : str, optional
            Path to .pt file.  Defaults to models/detection/best.pt.

        Returns
        -------
        dict
            Precision, Recall, mAP50, mAP50-95.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed.")
            raise

        if weights_path is None:
            weights_path = str(self.model_out_dir / "best.pt")

        logger.info("Running validation with weights: {}", weights_path)
        model = YOLO(weights_path)

        metrics = model.val(
            data   = str(self.dataset_yaml),
            split  = "test",
            imgsz  = self.img_size,
            device = self.device,
            verbose= True,
        )

        results = {
            "precision":  float(metrics.box.p.mean()),
            "recall":     float(metrics.box.r.mean()),
            "mAP50":      float(metrics.box.map50),
            "mAP50_95":   float(metrics.box.map),
        }

        logger.success("Validation Results:")
        for k, v in results.items():
            logger.success("  {:12s}: {:.4f}", k, v)

        return results

    # ----------------------------------------------------------

    def _plot_training_curves(self, results) -> None:
        """
        Plot loss and metric curves from Ultralytics results object.

        Parameters
        ----------
        results : ultralytics.engine.results.Results
            Object returned by model.train().
        """
        run_dir = next(Path("runs").rglob(self.project_name))
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            logger.warning("results.csv not found, skipping plot")
            return

        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()   # strip whitespace from column headers

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("YOLOv8 Training Curves — Sperm Detection", fontsize=14, fontweight="bold")

        metric_map = {
            "train/box_loss": (axes[0, 0], "Train Box Loss",   "#e74c3c"),
            "train/cls_loss": (axes[0, 1], "Train Class Loss", "#3498db"),
            "train/dfl_loss": (axes[0, 2], "Train DFL Loss",   "#9b59b6"),
            "metrics/precision(B)": (axes[1, 0], "Precision", "#2ecc71"),
            "metrics/recall(B)":    (axes[1, 1], "Recall",    "#f39c12"),
            "metrics/mAP50(B)":     (axes[1, 2], "mAP@0.5",  "#1abc9c"),
        }

        epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

        for col, (ax, title, color) in metric_map.items():
            if col in df.columns:
                ax.plot(df[epoch_col], df[col], color=color, linewidth=2)
                ax.set_title(title, fontweight="bold")
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f"{title} (N/A)")
                ax.text(0.5, 0.5, "data unavailable", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

        plt.tight_layout()
        save_path = self.plot_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved → {}", save_path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg     = load_config()
    trainer = SpermDetectionTrainer(cfg)
    best_pt = trainer.train()
    metrics = trainer.validate(best_pt)