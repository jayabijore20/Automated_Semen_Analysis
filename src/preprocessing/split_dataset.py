"""
src/preprocessing/split_dataset.py
====================================
Phase 1 · Data Preprocessing — Train / Validation / Test Splitter

Takes the unified YOLO-ready pool produced by xml_to_yolo.py and
organises it into:

    datasets/yolo_ready/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Split ratios are read from configs/config.yaml (default 70 / 15 / 15).

How to Run
----------
    python -m src.preprocessing.split_dataset
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataset Splitter
# ══════════════════════════════════════════════════════════════

class DatasetSplitter:
    """
    Splits a flat pool of image/label pairs into stratified subsets.

    Parameters
    ----------
    config : dict
        Loaded project configuration.
    """

    # Valid image extensions
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        ds_cfg = config["dataset"]
        paths_cfg = config["paths"]["datasets"]

        self.yolo_root  = Path(paths_cfg["yolo_ready"])
        self.img_pool   = self.yolo_root / "all_images"
        self.lbl_pool   = self.yolo_root / "all_labels"

        self.train_r = ds_cfg["train_ratio"]
        self.val_r   = ds_cfg["val_ratio"]
        self.test_r  = ds_cfg["test_ratio"]
        self.seed    = ds_cfg["random_seed"]

        # Validate ratios
        total = self.train_r + self.val_r + self.test_r
        assert abs(total - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total}"

    # ----------------------------------------------------------

    def split(self) -> Dict[str, List[Path]]:
        """
        Perform the split, copy files, and return the split mapping.

        Returns
        -------
        dict
            {"train": [...], "val": [...], "test": [...]}
            — each value is a list of image Paths in that split.
        """
        # 1. Collect valid pairs (both image AND label must exist)
        pairs = self._collect_valid_pairs()
        logger.info("Valid image-label pairs: {}", len(pairs))

        if not pairs:
            logger.error("No valid pairs found in {}", self.img_pool)
            return {"train": [], "val": [], "test": []}

        # 2. Shuffle deterministically
        random.seed(self.seed)
        random.shuffle(pairs)

        # 3. Compute split indices
        n = len(pairs)
        n_train = int(n * self.train_r)
        n_val   = int(n * self.val_r)

        splits = {
            "train": pairs[:n_train],
            "val":   pairs[n_train: n_train + n_val],
            "test":  pairs[n_train + n_val:],
        }

        # 4. Copy files to split directories
        for split_name, split_pairs in splits.items():
            self._copy_split(split_name, split_pairs)

        # 5. Log summary
        self._log_summary(splits)

        # 6. Persist the split manifest to CSV
        self._save_manifest(splits)

        return splits

    # ----------------------------------------------------------

    def _collect_valid_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Return (image_path, label_path) tuples where BOTH files exist
        and the image is non-empty.
        """
        pairs: List[Tuple[Path, Path]] = []

        if not self.img_pool.exists():
            logger.warning("Image pool not found: {}", self.img_pool)
            return pairs

        for img_path in sorted(self.img_pool.iterdir()):
            if img_path.suffix.lower() not in self.IMG_EXTS:
                continue

            lbl_path = self.lbl_pool / (img_path.stem + ".txt")
            if not lbl_path.exists():
                logger.debug("No label for {}, skipping", img_path.name)
                continue

            # Skip zero-byte label files
            if lbl_path.stat().st_size == 0:
                logger.debug("Empty label for {}, skipping", img_path.name)
                continue

            pairs.append((img_path, lbl_path))

        return pairs

    # ----------------------------------------------------------

    def _copy_split(self, split_name: str, pairs: List[Tuple[Path, Path]]) -> None:
        """
        Copy image and label files into split subdirectories.

        Parameters
        ----------
        split_name : "train" | "val" | "test"
        pairs : list of (img_path, lbl_path)
        """
        split_img_dir = ensure_dir(self.yolo_root / split_name / "images")
        split_lbl_dir = ensure_dir(self.yolo_root / split_name / "labels")

        for img_path, lbl_path in tqdm(pairs, desc=f"Copying {split_name}"):
            shutil.copy2(img_path, split_img_dir / img_path.name)
            shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

        logger.info("{}: {} samples copied", split_name, len(pairs))

    # ----------------------------------------------------------

    def _log_summary(self, splits: Dict[str, List]) -> None:
        """Print a summary table to the logger."""
        total = sum(len(v) for v in splits.values())
        logger.success("── Split Summary ──────────────────────")
        for name, pairs in splits.items():
            pct = 100 * len(pairs) / total if total else 0
            logger.success("  {:6s}: {:5d} samples ({:.1f}%)", name, len(pairs), pct)
        logger.success("  Total : {:5d} samples", total)

    # ----------------------------------------------------------

    def _save_manifest(self, splits: Dict[str, List[Tuple[Path, Path]]]) -> None:
        """Save a CSV manifest listing every file and its assigned split."""
        rows = []
        for split_name, pairs in splits.items():
            for img_path, lbl_path in pairs:
                rows.append({
                    "split":      split_name,
                    "image":      img_path.name,
                    "label":      lbl_path.name,
                })
        df = pd.DataFrame(rows)
        manifest_path = self.yolo_root / "split_manifest.csv"
        df.to_csv(manifest_path, index=False)
        logger.info("Split manifest saved → {}", manifest_path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    splitter = DatasetSplitter(cfg)
    splitter.split()