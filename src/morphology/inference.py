"""
src/morphology/inference.py
============================
Phase 5 · Morphology Classification — Inference Engine

Loads the trained EfficientNetB0 checkpoint and classifies
individual sperm crop images as Normal / Abnormal.

Also provides batch inference over a list of cropped images.

Usage
-----
    from src.morphology.inference import MorphologyClassifier

    clf = MorphologyClassifier()
    result = clf.classify_image("path/to/crop.jpg")
    print(result)   # {"class": "Normal", "confidence": 0.94, "probabilities": {...}}

    # Batch from detection boxes
    results = clf.classify_crops_from_frame(frame_bgr, detection_boxes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.helpers import load_config, read_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Morphology Classifier
# ══════════════════════════════════════════════════════════════

class MorphologyClassifier:
    """
    EfficientNetB0 morphology classifier (Normal / Abnormal).

    Parameters
    ----------
    config : dict, optional
    weights_path : str, optional
        Path to .pth checkpoint.  Defaults to models/morphology/efficientnet_morphology.pth
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 weights_path: Optional[str] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        morph_cfg = config["morphology"]
        self.img_size    = morph_cfg["image_size"]
        self.class_names = morph_cfg["class_names"]   # ["Normal", "Abnormal"]
        self.device_str  = morph_cfg["device"]

        if weights_path is None:
            weights_path = config["paths"]["models"]["morphology"]
        self.weights_path = Path(weights_path)

        self._model  = None
        self._device = None
        self._tf     = None   # torchvision transform

    # ----------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the checkpoint on first use."""
        if self._model is not None:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms
            from torchvision.models import efficientnet_b0
        except ImportError:
            raise ImportError("torch and torchvision are required.")

        if self.device_str == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Build model architecture (must match train_morphology.py)
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, len(self.class_names)),
        )

        if self.weights_path.exists():
            checkpoint = torch.load(str(self.weights_path),
                                    map_location=self._device)
            model.load_state_dict(checkpoint["model_state_dict"])
            # Override class_names if checkpoint stores them
            #if "class_names" in checkpoint:
               # self.class_names = checkpoint["class_names"]
            logger.info("Morphology model loaded from {}", self.weights_path)
        else:
            logger.warning(
                "Morphology weights not found at {}. "
                "Using random weights (train the model first).",
                self.weights_path
            )

        model.eval()
        self._model = model.to(self._device)

        # Inference transform (no augmentation)
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ----------------------------------------------------------

    def classify_image(self, image_path: str | Path | None = None,
                       image_array: Optional[np.ndarray] = None) -> Dict:
        """
        Classify a single sperm image as Normal or Abnormal.

        Parameters
        ----------
        image_path : str | Path, optional
        image_array : np.ndarray, optional
            BGR image.  Takes precedence over image_path.

        Returns
        -------
        dict
            {
              "class":       "Normal" | "Abnormal",
              "class_idx":   0 | 1,
              "confidence":  float (0-1),
              "probabilities": {"Normal": float, "Abnormal": float}
            }
        """
        import torch

        self._load()

        # ── Load image ─────────────────────────────────────────
        if image_array is not None:
            img_bgr = image_array
        elif image_path is not None:
            img_bgr = read_image(image_path)
            if img_bgr is None:
                return {"class": "Unknown", "confidence": 0.0, "class_idx": -1,
                        "probabilities": {}}
        else:
            raise ValueError("Provide image_path or image_array")

        # BGR → RGB for torchvision
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Pre-process ────────────────────────────────────────
        tensor = self._tf(img_rgb).unsqueeze(0).to(self._device)  # (1, 3, H, W)

        # ── Inference ──────────────────────────────────────────
        with torch.no_grad():
            logits = self._model(tensor)                   # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

        probs_np   = probs.cpu().numpy()
        class_idx  = int(np.argmax(probs_np))
        confidence = float(probs_np[class_idx])

        return {
            "class":         self.class_names[class_idx],
            "class_idx":     class_idx,
            "confidence":    round(confidence, 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.class_names, probs_np)
            },
        }

    # ----------------------------------------------------------

    def classify_crops_from_frame(self,
                                  frame: np.ndarray,
                                  boxes: List[List[float]],
                                  padding: int = 5) -> List[Dict]:
        """
        Classify all sperm crops detected in a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR microscope frame.
        boxes : list of [x1, y1, x2, y2]
            Detection bounding boxes.
        padding : int
            Extra pixels to add around each box.

        Returns
        -------
        list of dict
            One classification result per box.
        """
        h, w = frame.shape[:2]
        results = []

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            # Apply padding, clamp to frame
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                results.append({"class": "Unknown", "confidence": 0.0,
                                 "class_idx": -1, "probabilities": {}})
                continue

            crop = frame[y1:y2, x1:x2]
            results.append(self.classify_image(image_array=crop))

        return results

    # ----------------------------------------------------------

    def batch_summary(self, classifications: List[Dict]) -> Dict:
        """
        Aggregate a list of per-cell classification results.

        Parameters
        ----------
        classifications : list of dict

        Returns
        -------
        dict
            {normal_count, abnormal_count, normal_pct, abnormal_pct, total}
        """
        total    = len(classifications)
        normal   = sum(1 for r in classifications if r.get("class") == "Normal")
        abnormal = total - normal

        return {
            "total":         total,
            "normal_count":  normal,
            "abnormal_count": abnormal,
            "normal_pct":    round(100 * normal   / total, 2) if total else 0.0,
            "abnormal_pct":  round(100 * abnormal / total, 2) if total else 0.0,
        }
    