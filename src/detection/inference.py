"""
src/detection/inference.py
============================
Phase 2 · Sperm Detection — YOLOv8 Inference Pipeline

Loads the trained sperm-detection checkpoint (``models/detection/best.pt``)
and exposes a clean API for running detection on single frames, single
images, and batches of images.

Falls back to the pretrained ``yolov8n.pt`` weights (as configured under
``detection.model_variant`` in ``configs/config.yaml``) if the trained
checkpoint has not been produced yet.

How to Run
----------
    python -m src.detection.inference --image path/to/frame.jpg

Key outputs
-----------
    outputs/detections/<name>_annotated.jpg   ← annotated image(s)
    DetectionResult objects                   ← programmatic access
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from src.utils.helpers import ensure_dir, load_config, read_image, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """
    Container for the outcome of a single-frame sperm detection pass.

    Attributes
    ----------
    count : int
        Number of sperm detected in the frame.
    boxes : List[List[float]]
        Bounding boxes in ``[x1, y1, x2, y2]`` pixel-coordinate format.
    confidences : List[float]
        Confidence score for each corresponding box.
    class_ids : List[int]
        YOLO class index for each corresponding box.
    annotated_img : Optional[np.ndarray]
        Copy of the input frame with bounding boxes and confidence
        labels drawn on it. ``None`` if annotation was not requested.
    """

    count: int
    boxes: List[List[float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    annotated_img: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the result to a plain dictionary (excludes the image
        array, which is not JSON-serialisable).

        Returns
        -------
        dict
            Dictionary with ``count``, ``boxes``, ``confidences`` and
            ``class_ids`` keys.
        """
        return {
            "count": self.count,
            "boxes": self.boxes,
            "confidences": self.confidences,
            "class_ids": self.class_ids,
        }


# ══════════════════════════════════════════════════════════════
# Sperm Detector
# ══════════════════════════════════════════════════════════════

class SpermDetector:
    """
    Wraps a trained (or fallback pretrained) YOLOv8 model to perform
    sperm detection on frames, single images, and batches of images.

    Parameters
    ----------
    config : dict, optional
        Loaded project configuration. If ``None``, it is loaded from
        ``configs/config.yaml`` via :func:`src.utils.helpers.load_config`.
    weights_path : str, optional
        Explicit path to a ``.pt`` checkpoint. If ``None``, the path is
        resolved from ``paths.models.detection`` in the config, falling
        back to ``detection.model_variant`` (e.g. ``yolov8n.pt``) if the
        trained checkpoint does not exist on disk.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights_path: Optional[str] = None,
    ) -> None:
        if config is None:
            config = load_config()
        self.config = config

        det_cfg = config["detection"]
        paths_cfg = config["paths"]

        self.conf_threshold: float = det_cfg["conf_threshold"]
        self.iou_threshold: float = det_cfg["iou_threshold"]
        self.img_size: int = det_cfg["image_size"]
        self.device: str = det_cfg["device"]
        self.fallback_variant: str = det_cfg["model_variant"]

        self.class_names: List[str] = config.get("dataset", {}).get(
            "class_names", ["sperm"]
        )

        self.trained_weights_path: Path = Path(paths_cfg["models"]["detection"])
        self.output_dir: Path = ensure_dir(paths_cfg["outputs"]["detections"])

        self.weights_path: str = weights_path or self._resolve_weights_path()
        self.model = self._load_model()

    # ----------------------------------------------------------

    def _resolve_weights_path(self) -> str:
        """
        Determine which checkpoint to load: the trained ``best.pt`` if it
        exists, otherwise the pretrained fallback variant configured in
        ``detection.model_variant``.

        Returns
        -------
        str
            Path (or model name) to hand to ``ultralytics.YOLO``.
        """
        if self.trained_weights_path.exists():
            logger.info("Trained checkpoint found: {}", self.trained_weights_path)
            return str(self.trained_weights_path)

        logger.warning(
            "Trained checkpoint not found at {}. Falling back to '{}'.",
            self.trained_weights_path,
            self.fallback_variant,
        )
        return self.fallback_variant

    # ----------------------------------------------------------

    def _load_model(self) -> Any:
        """
        Instantiate and return the Ultralytics YOLO model from
        ``self.weights_path``.

        Returns
        -------
        ultralytics.YOLO
            Loaded YOLO model ready for inference.

        Raises
        ------
        ImportError
            If the ``ultralytics`` package is not installed.
        RuntimeError
            If the model fails to load for any other reason.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise ImportError(
                "The 'ultralytics' package is required for SpermDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        try:
            logger.info("Loading YOLO model: {}", self.weights_path)
            model = YOLO(self.weights_path)
            logger.success("YOLO model loaded successfully from {}", self.weights_path)
            return model
        except Exception as exc:
            logger.error("Failed to load YOLO model from {}: {}", self.weights_path, exc)
            raise RuntimeError(
                f"Could not load YOLO model from '{self.weights_path}'"
            ) from exc

    # ----------------------------------------------------------

    def detect_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> DetectionResult:
        """
        Run detection on a single in-memory frame (e.g. from a video
        stream or webcam).

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H, W, 3).
        draw : bool
            If True, populate ``DetectionResult.annotated_img`` with
            bounding boxes and confidence labels drawn on a copy of
            ``frame``.

        Returns
        -------
        DetectionResult
            Detection outcome for this frame.

        Raises
        ------
        ValueError
            If ``frame`` is None or empty.
        """
        if frame is None or frame.size == 0:
            raise ValueError("detect_frame() received an empty or None frame.")

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO prediction failed on frame: {}", exc)
            raise RuntimeError("Inference failed on the provided frame.") from exc

        boxes: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        if results:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)

                boxes = xyxy.tolist()
                confidences = conf.tolist()
                class_ids = cls.tolist()

        count = len(boxes)
        annotated_img: Optional[np.ndarray] = None

        if draw:
            annotated_img = self._draw_detections(frame, boxes, confidences, class_ids)

        logger.debug("detect_frame(): {} sperm detected", count)

        return DetectionResult(
            count=count,
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            annotated_img=annotated_img,
        )

    # ----------------------------------------------------------

    @timer
    def detect_image(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        draw: bool = True,
    ) -> DetectionResult:
        """
        Run detection on a single image file on disk.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.
        save : bool
            If True, save the annotated image to
            ``paths.outputs.detections``.
        draw : bool
            If True, produce an annotated image.

        Returns
        -------
        DetectionResult
            Detection outcome for this image.

        Raises
        ------
        FileNotFoundError
            If the image cannot be read from disk.
        """
        image_path = Path(image_path)
        frame = read_image(image_path)

        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        result = self.detect_frame(frame, draw=draw)

        if save and result.annotated_img is not None:
            out_path = self.output_dir / f"{image_path.stem}_annotated.jpg"
            try:
                cv2.imwrite(str(out_path), result.annotated_img)
                logger.info("Annotated image saved → {}", out_path)
            except Exception as exc:
                logger.warning("Failed to save annotated image {}: {}", out_path, exc)

        logger.info(
            "detect_image({}): {} sperm detected", image_path.name, result.count
        )
        return result

    # ----------------------------------------------------------

    @timer
    def detect_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        save: bool = True,
        draw: bool = True,
    ) -> List[DetectionResult]:
        """
        Run detection on a batch of images, each supplied either as a
        file path or an already-loaded ``np.ndarray`` frame.

        Parameters
        ----------
        images : sequence of (str | Path | np.ndarray)
            Images to process.
        save : bool
            If True and an item is a file path, save its annotated
            image to ``paths.outputs.detections``.
        draw : bool
            If True, produce annotated images for every item.

        Returns
        -------
        list of DetectionResult
            One result per input item, in the same order. Items that
            fail to process are skipped with a logged warning (no
            placeholder is inserted, so the returned list may be
            shorter than the input if failures occur).
        """
        results: List[DetectionResult] = []

        for idx, item in enumerate(images):
            try:
                if isinstance(item, (str, Path)):
                    result = self.detect_image(item, save=save, draw=draw)
                elif isinstance(item, np.ndarray):
                    result = self.detect_frame(item, draw=draw)
                else:
                    logger.warning(
                        "detect_batch(): unsupported item type at index {}: {}",
                        idx,
                        type(item),
                    )
                    continue

                results.append(result)
            except Exception as exc:
                logger.error("detect_batch(): failed on item {}: {}", idx, exc)
                continue

        logger.info(
            "detect_batch(): processed {}/{} items successfully",
            len(results),
            len(images),
        )
        return results

    # ----------------------------------------------------------

    def _draw_detections(
        self,
        frame: np.ndarray,
        boxes: List[List[float]],
        confidences: List[float],
        class_ids: List[int],
    ) -> np.ndarray:
        """
        Draw bounding boxes and confidence labels on a copy of *frame*.

        Parameters
        ----------
        frame : np.ndarray
            Source BGR image (not modified in-place).
        boxes : list of [x1, y1, x2, y2]
            Bounding boxes in pixel coordinates.
        confidences : list of float
            Confidence score for each box.
        class_ids : list of int
            Class index for each box.

        Returns
        -------
        np.ndarray
            Annotated copy of *frame*.
        """
        annotated = frame.copy()
        box_color = (0, 255, 0)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = [int(round(v)) for v in box]

            class_name = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else str(cls_id)
            )
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            label_y = max(y1, th + 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 6),
                (x1 + tw + 4, label_y),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        count_label = f"Count: {len(boxes)}"
        cv2.putText(
            annotated,
            count_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone script execution.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run sperm detection inference on an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the annotated image to disk.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = load_config()
    detector = SpermDetector(cfg)

    detection_result = detector.detect_image(
        args.image,
        save=not args.no_save,
    )

    logger.success(
        "Inference complete on '{}': {} sperm detected",
        args.image,
        detection_result.count,
    )