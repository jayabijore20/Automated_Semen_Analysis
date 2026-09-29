"""
src/utils/helpers.py
====================
General-purpose utility functions shared across all pipeline modules.
 
Includes:
  - YAML config loading
  - Path validation helpers
  - Image reading / resizing
  - Bounding-box drawing
  - Colour palette generation
  - Timing decorator
"""
 
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import cv2
import numpy as np
import yaml
 
from src.utils.logger import get_logger
 
logger = get_logger(__name__)
 
 
# ══════════════════════════════════════════════════════════════
# Configuration helpers
# ══════════════════════════════════════════════════════════════
 
def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load project YAML configuration.
 
    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
 
    Returns
    -------
    dict
        Parsed configuration dictionary.
 
    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
 
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
 
    logger.debug("Config loaded from {}", path)
    return cfg
 
 
# ══════════════════════════════════════════════════════════════
# Path helpers
# ══════════════════════════════════════════════════════════════
 
def ensure_dir(path: str | Path) -> Path:
    """
    Create *path* (and parents) if it does not exist.
 
    Parameters
    ----------
    path : str | Path
 
    Returns
    -------
    Path
        The same path as a resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
 
 
def get_image_paths(directory: str | Path,
                    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> List[Path]:
    """
    Recursively collect all image paths under *directory*.
 
    Parameters
    ----------
    directory : str | Path
    extensions : tuple of str
        File suffixes to include (lower-case).
 
    Returns
    -------
    list of Path
        Sorted list of matched paths.
    """
    d = Path(directory)
    if not d.exists():
        logger.warning("Directory not found: {}", d)
        return []
 
    paths = sorted([p for p in d.rglob("*") if p.suffix.lower() in extensions])
    logger.debug("Found {} images in {}", len(paths), d)
    return paths
 
 
# ══════════════════════════════════════════════════════════════
# Image utilities
# ══════════════════════════════════════════════════════════════
 
def read_image(path: str | Path, color: bool = True) -> Optional[np.ndarray]:
    """
    Read an image from disk with error handling.
 
    Parameters
    ----------
    path : str | Path
    color : bool
        If True, read as BGR colour; otherwise grayscale.
 
    Returns
    -------
    np.ndarray or None
        Image array, or None if reading failed.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        logger.warning("Failed to read image: {}", path)
    return img
 
 
def resize_image(image: np.ndarray,
                 target_size: Tuple[int, int],
                 keep_aspect: bool = True) -> np.ndarray:
    """
    Resize *image* to *target_size*.
 
    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C) or (H, W).
    target_size : (width, height)
    keep_aspect : bool
        If True, pad with black to preserve aspect ratio.
 
    Returns
    -------
    np.ndarray
        Resized image.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
 
    if keep_aspect:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to target size
        canvas = np.zeros((target_h, target_w, *image.shape[2:]), dtype=image.dtype)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas
    else:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
 
 
def is_valid_image(path: str | Path) -> bool:
    """
    Return True if *path* is a readable, non-corrupted image.
 
    Parameters
    ----------
    path : str | Path
 
    Returns
    -------
    bool
    """
    img = read_image(path)
    return img is not None and img.size > 0
 
 
# ══════════════════════════════════════════════════════════════
# Drawing utilities
# ══════════════════════════════════════════════════════════════
 
def generate_color_palette(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate *n* visually distinct BGR colours using HSV spacing.
 
    Parameters
    ----------
    n : int
        Number of colours to generate.
 
    Returns
    -------
    list of (B, G, R) tuples
    """
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors
 
 
def draw_bbox(image: np.ndarray,
              box: Tuple[int, int, int, int],
              label: str = "",
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box with optional label on *image*.
 
    Parameters
    ----------
    image : np.ndarray
        BGR image (modified in-place).
    box : (x1, y1, x2, y2)
        Pixel coordinates of the bounding box.
    label : str
        Text to draw above the box.
    color : (B, G, R)
    thickness : int
 
    Returns
    -------
    np.ndarray
        Annotated image (same object as *image*).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
 
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        # Background rectangle for readability
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image
 
 
# ══════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════
 
def timer(func):
    """
    Decorator that logs the wall-clock execution time of *func*.
 
    Usage
    -----
        @timer
        def train_model(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("{} completed in {:.2f}s", func.__qualname__, elapsed)
        return result
    return wrapper
 