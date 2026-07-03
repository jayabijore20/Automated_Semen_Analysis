"""
src/preprocessing/xml_to_yolo.py
=================================
Phase 1 · Data Preprocessing — XML → YOLO Converter
 
Reads Pascal VOC–format XML annotation files from the EVISAN dataset
and converts them to YOLO TXT format (one file per image).
 
YOLO label format per line:
    <class_id> <cx> <cy> <w> <h>          (all values normalised 0-1)
 
Input
-----
    datasets/EVISAN/images/       ← JPEG / PNG microscope images
    datasets/EVISAN/xml_file/     ← Pascal VOC XML annotations
 
Output
------
    datasets/yolo_ready/images/   ← copied images
    datasets/yolo_ready/labels/   ← converted YOLO .txt files
    datasets/yolo_ready/conversion_report.csv
 
How to Run
----------
    python -m src.preprocessing.xml_to_yolo
"""
 
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import pandas as pd
from tqdm import tqdm
 
from src.utils.helpers import ensure_dir, get_image_paths, is_valid_image, load_config
from src.utils.logger import get_logger
 
logger = get_logger(__name__)
 
 
# ══════════════════════════════════════════════════════════════
# Pascal VOC XML Parser
# ══════════════════════════════════════════════════════════════
 
class VOCAnnotation:
    """
    Represents a single Pascal VOC XML annotation file.
 
    Attributes
    ----------
    xml_path : Path
    image_path : Optional[Path]
    width : int
    height : int
    objects : list of dicts  →  {"name": str, "xmin": int, ...}
    """
 
    def __init__(self, xml_path: Path) -> None:
        self.xml_path = xml_path
        self.image_path: Optional[Path] = None
        self.width: int = 0
        self.height: int = 0
        self.objects: List[Dict] = []
        self._parse()
 
    def _parse(self) -> None:
        """Parse the XML tree and populate instance attributes."""
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
 
            # ── Image filename ────────────────────────────────
            filename_elem = root.find("filename")
            if filename_elem is not None and filename_elem.text:
                self.filename = filename_elem.text.strip()
            else:
                self.filename = self.xml_path.stem + ".jpg"
 
            # ── Image size ────────────────────────────────────
            size = root.find("size")
            if size is not None:
                self.width  = int(size.findtext("width",  default="0"))
                self.height = int(size.findtext("height", default="0"))
 
            # ── Objects / bounding boxes ──────────────────────
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                bndbox    = obj.find("bndbox")
                if name_elem is None or bndbox is None:
                    continue
 
                name = (name_elem.text or "sperm").strip().lower()
                xmin = float(bndbox.findtext("xmin", "0"))
                ymin = float(bndbox.findtext("ymin", "0"))
                xmax = float(bndbox.findtext("xmax", "0"))
                ymax = float(bndbox.findtext("ymax", "0"))
 
                # Sanity-check coordinates
                if xmax > xmin and ymax > ymin:
                    self.objects.append({
                        "name": name,
                        "xmin": xmin, "ymin": ymin,
                        "xmax": xmax, "ymax": ymax,
                    })
 
        except ET.ParseError as exc:
            logger.warning("XML parse error in {}: {}", self.xml_path, exc)
 
    # ----------------------------------------------------------
 
    def to_yolo_lines(self, class_map: Dict[str, int]) -> List[str]:
        """
        Convert all objects to YOLO format strings.
 
        Parameters
        ----------
        class_map : dict
            Maps class name → integer class index.
 
        Returns
        -------
        list of str
            One YOLO-format line per object.
        """
        lines: List[str] = []
        if self.width == 0 or self.height == 0:
            logger.warning("Zero image dimensions in {}", self.xml_path.name)
            return lines
 
        for obj in self.objects:
            class_name = obj["name"]
            class_id   = class_map.get(class_name, 0)   # default to 0 (sperm)
 
            # Centre coordinates + dimensions, normalised
            cx = (obj["xmin"] + obj["xmax"]) / 2.0 / self.width
            cy = (obj["ymin"] + obj["ymax"]) / 2.0 / self.height
            w  = (obj["xmax"] - obj["xmin"]) / self.width
            h  = (obj["ymax"] - obj["ymin"]) / self.height
 
            # Clamp to [0, 1]
            cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
 
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
 
        return lines
 
 
# ══════════════════════════════════════════════════════════════
# Converter Orchestrator
# ══════════════════════════════════════════════════════════════
 
class XMLtoYOLOConverter:
    """
    End-to-end converter from EVISAN Pascal VOC dataset to YOLO format.
 
    Parameters
    ----------
    config : dict
        Loaded project configuration (from configs/config.yaml).
    """
 
    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config
 
        paths_cfg = config["paths"]["datasets"]
 
        self.evisan_img_dir = Path(paths_cfg["evisan"]) / "images"
        self.evisan_xml_dir = Path(paths_cfg["evisan"]) / "xml_file"
        self.out_root       = Path(paths_cfg["yolo_ready"])
 
        self.out_img_dir    = self.out_root / "all_images"
        self.out_lbl_dir    = self.out_root / "all_labels"
 
        # Class mapping — extend this dict if more classes are added
        self.class_map: Dict[str, int] = {"sperm": 0}
 
        self._records: List[Dict] = []   # conversion log
 
    # ----------------------------------------------------------
 
    def convert(self) -> None:
        """
        Main entry point.  Converts all XML files and copies images.
        """
        logger.info("Starting VOC → YOLO conversion")
        logger.info("  XML dir  : {}", self.evisan_xml_dir)
        logger.info("  Image dir: {}", self.evisan_img_dir)
 
        ensure_dir(self.out_img_dir)
        ensure_dir(self.out_lbl_dir)
 
        xml_paths = sorted(self.evisan_xml_dir.glob("*.xml"))
        if not xml_paths:
            logger.error("No XML files found in {}", self.evisan_xml_dir)
            return
 
        logger.info("Found {} XML files", len(xml_paths))
 
        ok_count = 0
        skip_count = 0
 
        for xml_path in tqdm(xml_paths, desc="Converting XML → YOLO"):
            record = self._process_single(xml_path)
            self._records.append(record)
            if record["status"] == "ok":
                ok_count += 1
            else:
                skip_count += 1
 
        logger.success(
            "Conversion complete — {} ok, {} skipped", ok_count, skip_count
        )
        self._save_report()
        self._save_yaml()
 
    # ----------------------------------------------------------
 
    def _process_single(self, xml_path: Path) -> Dict:
        """
        Process one XML file: parse → find image → write YOLO label.
 
        Returns a log record dict.
        """
        record = {"xml": xml_path.name, "status": "ok", "boxes": 0, "note": ""}
 
        # 1. Parse annotation
        ann = VOCAnnotation(xml_path)
 
        # 2. Locate the corresponding image
        img_path = self._find_image(xml_path.stem)
        if img_path is None:
            record.update({"status": "skip", "note": "image not found"})
            return record
 
        # 3. Validate image (not corrupted)
        if not is_valid_image(img_path):
            record.update({"status": "skip", "note": "corrupted image"})
            return record
 
        # 4. If XML had zero dimensions, read from image itself
        if ann.width == 0 or ann.height == 0:
            import cv2
            img = cv2.imread(str(img_path))
            if img is not None:
                ann.height, ann.width = img.shape[:2]
 
        # 5. Convert to YOLO lines
        yolo_lines = ann.to_yolo_lines(self.class_map)
        if not yolo_lines:
            record.update({"status": "skip", "note": "no valid boxes"})
            return record
 
        # 6. Write label file
        out_lbl = self.out_lbl_dir / (xml_path.stem + ".txt")
        out_lbl.write_text("\n".join(yolo_lines))
 
        # 7. Copy image
        out_img = self.out_img_dir / img_path.name
        shutil.copy2(img_path, out_img)
 
        record["boxes"] = len(yolo_lines)
        return record
 
    # ----------------------------------------------------------
 
    def _find_image(self, stem: str) -> Optional[Path]:
        """
        Try common image extensions for a given file stem.
 
        Parameters
        ----------
        stem : str
            File name without extension (e.g. "frame_0001").
 
        Returns
        -------
        Path or None
        """
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
            candidate = self.evisan_img_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None
 
    # ----------------------------------------------------------
 
    def _save_report(self) -> None:
        """Save conversion log to CSV."""
        report_path = self.out_root / "conversion_report.csv"
        df = pd.DataFrame(self._records)
        df.to_csv(report_path, index=False)
        logger.info("Conversion report saved → {}", report_path)
 
    # ----------------------------------------------------------
 
    def _save_yaml(self) -> None:
        """
        Write a YOLO dataset YAML recognised by Ultralytics.
        The actual train/val/test splits are added by split_dataset.py.
        """
        yaml_content = (
            "# YOLO Dataset Config (generated by xml_to_yolo.py)\n"
            f"path: {self.out_root.resolve()}\n"
            "train: train/images\n"
            "val:   val/images\n"
            "test:  test/images\n\n"
            f"nc: {len(self.class_map)}\n"
            f"names: {list(self.class_map.keys())}\n"
        )
        yaml_path = self.out_root / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        logger.info("dataset.yaml written → {}", yaml_path)
 
 
# ══════════════════════════════════════════════════════════════
# Roboflow YOLO dataset integration helper
# ══════════════════════════════════════════════════════════════
 
def merge_roboflow_dataset(config: Optional[Dict] = None) -> None:
    """
    Copy Roboflow images and labels into the yolo_ready pool so
    the splitter sees all data at once.
 
    The Roboflow dataset already ships in YOLO format (images + labels).
 
    Parameters
    ----------
    config : dict
        Loaded project config.
    """
    if config is None:
        config = load_config()
 
    roboflow_dir = Path(config["paths"]["datasets"]["roboflow"])
    out_root     = Path(config["paths"]["datasets"]["yolo_ready"])
    out_img      = out_root / "all_images"
    out_lbl      = out_root / "all_labels"
 
    ensure_dir(out_img)
    ensure_dir(out_lbl)
 
    # Collect images from Roboflow (images may be in sub-folders)
    img_paths = get_image_paths(roboflow_dir)
    copied = 0
 
    for img_path in tqdm(img_paths, desc="Merging Roboflow images"):
        lbl_path = img_path.with_suffix(".txt")
        if not lbl_path.exists():
            # Labels might be in a sibling 'labels' folder
            lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
 
        if lbl_path.exists():
            shutil.copy2(img_path, out_img / img_path.name)
            shutil.copy2(lbl_path, out_lbl / lbl_path.name)
            copied += 1
 
    logger.success("Merged {} Roboflow samples into yolo_ready/", copied)
 
 
# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    cfg = load_config()
    converter = XMLtoYOLOConverter(cfg)
    converter.convert()
    merge_roboflow_dataset(cfg)