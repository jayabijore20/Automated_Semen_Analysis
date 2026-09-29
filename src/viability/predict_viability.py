"""
src/viability/predict_viability.py
=====================================
Phase 6 · Viability Estimation — Inference Module

Loads the trained viability model and predicts viability %
given motility and morphology feature dictionaries.

Usage
-----
    from src.viability.predict_viability import ViabilityPredictor

    predictor = ViabilityPredictor()
    viability = predictor.predict(
        motility_summary={"mean_VCL": 35.0, "progressive_pct": 58.0, ...},
        morphology_summary={"normal_pct": 62.0, ...}
    )
    print(f"Viability: {viability:.1f}%")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from src.viability.train_viability import FEATURE_NAMES, generate_viability_dataset
from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Viability Predictor
# ══════════════════════════════════════════════════════════════

class ViabilityPredictor:
    """
    Wraps the trained viability model for single-sample inference.

    Parameters
    ----------
    config : dict, optional
    weights_path : str, optional
        Override for model pickle path.
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 weights_path: Optional[str] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        if weights_path is None:
            weights_path = config["paths"]["models"]["viability"]
        self.weights_path = Path(weights_path)

        self._pipeline = None   # lazy-loaded

    # ----------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the model pipeline."""
        if self._pipeline is not None:
            return

        if self.weights_path.exists():
            self._pipeline = joblib.load(str(self.weights_path))
            logger.info("Viability model loaded from {}", self.weights_path)
        else:
            logger.warning(
                "Viability model not found at {}. "
                "Training a quick model on synthetic data...",
                self.weights_path
            )
            from src.viability.train_viability import ViabilityTrainer
            trainer = ViabilityTrainer(self.config)
            trainer.train()
            self._pipeline = joblib.load(str(self.weights_path))

    # ----------------------------------------------------------

    def predict(self,
                motility_summary: Dict,
                morphology_summary: Dict) -> float:
        """
        Predict viability percentage from analysis summaries.

        Parameters
        ----------
        motility_summary : dict
            Output of MotilityAnalyser._build_summary().
            Keys used: mean_VCL, mean_VSL, mean_VAP,
                       progressive_pct, nonprogressive_pct, immotile_pct

        morphology_summary : dict
            Output of MorphologyClassifier.batch_summary().
            Keys used: normal_pct, abnormal_pct

        Returns
        -------
        float
            Predicted viability %, clamped to [0, 100].
        """
        self._load()

        # Build feature vector in the same order as FEATURE_NAMES
        features = self._build_feature_vector(motility_summary, morphology_summary)
        X = np.array([features])

        viability = float(self._pipeline.predict(X)[0])
        viability = float(np.clip(viability, 0.0, 100.0))

        logger.debug("Predicted viability: {:.2f}%", viability)
        return round(viability, 2)

    # ----------------------------------------------------------

    def _build_feature_vector(self,
                               motility_summary: Dict,
                               morphology_summary: Dict) -> list:
        """
        Map summary dictionaries to the ordered FEATURE_NAMES vector.

        Missing keys default to 0.0.

        Parameters
        ----------
        motility_summary, morphology_summary : dict

        Returns
        -------
        list of float
        """
        combined = {**motility_summary, **morphology_summary}

        # Aliases from summary keys → feature names
        aliases = {
            "mean_VCL":            "VCL",
            "mean_VSL":            "VSL",
            "mean_VAP":            "VAP",
            # LIN, STR, WOB, ALH, mean_total_distance, mean_displacement
            # may not be in summary — default to 0 gracefully
        }

        resolved: Dict[str, float] = {}
        for feat in FEATURE_NAMES:
            if feat in combined:
                resolved[feat] = float(combined[feat])
            elif feat in aliases and aliases[feat] in combined:
                resolved[feat] = float(combined[aliases[feat]])
            else:
                resolved[feat] = 0.0

        return [resolved[feat] for feat in FEATURE_NAMES]


# ══════════════════════════════════════════════════════════════
# CLI quick test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    predictor = ViabilityPredictor(cfg)

    # Example input from a (hypothetical) analysis session
    motility = {
        "mean_VCL": 32.5, "mean_VSL": 18.0, "mean_VAP": 24.0,
        "progressive_pct": 55.0, "nonprogressive_pct": 25.0, "immotile_pct": 20.0,
    }
    morphology = {
        "normal_pct": 60.0, "abnormal_pct": 40.0,
    }

    result = predictor.predict(motility, morphology)
    print(f"Predicted Viability: {result:.1f}%")