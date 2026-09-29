"""
src/viability/train_viability.py
==================================
Phase 6 · Viability Estimation — ML Model Trainer

Trains a Random Forest (or XGBoost) model to predict sperm viability %
from combined motility and morphology features.

Feature vector per sample
--------------------------
  From motility: VCL, VSL, VAP, LIN, STR, WOB, ALH,
                 progressive_pct, nonprogressive_pct, immotile_pct,
                 mean_total_distance, mean_displacement

  From morphology: normal_pct, abnormal_pct

Target variable: viability_pct (0-100)

If real VISEM-annotated viability data is unavailable, the module
provides a WHO-formula synthetic generator for training + testing.

Output
------
    models/viability/viability_model.pkl
    models/viability/feature_importance.png

How to Run
----------
    python -m src.viability.train_viability
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, load_config, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Feature names (must match predict_viability.py)
# ══════════════════════════════════════════════════════════════

FEATURE_NAMES: List[str] = [
    "VCL", "VSL", "VAP",
    "LIN", "STR", "WOB", "ALH",
    "progressive_pct", "nonprogressive_pct", "immotile_pct",
    "mean_total_distance", "mean_displacement",
    "normal_pct", "abnormal_pct",
]


# ══════════════════════════════════════════════════════════════
# Synthetic data generator (WHO-inspired formula)
# ══════════════════════════════════════════════════════════════

def generate_viability_dataset(n_samples: int = 2000,
                               random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic labelled training data.

    The viability target is computed using a WHO-inspired formula:

        viability ≈ 0.4 × progressive_pct
                  + 0.3 × normal_pct
                  + 0.15 × (VCL / 50) * 100
                  + 0.15 × (VSL / 30) * 100
                  + Gaussian noise (σ = 3)

    Parameters
    ----------
    n_samples : int
    random_seed : int

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_NAMES + ["viability_pct"]
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    for _ in range(n_samples):
        # Simulate realistic motility metrics (µm/s)
        vcl = rng.uniform(0, 80)
        vsl = rng.uniform(0, vcl)
        vap = rng.uniform(vsl, max(vcl, vsl + 0.01))
        lin = vsl / vcl if vcl > 0 else 0.0
        str_ = vsl / vap if vap > 0 else 0.0
        wob  = vap / vcl if vcl > 0 else 0.0
        alh  = rng.uniform(0, 5)
        tot_dist  = vcl * rng.uniform(1, 10)
        displace  = vsl * rng.uniform(1, 10)

        # Classify into motility categories
        if vcl < 5:
            immotile, progressive, nonprogressive = 1.0, 0.0, 0.0
        elif vcl >= 25 and lin >= 0.5:
            progressive = rng.uniform(0.4, 0.9)
            immotile    = rng.uniform(0.0, 0.3)
            nonprogressive = 1 - progressive - immotile
        else:
            nonprogressive = rng.uniform(0.3, 0.7)
            immotile       = rng.uniform(0.0, 0.3)
            progressive    = 1 - nonprogressive - immotile

        progressive    = max(0, progressive)
        nonprogressive = max(0, nonprogressive)
        immotile       = max(0, immotile)
        total_mot = progressive + nonprogressive + immotile
        progressive    /= total_mot
        nonprogressive /= total_mot
        immotile       /= total_mot

        # Morphology
        normal_pct   = rng.uniform(0, 100)
        abnormal_pct = 100 - normal_pct

        # Viability target (WHO formula approximation)
        viability = (
            0.40 * progressive    * 100
            + 0.30 * normal_pct
            + 0.15 * min(vcl / 50, 1.0) * 100
            + 0.15 * min(vsl / 30, 1.0) * 100
            + rng.normal(0, 3)
        )
        viability = float(np.clip(viability, 0, 100))

        rows.append([
            vcl, vsl, vap,
            lin, str_, wob, alh,
            progressive    * 100,
            nonprogressive * 100,
            immotile       * 100,
            tot_dist, displace,
            normal_pct, abnormal_pct,
            viability,
        ])

    cols = FEATURE_NAMES + ["viability_pct"]
    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════
# Viability Trainer
# ══════════════════════════════════════════════════════════════

class ViabilityTrainer:
    """
    Trains a Random Forest / XGBoost regressor to predict viability %.

    Parameters
    ----------
    config : dict, optional
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        via_cfg   = config["viability"]
        paths_cfg = config["paths"]

        self.model_type  = via_cfg["model"]          # "random_forest" | "xgboost"
        self.n_est       = via_cfg["n_estimators"]
        self.max_depth   = via_cfg["max_depth"]
        self.seed        = via_cfg["random_seed"]

        self.out_dir   = ensure_dir(Path(paths_cfg["models"]["viability"]).parent)
        self.ckpt_path = self.out_dir / "viability_model.pkl"
        self.plot_dir  = ensure_dir(paths_cfg["outputs"]["plots"])

    # ----------------------------------------------------------

    def _build_model(self):
        """Instantiate the ML model."""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=self.n_est,
                    max_depth=self.max_depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.seed,
                    verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost not installed, falling back to RandomForest")

        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_est,
            max_depth=self.max_depth,
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=-1,
        )

    # ----------------------------------------------------------

    @timer
    def train(self,
              data: Optional[pd.DataFrame] = None,
              data_csv: Optional[str | Path] = None) -> str:
        """
        Train the viability model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Pre-built training table.  If None, uses synthetic data.
        data_csv : str | Path, optional
            CSV path to load.

        Returns
        -------
        str
            Path to saved model pickle.
        """
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ── Data ───────────────────────────────────────────────
        if data is not None:
            df = data
        elif data_csv is not None:
            df = pd.read_csv(data_csv)
        else:
            logger.info("No data provided — generating synthetic dataset")
            df = generate_viability_dataset(n_samples=2000, random_seed=self.seed)

        X = df[FEATURE_NAMES].values
        y = df["viability_pct"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        logger.info("Training samples: {}  |  Test samples: {}", len(X_train), len(X_test))

        # ── Pipeline ───────────────────────────────────────────
        base_model = self._build_model()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  base_model),
        ])

        # ── 5-fold CV on training set ─────────────────────────
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring="r2", n_jobs=-1)
        logger.info("5-fold CV R² = {:.4f} ± {:.4f}",
                    cv_scores.mean(), cv_scores.std())

        # ── Fit on full training set ───────────────────────────
        pipeline.fit(X_train, y_train)

        # ── Test evaluation ────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)

        logger.success("Test  MAE={:.3f}  R²={:.4f}", mae, r2)

        # ── Save pipeline ──────────────────────────────────────
        joblib.dump(pipeline, str(self.ckpt_path))
        logger.success("Viability model saved → {}", self.ckpt_path)

        # ── Plots ──────────────────────────────────────────────
        self._plot_feature_importance(pipeline, base_model)
        self._plot_predictions(y_test, y_pred)

        return str(self.ckpt_path)

    # ----------------------------------------------------------

    def _plot_feature_importance(self, pipeline, base_model) -> None:
        """Plot feature importances if available."""
        try:
            importances = base_model.feature_importances_
        except AttributeError:
            return   # XGBoost may need different accessor

        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(importances)),
               importances[indices],
               color="#3498db", edgecolor="white")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in indices], rotation=45, ha="right")
        ax.set_title("Viability Model — Feature Importances", fontweight="bold")
        ax.set_ylabel("Importance")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        save_path = self.plot_dir / "viability_feature_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Feature importance plot saved → {}", save_path)

    # ----------------------------------------------------------

    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Scatter plot of actual vs predicted viability."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, color="#9b59b6", s=15)
        lims = [0, 100]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Viability (%)")
        ax.set_ylabel("Predicted Viability (%)")
        ax.set_title("Viability: Actual vs Predicted", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = self.plot_dir / "viability_predictions.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Prediction plot saved → {}", save_path)


# ══════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg     = load_config()
    trainer = ViabilityTrainer(cfg)
    trainer.train()