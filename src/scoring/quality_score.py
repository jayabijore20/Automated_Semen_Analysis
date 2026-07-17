"""
src/scoring/quality_score.py
==============================
Phase 7 · Semen Quality Scoring — WHO-Aligned Scoring Engine

Produces a 0-100 quality score and categorical grade from the four
key semen parameters:

    Count       (sperm per mL or unique track count as proxy)
    Motility    (progressive %)
    Morphology  (normal %)
    Viability   (predicted %)

Weights and WHO reference thresholds are read from configs/config.yaml.

Usage
-----
    from src.scoring.quality_score import QualityScorer

    scorer = QualityScorer()
    result = scorer.score(
        count=45,
        progressive_pct=58.0,
        normal_pct=62.0,
        viability_pct=71.0
    )
    print(result)
    # {
    #   "score": 82.5,
    #   "category": "Excellent",
    #   "subscores": {"count": 100, "motility": 96.7, ...},
    #   "who_flags": [],
    #   "recommendations": [...]
    # }
"""

from typing import Dict, List, Optional

from src.utils.helpers import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# Quality Scorer
# ══════════════════════════════════════════════════════════════

class QualityScorer:
    """
    Weighted semen quality scoring engine aligned to WHO 2021 criteria.

    Parameters
    ----------
    config : dict, optional
        Loaded project config.  Auto-loaded if None.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if config is None:
            config = load_config()
        self.config = config

        scoring_cfg = config["scoring"]

        # Per-parameter weights (must sum to 1.0)

        if abs(sum(scoring_cfg["weights"].values()) - 1.0) > 1e-6:
         
           raise ValueError("Scoring weights must sum to 1.0")
        
        self.weights = scoring_cfg["weights"]

        # WHO 2021 reference lower limits
        who = scoring_cfg["who_reference"]
        self.who_count_min     = who["count_min_M_per_mL"]        # 16 M/mL
        self.who_prog_min      = who["progressive_motility_min"]  # 30 %
        self.who_morph_min     = who["normal_morphology_min"]     # 4 %
        self.who_viability_min = who["viability_min"]             # 54 %

        # Score → category mapping (inclusive upper bound)
        cats = scoring_cfg["categories"]
        self.categories = {
            "Excellent": tuple(cats["excellent"]),
            "Good":      tuple(cats["good"]),
            "Average":   tuple(cats["average"]),
            "Poor":      tuple(cats["poor"]),
        }

    # ----------------------------------------------------------

    def score(self,
              count: float,
              progressive_pct: float,
              normal_pct: float,
              viability_pct: float) -> Dict:
        """
        Compute composite quality score and grade.

        Parameters
        ----------
        count : float
            Total unique sperm count (tracks) or concentration in M/mL.
        progressive_pct : float
            Percentage of progressively motile sperm (0-100).
        normal_pct : float
            Percentage of morphologically normal sperm (0-100).
        viability_pct : float
            Predicted viability percentage (0-100).

        Returns
        -------
        dict
            Keys:
              score         float   — composite score 0-100
              category      str     — "Excellent" | "Good" | "Average" | "Poor"
              subscores     dict    — individual parameter scores
              who_flags     list    — list of WHO threshold violations
              recommendations list — clinical-style recommendations
        """
        # ── Sub-scores (each 0-100) ────────────────────────────
        subscores = {
            "count":      self._score_count(count),
            "motility":   self._score_motility(progressive_pct),
            "morphology": self._score_morphology(normal_pct),
            "viability":  self._score_viability(viability_pct),
        }

        # ── Weighted composite ─────────────────────────────────
        composite = (
            self.weights["count"]      * subscores["count"]
            + self.weights["motility"]   * subscores["motility"]
            + self.weights["morphology"] * subscores["morphology"]
            + self.weights["viability"]  * subscores["viability"]
        )
        composite = round(min(100.0, max(0.0, composite)), 2)

        # ── Category ───────────────────────────────────────────
        category = self._categorise(composite)

        # ── WHO flag any below-threshold parameters ────────────
        who_flags = self._check_who_thresholds(
            count, progressive_pct, normal_pct, viability_pct
        )

        # ── Recommendations ────────────────────────────────────
        recommendations = self._generate_recommendations(
            count, progressive_pct, normal_pct, viability_pct, who_flags
        )

        result = {
            "score":           composite,
            "category":        category,
            "subscores":       subscores,
            "who_flags":       who_flags,
            "recommendations": recommendations,
        }

        logger.info(
            "Quality score: {:.1f} / 100 → {} | WHO flags: {}",
            composite, category, len(who_flags)
        )
        return result

    # ----------------------------------------------------------

    def _score_count(self, count: float) -> float:
        """
        Score count parameter relative to WHO minimum.

        Parameters
        ----------
        count : float
            Number of unique tracks (used as sperm count proxy).

        Returns
        -------
        float  0-100
        """
        # Scale: WHO min = 50 points, 3× WHO min = 100 points
        if count <= 0:
            return 0.0
        # Use reference of 3× WHO minimum as "excellent" upper anchor
        excellent_anchor = self.who_count_min * 3
        score = min(100.0, (count / excellent_anchor) * 100)
        return round(score, 2)

    def _score_motility(self, progressive_pct: float) -> float:
        """Score progressive motility (WHO min 30%, excellent 70%)."""
        if progressive_pct <= 0:
            return 0.0
        excellent_anchor = 70.0
        score = min(100.0, (progressive_pct / excellent_anchor) * 100)
        return round(score, 2)

    def _score_morphology(self, normal_pct: float) -> float:
        """Score normal morphology (WHO min 4%, excellent 30%)."""
        if normal_pct <= 0:
            return 0.0
        excellent_anchor = 15.0
        score = min(100.0, (normal_pct / excellent_anchor) * 100)
        return round(score, 2)

    def _score_viability(self, viability_pct: float) -> float:
        """Score viability (WHO min 54%, excellent 85%)."""
        if viability_pct <= 0:
            return 0.0
        excellent_anchor = 85.0
        score = min(100.0, (viability_pct / excellent_anchor) * 100)
        return round(score, 2)

    # ----------------------------------------------------------

    def _categorise(self, score: float) -> str:
        """
        Map composite score to category label.

        Parameters
        ----------
        score : float  0-100

        Returns
        -------
        str
        """
        for category, (lo, hi) in self.categories.items():
            if lo <= score <= hi:
                return category
        return "Poor"

    # ----------------------------------------------------------

    def _check_who_thresholds(self,
                               count: float,
                               progressive_pct: float,
                               normal_pct: float,
                               viability_pct: float) -> List[str]:
        """
        Return a list of WHO-threshold violation strings.

        Parameters
        ----------
        count, progressive_pct, normal_pct, viability_pct : float

        Returns
        -------
        list of str
        """
        flags: List[str] = []

        if count < self.who_count_min:
            flags.append(
                f"Low sperm count ({count:.0f} < WHO min {self.who_count_min} M/mL)"
            )
        if progressive_pct < self.who_prog_min:
            flags.append(
                f"Low progressive motility ({progressive_pct:.1f}% < WHO min {self.who_prog_min}%)"
            )
        if normal_pct < self.who_morph_min:
            flags.append(
                f"Low normal morphology ({normal_pct:.1f}% < WHO min {self.who_morph_min}%)"
            )
        if viability_pct < self.who_viability_min:
            flags.append(
                f"Low viability ({viability_pct:.1f}% < WHO min {self.who_viability_min}%)"
            )

        return flags

    # ----------------------------------------------------------

    def _generate_recommendations(self,
                                   count: float,
                                   progressive_pct: float,
                                   normal_pct: float,
                                   viability_pct: float,
                                   who_flags: List[str]) -> List[str]:
        """
        Generate clinical-style recommendations based on results.

        Parameters
        ----------
        count, progressive_pct, normal_pct, viability_pct : float
        who_flags : list of str

        Returns
        -------
        list of str
        """
        recs: List[str] = []

        if not who_flags:
            recs.append(
                "All parameters are within WHO 2021 reference limits. "
                "Continue regular health monitoring."
            )
            return recs

        if count < self.who_count_min:
            recs.append(
                "Oligospermia detected. Consider hormonal evaluation, lifestyle modification "
                "(reduce heat exposure, alcohol, smoking), and repeat analysis in 3 months."
            )

        if progressive_pct < self.who_prog_min:
            recs.append(
                "Asthenozoospermia detected. Antioxidant supplementation (Vitamin C, E, CoQ10) "
                "and evaluation for varicocele or infection is recommended."
            )

        if normal_pct < self.who_morph_min:
            recs.append(
                "Teratozoospermia detected. Strict Kruger morphology assessment advised. "
                "ICSI may be considered if morphology is severely impaired."
            )

        if viability_pct < self.who_viability_min:
            recs.append(
                "Necrozoospermia detected. Sperm should be used fresh for ART procedures. "
                "Antioxidant therapy and oxidative stress markers should be evaluated."
            )

        recs.append(
            "This is an AI-assisted analysis. Clinical confirmation by a certified andrologist "
            "is strongly recommended before any medical decision."
        )

        return recs
    

if __name__=="__main__":
        scorer = QualityScorer()

        result = scorer.score(
           count=11592,
           progressive_pct=21.3,
           normal_pct=96.0,
           viability_pct=41.3,
        )

        from pprint import pprint
        pprint(result)