"""
src/analysis/concentration.py
================================
Sperm Concentration, Motile-Cell-Concentration (MCC), and Sustained
Motility Lifetime (SML) calculations.

SCIENTIFIC SCOPE AND HONESTY
-----------------------------
This module deliberately separates three distinct quantities that are
easy to conflate:

    1. SAMPLE VOLUME     -- the total volume of raw sample the operator
                             placed on the slide/chamber (µL). Entered
                             by the user. NOT the same as (2).
    2. OBSERVED VOLUME    -- the physical volume the microscope camera
                             actually imaged: field-of-view area ×
                             chamber/slide depth. Requires spatial
                             calibration (µm/pixel) AND a known depth.
    3. SPERM COUNT         -- the number of *unique* sperm cells counted
                             within the observed volume (from ByteTrack
                             track IDs, i.e. already deduplicated across
                             frames -- never a raw per-frame detection
                             sum).

Concentration is computed ONLY as:

    concentration_cells_per_ml = sperm_count / observed_volume_ml
    corrected = concentration_cells_per_ml * dilution_factor

There is no scenario in this module where `sample_volume_ul` is used
as a stand-in for observed volume. If the depth of the observation
(chamber depth, or an experimentally-validated slide depth) is not
configured, every function in this module that would need it returns
a result with ``status="calibration_required"`` and a human-readable
reason -- it never fabricates a number.

Chamber geometry
-----------------
``CHAMBER_CONFIG`` below lists a few counting-chamber DEPTHS that are
fixed, standard, manufacturer-specified geometry for the *named
device model* (Makler / Leja / Neubauer-improved hemocytometer). These
are not experimentally derived by this codebase -- they are published
specifications of a specific physical instrument, similar to citing
the length of a ruler. If your physical chamber is a different model
or variant, UPDATE THE CONFIGURATION before trusting the output; the
values here are a starting point, not a substitute for verifying your
own hardware.

"Standard Glass Slide" (a slide + coverslip with no fixed chamber
geometry) intentionally has ``depth_um = None``. A random coverslip
does not create a reproducible depth, so this module refuses to
compute an absolute concentration for that slide type until a
locally-validated effective depth is entered into
``CHAMBER_CONFIG["Standard Glass Slide"]``.

Counting-field model
---------------------
The current single-video pipeline analyses ONE continuous video
recording of ONE physical field of the slide/chamber; ByteTrack
already deduplicates detections into unique tracks over the full
recording, which is a *more* robust field count than sampling single
frames. This module therefore treats one video as one counting field
by default (``number_of_counting_fields=1``). The averaging-across-
fields machinery (``mean_sperm_per_field``) is implemented generically
so it is ready to use if a future workflow captures multiple separate
clips at different slide locations for the same sample -- but nothing
in the current app.py fabricates additional fields that were never
actually captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════
# FEATURE 16 — Configuration (single source of truth; no magic
# numbers scattered through the codebase)
# ══════════════════════════════════════════════════════════════

#: Target operating range for the whole system (Feature 1). This is a
#: *design/support* range, not a scientific validation claim -- see
#: `get_concentration_status()`.
TARGET_MIN_CONCENTRATION_M_PER_ML: float = 1.0
TARGET_MAX_CONCENTRATION_M_PER_ML: float = 550.0

#: Practical sample-volume range for a standard glass slide + coverslip.
NORMAL_SLIDE_MIN_VOLUME_UL: float = 6.5
NORMAL_SLIDE_MAX_VOLUME_UL: float = 10.0

#: Practical total sample-volume range across a counting-chamber setup.
CHAMBER_MIN_TOTAL_VOLUME_UL: float = 3.0
CHAMBER_MAX_TOTAL_VOLUME_UL: float = 6.0

#: Practical per-section volume range for a counting chamber.
CHAMBER_MIN_VOLUME_PER_SECTION_UL: float = 0.75
CHAMBER_MAX_VOLUME_PER_SECTION_UL: float = 1.5

#: Default dilution factor (1.0 == undiluted sample).
DEFAULT_DILUTION_FACTOR: float = 1.0

#: Default number of counting fields (see "Counting-field model" above).
DEFAULT_NUMBER_OF_COUNTING_FIELDS: int = 1

#: Minimum number of *valid* fields required before a concentration is
#: reported at all (Feature 15, warning #6).
MIN_VALID_FIELDS_REQUIRED: int = 1

#: SML threshold as a fraction of the initial/plateau progressive
#: motility (Feature 10 — Dynescan-style definition: 50% of baseline).
SML_THRESHOLD_FRACTION: float = 0.5

#: Minimum number of time points required to compute SML.
SML_MIN_TIME_POINTS: int = 2

#: 1 mL == 1000 mm^3 (exact unit conversion, not a tunable constant).
_MM3_PER_ML: float = 1000.0
#: 1 mm == 1000 µm (exact unit conversion).
_UM_PER_MM: float = 1000.0
#: 1,000,000 cells/mL == 1 M cells/mL (exact unit conversion).
_CELLS_PER_M_CELLS: float = 1_000_000.0

# ── Counting-field density thresholds used for the qualitative
#    "counting quality" indicator (Feature 19). These are density
#    BUCKETS for flagging likely-unreliable high-density counts to a
#    human operator -- they are heuristic UI guidance, not a
#    scientific accuracy claim, and are clearly labelled as such
#    wherever they are surfaced.
_DENSITY_QUALITY_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (150.0, "Excellent"),   # sperm per 10,000 µm^2 of field, approx.
    (400.0, "Good"),
    (800.0, "Moderate"),
)
_DENSITY_QUALITY_POOR_LABEL = "Poor"


class SlideCategory:
    """
    FEATURE (PART 3) — internal classification of a counting-chamber /
    slide type, used to decide which concentration-calculation
    pathway applies and to select correct result terminology
    (PART 29).

    PHYSICAL_CHAMBER: a manufactured, precision-machined counting
        chamber with a fixed, published depth (Makler / Leja /
        Hemocytometer). Concentration from these is labelled
        "Measured Concentration".
    CONTROLLED_ESTIMATION: a standard glass slide + coverslip. There
        is no precision-machined depth — only a THEORETICAL average
        depth derived from controlled sample volume and coverslip
        area (PART 9). Concentration from these is labelled
        "Estimated Concentration" until the complete measurement
        protocol has been empirically validated against a reference
        method, after which it becomes "Validated Controlled-Protocol
        Concentration" (PART 29).
    """
    PHYSICAL_CHAMBER = "physical_chamber"
    CONTROLLED_ESTIMATION = "controlled_estimation"


class ValidationState:
    """
    FEATURE (PART 44) — validation state for any calibration or
    protocol value used in a concentration calculation. ``None`` is
    NEVER treated as an implicit valid state — every calibration this
    module tracks carries an explicit state from this set.

    NOT_CONFIGURED: no value has been entered/measured at all.
    PROVISIONAL: a starting/default value is in use (e.g. a published
        manufacturer spec, or an initial protocol value) but has not
        been independently confirmed against a reference method for
        this specific hardware/protocol.
    VALIDATED: confirmed against an independent reference measurement
        (see the empirical-validation framework below) with
        documented results.
    EXPIRED: was previously VALIDATED but the validation is stale
        (e.g. hardware changed) and needs to be repeated.
    """
    NOT_CONFIGURED = "not_configured"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    EXPIRED = "expired"


@dataclass
class ChamberSpec:
    """
    Standard geometry for one named counting-chamber/slide type.

    Attributes
    ----------
    depth_um : float | None
        Nominal chamber depth in microns. ``None`` means "not
        verified / requires configuration" — concentration
        calculations MUST refuse to run until this is set.
    verified : bool
        True only for depths that are published, fixed manufacturer
        specifications for the named device. False for anything the
        operator must establish experimentally before trusting it.
        Kept for backward compatibility with the original single-flag
        model — ``validation_state`` below is the more expressive
        successor (PART 44) and is what new code should read.
    note : str
        Human-readable provenance / caveat, shown in the UI whenever
        this chamber is selected.
    category : str
        One of ``SlideCategory`` — determines which concentration
        pathway and result terminology apply (PART 3/29).
    validation_state : str
        One of ``ValidationState``. A *published manufacturer
        specification* (Makler/Leja/Hemocytometer depth) is marked
        PROVISIONAL, not VALIDATED — "verified" here means "a real
        published number for this device model", not "confirmed
        against a reference measurement on your specific unit". Only
        an actual empirical validation run (see
        ``ReferenceValidationRecord`` below) should ever set a chamber
        to VALIDATED.
    """
    depth_um: Optional[float]
    verified: bool
    note: str
    category: str = SlideCategory.PHYSICAL_CHAMBER
    validation_state: str = ValidationState.PROVISIONAL
    depth_source: str = ""
    geometry_validated: bool = False


#: PART 7/8 — preferred quantitative chamber for the controlled
#: workflow when a physical chamber is being used and the operator
#: has a choice. This is a workflow/UI hint, not a scientific claim —
#: it does not change how any other chamber's concentration is
#: calculated.
PREFERRED_QUANTITATIVE_CHAMBER: str = "Leja Chamber"

#: FEATURE 4 / 17 — Chamber geometry configuration.
#: DO NOT add an entry with an invented depth. If you don't have a
#: verified number, use `depth_um=None` and describe what needs to be
#: measured/entered in `note`.
CHAMBER_CONFIG: Dict[str, ChamberSpec] = {
    "Makler Chamber": ChamberSpec(
        depth_um=10.0,
        verified=True,
        category=SlideCategory.PHYSICAL_CHAMBER,
        validation_state=ValidationState.PROVISIONAL,
        depth_source="manufacturer_specification",
        geometry_validated=False,
        note=(
            "Standard nominal depth for a Makler Counting Chamber "
            "(fixed manufacturer specification). Verify against your "
            "specific chamber before relying on this for clinical use."
        ),
    ),
    "Leja Chamber": ChamberSpec(
        depth_um=20.0,
        verified=True,
        category=SlideCategory.PHYSICAL_CHAMBER,
        validation_state=ValidationState.PROVISIONAL,
        depth_source="manufacturer_specification",
        geometry_validated=False,
        note=(
            "Nominal depth for the widely-used 20 µm Leja slide "
            "variant (PART 8: preferred quantitative chamber for the "
            "controlled workflow). Leja slides are also manufactured "
            "in other depths (e.g. 10/50/100 µm) — confirm which "
            "variant you are using and update this value if it differs."
        ),
    ),
    "Hemocytometer": ChamberSpec(
        depth_um=100.0,
        verified=True,
        category=SlideCategory.PHYSICAL_CHAMBER,
        validation_state=ValidationState.PROVISIONAL,
        depth_source="manufacturer_specification",
        geometry_validated=False,
        note=(
            "Standard depth (0.1 mm) for a Neubauer-improved "
            "hemocytometer counting chamber (fixed manufacturer "
            "specification)."
        ),
    ),
    "Standard Glass Slide": ChamberSpec(
        depth_um=None,
        verified=False,
        category=SlideCategory.CONTROLLED_ESTIMATION,
        validation_state=ValidationState.NOT_CONFIGURED,
        depth_source="not_applicable_use_theoretical_depth",
        geometry_validated=False,
        note=(
            "A standard glass slide + coverslip has no fixed, "
            "reproducible depth — it is not a precision-machined "
            "chamber. See the Standard-Slide CASA-like controlled "
            "protocol (STANDARD_SLIDE_PROTOCOL) instead of entering "
            "a chamber depth here."
        ),
    ),
    "Other": ChamberSpec(
        depth_um=None,
        verified=False,
        category=SlideCategory.PHYSICAL_CHAMBER,
        validation_state=ValidationState.NOT_CONFIGURED,
        depth_source="unconfigured",
        geometry_validated=False,
        note="Unrecognised chamber — enter a verified depth to enable concentration.",
    ),
}

#: FEATURE 3 — Spatial calibration lookup, keyed by
#: (magnification, camera_resolution). ``None`` = not yet calibrated;
#: fill these in once a stage micrometer calibration has been
#: performed for that magnification/resolution combination. A manual
#: per-session override is always preferred over this table when
#: supplied (see `resolve_microns_per_pixel`).
#:
#: DO NOT invent values here. An unset entry is the correct, honest
#: default until someone measures it.
CALIBRATION_TABLE: Dict[Tuple[str, str], Optional[float]] = {
    ("10×",  "640×480"):   None,
    ("10×",  "1280×720"):  None,
    ("10×",  "1920×1080"): None,
    ("20×",  "640×480"):   None,
    ("20×",  "1280×720"):  None,
    ("20×",  "1920×1080"): None,
    ("40×",  "640×480"):   None,
    ("40×",  "1280×720"):  None,
    ("40×",  "1920×1080"): None,
    ("100×", "640×480"):   None,
    ("100×", "1280×720"):  None,
    ("100×", "1920×1080"): None,
}


def resolve_microns_per_pixel(
    magnification: str,
    camera_resolution: str,
    manual_override: Optional[float] = None,
    saved_calibration_um_per_pixel: Optional[float] = None,
) -> Optional[float]:
    """
    Resolve the microns-per-pixel calibration value to use.

    Priority order (highest first):
        1. `saved_calibration_um_per_pixel` -- a real, one-time
           stage-micrometer calibration saved for this EXACT hardware
           configuration (microscope + camera + magnification +
           resolution + digital zoom), looked up and passed in by the
           caller (see `src.calibration.optical_calibration` and
           `DatabaseManager.get_calibration`). This is the only tier
           the normal semen-analysis operator's workflow should ever
           populate.
        2. `manual_override` -- an admin/lab-only override (NOT part
           of the normal operator workflow; see app.py's dedicated
           Calibration Setup / Admin section). Retained for backward
           compatibility with existing callers/tests.
        3. `CALIBRATION_TABLE` lookup by (magnification, camera_resolution).

    Parameters
    ----------
    magnification : str        e.g. "40×"
    camera_resolution : str    e.g. "1280×720"
    manual_override : float | None
        An admin-supplied override value. Not populated by the normal
        analysis UI as of the calibration-workflow update.
    saved_calibration_um_per_pixel : float | None
        The resolved value from a saved, exact-configuration
        calibration record, if one exists. Always wins when present.

    Returns
    -------
    float | None
        None means "not calibrated" -- callers MUST treat this as
        "cannot compute an accurate concentration", never as zero.
    """
    if saved_calibration_um_per_pixel is not None and saved_calibration_um_per_pixel > 0:
        return float(saved_calibration_um_per_pixel)
    if manual_override is not None and manual_override > 0:
        return float(manual_override)
    return CALIBRATION_TABLE.get((magnification, camera_resolution))


# ══════════════════════════════════════════════════════════════
# PART 4/5 — Standard Glass Slide CASA-Like Controlled Protocol
# ══════════════════════════════════════════════════════════════
#
# Philosophy (see module docstring's "Counting-field model" section
# for the related field-counting rationale): a standard glass slide
# is not a precision-machined chamber, so it can never yield a true
# "measured" depth the way a Makler/Leja/Hemocytometer chamber does.
# Instead, this protocol controls everything ELSE tightly (a fixed
# pipetted volume within a tolerance, a fixed coverslip size, a fixed
# number of analysed fields) so that the resulting THEORETICAL AVERAGE
# DEPTH is at least reproducible run-to-run — and, critically, treats
# the result as an ESTIMATE until the complete protocol has been
# validated against an independent reference method (PART 22).

@dataclass
class StandardSlideProtocol:
    """
    Centralised, configurable definition of the standard-slide
    CASA-like controlled-analysis protocol (PART 5).

    ALL VALUES HERE ARE INITIAL/PROVISIONAL until experimentally
    validated — see ``validation_state``. They are not universal
    constants; change them in one place (this dataclass) rather than
    hard-coding them elsewhere.
    """
    sample_volume_ul: float = 8.0
    volume_tolerance_ul: float = 0.5
    coverslip_length_mm: float = 22.0
    coverslip_width_mm: float = 22.0
    number_of_fields: int = 10
    validation_state: str = ValidationState.PROVISIONAL
    protocol_version: str = "1.0-provisional"

    def in_volume_tolerance(self, actual_volume_ul: float) -> bool:
        """True if `actual_volume_ul` is within tolerance of the target."""
        return abs(actual_volume_ul - self.sample_volume_ul) <= self.volume_tolerance_ul


#: The active protocol instance. Import and use this rather than
#: hard-coding 8.0 / 0.5 / 22.0 / 10 elsewhere in the codebase.
STANDARD_SLIDE_PROTOCOL = StandardSlideProtocol()


def calculate_coverslip_area_mm2(length_mm: float, width_mm: float) -> float:
    """
    coverslip_area_mm2 = coverslip_length_mm × coverslip_width_mm

    Parameters
    ----------
    length_mm, width_mm : float   must both be > 0

    Returns
    -------
    float   area in mm^2 (22 × 22 -> 484.0)
    """
    if length_mm <= 0 or width_mm <= 0:
        raise ValueError("coverslip length and width must both be > 0")
    return length_mm * width_mm


def calculate_theoretical_depth_um(
    sample_volume_ul: float,
    coverslip_area_mm2: float,
) -> float:
    """
    PART 9 — Theoretical average depth for a standard glass slide.

    NOT a measured/machined chamber depth — an AVERAGE depth implied
    by spreading a known volume across a known coverslip area, assuming
    perfectly even spreading (which real coverslips only approximate).
    Always label results from this function "Theoretical Average
    Depth", never "Actual Chamber Depth" (PART 9).

        1 µL = 1 mm^3
        theoretical_depth_mm = sample_volume_ul / coverslip_area_mm2
        theoretical_depth_um = theoretical_depth_mm × 1000

    Worked example (PART 9/45): 8 µL over a 22×22 mm (484 mm^2)
    coverslip -> 8/484 mm = 0.016529... mm -> 16.53 µm.

    Parameters
    ----------
    sample_volume_ul : float        must be > 0
    coverslip_area_mm2 : float      must be > 0

    Returns
    -------
    float   theoretical average depth in microns
    """
    if sample_volume_ul <= 0:
        raise ValueError("sample_volume_ul must be > 0")
    if coverslip_area_mm2 <= 0:
        raise ValueError("coverslip_area_mm2 must be > 0")
    theoretical_depth_mm = sample_volume_ul / coverslip_area_mm2   # 1 µL == 1 mm^3
    return theoretical_depth_mm * _UM_PER_MM


@dataclass
class StandardSlideTechnicalParams:
    """
    Read-only technical diagnostics for a Standard Glass Slide session
    (PART 7/34/35 — never manually entered by the operator; always
    computed). Everything here is shown under "Technical Parameters /
    Advanced Diagnostics" in the UI, not in the primary workflow.
    """
    coverslip_area_mm2: float
    theoretical_depth_um: float
    effective_depth_um: Optional[float]      # None unless validated
    depth_correction_factor: Optional[float]  # None unless validated
    depth_validation_state: str
    microns_per_pixel: Optional[float]
    fov_width_um: Optional[float]
    fov_height_um: Optional[float]
    number_of_fields: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverslip_area_mm2":       self.coverslip_area_mm2,
            "theoretical_depth_um":     self.theoretical_depth_um,
            "effective_depth_um":       self.effective_depth_um,
            "depth_correction_factor":  self.depth_correction_factor,
            "depth_validation_state":   self.depth_validation_state,
            "microns_per_pixel":        self.microns_per_pixel,
            "fov_width_um":             self.fov_width_um,
            "fov_height_um":            self.fov_height_um,
            "number_of_fields":         self.number_of_fields,
        }


def get_standard_slide_effective_depth(
    theoretical_depth_um: float,
    manual_effective_depth_um: Optional[float] = None,
    depth_correction_factor: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """
    PART 10 — Resolve the EFFECTIVE depth to actually use for a
    Standard Glass Slide concentration calculation, honestly
    distinguishing it from the theoretical depth.

    There is deliberately NO default/fallback correction factor here.
    Unless the operator has supplied an experimentally-validated
    effective depth or correction factor, this returns
    ``(None, ValidationState.NOT_CONFIGURED)`` and the caller MUST
    refuse to compute an absolute concentration (PART 10).

    Parameters
    ----------
    theoretical_depth_um : float
        From ``calculate_theoretical_depth_um``.
    manual_effective_depth_um : float | None
        A directly-entered, experimentally-validated effective depth
        (takes priority over a correction factor if both are given).
    depth_correction_factor : float | None
        ``effective_depth_um = theoretical_depth_um * depth_correction_factor``,
        if a validated factor is available.

    Returns
    -------
    (effective_depth_um | None, validation_state)
    """
    if manual_effective_depth_um is not None and manual_effective_depth_um > 0:
        return manual_effective_depth_um, ValidationState.VALIDATED
    if depth_correction_factor is not None and depth_correction_factor > 0:
        return theoretical_depth_um * depth_correction_factor, ValidationState.VALIDATED
    return None, ValidationState.NOT_CONFIGURED


def get_concentration_label(category: str, validation_state: str) -> str:
    """
    PART 29 — Precise, non-overclaiming terminology for a
    concentration result.

    Returns
    -------
    str  one of:
        "Measured Concentration"                      (physical chamber)
        "Estimated Concentration"                      (standard slide, not yet validated)
        "Validated Controlled-Protocol Concentration"  (standard slide, validated)
    """
    if category == SlideCategory.PHYSICAL_CHAMBER:
        return "Measured Concentration"
    if validation_state == ValidationState.VALIDATED:
        return "Validated Controlled-Protocol Concentration"
    return "Estimated Concentration"


# ══════════════════════════════════════════════════════════════
# PART 13/14/27 — Multi-field statistics & quality control
# ══════════════════════════════════════════════════════════════

class FieldType:
    """
    PART 13 — distinguishes truly independent spatial fields (e.g.
    separate captures at different stage positions) from temporal
    frames/segments sampled from the SAME physical field (the current
    single-video-recording pipeline's default). Never claim
    INDEPENDENT_SPATIAL fields were analysed unless the acquisition
    method actually moved to a new physical location.
    """
    INDEPENDENT_SPATIAL = "independent_spatial"
    TEMPORAL_SAME_FIELD = "temporal_same_field"


@dataclass
class FieldQuality:
    """
    PART 14 — per-field quality-control record.

    ``valid`` is False if `rejection_reasons` is non-empty; a rejected
    field's `sperm_count` is excluded from
    ``calculate_field_statistics`` by the caller.
    """
    field_id: int
    sperm_count: int
    mean_detection_confidence: Optional[float] = None
    valid_track_count: Optional[int] = None
    tracking_quality: Optional[str] = None
    blur_metric: Optional[float] = None
    illumination_metric: Optional[float] = None
    field_type: str = FieldType.TEMPORAL_SAME_FIELD
    rejection_reasons: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.rejection_reasons) == 0


@dataclass
class FieldStatistics:
    """PART 27 — descriptive statistics across a set of field counts."""
    mean: float
    median: float
    std_dev: float
    cv_percent: float
    n_valid_fields: int
    n_rejected_fields: int
    high_variability_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_mean_count":       self.mean,
            "field_median_count":     self.median,
            "field_std_dev":          self.std_dev,
            "field_cv_percent":       self.cv_percent,
            "n_valid_fields":         self.n_valid_fields,
            "n_rejected_fields":      self.n_rejected_fields,
            "field_variability_warning": self.high_variability_warning,
        }


#: PART 27 — above this CV%, sperm distribution across fields is
#: flagged as non-uniform and a repeat preparation is recommended.
#: This threshold is a UI/QC heuristic, not a statistically derived
#: cutoff — adjust here if lab experience indicates otherwise.
FIELD_CV_WARNING_THRESHOLD_PERCENT: float = 30.0


def calculate_field_statistics(
    field_counts: Sequence[int],
    n_rejected_fields: int = 0,
) -> FieldStatistics:
    """
    PART 27 — mean / median / standard deviation / CV across valid
    field counts, with a non-uniformity warning when CV is excessive.

    Parameters
    ----------
    field_counts : sequence of int   sperm counts for each VALID field
    n_rejected_fields : int          count of fields excluded by QC

    Returns
    -------
    FieldStatistics
    """
    import statistics as _stats

    valid = [c for c in field_counts if c is not None and c >= 0]
    n = len(valid)
    if n == 0:
        return FieldStatistics(
            mean=0.0, median=0.0, std_dev=0.0, cv_percent=0.0,
            n_valid_fields=0, n_rejected_fields=n_rejected_fields,
            high_variability_warning="Insufficient valid fields for reliable concentration estimation.",
        )

    mean_val = _stats.mean(valid)
    median_val = _stats.median(valid)
    std_dev = _stats.stdev(valid) if n >= 2 else 0.0
    cv_percent = (std_dev / mean_val * 100.0) if mean_val > 0 else 0.0

    warning = None
    if cv_percent > FIELD_CV_WARNING_THRESHOLD_PERCENT:
        warning = (
            "Non-uniform sperm distribution detected "
            f"(field CV {cv_percent:.1f}% > {FIELD_CV_WARNING_THRESHOLD_PERCENT:g}%). "
            "Recommend repeat preparation/measurement."
        )

    return FieldStatistics(
        mean=mean_val, median=median_val, std_dev=std_dev, cv_percent=cv_percent,
        n_valid_fields=n, n_rejected_fields=n_rejected_fields,
        high_variability_warning=warning,
    )


# ══════════════════════════════════════════════════════════════
# PART 28 — Measurement Quality Score
# ══════════════════════════════════════════════════════════════

@dataclass
class MeasurementQualityResult:
    """
    PART 28 — a transparent, explainable 0-100 composite score.

    THIS IS NOT A STATISTICALLY VALIDATED PROBABILITY OR ACCURACY
    METRIC — it is a rule-based composite of QC signals intended to
    give the operator a quick, explainable read on whether to trust a
    given result, with the individual `reasons` always shown alongside
    the number so nothing is a black box.
    """
    score: int              # 0-100
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"measurement_quality_score": self.score,
                "measurement_quality_reasons": list(self.reasons)}


def calculate_measurement_quality_score(
    calibration_validation_state: str,
    n_valid_fields: int,
    expected_fields: int,
    field_cv_percent: float,
    mean_detection_confidence: Optional[float] = None,
    tracking_quality: Optional[str] = None,
    protocol_validation_state: str = ValidationState.NOT_CONFIGURED,
) -> MeasurementQualityResult:
    """
    PART 28 — composite Measurement Quality Score out of 100.

    Weighting (documented here so the score is auditable, not a black
    box): calibration 25 pts, field completeness 20 pts, field
    uniformity (CV) 20 pts, detection confidence 15 pts, tracking
    quality 10 pts, protocol validation 10 pts. Missing/unavailable
    inputs simply contribute 0 for that component (never guessed).
    """
    score = 0
    reasons: List[str] = []

    if calibration_validation_state == ValidationState.VALIDATED:
        score += 25
        reasons.append("✓ Optical calibration valid")
    elif calibration_validation_state == ValidationState.PROVISIONAL:
        score += 12
        reasons.append("~ Optical calibration provisional (not independently validated)")
    else:
        reasons.append("✗ Optical calibration missing or not configured")

    if expected_fields > 0:
        completeness = min(1.0, n_valid_fields / expected_fields)
        score += int(round(20 * completeness))
        reasons.append(f"{'✓' if completeness >= 1.0 else '~'} {n_valid_fields}/{expected_fields} valid fields")
    else:
        reasons.append("✗ Expected field count not configured")

    if n_valid_fields >= 2:
        if field_cv_percent <= FIELD_CV_WARNING_THRESHOLD_PERCENT:
            score += 20
            reasons.append(f"✓ Low field CV ({field_cv_percent:.1f}%)")
        else:
            reasons.append(f"✗ High field CV ({field_cv_percent:.1f}%) — non-uniform distribution")
    else:
        reasons.append("~ Field CV unavailable (fewer than 2 valid fields)")

    if mean_detection_confidence is not None:
        conf_pts = int(round(15 * max(0.0, min(1.0, mean_detection_confidence))))
        score += conf_pts
        reasons.append(f"{'✓' if mean_detection_confidence >= 0.7 else '~'} "
                       f"Mean detection confidence {mean_detection_confidence * 100:.0f}%")
    else:
        reasons.append("~ Detection confidence not available")

    if tracking_quality in ("Excellent", "Good"):
        score += 10
        reasons.append(f"✓ Tracking quality {tracking_quality}")
    elif tracking_quality is not None:
        reasons.append(f"~ Tracking quality {tracking_quality}")
    else:
        reasons.append("~ Tracking quality not available")

    if protocol_validation_state == ValidationState.VALIDATED:
        score += 10
        reasons.append("✓ Protocol validated")
    elif protocol_validation_state == ValidationState.PROVISIONAL:
        score += 5
        reasons.append("~ Protocol provisional (not yet empirically validated)")
    else:
        reasons.append("✗ Protocol validation status not configured")

    return MeasurementQualityResult(score=max(0, min(100, score)), reasons=reasons)


# ══════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════

@dataclass
class ConcentrationResult:
    """
    Full output of a concentration/MCC calculation attempt.

    ``status`` is always one of:
        "ok"                    -- concentration was calculated
        "calibration_required"  -- missing µm/pixel calibration
        "depth_required"        -- missing/unverified chamber depth
        "insufficient_fields"   -- not enough valid counting fields
    Only when ``status == "ok"`` are the concentration fields non-None.
    """
    status: str
    reason: str = ""

    # Inputs actually used (for transparency/audit trail)
    sperm_count: int = 0
    number_of_counting_fields: int = 0
    fov_width_um: Optional[float] = None
    fov_height_um: Optional[float] = None
    fov_area_mm2: Optional[float] = None
    chamber_depth_um: Optional[float] = None
    observed_volume_ml: Optional[float] = None
    dilution_factor: float = DEFAULT_DILUTION_FACTOR

    # Outputs
    total_concentration_cells_per_ml: Optional[float] = None
    total_concentration_m_per_ml: Optional[float] = None

    motile_percentage: Optional[float] = None
    progressive_percentage: Optional[float] = None
    non_progressive_percentage: Optional[float] = None
    immotile_percentage: Optional[float] = None

    motile_concentration_m_per_ml: Optional[float] = None
    progressive_concentration_m_per_ml: Optional[float] = None
    non_progressive_concentration_m_per_ml: Optional[float] = None
    immotile_concentration_m_per_ml: Optional[float] = None

    counting_quality: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # ── CASA-like architecture extensions (integration turn) ────
    # Populated by `compute_concentration_metrics_from_params` for
    # BOTH pathways (physical chamber and Standard Glass Slide) so
    # the UI/DB always has a consistent, centrally-computed set of
    # fields regardless of which pathway actually ran — app.py never
    # computes any of these itself.
    concentration_label: Optional[str] = None          # PART 29/56
    slide_category: Optional[str] = None                # SlideCategory
    depth_validation_state: Optional[str] = None         # ValidationState
    field_statistics: Optional[Dict[str, Any]] = None    # FieldStatistics.to_dict()
    technical_params: Optional[Dict[str, Any]] = None    # StandardSlideTechnicalParams.to_dict()
    measurement_quality: Optional[Dict[str, Any]] = None  # MeasurementQualityResult.to_dict()
    analysis_method: Optional[str] = None                 # PHASE 6 — human-readable method name
    protocol_version: Optional[str] = None                # PHASE 6 — e.g. "1.0-provisional"
    optical_calibration_status: Optional[str] = None      # PHASE 6 — ValidationState string

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict form, ready to merge into a pipeline results dict."""
        d = {
            "concentration_status": self.status,
            "concentration_status_reason": self.reason,
            "sperm_count_observed": self.sperm_count,
            "number_of_counting_fields": self.number_of_counting_fields,
            "field_of_view_width_um": self.fov_width_um,
            "field_of_view_height_um": self.fov_height_um,
            "field_of_view_area_mm2": self.fov_area_mm2,
            "chamber_depth_um": self.chamber_depth_um,
            "observed_volume_ml": self.observed_volume_ml,
            "dilution_factor": self.dilution_factor,
            "total_concentration_cells_per_ml": self.total_concentration_cells_per_ml,
            "total_concentration_m_cells_per_ml": self.total_concentration_m_per_ml,
            "motile_percentage": self.motile_percentage,
            "progressive_percentage": self.progressive_percentage,
            "non_progressive_percentage": self.non_progressive_percentage,
            "immotile_percentage": self.immotile_percentage,
            "motile_concentration_m_cells_per_ml": self.motile_concentration_m_per_ml,
            "progressive_concentration_m_cells_per_ml": self.progressive_concentration_m_per_ml,
            "non_progressive_concentration_m_cells_per_ml": self.non_progressive_concentration_m_per_ml,
            "immotile_concentration_m_cells_per_ml": self.immotile_concentration_m_per_ml,
            "concentration_counting_quality": self.counting_quality,
            "concentration_warnings": list(self.warnings),
            "concentration_label": self.concentration_label,
            "slide_category": self.slide_category,
            "depth_validation_state": self.depth_validation_state,
            "analysis_method": self.analysis_method,
            "protocol_version": self.protocol_version,
            "optical_calibration_status": self.optical_calibration_status,
        }
        if self.field_statistics:
            d.update(self.field_statistics)
        if self.technical_params:
            d.update(self.technical_params)
        if self.measurement_quality:
            d.update(self.measurement_quality)
        return d


@dataclass
class SMLResult:
    """Output of a Sustained Motility Lifetime calculation attempt."""
    status: str                      # "ok" | "insufficient_data"
    reason: str = ""
    initial_progressive_motility: Optional[float] = None
    sml_threshold_percentage: Optional[float] = None
    sml_minutes: Optional[float] = None
    observation_duration_min: Optional[float] = None
    number_of_time_points: int = 0
    time_series: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sml_status": self.status,
            "sml_status_reason": self.reason,
            "initial_progressive_motility": self.initial_progressive_motility,
            "sml_threshold_percentage": self.sml_threshold_percentage,
            "sml_minutes": self.sml_minutes,
            "sml_observation_duration_min": self.observation_duration_min,
            "sml_number_of_time_points": self.number_of_time_points,
            "sml_time_series": [
                {"minutes": t, "progressive_pct": p} for t, p in self.time_series
            ],
        }


# ══════════════════════════════════════════════════════════════
# FEATURE 4 — Field of view / observed volume
# ══════════════════════════════════════════════════════════════

def calculate_field_of_view(
    image_width_px: int,
    image_height_px: int,
    microns_per_pixel_x: float,
    microns_per_pixel_y: float,
) -> Dict[str, float]:
    """
    Calculate the physical field of view from pixel dimensions and
    spatial calibration.

    Parameters
    ----------
    image_width_px, image_height_px : int
        Frame dimensions in pixels (from the actual video/camera).
    microns_per_pixel_x, microns_per_pixel_y : float
        Calibrated scale (must be > 0).

    Returns
    -------
    dict with keys:
        fov_width_um, fov_height_um, fov_width_mm, fov_height_mm,
        fov_area_mm2

    Raises
    ------
    ValueError
        If any input is non-positive.
    """
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("image_width_px and image_height_px must be > 0")
    if microns_per_pixel_x <= 0 or microns_per_pixel_y <= 0:
        raise ValueError("microns_per_pixel_x and microns_per_pixel_y must be > 0")

    fov_width_um  = image_width_px  * microns_per_pixel_x
    fov_height_um = image_height_px * microns_per_pixel_y
    fov_width_mm  = fov_width_um  / _UM_PER_MM
    fov_height_mm = fov_height_um / _UM_PER_MM
    fov_area_mm2  = fov_width_mm * fov_height_mm

    return {
        "fov_width_um":  fov_width_um,
        "fov_height_um": fov_height_um,
        "fov_width_mm":  fov_width_mm,
        "fov_height_mm": fov_height_mm,
        "fov_area_mm2":  fov_area_mm2,
    }


def calculate_observed_volume(
    fov_area_mm2: float,
    chamber_depth_um: float,
) -> float:
    """
    Calculate the physical volume actually observed by the microscope
    for ONE field, given the field-of-view area and the chamber/slide
    depth.

    observed_volume_mm3 = fov_area_mm2 * (chamber_depth_um / 1000)
    observed_volume_mL  = observed_volume_mm3 / 1000

    Parameters
    ----------
    fov_area_mm2 : float        must be > 0
    chamber_depth_um : float    must be > 0

    Returns
    -------
    float   observed volume in mL

    Raises
    ------
    ValueError
        If either input is non-positive.
    """
    if fov_area_mm2 <= 0:
        raise ValueError("fov_area_mm2 must be > 0")
    if chamber_depth_um <= 0:
        raise ValueError("chamber_depth_um must be > 0")

    chamber_depth_mm  = chamber_depth_um / _UM_PER_MM
    observed_volume_mm3 = fov_area_mm2 * chamber_depth_mm
    observed_volume_ml  = observed_volume_mm3 / _MM3_PER_ML
    return observed_volume_ml


# ══════════════════════════════════════════════════════════════
# FEATURE 6 — Concentration
# ══════════════════════════════════════════════════════════════

def calculate_sperm_concentration(
    sperm_count: int,
    observed_volume_ml: float,
    dilution_factor: float = DEFAULT_DILUTION_FACTOR,
) -> float:
    """
    Core concentration formula.

        concentration_cells_per_ml =
            (sperm_count / observed_volume_ml) * dilution_factor

    Parameters
    ----------
    sperm_count : int                  unique cells observed, >= 0
    observed_volume_ml : float         must be > 0
    dilution_factor : float            must be > 0 (1.0 = undiluted)

    Returns
    -------
    float   concentration in cells/mL

    Raises
    ------
    ValueError
        If observed_volume_ml <= 0, dilution_factor <= 0, or
        sperm_count < 0.
    """
    if sperm_count < 0:
        raise ValueError("sperm_count cannot be negative")
    if observed_volume_ml <= 0:
        raise ValueError("observed_volume_ml must be > 0")
    if dilution_factor <= 0:
        raise ValueError("dilution_factor must be > 0")

    observed_concentration = sperm_count / observed_volume_ml
    return observed_concentration * dilution_factor


def cells_per_ml_to_m_per_ml(concentration_cells_per_ml: float) -> float:
    """Convert cells/mL to M cells/mL (division by 1,000,000)."""
    return concentration_cells_per_ml / _CELLS_PER_M_CELLS


def calculate_motile_concentration(
    total_concentration_m_per_ml: float,
    motile_fraction: float,
) -> float:
    """
    MCC = total_concentration * motile_fraction.

    Parameters
    ----------
    total_concentration_m_per_ml : float   >= 0
    motile_fraction : float                0.0-1.0 (NOT a percentage)

    Returns
    -------
    float   M cells/mL
    """
    if total_concentration_m_per_ml < 0:
        raise ValueError("total_concentration_m_per_ml cannot be negative")
    if not (0.0 <= motile_fraction <= 1.0):
        raise ValueError("motile_fraction must be between 0.0 and 1.0")
    return total_concentration_m_per_ml * motile_fraction


def _fraction_concentration(total_concentration_m_per_ml: float, fraction: float) -> float:
    """Shared implementation: concentration = total * fraction (0.0-1.0)."""
    if total_concentration_m_per_ml < 0:
        raise ValueError("total_concentration_m_per_ml cannot be negative")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError("fraction must be between 0.0 and 1.0")
    return total_concentration_m_per_ml * fraction


def calculate_progressive_concentration(
    total_concentration_m_per_ml: float,
    progressive_fraction: float,
) -> float:
    """Progressive concentration = total_concentration * progressive_fraction."""
    return _fraction_concentration(total_concentration_m_per_ml, progressive_fraction)


def calculate_non_progressive_concentration(
    total_concentration_m_per_ml: float,
    non_progressive_fraction: float,
) -> float:
    """Non-progressive concentration = total_concentration * non_progressive_fraction."""
    return _fraction_concentration(total_concentration_m_per_ml, non_progressive_fraction)


def calculate_immotile_concentration(
    total_concentration_m_per_ml: float,
    immotile_fraction: float,
) -> float:
    """Immotile concentration = total_concentration * immotile_fraction."""
    return _fraction_concentration(total_concentration_m_per_ml, immotile_fraction)


def mean_sperm_per_field(field_counts: Sequence[int]) -> Tuple[float, int]:
    """
    Average sperm count across multiple counting fields.

    Parameters
    ----------
    field_counts : sequence of int   one count per valid field

    Returns
    -------
    (mean_count, number_of_valid_fields)
    """
    valid = [c for c in field_counts if c is not None and c >= 0]
    if not valid:
        return 0.0, 0
    return sum(valid) / len(valid), len(valid)


def get_counting_quality(sperm_count: int, fov_area_mm2: float) -> str:
    """
    Heuristic, UI-facing counting-quality label based on detection
    density (sperm per unit area of the field of view).

    THIS IS A UI GUIDANCE HEURISTIC, NOT A VALIDATED ACCURACY METRIC.
    High density increases the chance of overlapping sperm causing
    missed/merged/duplicate YOLO detections and ByteTrack ID switches
    -- this function flags that risk to the operator, it does not
    measure it directly.

    Parameters
    ----------
    sperm_count : int
    fov_area_mm2 : float   must be > 0

    Returns
    -------
    str   one of "Excellent", "Good", "Moderate", "Poor"
    """
    if fov_area_mm2 <= 0:
        return _DENSITY_QUALITY_POOR_LABEL
    # Density normalised to sperm per 10,000 µm^2 (~ a 100x100 µm patch)
    area_um2 = fov_area_mm2 * (_UM_PER_MM ** 2)
    density_per_10k_um2 = (sperm_count / area_um2) * 10_000.0 if area_um2 > 0 else 0.0

    for threshold, label in _DENSITY_QUALITY_THRESHOLDS:
        if density_per_10k_um2 <= threshold:
            return label
    return _DENSITY_QUALITY_POOR_LABEL


# ══════════════════════════════════════════════════════════════
# FEATURE 15 — Validation
# ══════════════════════════════════════════════════════════════

def validate_sample_volume(
    slide_mode: str,
    sample_volume_ul: float,
) -> Optional[str]:
    """
    Validate a user-entered sample volume against the configured
    range for the given slide mode.

    Parameters
    ----------
    slide_mode : str        "normal_slide" | "chamber"
    sample_volume_ul : float

    Returns
    -------
    str | None   a warning message, or None if the volume is in range.
    """
    if slide_mode == "normal_slide":
        lo, hi = NORMAL_SLIDE_MIN_VOLUME_UL, NORMAL_SLIDE_MAX_VOLUME_UL
        label = "normal-slide"
    else:
        lo, hi = CHAMBER_MIN_TOTAL_VOLUME_UL, CHAMBER_MAX_TOTAL_VOLUME_UL
        label = "chamber"

    if sample_volume_ul < lo or sample_volume_ul > hi:
        return (
            f"Sample volume is outside the configured {label} range of "
            f"{lo:g}–{hi:g} µL."
        )
    return None


def get_concentration_status(concentration_m_per_ml: Optional[float]) -> Optional[str]:
    """
    Check a calculated concentration against the configured target
    operating range (Feature 1/15). Does NOT imply scientific
    validation at the extremes -- see module/Feature-1 docstring.

    Returns
    -------
    str | None   a warning message, or None if within range / unknown.
    """
    if concentration_m_per_ml is None:
        return None
    if concentration_m_per_ml < TARGET_MIN_CONCENTRATION_M_PER_ML:
        return (
            f"Calculated concentration ({concentration_m_per_ml:.2f} M/mL) is "
            f"below the configured target operating range of "
            f"{TARGET_MIN_CONCENTRATION_M_PER_ML:g}–{TARGET_MAX_CONCENTRATION_M_PER_ML:g} M/mL."
        )
    if concentration_m_per_ml > TARGET_MAX_CONCENTRATION_M_PER_ML:
        return (
            f"Calculated concentration ({concentration_m_per_ml:.2f} M/mL) is "
            f"above the configured target operating range of "
            f"{TARGET_MIN_CONCENTRATION_M_PER_ML:g}–{TARGET_MAX_CONCENTRATION_M_PER_ML:g} M/mL."
        )
    return None


# ══════════════════════════════════════════════════════════════
# Top-level orchestration — concentration + MCC
# ══════════════════════════════════════════════════════════════

def compute_concentration_metrics(
    sperm_count: int,
    motility_summary: Dict[str, float],
    image_width_px: int,
    image_height_px: int,
    slide_mode: str,
    chamber_name: str,
    magnification: str,
    camera_resolution: str,
    dilution_factor: float = DEFAULT_DILUTION_FACTOR,
    number_of_counting_fields: int = DEFAULT_NUMBER_OF_COUNTING_FIELDS,
    manual_microns_per_pixel: Optional[float] = None,
    manual_chamber_depth_um: Optional[float] = None,
    field_counts: Optional[Sequence[int]] = None,
    saved_calibration_um_per_pixel: Optional[float] = None,
) -> ConcentrationResult:
    """
    Compute total/motile/progressive/non-progressive/immotile
    concentration for a single analysis, honestly refusing when
    calibration or chamber-depth information is missing.

    Parameters
    ----------
    sperm_count : int
        Unique sperm cells observed (e.g. ``results["total_sperm"]``,
        already deduplicated by ByteTrack track IDs — never a raw
        per-frame detection sum).
    motility_summary : dict
        Must contain ``progressive_pct``/``nonprogressive_pct`` (or
        ``non_progressive_pct``)/``immotile_pct`` as 0-100 percentages.
    image_width_px, image_height_px : int
        Actual video/camera frame dimensions.
    slide_mode : str
        "normal_slide" | "chamber"
    chamber_name : str
        Key into ``CHAMBER_CONFIG`` (e.g. "Makler Chamber").
    magnification, camera_resolution : str
        Used to look up calibration in ``CALIBRATION_TABLE`` if no
        manual override is given.
    dilution_factor : float
    number_of_counting_fields : int
    manual_microns_per_pixel : float | None
        Directly-entered calibration value; overrides the lookup table.
    manual_chamber_depth_um : float | None
        Directly-entered depth; overrides ``CHAMBER_CONFIG``. Intended
        for "Standard Glass Slide" once a locally-validated effective
        depth is known, or to correct a non-standard chamber variant.
    field_counts : sequence of int | None
        Per-field sperm counts, if multiple fields were actually
        captured. When omitted, ``[sperm_count]`` is used (the current
        video treated as the one field observed) — see module
        docstring, "Counting-field model".

    Returns
    -------
    ConcentrationResult
    """
    warnings: List[str] = []

    # ── Resolve calibration (Feature 3) ─────────────────────────
    microns_per_pixel = resolve_microns_per_pixel(
        magnification, camera_resolution, manual_microns_per_pixel,
        saved_calibration_um_per_pixel,
    )
    if microns_per_pixel is None:
        return ConcentrationResult(
            status="calibration_required",
            reason=(
                "Concentration cannot be calculated accurately until "
                "microscope/camera spatial calibration is configured "
                f"for {magnification} @ {camera_resolution}."
            ),
            sperm_count=sperm_count,
            number_of_counting_fields=number_of_counting_fields,
        )

    # ── Resolve chamber depth (Feature 4/17/20) ─────────────────
    chamber_spec = CHAMBER_CONFIG.get(chamber_name, CHAMBER_CONFIG["Other"])
    depth_um = manual_chamber_depth_um if manual_chamber_depth_um else chamber_spec.depth_um
    if depth_um is None:
        if slide_mode == "normal_slide":
            reason = (
                "Absolute concentration requires a validated effective "
                "slide depth. Total sperm observed, motility, velocity, "
                "morphology, viability, and SML remain available; only "
                "the absolute concentration figures are withheld."
            )
        else:
            reason = (
                f"Concentration unavailable: no verified depth is "
                f"configured for chamber '{chamber_name}'. "
                f"{chamber_spec.note}"
            )
        return ConcentrationResult(
            status="depth_required",
            reason=reason,
            sperm_count=sperm_count,
            number_of_counting_fields=number_of_counting_fields,
        )

    # ── Field of view / observed volume ─────────────────────────
    try:
        fov = calculate_field_of_view(
            image_width_px, image_height_px,
            microns_per_pixel, microns_per_pixel,
        )
        observed_volume_ml_per_field = calculate_observed_volume(
            fov["fov_area_mm2"], depth_um,
        )
    except ValueError as exc:
        return ConcentrationResult(
            status="calibration_required",
            reason=f"Could not calculate observed volume: {exc}",
            sperm_count=sperm_count,
            number_of_counting_fields=number_of_counting_fields,
        )

    # ── Counting fields / sperm count to use ────────────────────
    counts = list(field_counts) if field_counts else [sperm_count]
    mean_count, n_valid_fields = mean_sperm_per_field(counts)
    if n_valid_fields < MIN_VALID_FIELDS_REQUIRED:
        return ConcentrationResult(
            status="insufficient_fields",
            reason="Insufficient valid fields for reliable concentration estimation.",
            sperm_count=sperm_count,
            number_of_counting_fields=number_of_counting_fields,
        )

    total_observed_volume_ml = observed_volume_ml_per_field  # one field's volume;
    # `mean_count` already averages the per-field counts onto that
    # single field's observed volume — see module docstring.

    # ── Core concentration ───────────────────────────────────────
    try:
        conc_cells_per_ml = calculate_sperm_concentration(
            int(round(mean_count)), total_observed_volume_ml, dilution_factor,
        )
    except ValueError as exc:
        return ConcentrationResult(
            status="calibration_required",
            reason=f"Could not calculate concentration: {exc}",
            sperm_count=sperm_count,
            number_of_counting_fields=number_of_counting_fields,
        )
    conc_m_per_ml = cells_per_ml_to_m_per_ml(conc_cells_per_ml)

    # ── Motility fractions (Feature 8/9) ────────────────────────
    progressive_pct = float(
        motility_summary.get("progressive_pct", motility_summary.get("progressive", 0.0)) or 0.0
    )
    nonprog_pct = float(
        motility_summary.get("nonprogressive_pct",
                             motility_summary.get("non_progressive_pct",
                                                  motility_summary.get("non_progressive", 0.0))) or 0.0
    )
    immotile_pct = float(
        motility_summary.get("immotile_pct", motility_summary.get("immotile", 0.0)) or 0.0
    )
    motile_pct = progressive_pct + nonprog_pct

    motile_conc      = calculate_motile_concentration(conc_m_per_ml, motile_pct / 100.0)
    progressive_conc = calculate_progressive_concentration(conc_m_per_ml, progressive_pct / 100.0)
    nonprog_conc      = calculate_non_progressive_concentration(conc_m_per_ml, nonprog_pct / 100.0)
    immotile_conc     = calculate_immotile_concentration(conc_m_per_ml, immotile_pct / 100.0)

    # ── Warnings (Feature 15) ────────────────────────────────────
    range_warning = get_concentration_status(conc_m_per_ml)
    if range_warning:
        warnings.append(range_warning)

    quality = get_counting_quality(int(round(mean_count)), fov["fov_area_mm2"])
    if quality in ("Moderate", "Poor"):
        warnings.append(
            "High sperm density may reduce counting accuracy. Consider "
            "dilution or a validated counting protocol."
        )
    if not chamber_spec.verified and manual_chamber_depth_um is None:
        warnings.append(
            f"Chamber depth for '{chamber_name}' is not an independently "
            f"verified value for your specific hardware — {chamber_spec.note}"
        )

    return ConcentrationResult(
        status="ok",
        reason="",
        sperm_count=sperm_count,
        number_of_counting_fields=n_valid_fields,
        fov_width_um=fov["fov_width_um"],
        fov_height_um=fov["fov_height_um"],
        fov_area_mm2=fov["fov_area_mm2"],
        chamber_depth_um=depth_um,
        observed_volume_ml=total_observed_volume_ml,
        dilution_factor=dilution_factor,
        total_concentration_cells_per_ml=conc_cells_per_ml,
        total_concentration_m_per_ml=conc_m_per_ml,
        motile_percentage=motile_pct,
        progressive_percentage=progressive_pct,
        non_progressive_percentage=nonprog_pct,
        immotile_percentage=immotile_pct,
        motile_concentration_m_per_ml=motile_conc,
        progressive_concentration_m_per_ml=progressive_conc,
        non_progressive_concentration_m_per_ml=nonprog_conc,
        immotile_concentration_m_per_ml=immotile_conc,
        counting_quality=quality,
        warnings=warnings,
    )


# ══════════════════════════════════════════════════════════════
# FEATURE 10 — Sustained Motility Lifetime (SML)
# ══════════════════════════════════════════════════════════════

def calculate_sml(
    time_points_min: Sequence[float],
    progressive_pct_values: Sequence[float],
    threshold_fraction: float = SML_THRESHOLD_FRACTION,
) -> SMLResult:
    """
    Calculate Sustained Motility Lifetime: the time at which
    progressive motility falls to ``threshold_fraction`` (default 50%)
    of the initial/plateau progressive motility, via linear
    interpolation between the two bracketing time points.

    Parameters
    ----------
    time_points_min : sequence of float
        Elapsed time in minutes for each measurement, in any order
        (will be sorted internally). Must correspond 1:1 with
        ``progressive_pct_values``.
    progressive_pct_values : sequence of float
        Progressive motility percentage (0-100) at each time point.
    threshold_fraction : float
        Fraction of the initial value defining the SML crossing
        (default 0.5, i.e. 50%).

    Returns
    -------
    SMLResult
        ``status="insufficient_data"`` with no invented value if fewer
        than ``SML_MIN_TIME_POINTS`` valid points are available, or if
        progressive motility never crosses the threshold within the
        observed window.
    """
    pairs = [
        (float(t), float(p))
        for t, p in zip(time_points_min, progressive_pct_values)
        if t is not None and p is not None
    ]
    pairs.sort(key=lambda tp: tp[0])

    if len(pairs) < SML_MIN_TIME_POINTS:
        return SMLResult(
            status="insufficient_data",
            reason="SML unavailable — insufficient time-series measurements.",
            number_of_time_points=len(pairs),
            time_series=pairs,
        )

    t0, initial_pm = pairs[0]
    threshold = initial_pm * threshold_fraction
    duration = pairs[-1][0] - t0

    if initial_pm <= 0:
        return SMLResult(
            status="insufficient_data",
            reason="SML unavailable — initial progressive motility is zero.",
            initial_progressive_motility=initial_pm,
            sml_threshold_percentage=threshold,
            observation_duration_min=duration,
            number_of_time_points=len(pairs),
            time_series=pairs,
        )

    # Find the first pair of consecutive points that bracket the
    # threshold crossing (PM1 >= threshold >= PM2, i.e. declining
    # through the threshold).
    for (t1, pm1), (t2, pm2) in zip(pairs[:-1], pairs[1:]):
        if pm1 >= threshold >= pm2:
            if pm2 == pm1:
                t_sml = t1  # no slope; crossing coincides with t1
            else:
                t_sml = t1 + (threshold - pm1) / (pm2 - pm1) * (t2 - t1)
            return SMLResult(
                status="ok",
                initial_progressive_motility=initial_pm,
                sml_threshold_percentage=threshold,
                sml_minutes=t_sml,
                observation_duration_min=duration,
                number_of_time_points=len(pairs),
                time_series=pairs,
            )

    return SMLResult(
        status="insufficient_data",
        reason=(
            "SML unavailable — progressive motility did not fall to the "
            f"{threshold_fraction * 100:.0f}% threshold within the "
            "observed time window."
        ),
        initial_progressive_motility=initial_pm,
        sml_threshold_percentage=threshold,
        observation_duration_min=duration,
        number_of_time_points=len(pairs),
        time_series=pairs,
    )


# ══════════════════════════════════════════════════════════════
# PART 4/20/21 — Standard Glass Slide CASA-like concentration
# ══════════════════════════════════════════════════════════════

def compute_standard_slide_concentration(
    field_sperm_counts: Sequence[int],
    motility_summary: Dict[str, float],
    microns_per_pixel: Optional[float],
    image_width_px: int,
    image_height_px: int,
    protocol: StandardSlideProtocol = STANDARD_SLIDE_PROTOCOL,
    manual_effective_depth_um: Optional[float] = None,
    depth_correction_factor: Optional[float] = None,
    dilution_factor: float = DEFAULT_DILUTION_FACTOR,
    n_rejected_fields: int = 0,
    mean_detection_confidence: Optional[float] = None,
    tracking_quality: Optional[str] = None,
) -> Tuple[ConcentrationResult, FieldStatistics, StandardSlideTechnicalParams, MeasurementQualityResult]:
    """
    PART 20/21 — Standard Glass Slide CASA-like controlled-estimation
    concentration pathway.

    Distinct from ``compute_concentration_metrics`` (the
    physical-chamber pathway) in three ways:
      1. Depth comes from ``get_standard_slide_effective_depth`` (a
         THEORETICAL average depth from the controlled protocol,
         optionally corrected by a *validated* factor) — never a
         chamber's published spec.
      2. Concentration is estimated from MULTIPLE fields (mean count
         across ``field_sperm_counts``), not a single field, with
         field statistics (mean/median/SD/CV) always returned
         alongside so variability is visible (PART 27).
      3. The result is always labelled via ``get_concentration_label``
         — "Estimated Concentration" unless the depth has actually
         been validated (PART 29) — and comes with a
         ``MeasurementQualityResult`` (PART 28).

    Parameters
    ----------
    field_sperm_counts : sequence of int
        Unique sperm count for each analysed field (see module
        docstring re: temporal vs. independent-spatial fields — the
        caller is responsible for recording which kind these are).
    motility_summary : dict
        Same shape as ``compute_concentration_metrics``.
    microns_per_pixel, image_width_px, image_height_px :
        Same as ``compute_concentration_metrics`` — spatial
        calibration is REQUIRED here exactly as for a physical
        chamber; a standard slide does not exempt you from optical
        calibration (PART 11).
    protocol : StandardSlideProtocol
        Defaults to the module-level ``STANDARD_SLIDE_PROTOCOL``.
    manual_effective_depth_um, depth_correction_factor :
        See ``get_standard_slide_effective_depth`` — both default to
        ``None``, meaning concentration is withheld
        (``status="depth_required"``) until one is supplied.
    n_rejected_fields, mean_detection_confidence, tracking_quality :
        Forwarded into the Measurement Quality Score (PART 28).

    Returns
    -------
    (ConcentrationResult, FieldStatistics, StandardSlideTechnicalParams, MeasurementQualityResult)
    """
    coverslip_area = calculate_coverslip_area_mm2(
        protocol.coverslip_length_mm, protocol.coverslip_width_mm,
    )
    theoretical_depth_um = calculate_theoretical_depth_um(
        protocol.sample_volume_ul, coverslip_area,
    )
    effective_depth_um, depth_state = get_standard_slide_effective_depth(
        theoretical_depth_um, manual_effective_depth_um, depth_correction_factor,
    )

    field_stats = calculate_field_statistics(field_sperm_counts, n_rejected_fields)

    calibration_state = (
        ValidationState.PROVISIONAL if microns_per_pixel else ValidationState.NOT_CONFIGURED
    )

    tech_params = StandardSlideTechnicalParams(
        coverslip_area_mm2=coverslip_area,
        theoretical_depth_um=theoretical_depth_um,
        effective_depth_um=effective_depth_um,
        depth_correction_factor=depth_correction_factor,
        depth_validation_state=depth_state,
        microns_per_pixel=microns_per_pixel,
        fov_width_um=None,
        fov_height_um=None,
        number_of_fields=protocol.number_of_fields,
    )

    quality = calculate_measurement_quality_score(
        calibration_validation_state=calibration_state,
        n_valid_fields=field_stats.n_valid_fields,
        expected_fields=protocol.number_of_fields,
        field_cv_percent=field_stats.cv_percent,
        mean_detection_confidence=mean_detection_confidence,
        tracking_quality=tracking_quality,
        protocol_validation_state=protocol.validation_state,
    )

    # ── Refuse honestly if calibration or depth is missing ──────
    if not microns_per_pixel:
        result = ConcentrationResult(
            status="calibration_required",
            reason="Optical calibration required.",
            sperm_count=int(round(field_stats.mean)),
            number_of_counting_fields=field_stats.n_valid_fields,
        )
        return result, field_stats, tech_params, quality

    if effective_depth_um is None:
        result = ConcentrationResult(
            status="depth_required",
            reason=(
                "Effective slide depth calibration not validated. "
                f"Theoretical average depth is {theoretical_depth_um:.2f} µm "
                "(from the controlled protocol), but this is not a "
                "validated effective depth — see PART 10/22."
            ),
            sperm_count=int(round(field_stats.mean)),
            number_of_counting_fields=field_stats.n_valid_fields,
            chamber_depth_um=None,
        )
        return result, field_stats, tech_params, quality

    if field_stats.n_valid_fields < MIN_VALID_FIELDS_REQUIRED:
        result = ConcentrationResult(
            status="insufficient_fields",
            reason="Insufficient valid fields for reliable concentration estimation.",
            sperm_count=0,
            number_of_counting_fields=field_stats.n_valid_fields,
        )
        return result, field_stats, tech_params, quality

    try:
        fov = calculate_field_of_view(
            image_width_px, image_height_px, microns_per_pixel, microns_per_pixel,
        )
        tech_params.fov_width_um = fov["fov_width_um"]
        tech_params.fov_height_um = fov["fov_height_um"]
        observed_volume_ml = calculate_observed_volume(fov["fov_area_mm2"], effective_depth_um)
        conc_cells_per_ml = calculate_sperm_concentration(
            int(round(field_stats.mean)), observed_volume_ml, dilution_factor,
        )
    except ValueError as exc:
        result = ConcentrationResult(
            status="calibration_required",
            reason=f"Could not calculate concentration: {exc}",
            sperm_count=int(round(field_stats.mean)),
            number_of_counting_fields=field_stats.n_valid_fields,
        )
        return result, field_stats, tech_params, quality

    conc_m_per_ml = cells_per_ml_to_m_per_ml(conc_cells_per_ml)

    progressive_pct = float(motility_summary.get("progressive_pct", 0.0) or 0.0)
    nonprog_pct = float(motility_summary.get("nonprogressive_pct", 0.0) or 0.0)
    immotile_pct = float(motility_summary.get("immotile_pct", 0.0) or 0.0)
    motile_pct = progressive_pct + nonprog_pct

    warnings: List[str] = []
    if field_stats.high_variability_warning:
        warnings.append(field_stats.high_variability_warning)
    range_warning = get_concentration_status(conc_m_per_ml)
    if range_warning:
        warnings.append(range_warning)
    label = get_concentration_label(SlideCategory.CONTROLLED_ESTIMATION, depth_state)
    warnings.append(
        f"Result type: {label}. Theoretical average depth "
        f"({theoretical_depth_um:.2f} µm) is a controlled-protocol "
        "estimate, not a precision-machined chamber measurement — "
        "see PART 10/22 for validation status."
    )

    result = ConcentrationResult(
        status="ok",
        sperm_count=int(round(field_stats.mean)),
        number_of_counting_fields=field_stats.n_valid_fields,
        fov_width_um=fov["fov_width_um"],
        fov_height_um=fov["fov_height_um"],
        fov_area_mm2=fov["fov_area_mm2"],
        chamber_depth_um=effective_depth_um,
        observed_volume_ml=observed_volume_ml,
        dilution_factor=dilution_factor,
        total_concentration_cells_per_ml=conc_cells_per_ml,
        total_concentration_m_per_ml=conc_m_per_ml,
        motile_percentage=motile_pct,
        progressive_percentage=progressive_pct,
        non_progressive_percentage=nonprog_pct,
        immotile_percentage=immotile_pct,
        motile_concentration_m_per_ml=calculate_motile_concentration(conc_m_per_ml, motile_pct / 100.0),
        progressive_concentration_m_per_ml=calculate_progressive_concentration(conc_m_per_ml, progressive_pct / 100.0),
        non_progressive_concentration_m_per_ml=calculate_non_progressive_concentration(conc_m_per_ml, nonprog_pct / 100.0),
        immotile_concentration_m_per_ml=calculate_immotile_concentration(conc_m_per_ml, immotile_pct / 100.0),
        counting_quality=get_counting_quality(int(round(field_stats.mean)), fov["fov_area_mm2"]),
        warnings=warnings,
    )
    return result, field_stats, tech_params, quality


# ══════════════════════════════════════════════════════════════
# PART 22-24, 38-40 — Empirical validation / reference-method
# framework
# ══════════════════════════════════════════════════════════════
#
# Pure statistics only. NOTHING here trains or applies a calibration
# model automatically (PART 24/25) — these functions compute error
# metrics FROM a dataset you supply; deciding whether/how to correct
# for that error (constant bias vs. proportional vs. a regression
# model) is a human, evidence-based decision made by inspecting the
# results of these functions, not something this module does for you.

@dataclass
class ReferenceValidationRecord:
    """
    PART 22 — one paired (reference, system) measurement for
    empirical validation. ``reference_concentration_m_per_ml`` MUST
    come from a trusted/validated laboratory method (PART 22/23) —
    never from this application's own output.
    """
    sample_id: str
    reference_concentration_m_per_ml: float
    system_concentration_m_per_ml: float
    reference_method: str                      # PART 23, e.g. "Hemocytometer/manual"
    sample_volume_ul: Optional[float] = None
    chamber: Optional[str] = None
    microscope: Optional[str] = None
    camera: Optional[str] = None
    objective: Optional[str] = None
    camera_resolution: Optional[str] = None
    microns_per_pixel: Optional[float] = None
    number_of_fields: Optional[int] = None
    field_counts: List[int] = field(default_factory=list)
    detection_quality: Optional[str] = None
    tracking_quality: Optional[str] = None
    operator: Optional[str] = None
    date: Optional[str] = None

    @property
    def error_m_per_ml(self) -> float:
        """system - reference (signed)."""
        return self.system_concentration_m_per_ml - self.reference_concentration_m_per_ml

    @property
    def relative_error_percent(self) -> Optional[float]:
        if self.reference_concentration_m_per_ml == 0:
            return None
        return (self.error_m_per_ml / self.reference_concentration_m_per_ml) * 100.0


def calculate_bias(records: Sequence[ReferenceValidationRecord]) -> Optional[float]:
    """Mean signed error (system - reference), in M/mL. None if no records."""
    if not records:
        return None
    errors = [r.error_m_per_ml for r in records]
    return sum(errors) / len(errors)


def calculate_mae(records: Sequence[ReferenceValidationRecord]) -> Optional[float]:
    """Mean Absolute Error, in M/mL."""
    if not records:
        return None
    errors = [abs(r.error_m_per_ml) for r in records]
    return sum(errors) / len(errors)


def calculate_rmse(records: Sequence[ReferenceValidationRecord]) -> Optional[float]:
    """Root Mean Squared Error, in M/mL."""
    if not records:
        return None
    sq_errors = [r.error_m_per_ml ** 2 for r in records]
    return (sum(sq_errors) / len(sq_errors)) ** 0.5


def calculate_mape(records: Sequence[ReferenceValidationRecord]) -> Optional[float]:
    """
    Mean Absolute Percentage Error. Records with a zero reference
    value are excluded (undefined percentage error); returns None if
    no records have a usable relative error.
    """
    rel_errors = [
        abs(r.relative_error_percent) for r in records
        if r.relative_error_percent is not None
    ]
    if not rel_errors:
        return None
    return sum(rel_errors) / len(rel_errors)


def calculate_correlation(records: Sequence[ReferenceValidationRecord]) -> Optional[float]:
    """Pearson correlation coefficient between reference and system values."""
    if len(records) < 2:
        return None
    xs = [r.reference_concentration_m_per_ml for r in records]
    ys = [r.system_concentration_m_per_ml for r in records]
    n = len(records)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return cov / denom


@dataclass
class BlandAltmanResult:
    """PART 24/38 — classic Bland-Altman agreement analysis."""
    mean_difference: float          # bias
    std_dev_difference: float
    upper_limit_of_agreement: float  # mean_diff + 1.96*SD
    lower_limit_of_agreement: float  # mean_diff - 1.96*SD
    points: List[Tuple[float, float]]  # (mean_of_pair, difference) per record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bland_altman_mean_difference": self.mean_difference,
            "bland_altman_std_dev":         self.std_dev_difference,
            "bland_altman_upper_loa":       self.upper_limit_of_agreement,
            "bland_altman_lower_loa":       self.lower_limit_of_agreement,
            "bland_altman_points": [
                {"mean": m, "difference": d} for m, d in self.points
            ],
        }


def calculate_bland_altman(
    records: Sequence[ReferenceValidationRecord],
) -> Optional[BlandAltmanResult]:
    """
    PART 24/38 — Bland-Altman agreement analysis between reference and
    system measurements. Returns None if fewer than 2 records (a
    standard deviation is not meaningful with 0-1 points).
    """
    import statistics as _stats

    if len(records) < 2:
        return None

    points: List[Tuple[float, float]] = []
    diffs: List[float] = []
    for r in records:
        mean_of_pair = (r.reference_concentration_m_per_ml + r.system_concentration_m_per_ml) / 2.0
        diff = r.error_m_per_ml
        points.append((mean_of_pair, diff))
        diffs.append(diff)

    mean_diff = _stats.mean(diffs)
    sd_diff = _stats.stdev(diffs)
    return BlandAltmanResult(
        mean_difference=mean_diff,
        std_dev_difference=sd_diff,
        upper_limit_of_agreement=mean_diff + 1.96 * sd_diff,
        lower_limit_of_agreement=mean_diff - 1.96 * sd_diff,
        points=points,
    )


@dataclass
class RepeatabilityResult:
    """PART 39 — repeated-measurement statistics for one sample."""
    sample_id: str
    n_runs: int
    mean_m_per_ml: float
    std_dev_m_per_ml: float
    cv_percent: float
    values_m_per_ml: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_runs": self.n_runs,
            "repeatability_mean_m_ml": self.mean_m_per_ml,
            "repeatability_std_dev_m_ml": self.std_dev_m_per_ml,
            "repeatability_cv_percent": self.cv_percent,
            "repeatability_values_m_ml": list(self.values_m_per_ml),
        }


def calculate_repeatability(
    sample_id: str,
    repeated_concentrations_m_per_ml: Sequence[float],
) -> Optional[RepeatabilityResult]:
    """
    PART 39 — mean / SD / CV across repeated runs of the SAME sample
    (same preparation, ideally same day/operator — see PART 40 for
    cross-operator/day/instrument REPRODUCIBILITY, which is this same
    calculation applied to a differently-grouped dataset).

    Returns None if fewer than 2 runs are supplied (repeatability is
    not meaningful from a single measurement).
    """
    import statistics as _stats

    values = [v for v in repeated_concentrations_m_per_ml if v is not None]
    if len(values) < 2:
        return None

    mean_val = _stats.mean(values)
    std_dev = _stats.stdev(values)
    cv_percent = (std_dev / mean_val * 100.0) if mean_val > 0 else 0.0

    return RepeatabilityResult(
        sample_id=sample_id, n_runs=len(values), mean_m_per_ml=mean_val,
        std_dev_m_per_ml=std_dev, cv_percent=cv_percent, values_m_per_ml=list(values),
    )


# ══════════════════════════════════════════════════════════════
# Convenience: build a ConcentrationResult from a pipeline results
# dict + a session's concentration_params dict (used by both
# run_analysis.py and app.py's monitoring-capture enrichment path).
# ══════════════════════════════════════════════════════════════

def compute_concentration_metrics_from_params(
    results: Dict[str, Any],
    concentration_params: Dict[str, Any],
) -> ConcentrationResult:
    """
    Central entry point: pull sperm count + motility percentages out
    of a pipeline `results` dict (flat `AutomatedSemenAnalysis` schema)
    and combine with a `concentration_params` dict (as built by
    app.py's intake form) to compute concentration/MCC.

    PHASE 1 ROUTING FIX: this function now explicitly branches on
    `concentration_params["slide_mode"]`:

        "normal_slide"  -> compute_standard_slide_concentration()
                           (Standard Glass Slide CASA-like controlled
                           protocol: theoretical average depth, field
                           statistics, measurement quality score,
                           honest ESTIMATED/PROVISIONAL labeling)

        "chamber" (or anything else) -> compute_concentration_metrics()
                           (physical chamber: Makler / Leja /
                           Hemocytometer / Other -- UNCHANGED behavior)

    Previously this always called `compute_concentration_metrics()`
    regardless of slide_mode, which meant Standard Glass Slide could
    never produce a result (`CHAMBER_CONFIG["Standard Glass
    Slide"].depth_um` is `None`, so it always returned
    `status="depth_required"` no matter what the operator entered in
    the "Validated effective depth" / "Depth correction factor"
    fields -- those inputs were inert). This routes to the function
    that actually reads them.

    Parameters
    ----------
    results : dict
        A pipeline results dict containing at least "total_sperm" and
        the motility percentage keys.
    concentration_params : dict
        {
            "slide_mode": "normal_slide" | "chamber",
            "chamber_name": str,
            "magnification": str,
            "camera_resolution": str,
            "image_width_px": int,
            "image_height_px": int,
            "dilution_factor": float,
            "number_of_counting_fields": int,
            "manual_microns_per_pixel": float | None,
            "manual_chamber_depth_um": float | None,       # chamber mode only
            "manual_effective_depth_um": float | None,      # normal_slide mode only
            "depth_correction_factor": float | None,        # normal_slide mode only
        }

    Returns
    -------
    ConcentrationResult
        For "normal_slide", `concentration_label` is always either
        "Estimated Concentration" or "Validated Controlled-Protocol
        Concentration" (never "Measured") -- see
        `get_concentration_label`, which requires an actual validated
        effective depth (PHASE 2/3: manual override or correction
        factor) before it will ever say "Validated". Absent that, it
        is always "Estimated Concentration" -- software implementation
        alone never becomes a scientific validation claim.
    """
    slide_mode = concentration_params.get("slide_mode", "chamber")
    image_width_px = int(concentration_params.get("image_width_px", 0) or 0)
    image_height_px = int(concentration_params.get("image_height_px", 0) or 0)

    sperm_count = int(results.get("total_sperm", 0) or 0)
    motility_summary = {
        "progressive_pct": results.get("progressive", 0.0),
        "nonprogressive_pct": results.get("non_progressive", 0.0),
        "immotile_pct": results.get("immotile", 0.0),
    }

    if slide_mode == "normal_slide":
        # PHASE 2/3: the current single-video pipeline provides ONE
        # aggregate sperm count (ByteTrack-deduplicated across the
        # whole recording) -- treated as ONE observed field, exactly
        # as documented in compute_standard_slide_concentration's
        # "Counting-field model". True independent multi-field capture
        # is not yet implemented (see audit) -- this does not
        # fabricate additional fields that were never captured.
        microns_per_pixel = resolve_microns_per_pixel(
            concentration_params.get("magnification", ""),
            concentration_params.get("camera_resolution", ""),
            concentration_params.get("manual_microns_per_pixel"),
            concentration_params.get("saved_calibration_um_per_pixel"),
        )
        conc_result, field_stats, tech_params, quality = compute_standard_slide_concentration(
            field_sperm_counts=[sperm_count],
            motility_summary=motility_summary,
            microns_per_pixel=microns_per_pixel,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            protocol=STANDARD_SLIDE_PROTOCOL,
            manual_effective_depth_um=concentration_params.get("manual_effective_depth_um"),
            depth_correction_factor=concentration_params.get("depth_correction_factor"),
            dilution_factor=float(
                concentration_params.get("dilution_factor", DEFAULT_DILUTION_FACTOR)
                or DEFAULT_DILUTION_FACTOR
            ),
            n_rejected_fields=0,
            mean_detection_confidence=None,
            tracking_quality=None,
        )
        conc_result.slide_category = SlideCategory.CONTROLLED_ESTIMATION
        conc_result.depth_validation_state = tech_params.depth_validation_state
        # IMPORTANT (scientific honesty): the concentration LABEL must
        # NOT flip to "Validated Controlled-Protocol Concentration"
        # just because the operator typed a number into "Validated
        # effective depth" -- that is a self-reported override, not
        # proof of laboratory reference-method validation (PART
        # 22/38-40's ReferenceValidationRecord framework, which has no
        # UI/entry-point yet -- see audit). The label is driven by the
        # PROTOCOL's own validation_state (STANDARD_SLIDE_PROTOCOL.
        # validation_state = PROVISIONAL, and nothing in this codebase
        # currently changes that), so it stays "Estimated Concentration"
        # until an actual validation workflow explicitly promotes the
        # protocol -- never merely because a field was filled in.
        # `depth_validation_state` above still honestly reports whether
        # THIS RUN used a manually-entered depth, as a separate,
        # visible diagnostic -- it just doesn't drive the label.
        conc_result.concentration_label = get_concentration_label(
            SlideCategory.CONTROLLED_ESTIMATION, STANDARD_SLIDE_PROTOCOL.validation_state,
        )
        conc_result.field_statistics = field_stats.to_dict()
        conc_result.technical_params = tech_params.to_dict()
        conc_result.measurement_quality = quality.to_dict()
        conc_result.analysis_method = "Standard Glass Slide – CASA-Like Controlled Analysis"
        conc_result.protocol_version = STANDARD_SLIDE_PROTOCOL.protocol_version
        saved_um_per_px_for_status = concentration_params.get("saved_calibration_um_per_pixel")
        if saved_um_per_px_for_status is not None and saved_um_per_px_for_status > 0:
            conc_result.optical_calibration_status = ValidationState.VALIDATED
        elif microns_per_pixel:
            conc_result.optical_calibration_status = ValidationState.PROVISIONAL
        else:
            conc_result.optical_calibration_status = ValidationState.NOT_CONFIGURED
        return conc_result

    # ── Physical chamber pathway (Makler / Leja / Hemocytometer /
    #    Other) — UNCHANGED calculation, now also decorated with a
    #    label and a measurement-quality score for UI consistency
    #    with the Standard-Slide pathway (PHASE 4/5). ─────────────
    chamber_name = concentration_params.get("chamber_name", "Other")
    conc_result = compute_concentration_metrics(
        sperm_count=sperm_count,
        motility_summary=motility_summary,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        slide_mode=slide_mode,
        chamber_name=chamber_name,
        magnification=concentration_params.get("magnification", ""),
        camera_resolution=concentration_params.get("camera_resolution", ""),
        dilution_factor=float(
            concentration_params.get("dilution_factor", DEFAULT_DILUTION_FACTOR)
            or DEFAULT_DILUTION_FACTOR
        ),
        number_of_counting_fields=int(
            concentration_params.get("number_of_counting_fields", DEFAULT_NUMBER_OF_COUNTING_FIELDS)
            or DEFAULT_NUMBER_OF_COUNTING_FIELDS
        ),
        manual_microns_per_pixel=concentration_params.get("manual_microns_per_pixel"),
        manual_chamber_depth_um=concentration_params.get("manual_chamber_depth_um"),
        saved_calibration_um_per_pixel=concentration_params.get("saved_calibration_um_per_pixel"),
    )

    chamber_spec = CHAMBER_CONFIG.get(chamber_name, CHAMBER_CONFIG["Other"])
    conc_result.slide_category = chamber_spec.category
    conc_result.depth_validation_state = chamber_spec.validation_state
    conc_result.concentration_label = get_concentration_label(
        chamber_spec.category, chamber_spec.validation_state,
    )

    saved_um_per_px = concentration_params.get("saved_calibration_um_per_pixel")
    manual_um_per_px = concentration_params.get("manual_microns_per_pixel")
    if saved_um_per_px is not None and saved_um_per_px > 0:
        # A real one-time stage-micrometer calibration for this exact
        # configuration -- genuinely validated (calibration workflow),
        # not merely provisional.
        calibration_state = ValidationState.VALIDATED
    elif resolve_microns_per_pixel(
        concentration_params.get("magnification", ""),
        concentration_params.get("camera_resolution", ""),
        manual_um_per_px,
    ):
        calibration_state = ValidationState.PROVISIONAL
    else:
        calibration_state = ValidationState.NOT_CONFIGURED
    n_valid = 1 if conc_result.status == "ok" else 0
    quality = calculate_measurement_quality_score(
        calibration_validation_state=calibration_state,
        n_valid_fields=n_valid,
        expected_fields=int(
            concentration_params.get("number_of_counting_fields", DEFAULT_NUMBER_OF_COUNTING_FIELDS)
            or DEFAULT_NUMBER_OF_COUNTING_FIELDS
        ),
        field_cv_percent=0.0,  # a single field has no CV -- honestly reported as unavailable
        mean_detection_confidence=None,
        tracking_quality=conc_result.counting_quality,
        protocol_validation_state=chamber_spec.validation_state,
    )
    conc_result.measurement_quality = quality.to_dict()
    conc_result.analysis_method = f"Physical Chamber – {chamber_name}"
    conc_result.protocol_version = None  # no versioned protocol concept for physical chambers
    conc_result.optical_calibration_status = calibration_state
    return conc_result