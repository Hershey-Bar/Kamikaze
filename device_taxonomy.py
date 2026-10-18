"""Structural smartphone taxonomy, asymmetric phone view weighting and detector domain gating for Kamikaze.

Pure Python (no torch / network). receiver.py uses it to
  * score smartphone classes with rear-camera-weighted view centroids (the front screen carries little identity),
  * cap confidence and prompt a flip when the live frame matches a phone's screen better than its back,
  * pick tier-specific orientation prompts when two enrolled phones are too close to call,
  * restrict the prototype search to the domain the NPU object detector reports (detector = domain gate only).
"""

from __future__ import annotations

import re

PHONE_STRUCTURAL_TAXONOMY = {
    # APPLE IPHONE ECOSYSTEM
    "iphone_standard": {
        "models": ["iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15", "iPhone 16"],
        "lens_count": 2,
        "lidar": False,
        "frame_material": "Anodized aluminum (matte)",
        "back_finish": "Glossy glass (12-14) or color-infused matte (15-16)",
        "lens_layout": {
            "iPhone 12": "Vertical stacked pill",
            "iPhone 13/14/15": "Diagonal arrangement",
            "iPhone 16": "Vertical pill module with unified island"
        },
        "guidance_rule": "If 3 lenses or black circular LiDAR aperture detected, reject standard tier."
    },
    "iphone_pro": {
        "models": ["iPhone 12 Pro", "iPhone 13 Pro", "iPhone 14 Pro", "iPhone 15 Pro", "iPhone 16 Pro", "Max models"],
        "lens_count": 3,
        "lidar": True,  # Dark circular sensor cutout at 4 o'clock
        "frame_material": "Polished stainless steel (12-14 Pro) or brushed titanium (15-16 Pro)",
        "back_finish": "Textured matte back glass",
        "lens_layout": "Triangular 3-lens cluster (Wide, Ultra-Wide, Telephoto)",
        "prompt": "Tilt phone 30° toward light to expose the dark circular LiDAR sensor below the bottom-right lens."
    },

    # SAMSUNG GALAXY S-SERIES ECOSYSTEM
    "samsung_standard_plus": {
        "models": ["Galaxy S20", "Galaxy S21", "Galaxy S22", "Galaxy S23", "Galaxy S24", "Galaxy S25", "Plus models"],
        "lens_count": 3,
        "corners": "Rounded body corners",
        "lens_layout": "3 vertical circular islands aligned in a single column",
        "periscope": False,
        "laser_af": False,
        "guidance_rule": "If corner silhouette is sharp 90-degree boxy profile, reject standard tier."
    },
    "samsung_ultra": {
        "models": ["Galaxy S20 Ultra", "Galaxy S21 Ultra", "Galaxy S22 Ultra", "Galaxy S23 Ultra", "Galaxy S24 Ultra", "Galaxy S25 Ultra"],
        "lens_count": 4,  # Wide, Ultra-Wide, 3x Telephoto, 5x/10x Periscope
        "corners": "Distinct sharp, 90-degree boxy rectangular corners",
        "periscope": True,  # Rectangular dark periscope cutout
        "laser_af": True,   # Extra sensor ring above LED flash
        "stylus": "Integrated S-Pen silo opening on bottom rail",
        "prompt": "Align silhouette to highlight sharp 90° boxy corners and rectangular periscope camera cut-out."
    }
}

# S_phone = 0.65 S_back + 0.15 S_side + 0.10 S_top + 0.10 S_front: the rear shell / camera island dominates identity.
PHONE_VIEW_WEIGHTS = {"back": 0.65, "side": 0.15, "top": 0.10, "front": 0.10}

# A weighted *average* of view similarities can never be high for a single physical pose (the camera sees one side at
# a time), so the weights are applied as per-view reliability: the back view is trusted fully and every other view is
# discounted in proportion to its weight relative to the back. kappa = 0.25 puts side at 0.81 and top / front at 0.79,
# which keeps them below the IDENTIFIED floor (calibrated 0.70 needs raw 0.856) - only the rear shell can verify a phone.
VIEW_RELIABILITY_KAPPA = 0.25

FRONT_CONFIDENCE_CAP = 0.35
MSG_FRONT_SCREEN = "Front screen detected. Flip phone 180° to expose camera island and sensors."
MSG_VERIFY_GEOMETRY = "Verifying object geometry... Hold steady in center."
MSG_SHOW_BACK = "Show the rear shell and camera island squarely to the camera."
MSG_IPHONE_LAYOUT = "Hold the camera island flat to the camera: lens layout (stacked pill vs diagonal) separates these iPhones."
MSG_CROSS_BRAND = "Show the rear camera island: lens count and layout separate Apple from Samsung."

DETECTOR_GATE_MIN = 0.50
PHONE_DOMAIN = ("smartphone",)
BEVERAGE_DOMAIN = ("beverage_can", "beverage_bottle")
DETECTOR_DOMAINS = {
    "cell phone": PHONE_DOMAIN, "phone": PHONE_DOMAIN, "smartphone": PHONE_DOMAIN,
    "cup": BEVERAGE_DOMAIN, "bottle": BEVERAGE_DOMAIN, "can": BEVERAGE_DOMAIN, "soda can": BEVERAGE_DOMAIN,
    "beverage can": BEVERAGE_DOMAIN,
}


def view_reliability(view: str | None) -> float:
    """Multiplier applied to a smartphone's view similarity (1.0 for the back, < 1 for less informative views)."""
    if view is None:
        return 1.0
    weight = PHONE_VIEW_WEIGHTS.get(view)
    if weight is None:
        return 1.0
    return 1.0 - VIEW_RELIABILITY_KAPPA * (1.0 - weight / PHONE_VIEW_WEIGHTS["back"])


def weighted_phone_score(view_sims: dict[str, float]) -> tuple[float, str | None]:
    """(score, view) - best reliability-weighted view similarity of a smartphone class."""
    best, best_view = -2.0, None
    for view, sim in view_sims.items():
        score = sim * view_reliability(view) if sim > 0 else sim
        if score > best:
            best, best_view = score, view
    return best, best_view


def weighted_average(view_sims: dict[str, float]) -> float:
    """Literal 0.65/0.15/0.10/0.10 blend over the views that exist (reported as a diagnostic)."""
    total = sum(PHONE_VIEW_WEIGHTS.get(v, 0.0) for v in view_sims)
    if total <= 0:
        return 0.0
    return sum(PHONE_VIEW_WEIGHTS.get(v, 0.0) * s for v, s in view_sims.items()) / total


def front_dominates(view_sims: dict[str, float]) -> bool:
    """True when the live frame matches the screen centroid better than the rear centroid."""
    front, back = view_sims.get("front"), view_sims.get("back")
    return front is not None and back is not None and front > back


def tier_for(meta: dict | None) -> str | None:
    """Map enrollment metadata to a PHONE_STRUCTURAL_TAXONOMY key (None for non-phones / unknown makers)."""
    if not meta or meta.get("category") != "smartphone":
        return None
    brand = (meta.get("brand") or "").lower()
    text = f"{meta.get('raw_name', '')} {meta.get('variant', '')}".lower()
    if brand == "apple" or "iphone" in text:
        return "iphone_pro" if re.search(r"\bpro\b|pro_max|\bmax\b", text) else "iphone_standard"
    if brand == "samsung" or "galaxy" in text or re.search(r"\bs\s?2\d\b", text):
        return "samsung_ultra" if "ultra" in text else "samsung_standard_plus"
    return None


def disambiguation_prompt(meta_a: dict | None, meta_b: dict | None) -> str | None:
    """Physical orientation prompt that best separates two close smartphone matches (None = no tier-specific cue)."""
    ta, tb = tier_for(meta_a), tier_for(meta_b)
    tiers = {t for t in (ta, tb) if t}
    if not tiers:
        return None
    if "iphone_pro" in tiers and tiers <= {"iphone_standard", "iphone_pro"}:
        return PHONE_STRUCTURAL_TAXONOMY["iphone_pro"]["prompt"]
    if "samsung_ultra" in tiers and tiers <= {"samsung_ultra", "samsung_standard_plus"}:
        return PHONE_STRUCTURAL_TAXONOMY["samsung_ultra"]["prompt"]
    if ta == tb == "iphone_standard":
        return MSG_IPHONE_LAYOUT
    if ta and tb and ta.split("_")[0] != tb.split("_")[0]:
        return MSG_CROSS_BRAND
    return None


def domain_for(detection: dict | None, min_score: float = DETECTOR_GATE_MIN) -> tuple[str, ...] | None:
    """Category domain implied by a detector hit ({label, score}); None when the detector should not gate."""
    if not detection:
        return None
    try:
        score = float(detection.get("score") or 0.0)
    except (TypeError, ValueError):
        return None
    if score < min_score:
        return None
    return DETECTOR_DOMAINS.get(str(detection.get("label") or "").lower())


def describe(tier: str | None) -> dict | None:
    return dict(PHONE_STRUCTURAL_TAXONOMY[tier]) if tier in PHONE_STRUCTURAL_TAXONOMY else None
