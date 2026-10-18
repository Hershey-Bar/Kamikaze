"""Entity resolver: free-text product descriptions -> one canonical, stable product record.

    resolve("coke zero can 330ml")            -> class_id "coca_cola_zero_330ml_can"
    resolve("iphone 14 pro max deep purple")  -> class_id "iphone_14_pro_max", color "deep purple"
    resolve("s24 ultra")                      -> class_id "galaxy_s24_ultra"

The class_id is the key used everywhere: the prototype bank, enrolled_classes.pt, static/models/<class_id>.glb and
the dashboard. Pure Python (no torch / network). Built on taxonomy.parse_metadata so both stay consistent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import taxonomy

_COLORS = (
    "deep purple", "space black", "natural titanium", "black titanium", "blue titanium", "white titanium",
    "desert titanium", "titanium gray", "titanium grey", "titanium black", "titanium violet", "titanium yellow",
    "titanium blue", "titanium green", "titanium silver", "phantom black", "phantom white", "phantom silver",
    "phantom violet", "sierra blue", "alpine green", "pacific blue", "product red", "midnight green",
    "cosmic black", "cosmic gray", "cream", "lavender", "midnight", "starlight", "graphite", "violet", "burgundy",
    "ultramarine", "teal", "black", "white", "blue", "purple", "gold", "silver", "green", "pink", "yellow",
    "red", "orange", "gray", "grey",
)
_COLOR_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(_COLORS, key=len, reverse=True)) + r")\b")

_CAN_VARIANT_NAMES = {"original": "Original Taste", "zero": "Zero Sugar", "diet": "Diet"}
_CAN_SUB_TIER = {"original": "regular", "zero": "zero", "diet": "diet"}


@dataclass(frozen=True)
class CanonicalProduct:
    raw_query: str
    class_id: str
    canonical_name: str
    category: str
    brand: str
    model_or_variant: str
    sub_tier: str
    cad_search_terms: tuple[str, ...]
    package_type: str = ""
    container_volume: str = ""
    color: str = ""

    def metadata(self) -> dict:
        """taxonomy-compatible enrollment metadata (the structure stored with each prototype)."""
        meta = {
            "raw_name": self.raw_query, "category": self.category, "brand": self.brand,
            "variant": self.sub_tier if self.category == "smartphone" else {"regular": "original"}.get(self.sub_tier, self.sub_tier),
            "package_type": self.package_type, "container_volume": self.container_volume,
        }
        if self.color:
            meta["color"] = self.color
        return meta

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cad_search_terms"] = list(self.cad_search_terms)
        return data


def _phone_variant_name(meta: dict) -> str:
    model, _ = taxonomy._phone_model(meta.get("brand", ""), f"{meta.get('raw_name', '')} {meta.get('variant', '')}")
    return model


def resolve(text: str, category: str = "", brand: str = "", variant: str = "", package_type: str = "",
            container_volume: str = "") -> CanonicalProduct:
    """Normalise a free-text description (explicit fields override inference). Raises taxonomy.MetadataError."""
    request = {"raw_name": text, "category": category, "brand": brand, "variant": variant,
               "package_type": package_type, "container_volume": container_volume}
    meta = taxonomy.parse_metadata(request)
    raw = re.sub(r"\s+", " ", str(text or "")).strip() or meta["raw_name"]
    class_id = taxonomy.label_from_metadata(meta)
    name = taxonomy.display_name(meta)

    color = ""
    if meta["category"] == "smartphone":
        found = _COLOR_RE.search(raw.lower())
        color = found.group(1) if found else ""
        model_or_variant = _phone_variant_name(meta) or meta["raw_name"]
        sub_tier = meta["variant"] or "standard"
    else:
        model_or_variant = _CAN_VARIANT_NAMES.get(meta["variant"], meta["variant"].title())
        sub_tier = _CAN_SUB_TIER.get(meta["variant"], meta["variant"])

    terms = list(taxonomy.search_queries(meta))
    if meta["category"] == "beverage_can" and meta["brand"] == "Coca-Cola" and meta["variant"] in ("zero", "diet"):
        terms.insert(0, "Coke Zero can" if meta["variant"] == "zero" else "Diet Coke can")
    if meta["category"] != "smartphone" and meta["container_volume"]:
        terms.insert(0, f"{name} 3D model")
    seen, ordered = set(), []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(term)

    return CanonicalProduct(
        raw_query=raw, class_id=class_id, canonical_name=name, category=meta["category"], brand=meta["brand"],
        model_or_variant=model_or_variant, sub_tier=sub_tier, cad_search_terms=tuple(ordered),
        package_type=meta["package_type"], container_volume=meta["container_volume"], color=color,
    )


def resolve_metadata(meta: dict) -> CanonicalProduct:
    """Canonical record for an already-structured metadata dict (e.g. loaded from enrolled_classes.pt)."""
    product = resolve(meta.get("raw_name", ""), meta.get("category", ""), meta.get("brand", ""), meta.get("variant", ""),
                      meta.get("package_type", ""), meta.get("container_volume", ""))
    if meta.get("color") and not product.color:
        object.__setattr__(product, "color", str(meta["color"]))
    return product
