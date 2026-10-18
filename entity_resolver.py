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


QUERY_LEVELS = ("exact", "similar", "category")
_SEARCH_BRAND = {"Coca-Cola": "Coca Cola"}
_SEARCH_NICK = {"Coca-Cola": "Coke"}
_ZERO_PHRASES = {"Pepsi": ("Pepsi Zero Sugar", "Pepsi Zero", "Pepsi Max"), "Red Bull": ("Red Bull Sugarfree", "Red Bull Zero")}
_APPLE_SIBLING = {"pro_max": "Pro", "pro": "Pro Max", "standard": "Plus", "plus": "", "mini": ""}
_SAMSUNG_SIBLING = {"plus": "", "standard": "Plus", "fe": ""}


def _iphone_label_exists(gen: int, label: str) -> bool:
    low = label.lower()
    return (low != "plus" or gen >= 14) and (low != "mini" or gen in (12, 13)) and gen >= 11


def _dedupe_plan(plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for query, level in plan:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query.lower() not in seen:
            seen.add(query.lower())
            out.append((query, level))
    return out


def _phone_plan(brand: str, model: str, tier: str) -> list[tuple[str, str]]:
    m = re.match(r"(iPhone|Galaxy S)\s?(\d+)\s*(.*)$", model or "")
    if not m:
        return [(f"{brand} {model}".strip(), "exact"), ("Smartphone", "category")]
    family, gen, label = m.group(1), int(m.group(2)), m.group(3).strip()
    if family == "iPhone":
        plan = [(model, "exact"), (f"Apple {model}", "exact")]
        sibling = _APPLE_SIBLING.get(tier)
        if tier == "standard" and gen in (12, 13):
            sibling = "mini"
        if sibling is not None and (sibling or label) and _iphone_label_exists(gen, sibling):
            plan.append((f"iPhone {gen} {sibling}", "similar"))
        if _iphone_label_exists(gen - 1, label):
            plan.append((f"iPhone {gen - 1} {label}", "similar"))
        elif gen > 11:
            plan.append((f"iPhone {gen - 1}", "similar"))
    else:
        plan = [(f"Samsung {model}", "exact"), (f"Samsung S{gen} {label}", "exact"), (model, "exact")]
        if gen > 20:
            plan.append((f"Galaxy S{gen - 1} {label}", "similar"))
        sibling = _SAMSUNG_SIBLING.get(tier)
        if sibling is not None:
            plan.append((f"Galaxy S{gen} {sibling}", "similar"))
    plan.append(("Smartphone", "category"))
    return plan


def _beverage_plan(category: str, brand: str, variant: str, volume: str) -> list[tuple[str, str]]:
    pkg = "Bottle" if category == "beverage_bottle" else "Can"
    name = _SEARCH_BRAND.get(brand, brand)
    nick = _SEARCH_NICK.get(brand, name)
    if variant == "zero":
        phrases = list(_ZERO_PHRASES.get(brand, (f"{name} Zero", f"{nick} Zero")))
    elif variant == "diet":
        phrases = [f"Diet {nick}", f"{name} Diet"]
    else:
        phrases = [name, nick]
    plan = [(f"{phrases[0]} {pkg}", "exact")]
    if volume:
        plan.append((f"{phrases[1]} {volume} {pkg}", "exact"))
    plan += [(f"{p} {pkg}", "exact") for p in phrases[1:]]
    if variant not in ("", "original"):
        plan.append((f"{name} {pkg}", "similar"))
    if category == "beverage_bottle":
        plan += [("Soda Bottle", "category"), ("Plastic Bottle", "category")]
    else:
        plan.append(("Energy Drink Can" if brand == "Red Bull" else "Soda Can", "category"))
    return plan


def query_plan(category: str, brand: str, model_or_variant: str, sub_tier: str, container_volume: str = "") -> list[tuple[str, str]]:
    """Ordered [(query, level)] for CAD search: exact product -> similar product -> generic category (most specific first).

    "coke zero can 330ml" -> Coca Cola Zero Can, Coke Zero 330ml Can, Coke Zero Can | Coca Cola Can | Soda Can
    "s25 ultra"           -> Samsung Galaxy S25 Ultra, Samsung S25 Ultra, Galaxy S25 Ultra | Galaxy S24 Ultra | Smartphone
    "iphone 15 pro max"   -> iPhone 15 Pro Max, Apple iPhone 15 Pro Max | iPhone 15 Pro, iPhone 14 Pro Max | Smartphone
    """
    if category == "smartphone":
        plan = _phone_plan(brand, model_or_variant, sub_tier)
    else:
        variant = {"regular": "original"}.get(sub_tier, sub_tier)
        plan = _beverage_plan(category, brand, variant, container_volume)
    return _dedupe_plan(plan)


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

    plan = query_plan(meta["category"], meta["brand"], model_or_variant, sub_tier, meta["container_volume"])
    ordered = [query for query, _ in plan]
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
