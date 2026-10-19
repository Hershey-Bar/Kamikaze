"""Bounded, RAM-only evidence for products with no enrolled prototype. No network or file writes."""

from __future__ import annotations

import collections
import re

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

import citrus_classifier as citrus
import device_cache
import entity_resolver
import spec_search
import taxonomy

_BRANDS = {
    "Kirkland": r"\bkirkland\b",
    "Coca-Cola": r"\b(?:coke|coca[\s-]*cola)\b",
    "Pepsi": r"\bpepsi\b",
    "Sprite": r"\bsprite\b",
    "Red Bull": r"\bred[\s-]*bull\b",
    "Monster": r"\bmonster\b",
    "La Croix": r"\bla[\s-]*croix\b",
}
_PHONE = re.compile(
    r"\b(?:iphone\s*\d{1,2}\s*(?:pro\s*max|pro|plus|mini)?|"
    r"(?:samsung\s+)?(?:galaxy\s+)?s\s*2[0-5]\s*(?:ultra|plus|\+)?|"
    r"(?:galaxy\s+)?z?\s*fold\s*[3-6])\b", re.I)
_VOLUME = re.compile(r"\d+(?:[.,]\d+)?\s*(?:ml|cl|fl\.?\s*oz|oz|l)\b", re.I)


def detected_category(target: dict | None) -> str | None:
    name = str((target or {}).get("label") or "").lower().replace("_", " ")
    if name in ("smartphone", "cell phone", "phone"):
        return "smartphone"
    if name in ("beverage can", "soda can", "can"):
        return "beverage_can"
    if name in ("bottle", "beverage bottle"):
        return "beverage_bottle"
    return None


class ScanEvidence:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tokens = collections.Counter()
        self.texts: collections.deque[str] = collections.deque(maxlen=32)
        self.colors: collections.deque[dict] = collections.deque(maxlen=30)
        self.crops: dict[str, tuple[float, np.ndarray]] = {}
        self.last_run = -1
        self.ocr_passes = 0
        self.message = ""
        self.identity: dict | None = None
        self.category: str | None = None

    def add(self, ocr: dict | None, color: dict | None, crop=None, view=None) -> None:
        ocr = ocr or {}
        run = ocr.get("last_read_at", ocr.get("runs", 0))
        if run != self.last_run:
            self.last_run = run
            texts = ocr.get("corrected") or ocr.get("texts") or []
            if texts or ocr.get("tokens"):
                self.ocr_passes += 1
                self.tokens.update(set(ocr.get("tokens") or ()))
                self.texts.extend(str(t)[:256] for t in texts[:8])
        if color:
            self.colors.append(color)
        if crop is not None and np.asarray(crop).size:
            rgb = np.asarray(crop)
            h, w = rgb.shape[:2]
            scale = min(1.0, 320 / max(h, w))
            if cv2 is not None:
                small = cv2.resize(rgb, (max(4, round(w * scale)), max(4, round(h * scale))))
                clarity = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
            else:
                from PIL import Image
                img = Image.fromarray(rgb)
                sw, sh = max(4, round(w * scale)), max(4, round(h * scale))
                small = np.asarray(img.resize((sw, sh), Image.Resampling.BILINEAR))
                gray = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
                lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
                clarity = float(lap.var()) if lap.size else 0.0
            key = view if view in ("front", "back", "side") else "front"
            if key not in self.crops or clarity > self.crops[key][0]:
                self.crops[key] = (clarity, small.copy())

    def color(self) -> dict | None:
        if not self.colors:
            return None
        scores = {f: sum(c["scores"].get(f, 0) for c in self.colors) / len(self.colors) for f in citrus.FLAVORS}
        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        verdict = best if scores[best] >= citrus.VERDICT_MIN and (
            scores[best] - scores[ranked[1]] >= citrus.VERDICT_MARGIN) else None
        out = {"scores": scores, "verdict": verdict, "strong": bool(verdict and scores[best] >= citrus.STRONG)}
        samples = [c["kirkland"] for c in list(self.colors)[-12:] if "kirkland" in c]
        if samples:
            sc = {f: sum(c["scores"].get(f, 0) for c in samples) / len(samples) for f in citrus.FLAVORS}
            order = sorted(sc, key=sc.get, reverse=True)
            winner = order[0] if sc[order[0]] >= citrus.KIRKLAND_MIN and (
                sc[order[0]] - sc[order[1]] >= citrus.KIRKLAND_MARGIN) else None
            out["kirkland"] = {"scores": sc, "verdict": winner,
                               "strong": bool(winner and sc[winner] >= citrus.STRONG)}
        return out

    def resolve(self, category: str | None, phone: dict | None = None,
                known_kirkland_line: bool = False) -> dict | None:
        self.identity = None
        self.category = category
        self.message = "Rotate slowly to show the brand, flavour and volume."
        if category == "smartphone":
            return self._phone(phone)
        if category not in ("beverage_can", "beverage_bottle"):
            self.message = "Present a phone, beverage can or bottle inside the reticle."
            return None
        text = " ".join(self.texts).lower()
        known_kirkland_line = known_kirkland_line and not citrus.KIRKLAND_OTHER_TYPES.intersection(
            re.findall(r"[a-z]+", text))
        brands = [brand for brand, pattern in _BRANDS.items() if re.search(pattern, text)]
        color = self.color()
        fixed = citrus.kirkland_flavor(color)
        visual_brand = known_kirkland_line and category == "beverage_can" and fixed and not brands
        if visual_brand:
            brands = ["Kirkland"]
        if len(brands) != 1 or (self.ocr_passes < 2 and not visual_brand):
            self.message = "Show the brand clearly; colour alone cannot identify a product."
            return None
        brand = brands[0]
        fused = citrus.fuse(color, citrus.ocr_scores(recent=self.tokens))
        flavor = fused["flavor"] or taxonomy.find_flavor(text)
        kirkland_line = brand == "Kirkland" and category == "beverage_can" and (
            known_kirkland_line or citrus.is_kirkland_citrus_can(meta={"raw_name": text, "brand": brand,
                                                                     "category": category}))
        if kirkland_line:
            # Identical bodies and stale/misread flavour text must not substitute for visible artwork.
            if not fixed:
                self.message = "Rotate to show the coloured flavour artwork clearly."
                return None
            flavor = fixed
        # A green Sprite or a blue/silver energy drink is not a citrus-flavour declaration.
        if brand not in ("Kirkland", "La Croix"):
            flavor = taxonomy.find_flavor(text)
        if brand in ("Kirkland", "La Croix") and not flavor:
            self.message = "Brand read. Rotate to expose the flavour colour or flavour name."
            return None
        volumes = {spec_search.volume_ml(re.sub(r"\bfl\.", "fl", m.group(), flags=re.I))
                   for m in _VOLUME.finditer(text)}
        volumes.discard(None)
        if len(volumes) > 1:
            self.message = "Conflicting package volumes read; hold the volume marking steady."
            return None
        volume = f"{next(iter(volumes))}ml" if volumes else ""
        tier = "diet" if re.search(r"\bdiet\b", text) else "zero" if re.search(
            r"\b(?:zero|sugarfree|sugar\s*free)\b", text) else ""
        kind = "sparkling water" if kirkland_line or brand == "La Croix" or "sparkling water" in text else ""
        package = "bottle" if category == "beverage_bottle" else "can"
        product = entity_resolver.resolve(" ".join(filter(None, (brand, tier, flavor, kind, volume, package))), category)
        return self._result(product, 0.88, "vision_color" if visual_brand else "ocr_color" if color else "ocr", volume)

    def _phone(self, phone: dict | None) -> dict | None:
        self.message = "Show the camera island, then the screen cutout or model name."
        ids = {device_cache.device_id_for(m.group()) for text in self.texts for m in _PHONE.finditer(text)}
        ids.discard(None)
        if len(ids) == 1 and self.ocr_passes >= 2:
            dev = next(iter(ids))
            product = entity_resolver.resolve(dev.replace("_", " "), "smartphone")
            return self._result(product, 0.92, "model_text")
        if len(ids) > 1:
            self.message = "Conflicting phone model names read; show only one device."
            return None
        phone = phone or {}
        runners = phone.get("runner_up") or []
        gap = phone.get("score", 0) - (runners[0]["score"] if runners else 1)
        # Near-identical generations and screen-only matches must remain ambiguous.
        if (phone.get("frames", 0) >= 3 and phone.get("dino") and phone.get("rear_frames", 0) >= 3
                and phone.get("raw_similarity", 0) >= 0.70 and phone.get("raw_margin", 0) >= 0.04
                and phone.get("score", 0) >= 0.80 and gap >= 0.08 and phone.get("cues", {}).get("layout")):
            product = entity_resolver.resolve(phone["id"].replace("_", " "), "smartphone")
            return self._result(product, min(0.9, phone["score"]), "camera_geometry_dinov2")
        if phone.get("id"):
            self.message = "Phone family found, exact model uncertain. Show the model name or a clearer rear view."
        return None

    def _result(self, product, confidence, source, volume="") -> dict:
        self.message = f"Reading {product.canonical_name}; keep rotating slowly."
        meta = {**product.metadata(), "raw_name": product.raw_query, "display_name": product.canonical_name,
                "autonomous": True, "identity_source": source}
        self.identity = {"label": product.class_id, "confidence": confidence, "category": product.category,
                         "metadata": meta, "source": source, "volume": volume}
        return self.identity

    def snapshot(self) -> dict:
        return {"ocr_passes": self.ocr_passes, "tokens": self.tokens.most_common(16),
                "color": self.color(), "volume": (self.identity or {}).get("volume", ""),
                "source": (self.identity or {}).get("source"),
                "crops_in_ram": len(self.crops), "crop_bytes": sum(a.nbytes for _, a in self.crops.values()),
                "message": self.message}
