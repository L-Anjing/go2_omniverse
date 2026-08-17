"""Semantic-name inference for external Isaac environment assets."""

from __future__ import annotations

import re


_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Within one prim name structural obstacles win. Path components are
    # evaluated leaf-first below, so a Rail parent cannot mask a Floor child.
    ("wall", ("wall", "ceiling", "column", "pillar", "fence", "barrier", "rail", "window", "partition")),
    ("floor", ("floor", "ground", "walkway", "corridor", "hallway", "pavement", "platform", "carpet", "rug")),
    ("stairs", ("stair", "step")),
    ("ramp", ("ramp", "slope", "incline")),
    ("door", ("door", "elevator", "escalator")),
    ("pallet", ("pallet",)),
    ("forklift", ("forklift", "wheelchair", "gurney", "cart", "trolley")),
    ("sign", ("sign", "placard", "poster")),
    ("traffic_cone", ("cone", "bollard", "iv_stand")),
    ("furniture", ("table", "chair", "desk", "bed", "cabinet", "shelf", "rack", "sofa", "couch", "bench", "sink", "toilet")),
)

_CANONICAL_ALIASES: dict[str, str] = {
    "ground": "floor",
    "walkway": "floor",
    "corridor": "floor",
    "hallway": "floor",
    "platform": "floor",
    "carpet": "floor",
    "railing": "wall",
    "handrail": "wall",
    "guardrail": "wall",
    "fence": "wall",
    "barrier": "wall",
    "column": "wall",
    "pillar": "wall",
    "ceiling": "wall",
    "stair": "stairs",
    "staircase": "stairs",
    "stairwell": "stairs",
    "step": "stairs",
    "steps": "stairs",
    "slope": "ramp",
    "incline": "ramp",
}


def canonical_semantic_class(class_name: str | None) -> str:
    """Map authored source labels to classes used by spatial extraction."""
    normalized = re.sub(r"[^a-z0-9]+", "_", (class_name or "").lower()).strip("_")
    if not normalized:
        return "misc"
    return _CANONICAL_ALIASES.get(normalized, normalized)


def is_traversal_surface_class(class_name: str | None) -> bool:
    """Return whether geometry represents an actual height-transition surface."""
    return canonical_semantic_class(class_name) in {"stairs", "ramp"}


def _infer_component(component: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
    tokens = set(normalized.split("_"))
    for class_name, keywords in _RULES:
        if any(keyword in tokens or keyword in normalized for keyword in keywords):
            return class_name
    return None


def infer_semantic_class(prim_path: str, profile_hint: str = "") -> str:
    """Infer a class leaf-first so broad parent names cannot mask a child mesh."""
    components = [part for part in prim_path.replace("\\", "/").split("/") if part]
    for component in reversed(components):
        inferred = _infer_component(component)
        if inferred is not None:
            return inferred

    # Never infer geometry from the profile name: `hospital_stairs` also
    # contains walls, furniture, and railings.
    return "misc"
