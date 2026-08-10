"""Semantic-name inference for external Isaac environment assets."""

from __future__ import annotations

import re


_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Structural obstacles must win over a parent path such as
    # /StairsPlaneHalfRail/...Floor; otherwise rail meshes become walkable.
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


def infer_semantic_class(prim_path: str, profile_hint: str = "") -> str:
    """Infer the stable source class from a USD prim path."""
    normalized = re.sub(r"[^a-z0-9]+", "_", prim_path.lower())
    tokens = set(normalized.strip("_").split("_"))
    for class_name, keywords in _RULES:
        if any(keyword in tokens or keyword in normalized for keyword in keywords):
            return class_name

    profile = profile_hint.lower()
    if "stair" in profile:
        return "stairs"
    if "slope" in profile or "ramp" in profile:
        return "ramp"
    return "misc"
