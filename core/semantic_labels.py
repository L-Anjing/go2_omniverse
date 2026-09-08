"""Semantic-name inference for external Isaac environment assets."""

from __future__ import annotations

import re


_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # S_WetFloorSign is a physical warning sign, not a floor surface.
    ("sign", ("sign", "placard", "poster")),
    # Within one prim name structural obstacles win. Path components are
    # evaluated leaf-first below, so a Rail parent cannot mask a Floor child.
    ("wall", ("wall", "ceiling", "column", "pillar", "fence", "barrier", "rail", "window", "partition")),
    ("floor", ("floor", "ground", "walkway", "corridor", "hallway", "pavement", "platform", "carpet", "rug")),
    ("stairs", ("stair", "step")),
    ("ramp", ("ramp", "slope", "incline")),
    ("door", ("door", "elevator", "escalator")),
    ("pallet", ("pallet",)),
    ("forklift", ("forklift", "wheelchair", "gurney", "cart", "trolley")),
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

# Geometry-audited Hospital second-floor slabs.  Their asset name says
# ``Roof`` because the same solid also forms the first-floor ceiling, but the
# top face is extracted into a sibling SupportSurface by the USD pipeline;
# the original mesh retains only structural faces. Keep this list exact:
# broad ``roof``/``platform`` matching would
# incorrectly relabel non-walkable structure in other scenes.
_HOSPITAL_UPPER_FLOOR_MESH_SUFFIXES = (
    "/Root/Geo_Roof_Costum1_34/Geo_Roof_Costum1/Geo_Roof_Costum1",
    "/Root/Geo_Roof_Costum6_19/Geo_Roof_Costum6/Geo_Roof_Costum6",
)


def hospital_upper_floor_semantic_class(prim_path: str) -> str | None:
    """Label only extracted support faces; solid slab remainders are walls."""
    normalized = prim_path.replace("\\", "/").rstrip("/")
    return (
        "wall"
        if any(normalized.endswith(suffix) for suffix in _HOSPITAL_UPPER_FLOOR_MESH_SUFFIXES)
        else ("floor" if any(normalized == suffix + "_SupportSurface" for suffix in _HOSPITAL_UPPER_FLOOR_MESH_SUFFIXES) else None)
    )


def canonical_semantic_class(class_name: str | None) -> str:
    """Map authored source labels to classes used by spatial extraction."""
    normalized = re.sub(r"[^a-z0-9]+", "_", (class_name or "").lower()).strip("_")
    if not normalized:
        return "misc"
    return _CANONICAL_ALIASES.get(normalized, normalized)


def is_traversal_surface_class(class_name: str | None) -> bool:
    """Return whether geometry represents an actual height-transition surface."""
    return canonical_semantic_class(class_name) in {"stairs", "ramp"}


def direct_navigation_semantic_class(prim_path: str) -> str | None:
    """Return the direct label required by a named navigation mesh."""
    audited_upper_floor = hospital_upper_floor_semantic_class(prim_path)
    if audited_upper_floor is not None:
        return audited_upper_floor
    # A number of Hospital rail instances reference a mesh whose internal
    # name ends in ``Floor``.  The enclosing instance name is authoritative:
    # never turn that obstacle back into walkable floor.
    if direct_railing_semantic_class(prim_path) is not None:
        return None
    # Navigation surfaces must identify themselves on the rendered leaf prim;
    # a generic child must not inherit stairs from a StairAssembly ancestor.
    leaf_name = prim_path.rstrip("/").rsplit("/", 1)[-1]
    inferred = _infer_component(leaf_name)
    return inferred if inferred in {"floor", "ramp", "stairs"} else None


def direct_railing_semantic_class(prim_path: str) -> str | None:
    """Label named rail/fence render geometry as a structural obstacle."""
    normalized = re.sub(r"[^a-z0-9]+", "_", prim_path.lower())
    # Match compound asset names such as StairsPlaneHalfRail2 as well as
    # underscore-delimited names.  Do not re-run generic leaf-first inference:
    # these rail instances contain a misleading child named ``...Floor``.
    is_named_railing = re.search(
        r"(?:railing|handrail|guardrail|fence|barrier|rail)(?:\d|_|$)",
        normalized,
    )
    return "wall" if is_named_railing else None


def direct_door_semantic_class(prim_path: str) -> str | None:
    """Treat door panels, frames, and jambs as structure, never as floor."""
    components = [
        re.sub(r"[^a-z0-9]+", "_", part.lower()).strip("_")
        for part in prim_path.replace("\\", "/").split("/")
        if part
    ]
    is_door_structure = any(
        "door" in part or "doorframe" in part or "jamb" in part or "elevator" in part
        for part in components
    )
    return "wall" if is_door_structure else None


def stair_section_semantic_class(prim_path: str) -> str | None:
    """Return audited semantics for anonymous Hospital/Office stair sections."""
    components = [
        part.lower()
        for part in prim_path.replace("\\", "/").split("/")
        if part
    ]
    if not components:
        return None
    if any("stairsplanehalfrail" in part for part in components):
        return "wall"
    if any(part.startswith("sm_stairscarpet") for part in components):
        return "stairs"
    leaf = components[-1]
    if "geo_m_stairs" in components:
        if leaf == "section1":
            return "wall"
        if leaf in {"section0", "section2"}:
            return "stairs"
    if leaf == "section0" and any(
        part.startswith("geo_m_stairsplane") for part in components
    ):
        return "wall"
    if leaf in {"section1", "section2"} and any(
        part.startswith("geo_m_stairsplane") for part in components
    ):
        return "stairs"
    office_stair = any(
        part.startswith(("sm_stairsmain", "sm_stairssmall"))
        and "_walls" not in part
        for part in components
    )
    if office_stair and leaf == "section0":
        return "stairs"
    if office_stair and leaf in {"section1", "section2"}:
        return "wall"
    return None


# Compatibility for callers created before the Office scene was audited.
hospital_stair_section_class = stair_section_semantic_class


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
