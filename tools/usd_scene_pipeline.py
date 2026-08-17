#!/usr/bin/env python3
"""Audit Isaac USD scenes and create non-destructive open-door overlays.

The script runs with Isaac Lab's Python interpreter.  It automatically exposes
the bundled USD runtime, so it does not need to launch SimulationApp just to
inspect or edit a USD layer.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any


def _bootstrap_usd_runtime() -> None:
    """Re-exec once with Isaac Sim's standalone pxr runtime on the path."""
    if os.environ.get("GO2_USD_RUNTIME_READY") == "1":
        return
    candidates = glob.glob(
        str(Path(sys.prefix) / "lib/python*/site-packages/isaacsim/extscache/omni.usd.libs-*")
    )
    if not candidates:
        raise RuntimeError(
            "Isaac USD runtime was not found. Run with the Isaac Lab Python interpreter."
        )
    usd_lib = Path(sorted(candidates)[-1])
    env = os.environ.copy()
    env["GO2_USD_RUNTIME_READY"] = "1"
    env["PYTHONPATH"] = f"{usd_lib}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["LD_LIBRARY_PATH"] = f"{usd_lib / 'bin'}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


_bootstrap_usd_runtime()

from pxr import Gf, Usd, UsdGeom  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.semantic_labels import (  # noqa: E402
    canonical_semantic_class,
    infer_semantic_class,
)


PANEL_PREFIXES = ("sm_door_", "glassdoor")


def _vec3(value: Gf.Vec3d) -> list[float]:
    return [round(float(value[index]), 4) for index in range(3)]


def _bbox_data(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict[str, list[float]] | None:
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    minimum = box.GetMin()
    maximum = box.GetMax()
    return {
        "min": _vec3(minimum),
        "max": _vec3(maximum),
        "size": [round(float(maximum[index] - minimum[index]), 4) for index in range(3)],
    }


def _is_door_panel(prim: Usd.Prim) -> bool:
    name = prim.GetName().lower()
    if name.startswith("glassdoor"):
        return True
    if not name.startswith("sm_door") or "handle" in name or "glass_main" in name:
        return False
    return name.startswith("sm_door_") or name.endswith("_door")


def _is_measurement_geometry(prim: Usd.Prim) -> bool:
    """Skip skydomes and similar presentation-only geometry."""
    name = prim.GetName().lower()
    return not any(token in name for token in ("skysphere", "skydome", "sky_sphere"))


def _has_selected_ancestor(prim: Usd.Prim, selected_paths: set[str]) -> bool:
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if str(parent.GetPath()) in selected_paths:
            return True
        parent = parent.GetParent()
    return False


def _combined_bounds(bounds: list[dict[str, list[float]]]) -> dict[str, list[float]] | None:
    if not bounds:
        return None
    minimum = [min(item["min"][index] for item in bounds) for index in range(3)]
    maximum = [max(item["max"][index] for item in bounds) for index in range(3)]
    return {
        "min": [round(value, 4) for value in minimum],
        "max": [round(value, 4) for value in maximum],
        "size": [round(maximum[index] - minimum[index], 4) for index in range(3)],
    }


def audit_scene(scene_path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD scene: {scene_path}")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        roots = stage.GetPseudoRoot().GetChildren()
        if not roots:
            raise RuntimeError(f"USD scene has no root prim: {scene_path}")
        root = roots[0]

    raw_stage_bbox = _bbox_data(cache, root)
    if raw_stage_bbox is None:
        raise RuntimeError(f"USD scene has an empty world bound: {scene_path}")
    top_level_prims = []
    for child in root.GetChildren():
        if not _is_measurement_geometry(child):
            continue
        child_bbox = _bbox_data(cache, child)
        if child_bbox is not None:
            top_level_prims.append({"path": str(child.GetPath()), "bbox": child_bbox})
    top_level_bounds = [item["bbox"] for item in top_level_prims]
    effective_bbox = _combined_bounds(top_level_bounds) or raw_stage_bbox

    mesh_count = 0
    gprim_count = 0
    floor_bands: set[float] = set()
    door_panels: list[dict[str, Any]] = []
    named_doors = 0
    selected_paths: set[str] = set()
    scene_min: list[float] | None = None
    scene_max: list[float] | None = None
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if "door" in name:
            named_doors += 1
        if prim.IsA(UsdGeom.Gprim) and _is_measurement_geometry(prim):
            gprim_count += 1
            prim_bbox = _bbox_data(cache, prim)
            if prim_bbox:
                if scene_min is None:
                    scene_min = list(prim_bbox["min"])
                    scene_max = list(prim_bbox["max"])
                else:
                    scene_min = [min(scene_min[index], prim_bbox["min"][index]) for index in range(3)]
                    scene_max = [max(scene_max[index], prim_bbox["max"][index]) for index in range(3)]
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            prim_bbox = _bbox_data(cache, prim)
            if prim_bbox and prim_bbox["size"][0] > 0.4 and prim_bbox["size"][1] > 0.4:
                floor_bands.add(round(prim_bbox["min"][2] * 2.0) / 2.0)
        if _is_door_panel(prim) and not _has_selected_ancestor(prim, selected_paths):
            panel_bbox = _bbox_data(cache, prim)
            if panel_bbox:
                door_panels.append(
                    {"path": str(prim.GetPath()), "name": prim.GetName(), "bbox": panel_bbox}
                )
                selected_paths.add(str(prim.GetPath()))

    if scene_min is None or scene_max is None:
        raise RuntimeError(f"USD scene has no measurable geometry: {scene_path}")

    size = effective_bbox["size"]
    xy_area = round(size[0] * size[1], 2)
    vertical_span = size[2]
    return {
        "scene": str(scene_path.resolve()),
        "default_prim": str(root.GetPath()),
        "world_bbox": effective_bbox,
        "raw_stage_bbox": raw_stage_bbox,
        "top_level_prims": top_level_prims,
        "xy_area_m2": xy_area,
        "vertical_span_m": vertical_span,
        "gprim_count": gprim_count,
        "mesh_count": mesh_count,
        "floor_bands_m": sorted(floor_bands),
        "named_door_prims": named_doors,
        "openable_door_panels": door_panels,
        "classification": {
            "large_scene": xy_area >= 100.0,
            "vertical_scene": vertical_span >= 3.5,
            "candidate_for_indoor_tomogram": xy_area >= 100.0 and gprim_count >= 10,
        },
    }


def _direct_semantic_label(prim: Usd.Prim) -> str | None:
    for attribute in prim.GetAttributes():
        if "semanticData" in attribute.GetName():
            value = attribute.Get()
            if value:
                return str(value)
    return None


def _nearest_semantic_label(prim: Usd.Prim) -> str | None:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        label = _direct_semantic_label(current)
        if label:
            return label
        current = current.GetParent()
    return None


def audit_semantics(scene_path: Path, max_examples: int = 25) -> dict[str, Any]:
    """Compare authored labels with leaf-first navigation classifications."""
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD scene: {scene_path}")
    counts: dict[str, int] = {}
    authored_counts: dict[str, int] = {}
    conflicts: list[dict[str, str]] = []
    conflict_count = 0
    unlabeled_examples: list[str] = []
    class_examples: dict[str, list[dict[str, Any]]] = {}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    geometry_count = 0
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        geometry_count += 1
        inferred = infer_semantic_class(str(prim.GetPath()))
        inherited = _nearest_semantic_label(prim)
        authored = canonical_semantic_class(inherited)
        effective = inferred if inferred != "misc" else authored
        counts[effective] = counts.get(effective, 0) + 1
        examples = class_examples.setdefault(effective, [])
        if len(examples) < max_examples:
            examples.append(
                {
                    "path": str(prim.GetPath()),
                    "bbox": _bbox_data(cache, prim),
                }
            )
        if inherited:
            authored_counts[authored] = authored_counts.get(authored, 0) + 1
        elif len(unlabeled_examples) < max_examples:
            unlabeled_examples.append(str(prim.GetPath()))
        if (
            inferred != "misc"
            and inherited
            and authored != inferred
        ):
            conflict_count += 1
            if len(conflicts) < max_examples:
                conflicts.append(
                    {
                        "path": str(prim.GetPath()),
                        "authored": inherited,
                        "canonical_authored": authored,
                        "leaf_first": inferred,
                    }
                )
    return {
        "scene": str(scene_path.resolve()),
        "geometry_count": geometry_count,
        "effective_navigation_counts": dict(sorted(counts.items())),
        "direct_authored_counts": dict(sorted(authored_counts.items())),
        "class_examples": dict(sorted(class_examples.items())),
        "conflict_count": conflict_count,
        "conflict_count_shown": len(conflicts),
        "conflicts": conflicts,
        "unlabeled_examples": unlabeled_examples,
    }


def _relative_asset_path(output_path: Path, source_path: Path) -> str:
    # Preserve a workspace symlink such as assets/env/raw/Hospital. Resolving it
    # here would make the derived layer point back to the external asset store.
    return os.path.relpath(os.path.abspath(source_path), os.path.abspath(output_path.parent))


def write_open_door_overlay(source_path: Path, output_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    panels = report["openable_door_panels"]
    if not panels:
        raise RuntimeError(f"No recognised door panels found in {source_path}; no overlay was written.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    layer = stage.GetRootLayer()
    layer.subLayerPaths.append(_relative_asset_path(output_path, source_path))
    stage.SetMetadata("comment", "Derived overlay: recognised door panels are inactive; source USD is unchanged.")
    for panel in panels:
        override = stage.OverridePrim(panel["path"])
        override.SetActive(False)
    stage.GetRootLayer().Save()
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "disabled_door_panels": [panel["path"] for panel in panels],
    }


def _write_json(data: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Inspect a composed USD scene")
    audit.add_argument("scene", type=Path)
    audit.add_argument("--json", type=Path, help="Write the audit report to this path")
    open_doors = subparsers.add_parser("open-doors", help="Create a non-destructive door-open overlay")
    open_doors.add_argument("scene", type=Path)
    open_doors.add_argument("--output", type=Path, required=True)
    open_doors.add_argument("--json", type=Path, help="Write the change manifest to this path")
    semantics = subparsers.add_parser(
        "audit-semantics", help="Audit authored labels against navigation semantics"
    )
    semantics.add_argument("scene", type=Path)
    semantics.add_argument("--json", type=Path, help="Write the audit report to this path")
    semantics.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    if args.command == "audit-semantics":
        _write_json(audit_semantics(args.scene, args.max_examples), args.json)
        return 0
    report = audit_scene(args.scene)
    if args.command == "audit":
        _write_json(report, args.json)
        return 0
    manifest = write_open_door_overlay(args.scene, args.output, report)
    manifest["audit"] = report
    _write_json(manifest, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
