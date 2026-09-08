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

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.semantic_labels import (  # noqa: E402
    canonical_semantic_class,
    direct_door_semantic_class,
    direct_navigation_semantic_class,
    direct_railing_semantic_class,
    hospital_upper_floor_semantic_class,
    stair_section_semantic_class,
    infer_semantic_class,
)


PANEL_PREFIXES = ("sm_door_", "glassdoor")


def _author_semantic_class(prim: Usd.Prim, class_name: str) -> None:
    """Author a Replicator-visible ``class`` SemanticsAPI instance."""
    schema = "SemanticsAPI:Semantics"
    # Preserve unrelated applied schemas and remove competing class instances.
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if name.endswith(":params:semanticType") and attribute.Get() == "class":
            instance = name.split(":")[1]
            if instance != "Semantics":
                prim.RemoveAppliedSchema(f"SemanticsAPI:{instance}")
    prim.AddAppliedSchema(schema)
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String
    ).Set("class")
    prim.CreateAttribute(
        "semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String
    ).Set(class_name)


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
        prim_path = str(prim.GetPath())
        inferred = stair_section_semantic_class(prim_path)
        if inferred is None:
            inferred = infer_semantic_class(prim_path)
        inherited = _nearest_semantic_label(prim)
        authored = canonical_semantic_class(inherited)
        effective = inferred if inferred != "misc" else authored
        counts[effective] = counts.get(effective, 0) + 1
        examples = class_examples.setdefault(effective, [])
        if len(examples) < max_examples:
            examples.append(
                {
                    "path": prim_path,
                    "bbox": _bbox_data(cache, prim),
                }
            )
        if inherited:
            authored_counts[authored] = authored_counts.get(authored, 0) + 1
        elif len(unlabeled_examples) < max_examples:
            unlabeled_examples.append(prim_path)
        if (
            inferred != "misc"
            and inherited
            and authored != inferred
        ):
            conflict_count += 1
            if len(conflicts) < max_examples:
                conflicts.append(
                    {
                        "path": prim_path,
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


def write_navigation_semantic_overlay(
    source_path: Path,
    output_path: Path,
    *,
    open_doors: bool = False,
    all_semantics: bool = False,
) -> dict[str, Any]:
    """Write direct mesh semantics and optional open-door overrides."""
    source = Usd.Stage.Open(str(source_path))
    if source is None:
        raise RuntimeError(f"Unable to open USD scene: {source_path}")
    root = source.GetDefaultPrim()
    if not root or not root.IsValid():
        root = source.GetPseudoRoot()

    labels: dict[str, str] = {}
    door_semantic_prims: set[str] = set()
    instance_roots: set[str] = set()
    stair_semantic_ancestors: set[str] = set()
    class_ancestors: set[str] = set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        if direct_door_semantic_class(prim_path) is not None and (
            prim.IsA(UsdGeom.Gprim) or prim.GetTypeName() == "Xform"
        ):
            door_semantic_prims.add(prim_path)
        if (
            "stair" in prim_path.lower()
            and not prim.IsA(UsdGeom.Gprim)
            and canonical_semantic_class(_direct_semantic_label(prim)) == "stairs"
        ):
            stair_semantic_ancestors.add(prim_path)
        if not prim.IsA(UsdGeom.Gprim):
            continue
        audited_stair_label = stair_section_semantic_class(prim_path)
        audited_door_label = direct_door_semantic_class(prim_path)
        audited_navigation_label = direct_navigation_semantic_class(prim_path)
        # Audited Hospital/Warehouse warning-sign mesh. Older warehouse
        # source labels classified S_WetFloorSign as floor before sign.
        audited_sign_label = (
            "sign" if prim.GetName().lower() == "s_wetfloorsign" else None
        )
        label = (
            audited_sign_label
            or audited_stair_label
            or audited_door_label
            or direct_railing_semantic_class(prim_path)
            or audited_navigation_label
            or _nearest_semantic_label(prim)
            or infer_semantic_class(prim_path)
        )
        if not all_semantics and label not in {"floor", "stairs", "ramp"} and not (
            audited_stair_label == "wall"
        ):
            continue
        if label == "misc" and not all_semantics:
            continue
        labels[prim_path] = label
        if all_semantics:
            ancestor = prim.GetParent()
            while ancestor and ancestor.IsValid() and not ancestor.IsPseudoRoot():
                class_ancestors.add(str(ancestor.GetPath()))
                ancestor = ancestor.GetParent()
        current = prim
        while current and current.IsValid() and not current.IsPseudoRoot():
            if current.IsInstance() or current.IsInstanceProxy():
                instance_root = current
                while instance_root.IsInstanceProxy():
                    instance_root = instance_root.GetParent()
                if instance_root and instance_root.IsValid():
                    instance_roots.add(str(instance_root.GetPath()))
                break
            current = current.GetParent()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = Usd.Stage.CreateNew(str(output_path))
    overlay.GetRootLayer().subLayerPaths.append(
        _relative_asset_path(output_path, source_path)
    )
    source_default = source.GetDefaultPrim()
    if source_default and source_default.IsValid():
        composed_default = overlay.GetPrimAtPath(source_default.GetPath())
        if not composed_default or not composed_default.IsValid():
            raise RuntimeError(
                f"Overlay cannot resolve source defaultPrim {source_default.GetPath()}"
            )
        overlay.SetDefaultPrim(composed_default)
    else:
        raise RuntimeError(f"Source scene has no valid defaultPrim: {source_path}")
    overlay.SetMetadata(
        "comment",
        "Direct render-mesh semantics with optional open-door overrides.",
    )
    for path in sorted(instance_roots):
        overlay.OverridePrim(path).SetInstanceable(False)
    # Replicator may prefer an instance/Xform semantic over a rendered mesh's
    # direct semantic.  Block broad stair-assembly labels so Section-level
    # stairs/wall labels are the only available classification.
    for path in sorted(stair_semantic_ancestors):
        override = overlay.OverridePrim(path)
        override.CreateAttribute(
            "semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String
        ).Set(Sdf.ValueBlock())
        override.CreateAttribute(
            "semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String
        ).Set(Sdf.ValueBlock())
    # Door panels, frames, jambs, and misleading children such as DoorFloor
    # are physical structure. Open-door state never makes the asset navigable.
    for path in sorted(door_semantic_prims):
        _author_semantic_class(overlay.OverridePrim(path), "wall")
    disabled_door_panels: list[str] = []
    if open_doors:
        scene_report = audit_scene(source_path)
        disabled_door_panels = [
            panel["path"] for panel in scene_report["openable_door_panels"]
        ]
    overlay.GetRootLayer().Save()

    # Reopen after de-instancing so child mesh overrides are authorable.
    overlay = Usd.Stage.Open(str(output_path))
    for path, label in sorted(labels.items()):
        _author_semantic_class(overlay.OverridePrim(path), label)
    # All renderable leaves now own their class. Remove the ancestor APIs,
    # not merely their values: empty APIs can produce ':' and parent classes
    # can combine with leaf classes (e.g. 'wheelchair,forklift').
    for path in sorted(class_ancestors | stair_semantic_ancestors):
        prim = overlay.GetPrimAtPath(path)
        if not prim or prim.IsA(UsdGeom.Gprim):
            continue
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if name.endswith(":params:semanticType") and attribute.Get() in (None, "class"):
                instance = name.split(":")[1]
                prim.RemoveAppliedSchema(f"SemanticsAPI:{instance}")
                attribute.Set(Sdf.ValueBlock())
                data = prim.GetAttribute(f"semantic:{instance}:params:semanticData")
                if data:
                    data.Set(Sdf.ValueBlock())
    # Deactivate doors only after descendant mesh labels have been authored;
    # an inactive ancestor makes its descendants unavailable for overrides.
    for path in disabled_door_panels:
        overlay.OverridePrim(path).SetActive(False)
    support_splits = _split_hospital_support_surfaces(overlay)
    for split in support_splits:
        labels[split["source_mesh"]] = "wall"
        labels[split["support_mesh"]] = "floor"
    overlay.GetRootLayer().Save()
    return {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "expanded_navigation_instances": len(instance_roots),
        "blocked_stair_ancestor_labels": len(stair_semantic_ancestors),
        "disabled_door_panels": disabled_door_panels,
        "support_surface_splits": support_splits,
        "direct_mesh_labels": {
            name: sum(value == name for value in labels.values())
            for name in sorted(set(labels.values()) | {"floor", "stairs", "ramp", "wall"})
        },
    }


def _split_hospital_support_surfaces(stage: Usd.Stage) -> list[dict[str, Any]]:
    """Separate audited slab support faces from ceiling/side faces offline.

    Exact asset identities and measured top heights limit this operation to the
    two Hospital slabs connected to the audited upper-floor waypoints. Normals
    alone never authorize another roof, beam, furniture or overhead surface.
    """
    expected_tops = {"Geo_Roof_Costum1": 3.3338134078, "Geo_Roof_Costum6": 3.3374141768}
    candidates = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
                  and hospital_upper_floor_semantic_class(str(p.GetPath())) == "wall"]
    if not candidates:
        return []
    composed = stage.Flatten()
    reports = []
    for prim in candidates:
        mesh = UsdGeom.Mesh(prim)
        # The audited assets use vertex normals/UVs and have no subsets,
        # holes or creases. Refuse changed topology rather than corrupt data.
        if prim.GetChildren() or mesh.GetHoleIndicesAttr().Get():
            raise ValueError(f"Unexpected slab subsets/holes: {prim.GetPath()}")
        for attr in prim.GetAttributes():
            if attr.GetMetadata("interpolation") in {"uniform", "faceVarying"}:
                raise ValueError(f"Unsupported face data: {attr.GetPath()}")
        points = mesh.GetPointsAttr().Get()
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        world = [transform.Transform(Gf.Vec3d(p)) for p in points]
        top = max(p[2] for p in world)
        tolerance = max((top - min(p[2] for p in world)) * 0.001, 1e-6)
        if abs(top - expected_tops[prim.GetName()]) > tolerance:
            raise ValueError(f"Audited slab height changed: {prim.GetPath()}")
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        groups = {"floor": [], "wall": []}
        offset = 0
        for count in counts:
            face = list(indices[offset:offset + count])
            offset += count
            vertices = [world[i] for i in face]
            normal = Gf.Cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).GetNormalized()
            support = normal[2] > 0.99 and all(abs(v[2] - top) <= tolerance for v in vertices)
            groups["floor" if support else "wall"].append(face)
        if not all(groups.values()):
            raise ValueError(f"Slab must have both support and structure: {prim.GetPath()}")
        support_path = str(prim.GetPath()) + "_SupportSurface"
        Sdf.CopySpec(composed, prim.GetPath(), stage.GetRootLayer(), Sdf.Path(support_path))
        for label, path in (("floor", support_path), ("wall", str(prim.GetPath()))):
            target = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            faces = groups[label]
            target.GetFaceVertexCountsAttr().Set([len(f) for f in faces])
            target.GetFaceVertexIndicesAttr().Set([i for f in faces for i in f])
            used = [points[i] for f in faces for i in f]
            target.GetExtentAttr().Set([
                Gf.Vec3f(*(min(p[a] for p in used) for a in range(3))),
                Gf.Vec3f(*(max(p[a] for p in used) for a in range(3))),
            ])
            _author_semantic_class(target.GetPrim(), label)
        reports.append({"source_mesh": str(prim.GetPath()), "support_mesh": support_path,
                        "support_world_z": top, "floor_faces": len(groups["floor"]),
                        "structure_faces": len(groups["wall"]), "original_faces": len(counts)})
    return reports


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
    nav_overlay = subparsers.add_parser(
        "navigation-overlay",
        help="Create an overlay with direct render-mesh navigation labels",
    )
    nav_overlay.add_argument("scene", type=Path)
    nav_overlay.add_argument("--output", type=Path, required=True)
    nav_overlay.add_argument("--json", type=Path)
    nav_overlay.add_argument(
        "--open-doors", action="store_true", help="Deactivate audited door panels"
    )
    nav_overlay.add_argument(
        "--all-semantics",
        action="store_true",
        help="Author every recognized semantic class, not only navigation classes",
    )
    args = parser.parse_args()

    if args.command == "audit-semantics":
        _write_json(audit_semantics(args.scene, args.max_examples), args.json)
        return 0
    if args.command == "navigation-overlay":
        report = write_navigation_semantic_overlay(
            args.scene,
            args.output,
            open_doors=args.open_doors,
            all_semantics=args.all_semantics,
        )
        _write_json(report, args.json)
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
