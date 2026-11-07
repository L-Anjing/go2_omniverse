# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Script to deploy an RSL-RL checkpoint in Isaac Sim."""

from __future__ import annotations



"""Shared Hospital scene loading; import only after AppLauncher initialization."""


import os
from pxr import Semantics, Usd, UsdGeom
from core.semantic_labels import (
    canonical_semantic_class, direct_door_semantic_class,
    direct_navigation_semantic_class, direct_railing_semantic_class,
    infer_semantic_class,
)

def _direct_semantic_class(prim) -> str:
    for attribute in prim.GetAttributes():
        if "semanticData" not in attribute.GetName():
            continue
        value = attribute.Get()
        if value:
            return canonical_semantic_class(str(value))
    return "misc"



def _spatial_semantic_class(prim, profile_hint: str = "") -> str:
    """Resolve geometry leaf-first, then fall back to authored ancestor labels."""
    inferred = infer_semantic_class(str(prim.GetPath()), profile_hint)
    if inferred != "misc":
        return inferred
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        authored = _direct_semantic_class(current)
        if authored != "misc":
            return authored
        current = current.GetParent()
    return "misc"



def _deactivate_hospital_collision_planes(stage) -> int:
    """Disable duplicate ground-plane prims under the imported hospital subtree.

    The labeled hospital USD does not reliably expose the original
    ``/World/hospital/Root/CollisionPlane`` prim. In practice the duplicate
    hospital-local ground may appear as either ``CollisionPlane`` or
    ``WorldGrid`` after export/composition, so resolve by a small allowlist of
    prim names instead of relying on one hard-coded path.
    """
    hospital_root = stage.GetPrimAtPath("/World/hospital")
    if not hospital_root or not hospital_root.IsValid():
        return 0

    candidate_names = {"CollisionPlane", "WorldGrid"}
    deactivated = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/hospital/"):
            continue
        if prim.GetName() not in candidate_names:
            continue
        prim.SetActive(False)
        deactivated += 1
        print(f"[INFO] Hospital duplicate ground prim deactivated: {path}")
    return deactivated



def _has_direct_semantic_annotation(prim) -> bool:
    return any("semanticData" in attr.GetName() for attr in prim.GetAttributes())



def _reconcile_navigation_semantics(stage, root_path: str) -> int:
    """Repair stale structural/walkable labels using leaf-first prim names."""
    if os.environ.get("SEMANTIC_LABEL_MODE", "").strip() == "manual_labeled":
        return 0
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    repaired = 0
    walkable = {"floor", "ramp", "stairs"}
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        expected = direct_door_semantic_class(path)
        if expected is None:
            expected = direct_railing_semantic_class(path)
        if expected is None:
            expected = direct_navigation_semantic_class(path)
        if expected is None:
            continue
        # Never restore a broad navigation class on an assembly Xform.  The
        # navigation overlay deliberately blocks stair-root semantics so the
        # rendered child meshes can distinguish tread from railing.  Rewriting
        # ``Geo_M_Stairs*`` here made Replicator classify the whole assembly as
        # stairs and overrode the child wall label at runtime.
        if not prim.IsA(UsdGeom.Gprim):
            continue
        updated = False
        for attribute in prim.GetAttributes():
            if "semanticData" in attribute.GetName():
                current = canonical_semantic_class(str(attribute.Get() or ""))
                if (current in walkable) != (expected in walkable):
                    attribute.Set(expected)
                    updated = True
        if updated:
            repaired += 1
    return repaired



def _count_semantic_annotations(stage, root_path: str) -> int:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    return sum(
        1
        for prim in Usd.PrimRange(root)
        if _has_direct_semantic_annotation(prim)
    )



def _count_stair_obstacle_labels(stage, root_path: str) -> int:
    """Count direct obstacle labels authored inside named stair assemblies."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if "stair" not in str(prim.GetPath()).lower():
            continue
        for attribute in prim.GetAttributes():
            if "semanticdata" not in attribute.GetName().lower():
                continue
            if canonical_semantic_class(attribute.Get()) == "wall":
                count += 1
                break
    return count



def _label_named_navigation_geometry(stage, root_path: str) -> dict[str, int]:
    """Author direct labels on renderable floor and transition prims.

    The hospital asset has structural labels on broad ancestor prims.  A mesh
    named ``Geo_Floor_*`` can therefore inherit ``structure`` even though name
    inference correctly identifies it as a floor.  Reconciliation alone does
    not fix that case because it only edits semantic attributes already
    authored on the mesh.  Replicator resolves the rendered leaf prim, so put
    the navigation label directly on each renderable leaf before annotators
    are created.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {}
    counts: dict[str, int] = {}
    for candidate in Usd.PrimRange(root):
        if not candidate.IsActive():
            continue
        if not candidate.IsA(UsdGeom.Gprim):
            continue
        if os.environ.get("SEMANTIC_LABEL_MODE", "").strip() == "manual_labeled":
            # Count authored surfaces without rewriting audited source labels.
            # S_WetFloorSign used to be overwritten as floor here after load.
            value = candidate.GetAttribute("semantic:Semantics:params:semanticData").Get()
            if value in {"floor", "stairs", "ramp"}:
                counts[value] = counts.get(value, 0) + 1
            continue
        class_name = direct_navigation_semantic_class(str(candidate.GetPath()))
        if class_name is None:
            continue
        if _set_semantic_class(candidate, class_name):
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts



def _label_named_railing_geometry(stage, root_path: str) -> dict[str, int]:
    """Directly label rail/fence leaves so stair-parent labels cannot leak."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {}
    counts: dict[str, int] = {}
    for candidate in Usd.PrimRange(root):
        if not candidate.IsActive():
            continue
        if not candidate.IsA(UsdGeom.Gprim):
            continue
        class_name = direct_railing_semantic_class(str(candidate.GetPath()))
        if class_name is None:
            continue
        if _set_semantic_class(candidate, class_name):
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts



def _set_semantic_class(prim, class_name: str) -> bool:
    """Attach one unambiguous Replicator class label to a prim."""
    if not prim or not prim.IsValid():
        return False
    # Use the same canonical API instance as the statically labeled assets.
    # Creating class_floor/class_stairs beside an existing ``Semantics`` API
    # gives Replicator multiple class values for one rendered prim and the ROS
    # bridge can publish that conflict as background.
    semantics = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    semantics.CreateSemanticTypeAttr().Set("class")
    semantics.CreateSemanticDataAttr().Set(class_name)
    return True



def _label_ground_subtree(stage, root_path: str, class_name: str) -> int:
    """Apply a semantic class to every renderable prim under root_path.

    Generated terrains put the actual mesh on child prims (e.g.
    /World/ground/flat); the root prim itself is only an Xform.  Replicator
    semantics are per-prim, so labelling just the root leaves the rendered
    floor unlabeled (it then merges with structure/background, which makes
    flat scenes look permanently "blocked" to semantic-band avoidance).
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[WARN] Cannot label ground: {root_path} missing")
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsActive():
            continue
        if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Mesh)):
            continue
        if _set_semantic_class(prim, class_name):
            count += 1
    if count:
        print(f"[INFO] Semantic class applied: {root_path}/** -> {class_name} ({count} prims)")
    return count



def _spawn_hospital_usd(prim_path: str, cfg, translation=None, orientation=None, on_loaded=None):
    """Spawn hospital USD and immediately deactivate its duplicate ground plane.

    Using this as the spawn func inside AssetBaseCfg lets us load the hospital
    BEFORE gym.make() (so robot and scene appear simultaneously) while still
    removing the CollisionPlane before PhysX runs its first broad-phase reset.

    Without this, two coincident collision planes exist during PhysX init and
    crash scene loading.  Without pre-gym.make loading, hospital collision meshes
    appear mid-episode and trigger base_contact every step.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    import omni.usd
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)
    stage = omni.usd.get_context().get_stage()
    repaired_labels = _reconcile_navigation_semantics(stage, prim_path)
    navigation_labels = _label_named_navigation_geometry(stage, prim_path)
    railing_labels = _label_named_railing_geometry(stage, prim_path)
    if navigation_labels.get("floor", 0) == 0:
        raise RuntimeError(
            "Hospital USD exposes no directly labeled renderable floor prims; "
            "refusing to start semantic collection"
        )
    stair_obstacle_labels = _count_stair_obstacle_labels(stage, prim_path)
    if stair_obstacle_labels == 0:
        raise RuntimeError(
            "Hospital stair geometry exposes no direct wall/railing labels; "
            "refusing to start semantic collection"
        )
    annotation_count = _count_semantic_annotations(stage, prim_path)
    if annotation_count == 0:
        raise RuntimeError(f"Labeled hospital USD has no semantic annotations: {cfg.usd_path}")
    num_planes = _deactivate_hospital_collision_planes(stage)
    from core.ground_friction import apply_scene_ground_friction
    apply_scene_ground_friction(stage, prim_path)
    if on_loaded is not None:
        on_loaded(stage, prim_path)
    if num_planes == 0:
        print("[WARN] Hospital CollisionPlane not found during spawn — check USD prim path")
    else:
        print(
            f"[INFO] Hospital spawned with {annotation_count} semantic annotations; "
            f"{num_planes} duplicate ground prim(s) deactivated; "
            f"{repaired_labels} navigation label(s) reconciled; "
            f"direct navigation labels={navigation_labels}; "
            f"direct railing labels={railing_labels}; "
            f"stair obstacle labels={stair_obstacle_labels}"
        )
    return prim



def configure_default_ground(stage, floor_bounds, metadata) -> bool:
    """Hide the fallback terrain when the imported USD provides its own floor.

    The plane remains active for conservative foot-contact support, while USD
    visibility inheritance removes it from RGB, depth, and semantic rendering.
    """

    if os.environ.get("HIDE_DEFAULT_GROUND", "false").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    ground = stage.GetPrimAtPath("/World/ground")
    if not ground or not ground.IsValid():
        raise RuntimeError("HIDE_DEFAULT_GROUND=true but /World/ground is missing")
    if not floor_bounds:
        raise RuntimeError(
            "Refusing to hide /World/ground: the custom USD exposes no floor meshes"
        )
    imageables = []
    for prim in Usd.PrimRange(ground):
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        imageables.append((prim, imageable))
    if not imageables:
        raise RuntimeError(
            "HIDE_DEFAULT_GROUND=true but /World/ground has no imageable prims"
        )

    # Author visibility on the root and concrete descendants. Root visibility
    # is normally sufficient through inheritance, while descendant opinions
    # make the invariant robust to imported terrain assets with local values.
    for _, imageable in imageables:
        imageable.MakeInvisible()

    still_visible = [
        str(prim.GetPath())
        for prim, imageable in imageables
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
    ]
    if still_visible:
        raise RuntimeError(
            "Default ground is still sensor-visible after hiding: "
            + ", ".join(still_visible[:8])
        )
    print(
        f"[INFO] Default /World/ground hidden from sensors "
        f"({len(imageables)} imageable prims verified invisible); "
        f"the custom USD supplies {len(floor_bounds)} rendered floor meshes"
    )
    metadata["default_ground_sensor_visibility"] = "invisible"
    metadata["custom_floor_mesh_count"] = len(floor_bounds)
    return True



def configure_hospital_scene(scene, usd_path, spawn_func=None):
    """Use the same ground, Hospital spawner and lighting in both entries."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.terrains import TerrainImporterCfg
    from core.ground_friction import GroundFriction

    friction = GroundFriction.from_env()
    scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", debug_vis=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=friction.static, dynamic_friction=friction.dynamic,
        ),
    )
    hospital = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    hospital.func = spawn_func or _spawn_hospital_usd
    scene.custom_env = AssetBaseCfg(prim_path="/World/hospital", spawn=hospital)
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
