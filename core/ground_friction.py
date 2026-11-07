"""Ground contact tuning, independent of policy and actuator joint friction."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class GroundFriction:
    static: float = 0.9
    dynamic: float = 0.8

    @classmethod
    def from_env(cls):
        value = cls(float(os.environ.get('GROUND_STATIC_FRICTION', '0.9')),
                    float(os.environ.get('GROUND_DYNAMIC_FRICTION', '0.8')))
        if not (math.isfinite(value.static) and math.isfinite(value.dynamic)
                and 0 <= value.dynamic <= value.static):
            raise ValueError('Ground friction must be finite and satisfy 0 <= dynamic <= static')
        return value


def apply_scene_ground_friction(stage, root_path):
    """Override static walkable collision materials in this composed stage only.

    Keep restitution/combine settings and shared original materials intact.
    Instance proxies cannot be authored independently; report them rather than
    changing an entire assembly (which might also contain walls or robots).
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from core.usd_terrain import semantic_class, static_collision

    friction = GroundFriction.from_env()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    candidates = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.IsA(UsdGeom.Gprim) and '/Robot' not in str(p.GetPath())
                  and static_collision(p) and semantic_class(p) in ('floor', 'stairs', 'ramp')]
    materials = {}
    applied = skipped = 0
    for prim in candidates:
        if prim.IsInstanceProxy():
            skipped += 1
            continue
        source, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        key = str(source.GetPath()) if source else ''
        if key not in materials:
            # A separate namespace under this asset avoids cross-scene coupling.
            material = UsdShade.Material.Define(
                stage, f'{root_path}/GroundFrictionMaterials/material_{len(materials)}')
            target = material.GetPrim()
            UsdPhysics.MaterialAPI.Apply(target)
            if source:
                if source.GetPrim().HasAPI(PhysxSchema.PhysxMaterialAPI):
                    PhysxSchema.PhysxMaterialAPI.Apply(target)
                for attr in source.GetPrim().GetAttributes():
                    if attr.GetName().startswith(('physics:', 'physxMaterial:')) and attr.HasAuthoredValueOpinion():
                        target.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
            api = UsdPhysics.MaterialAPI(target)
            api.CreateStaticFrictionAttr().Set(friction.static)
            api.CreateDynamicFrictionAttr().Set(friction.dynamic)
            materials[key] = material
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            materials[key], bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose='physics')
        applied += 1
    print(f'[ground friction] root={root_path} static={friction.static:g} '
          f'dynamic={friction.dynamic:g} bound={applied} skipped_instance_proxies={skipped}')
    return applied
