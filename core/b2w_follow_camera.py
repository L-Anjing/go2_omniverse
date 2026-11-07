"""Low mounted chase view. Parenting handles tracking without PhysX callbacks."""
import math
import os

import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from pxr import Usd, UsdGeom


def add_b2w_follow_cameras(scene):
    setback = float(os.environ.get("B2W_FOLLOW_DISTANCE", "0.7"))
    height = float(os.environ.get("B2W_FOLLOW_HEIGHT", "0.4"))
    pitch = float(os.environ.get("B2W_FOLLOW_PITCH_DEG", "20"))
    if not all(math.isfinite(v) for v in (setback, height, pitch)) or setback <= 0 or height < 0 or not 0 <= pitch < 90:
        raise ValueError("Follow camera requires distance > 0, height >= 0, and pitch in [0,90)")
    half_pitch = math.radians(pitch) / 2
    rotation = convert_camera_frame_orientation_convention(
        torch.tensor([[math.cos(half_pitch), 0., math.sin(half_pitch), 0.]]),
        origin="world", target="opengl",
    )[0].tolist()
    stage = omni.usd.get_context().get_stage()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    paths = []
    for root in scene.env_prim_paths:
        mount = f"{root}/Robot/base_link"
        prim = stage.GetPrimAtPath(mount)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"B2W follow camera mount missing: {mount}")
        body = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if body.IsEmpty():
            raise RuntimeError(f"B2W body has no bounds: {mount}")
        position = (float(body.GetMin()[0]) - setback, 0., float(body.GetMax()[2]) + height)
        camera = sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=36.0,
            vertical_aperture=24.0, clipping_range=(0.01, 1000.0),
            lock_camera=True,
        )
        path = f"{mount}/follow_cam"
        camera.func(path, camera, translation=position, orientation=tuple(rotation))
        paths.append(path)
        print(f"[camera] B2W follow: {path}, body-relative xyz={position}, down={pitch:g} deg", flush=True)
    if paths and os.environ.get("B2W_FOLLOW_VIEW", "1").lower() in {"1", "true", "yes"}:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(paths[0])
