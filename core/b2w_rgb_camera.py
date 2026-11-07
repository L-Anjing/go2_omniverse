"""Mounted RGB observation camera sharing the B2W raycast depth configuration."""
import torch
import omni.graph.core as og
import isaaclab.sim as sim_utils
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from pxr import Sdf

from core.robot_profiles import robot_namespace


def add_b2w_rgb_cameras(scene, depth_config):
    mount_pattern = depth_config.prim_path.replace("{ENV_REGEX_NS}", scene.env_regex_ns)
    mounts = [Sdf.Path(path) for path in sim_utils.find_matching_prim_paths(mount_pattern)]
    camera_paths = []
    for env_path in scene.env_prim_paths:
        candidates = [path for path in mounts if path.HasPrefix(Sdf.Path(env_path))]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one depth camera mount in {env_path}, found {candidates}")
        camera_paths.append(str(candidates[0].AppendChild("front_cam")))
    pattern = depth_config.pattern_cfg
    width, height = pattern.width * 10, pattern.height * 10
    rotation = convert_camera_frame_orientation_convention(
        torch.tensor([depth_config.offset.rot]),
        origin=depth_config.offset.convention, target="opengl",
    )[0].tolist()
    lens = sim_utils.PinholeCameraCfg(
        focal_length=pattern.focal_length,
        horizontal_aperture=pattern.horizontal_aperture,
        vertical_aperture=pattern.vertical_aperture,
        horizontal_aperture_offset=pattern.horizontal_aperture_offset,
        vertical_aperture_offset=pattern.vertical_aperture_offset,
        clipping_range=(0.01, depth_config.max_distance),
        lock_camera=True,
    )
    for index, camera_path in enumerate(camera_paths):
        lens.func(camera_path, lens, translation=depth_config.offset.pos, orientation=tuple(rotation))
        prefix = f"{robot_namespace(index)}/front_cam"
        keys = og.Controller.Keys
        connections = [("Tick.outputs:tick", "Render.inputs:execIn")]
        for publisher in ("RGB", "Info"):
            connections.extend([
                ("Render.outputs:execOut", f"{publisher}.inputs:execIn"),
                ("Render.outputs:renderProductPath", f"{publisher}.inputs:renderProductPath"),
            ])
        og.Controller.edit(
            {"graph_path": f"/ROS_b2w{index}_front_rgb", "evaluator_name": "execution",
             "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION},
            {
                keys.CREATE_NODES: [
                    ("Tick", "omni.graph.action.OnPlaybackTick"),
                    ("Render", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("RGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ],
                keys.SET_VALUES: [
                    ("Render.inputs:cameraPrim", camera_path),
                    ("Render.inputs:enabled", True),
                    ("Render.inputs:width", width), ("Render.inputs:height", height),
                    ("RGB.inputs:type", "rgb"),
                    ("RGB.inputs:topicName", f"{prefix}/rgb"),
                    ("RGB.inputs:frameId", prefix), ("RGB.inputs:useSystemTime", False),
                    ("Info.inputs:topicName", f"{prefix}/rgb/camera_info"),
                    ("Info.inputs:frameId", prefix), ("Info.inputs:useSystemTime", False),
                ],
                keys.CONNECT: connections,
            },
        )
        print(f"[camera] B2W RGB: {camera_path}, {width}x{height}, /{prefix}/rgb; shared depth mount/lens", flush=True)
