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


import omni
import omni.graph.core as og

# Shared camera resolution — lowered from 1280×720 to reduce RTX render load.
# 640×360 keeps the same 16:9 aspect ratio while cutting pixel count by 75%.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360


def _build_graph_name(robot_num: int, cam_prim_name: str, stream: str) -> str:
    """Build a unique OmniGraph path for a camera stream.

    >>> _build_graph_name(0, "front_cam", "rgb")
    '/ROS_front_cam0_rgb'
    >>> _build_graph_name(0, "front_cam_left", "depth")
    '/ROS_front_cam0_left_depth'
    """
    # Use the last segment of the prim name to keep graph paths short.
    # front_cam → cam,  front_cam_left → left,  front_cam_right → right
    parts = cam_prim_name.split("_")
    if parts[-1] in ("left", "right"):
        short = parts[-1]  # left / right
    else:
        short = parts[-1]  # cam (or whatever the last token is)
    return f"/ROS_front_cam{robot_num}_{short}_{stream}"


def _create_one_camera_omnigraphs(
    robot_num: int,
    cam_prim_name: str,
    enable_depth: bool,
):
    """Create one shared Omnigraph for a single camera prim.

    RGB, depth, semantic segmentation, and camera info share one render
    product and playback tick. Besides avoiding redundant RTX renders, this
    keeps image timestamps aligned for downstream RGBD synchronization.

    Parameters
    ----------
    robot_num : int
        Environment index.
    cam_prim_name : str
        Camera prim name, e.g. ``"front_cam"``, ``"front_cam_left"``.
    enable_depth : bool
        Whether to create the depth publishing graph.
    """
    keys = og.Controller.Keys
    camera_prim = f"/World/envs/env_{robot_num}/Robot/base/{cam_prim_name}"

    # Topic prefix uses the prim name directly so ROS consumers can
    # distinguish left / right:  robot0/front_cam_left/rgb  etc.
    topic_prefix = f"robot{robot_num}/{cam_prim_name}"
    # Frame id follows the same scheme.
    frame_id = f"robot{robot_num}/{cam_prim_name}"

    publishers = ["RGBPublisher", "SemanticPublisher", "CameraInfoPublisher"]
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("RGBPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("SemanticPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("CameraInfoPublisher", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
    ]
    if enable_depth:
        publishers.insert(1, "DepthPublisher")
        create_nodes.insert(
            3, ("DepthPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper")
        )

    set_values = [
        ("IsaacCreateRenderProduct.inputs:cameraPrim", camera_prim),
        ("IsaacCreateRenderProduct.inputs:enabled", True),
        ("IsaacCreateRenderProduct.inputs:width", CAMERA_WIDTH),
        ("IsaacCreateRenderProduct.inputs:height", CAMERA_HEIGHT),
        ("RGBPublisher.inputs:type", "rgb"),
        ("RGBPublisher.inputs:topicName", f"{topic_prefix}/rgb"),
        ("RGBPublisher.inputs:frameId", frame_id),
        ("RGBPublisher.inputs:useSystemTime", False),
        ("SemanticPublisher.inputs:type", "semantic_segmentation"),
        (
            "SemanticPublisher.inputs:topicName",
            f"{topic_prefix}/semantic_segmentation",
        ),
        ("SemanticPublisher.inputs:frameId", frame_id),
        ("SemanticPublisher.inputs:useSystemTime", False),
        ("SemanticPublisher.inputs:enableSemanticLabels", True),
        (
            "SemanticPublisher.inputs:semanticLabelsTopicName",
            f"{topic_prefix}/semantic_segmentation_labels",
        ),
        ("CameraInfoPublisher.inputs:topicName", f"{topic_prefix}/camera_info"),
        ("CameraInfoPublisher.inputs:frameId", frame_id),
        ("CameraInfoPublisher.inputs:useSystemTime", False),
    ]
    if enable_depth:
        set_values.extend(
            [
                ("DepthPublisher.inputs:type", "depth"),
                ("DepthPublisher.inputs:topicName", f"{topic_prefix}/depth"),
                ("DepthPublisher.inputs:frameId", frame_id),
                ("DepthPublisher.inputs:useSystemTime", False),
            ]
        )

    connections = [
        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn")
    ]
    for publisher in publishers:
        connections.extend(
            [
                (
                    "IsaacCreateRenderProduct.outputs:execOut",
                    f"{publisher}.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:renderProductPath",
                    f"{publisher}.inputs:renderProductPath",
                ),
            ]
        )

    og.Controller.edit(
        {
            "graph_path": _build_graph_name(robot_num, cam_prim_name, "streams"),
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: create_nodes,
            keys.SET_VALUES: set_values,
            keys.CONNECT: connections,
        },
    )


def create_front_cam_omnigraph(
    robot_num: int,
    camera_mode: str = "single",
    enable_depth: bool = False,
):
    """Create OmniGraph pipelines for robot front camera(s).

    Parameters
    ----------
    robot_num : int
        Environment index (0, 1, …).
    camera_mode : str
        ``"single"`` — one center camera (``front_cam``).
        ``"double"`` — stereo pair (``front_cam_left``, ``front_cam_right``).
    enable_depth : bool
        When True, publish ``/depth`` topics alongside ``/rgb``.
        Default False (controlled by ``--enable_depth`` / ``ENABLE_DEPTH``).

    Published topics (single)
    -------------------------
    - ``robot{N}/front_cam/rgb``
    - ``robot{N}/front_cam/depth``             (only if *enable_depth*)
    - ``robot{N}/front_cam/semantic_segmentation``
    - ``robot{N}/front_cam/semantic_segmentation_labels``
    - ``robot{N}/front_cam/camera_info``

    Published topics (double)
    -------------------------
    - ``robot{N}/front_cam_left/rgb``
    - ``robot{N}/front_cam_right/rgb``
    - ``robot{N}/front_cam_left/depth``        (only if *enable_depth*)
    - ``robot{N}/front_cam_right/depth``       (only if *enable_depth*)
    - ``robot{N}/front_cam_left/semantic_segmentation``
    - ``robot{N}/front_cam_right/semantic_segmentation``
    - ``robot{N}/front_cam_left/semantic_segmentation_labels``
    - ``robot{N}/front_cam_right/semantic_segmentation_labels``
    - ``robot{N}/front_cam_left/camera_info``
    - ``robot{N}/front_cam_right/camera_info``
    """
    if camera_mode == "double":
        cam_prim_names = ["front_cam_left", "front_cam_right"]
    else:
        cam_prim_names = ["front_cam"]

    for cam_prim_name in cam_prim_names:
        _create_one_camera_omnigraphs(robot_num, cam_prim_name, enable_depth)
