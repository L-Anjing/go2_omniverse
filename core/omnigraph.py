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
    """
    short = cam_prim_name.split("_")[-1]
    return f"/ROS_front_cam{robot_num}_{short}_{stream}"


def _create_one_camera_omnigraphs(
    robot_num: int,
    cam_prim_name: str,
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
        Camera prim name. The supported sensor is ``"front_cam"``.
    """
    keys = og.Controller.Keys
    camera_prim = f"/World/envs/env_{robot_num}/Robot/base/{cam_prim_name}"

    topic_prefix = f"robot{robot_num}/{cam_prim_name}"
    # Frame id follows the same scheme.
    frame_id = f"robot{robot_num}/{cam_prim_name}"

    publishers = [
        "RGBPublisher",
        "DepthPublisher",
        "SemanticPublisher",
        "CameraInfoPublisher",
    ]
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("RGBPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("DepthPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("SemanticPublisher", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("CameraInfoPublisher", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
    ]

    set_values = [
        ("IsaacCreateRenderProduct.inputs:cameraPrim", camera_prim),
        ("IsaacCreateRenderProduct.inputs:enabled", True),
        ("IsaacCreateRenderProduct.inputs:width", CAMERA_WIDTH),
        ("IsaacCreateRenderProduct.inputs:height", CAMERA_HEIGHT),
        ("RGBPublisher.inputs:type", "rgb"),
        ("RGBPublisher.inputs:topicName", f"{topic_prefix}/rgb"),
        ("RGBPublisher.inputs:frameId", frame_id),
        ("RGBPublisher.inputs:useSystemTime", False),
        ("DepthPublisher.inputs:type", "depth"),
        ("DepthPublisher.inputs:topicName", f"{topic_prefix}/depth"),
        ("DepthPublisher.inputs:frameId", frame_id),
        ("DepthPublisher.inputs:useSystemTime", False),
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


def create_front_cam_omnigraph(robot_num: int):
    """Create the OmniGraph pipelines for one robot front_cam.

    Parameters
    ----------
    robot_num : int
        Environment index (0, 1, …).
    Published topics
    ----------------
    - ``robot{N}/front_cam/rgb``
    - ``robot{N}/front_cam/depth``
    - ``robot{N}/front_cam/semantic_segmentation``
    - ``robot{N}/front_cam/semantic_segmentation_labels``
    - ``robot{N}/front_cam/camera_info``

    """
    _create_one_camera_omnigraphs(robot_num, "front_cam")
