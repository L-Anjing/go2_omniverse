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


def create_front_cam_omnigraph(robot_num):
    """Create two independent OmniGraphs for the front camera.

    Graph 1  (/ROS_front_cam{N}_rgb):
        Tick → IsaacCreateRenderProduct → ROS2CameraHelper(rgb)
        Publishes: robot{N}/front_cam/rgb

    Graph 2  (/ROS_front_cam{N}_seg):
        Tick → IsaacCreateRenderProduct → ROS2CameraHelper(semantic_segmentation)
        Publishes: robot{N}/front_cam/semantic_segmentation  [32SC1]

    Two separate graphs are used because ROS2CameraHelper has no execOut port,
    so the nodes cannot be chained inside a single graph.  Each graph gets its
    own IsaacCreateRenderProduct (and therefore its own render product handle)
    so there is no shared-state race between the two helpers.

    The raw 32SC1 semantic image can be converted to rgb8 for RViz2 display
    by running seg_to_rgb8.py in a separate terminal.
    """

    keys = og.Controller.Keys
    camera_prim = f"/World/envs/env_{robot_num}/Robot/base/front_cam"

    # ------------------------------------------------------------------
    # Graph 1 — RGB
    # ------------------------------------------------------------------
    og.Controller.edit(
        {
            "graph_path": f"/ROS_front_cam{robot_num}_rgb",
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "IsaacCreateRenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("IsaacCreateRenderProduct.inputs:cameraPrim", camera_prim),
                ("IsaacCreateRenderProduct.inputs:enabled", True),
                ("IsaacCreateRenderProduct.inputs:width", CAMERA_WIDTH),
                ("IsaacCreateRenderProduct.inputs:height", CAMERA_HEIGHT),
                ("ROS2CameraHelper.inputs:type", "rgb"),
                (
                    "ROS2CameraHelper.inputs:topicName",
                    f"robot{robot_num}/front_cam/rgb",
                ),
                ("ROS2CameraHelper.inputs:frameId", f"robot{robot_num}"),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "IsaacCreateRenderProduct.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:execOut",
                    "ROS2CameraHelper.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:renderProductPath",
                    "ROS2CameraHelper.inputs:renderProductPath",
                ),
            ],
        },
    )

    # ------------------------------------------------------------------
    # Graph 2 — Semantic segmentation (published as 32SC1)
    # ------------------------------------------------------------------
    og.Controller.edit(
        {
            "graph_path": f"/ROS_front_cam{robot_num}_seg",
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "IsaacCreateRenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("IsaacCreateRenderProduct.inputs:cameraPrim", camera_prim),
                ("IsaacCreateRenderProduct.inputs:enabled", True),
                ("IsaacCreateRenderProduct.inputs:width", CAMERA_WIDTH),
                ("IsaacCreateRenderProduct.inputs:height", CAMERA_HEIGHT),
                ("ROS2CameraHelper.inputs:type", "semantic_segmentation"),
                (
                    "ROS2CameraHelper.inputs:topicName",
                    f"robot{robot_num}/front_cam/semantic_segmentation",
                ),
                ("ROS2CameraHelper.inputs:frameId", f"robot{robot_num}"),
                # Publishes {id: {class: name}} JSON on
                # robot{N}/front_cam/semantic_segmentation_labels (std_msgs/String)
                ("ROS2CameraHelper.inputs:enableSemanticLabels", True),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "IsaacCreateRenderProduct.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:execOut",
                    "ROS2CameraHelper.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:renderProductPath",
                    "ROS2CameraHelper.inputs:renderProductPath",
                ),
            ],
        },
    )

    # ------------------------------------------------------------------
    # Graph 3 — Camera Info
    # Publishes: robot{N}/front_cam/camera_info  (sensor_msgs/CameraInfo)
    # ------------------------------------------------------------------
    og.Controller.edit(
        {
            "graph_path": f"/ROS_front_cam{robot_num}_info",
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "IsaacCreateRenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("IsaacCreateRenderProduct.inputs:cameraPrim", camera_prim),
                ("IsaacCreateRenderProduct.inputs:enabled", True),
                ("IsaacCreateRenderProduct.inputs:width", CAMERA_WIDTH),
                ("IsaacCreateRenderProduct.inputs:height", CAMERA_HEIGHT),
                ("ROS2CameraHelper.inputs:type", "camera_info"),
                (
                    "ROS2CameraHelper.inputs:topicName",
                    f"robot{robot_num}/front_cam/camera_info",
                ),
                ("ROS2CameraHelper.inputs:frameId", f"robot{robot_num}/front_cam"),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "IsaacCreateRenderProduct.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:execOut",
                    "ROS2CameraHelper.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:renderProductPath",
                    "ROS2CameraHelper.inputs:renderProductPath",
                ),
            ],
        },
    )
