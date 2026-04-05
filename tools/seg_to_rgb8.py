#!/usr/bin/env python3
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

"""Semantic segmentation 32SC1 → rgb8 converter for RViz2.

Isaac Sim publishes semantic segmentation images with encoding ``32SC1``
(int32 per pixel, value = semantic class ID).  RViz2 cannot display this
format directly.  This node subscribes to the raw topic, applies a
deterministic ID→colour mapping, and republishes as ``rgb8``.

Topic mapping (one pair per robot):
  Input:  robot{i}/front_cam/semantic_segmentation   [sensor_msgs/Image, 32SC1]
  Output: robot{i}/front_cam/semantic_segmentation_rgb8 [sensor_msgs/Image, rgb8]

Usage (run in a separate terminal alongside the Isaac Sim session):
  python3 seg_to_rgb8.py --num_robots 1

The colour palette is derived by hashing the semantic ID with a fixed random
seed so that:
  • ID 0 (background / unlabelled) is always black.
  • Every other ID maps to a stable, visually distinct colour across frames.
"""

import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------

def _id_to_rgb(semantic_id: int) -> tuple[int, int, int]:
    """Map a 32-bit signed semantic class ID to a stable RGB triple.

    Uses numpy's legacy random generator seeded with the masked ID so that
    the mapping is deterministic and independent of call order.
    """
    if semantic_id == 0:
        return (0, 0, 0)  # background → black
    rng = np.random.RandomState(int(semantic_id) & 0x7FFFFFFF)
    r, g, b = rng.randint(40, 256, 3)
    return (int(r), int(g), int(b))


# Pre-build a small cache so repeated calls for the same ID are free.
_colour_cache: dict[int, tuple[int, int, int]] = {}


def _get_colour(semantic_id: int) -> tuple[int, int, int]:
    if semantic_id not in _colour_cache:
        _colour_cache[semantic_id] = _id_to_rgb(semantic_id)
    return _colour_cache[semantic_id]


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class SegToRgb8Node(Node):
    """Subscribes to 32SC1 semantic images and republishes them as rgb8."""

    def __init__(self, num_robots: int) -> None:
        super().__init__("seg_to_rgb8_node")

        self._publishers: list = []

        for i in range(num_robots):
            pub = self.create_publisher(
                Image,
                f"robot{i}/front_cam/semantic_segmentation_rgb8",
                10,
            )
            self._publishers.append(pub)

            self.create_subscription(
                Image,
                f"robot{i}/front_cam/semantic_segmentation",
                lambda msg, idx=i: self._callback(msg, idx),
                10,
            )

        self.get_logger().info(
            f"seg_to_rgb8 ready — watching {num_robots} robot(s).\n"
            "  Input  encoding: 32SC1\n"
            "  Output encoding: rgb8"
        )

    # ------------------------------------------------------------------

    def _callback(self, msg: Image, robot_idx: int) -> None:
        if msg.encoding not in ("32SC1", "mono32"):
            self.get_logger().warn(
                f"[robot{robot_idx}] Unexpected encoding '{msg.encoding}', "
                "expected '32SC1'.  Skipping frame.",
                throttle_duration_sec=5.0,
            )
            return

        # Decode the raw int32 pixel buffer
        raw: np.ndarray = np.frombuffer(msg.data, dtype=np.int32).reshape(
            msg.height, msg.width
        )

        # Vectorised colourmap application:
        #   1. Find unique IDs in this frame (typically << 256 classes).
        #   2. Build an RGB output array by masked assignment.
        # This avoids a Python loop per pixel while keeping memory modest.
        rgb = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
        for uid in np.unique(raw):
            colour = _get_colour(int(uid))
            mask = raw == uid
            rgb[mask] = colour

        out = Image()
        out.header = msg.header  # preserves timestamp and frame_id
        out.height = msg.height
        out.width = msg.width
        out.encoding = "rgb8"
        out.is_bigendian = 0
        out.step = msg.width * 3
        out.data = rgb.tobytes()

        self._publishers[robot_idx].publish(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Isaac Sim 32SC1 semantic segmentation to rgb8 for RViz2."
    )
    parser.add_argument(
        "--num_robots",
        type=int,
        default=1,
        help="Number of robots (= number of topic pairs to bridge). Default: 1",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = SegToRgb8Node(args.num_robots)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
