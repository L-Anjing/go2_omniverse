"""Robot-specific ROS names; the Go2 public interface stays unchanged."""
import math
import os


def robot_type():
    value = os.environ.get("SIM_ROBOT_TYPE", "go2")
    if value not in {"go2", "b2w"}:
        raise ValueError(f"Unsupported robot type: {value}")
    return value


def robot_namespace(index):
    return f"b2w{index}" if robot_type() == "b2w" else f"robot{index}"


def b2w_camera_pose():
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    # Base-to-optical = pitch(20 degrees) * ROS optical-axis rotation.
    return (0.387, 0.0, 0.28), (0.5*(c-s), -0.5*(c+s), 0.5*(c+s), -0.5*(c-s))
