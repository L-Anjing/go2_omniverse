"""Camera optical-to-base extrinsic; shared by rendered camera and ROS TF."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'robot_extrinsic.yaml'


def load_camera_transform(path=CONFIG_PATH):
    with Path(path).open(encoding='utf-8') as handle:
        document = yaml.safe_load(handle)
    try:
        extrinsic = document['robot_extrinsic']
        rotation = np.asarray(extrinsic['camera_to_robot_R'], dtype=np.float64)
        translation = np.asarray(extrinsic['camera_to_robot_T'], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError(f'{path}: expected robot_extrinsic.camera_to_robot_R/T') from error
    if rotation.shape != (9,) or translation.shape != (3,):
        raise ValueError(f'{path}: camera_to_robot_R requires 9 row-major values and T requires 3')
    rotation = rotation.reshape(3, 3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError(f'{path}: camera extrinsic must contain finite values')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError(f'{path}: camera rotation must be orthonormal with det(R)=1')
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    return transform


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, ...]:
    """Convert a proper 3×3 rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)



@lru_cache(maxsize=1)
def front_camera_pose():
    """Load once so sensor spawn and static TF always use the same extrinsic."""
    transform = load_camera_transform()
    return tuple(float(v) for v in transform[:3, 3]), _rotation_matrix_to_quaternion_wxyz(transform[:3, :3])
