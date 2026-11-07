"""Camera optical-to-base extrinsic; shared by rendered camera and ROS TF."""
import math
import os
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


def stereo_camera_settings(path=CONFIG_PATH):
    """Pure configuration helper shared by the sensor, TF and graph setup."""
    with Path(path).open(encoding='utf-8') as handle:
        cfg = (yaml.safe_load(handle) or {}).get('stereo_rgb', {})
    enabled = cfg.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('stereo_rgb.enabled must be a boolean')
    if 'STEREO_RGB' in os.environ:
        value = os.environ['STEREO_RGB'].lower()
        if value not in ('0', '1', 'false', 'true'):
            raise ValueError('STEREO_RGB must be 0/1/false/true')
        enabled = value in ('1', 'true')
    baseline = float(os.getenv('STEREO_BASELINE', cfg.get('baseline_m', .12)))
    if not math.isfinite(baseline) or not .02 <= baseline <= .5:
        raise ValueError('stereo baseline must be finite and within [0.02, 0.5] m')
    return enabled, baseline


def camera_profile():
    profile = os.getenv('CAMERA_RIG', 'legacy')
    if profile not in ('legacy', 'd455f_twin612'):
        raise ValueError('CAMERA_RIG must be legacy or d455f_twin612')
    return profile


def front_camera_poses():
    """Always preserve the SRU camera; optional mapping cameras have separate TFs."""
    position, quaternion = front_camera_pose()
    poses = [('front_cam', position, quaternion)]
    enabled, baseline = stereo_camera_settings()
    if not enabled:
        return poses
    transform = load_camera_transform()
    if camera_profile() == 'legacy':
        right = np.asarray(position) + transform[:3, :3] @ np.array([baseline, 0., 0.])
        poses.append(('front_cam_right', tuple(float(x) for x in right), quaternion))
        return poses
    with CONFIG_PATH.open(encoding='utf-8') as handle:
        rig = yaml.safe_load(handle).get('mapping_rig', {})
    # Offsets are provisional mechanical mounting values, never factory calibration.
    left_offset = np.asarray(rig.get('thermal_left_offset_m', [0., 0., .1]), dtype=float)
    depth_offset = np.asarray(rig.get('depth_offset_m', [0., 0., 0.]), dtype=float)
    for value in (left_offset, depth_offset):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('mapping camera offsets must contain three finite values')
    left = np.asarray(position) + left_offset
    right = left + transform[:3, :3] @ np.array([baseline, 0., 0.])
    poses.extend([
        ('d455f_depth', tuple(np.asarray(position) + depth_offset), quaternion),
        ('twin612_left', tuple(left), quaternion),
        ('twin612_right', tuple(right), quaternion),
    ])
    return poses


def camera_sensor_parameters(name):
    """Pinhole reference profiles; exposure/noise/thermal physics are not simulated."""
    if name in ('front_cam', 'front_cam_right'):
        return dict(width=640, height=360, focal_length=24., horizontal_aperture=20.955,
                    vertical_aperture=None if name == 'front_cam' else 20.955*360/640, min_range=.1, max_range=30.)
    if name == 'd455f_depth':
        # Half-resolution rendering of the 1280x720 HD depth geometry, not a
        # claim that 640x360 is a native D455f streaming mode.
        focal = 12.
        return dict(width=640, height=360, focal_length=focal,
                    horizontal_aperture=2*focal*math.tan(math.radians(87/2)),
                    vertical_aperture=2*focal*math.tan(math.radians(58/2)),
                    min_range=.6, max_range=6.)
    if name in ('twin612_left', 'twin612_right'):
        with CONFIG_PATH.open(encoding='utf-8') as handle:
            cfg = yaml.safe_load(handle).get('mapping_rig', {})
        lens = float(os.getenv('THERMAL_LENS_MM', cfg.get('thermal_lens_mm', 25.)))
        if not math.isfinite(lens) or not 6 <= lens <= 100:
            raise ValueError('THERMAL_LENS_MM must be finite and within [6,100] mm')
        return dict(width=640, height=512, focal_length=lens,
                    horizontal_aperture=640*.012, vertical_aperture=512*.012,
                    min_range=.1, max_range=30.)
    raise ValueError('Unknown camera profile: '+name)
