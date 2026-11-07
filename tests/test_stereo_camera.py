"""Isaac-independent camera/TF/OmniGraph configuration contract tests."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.camera_extrinsic import front_camera_pose, front_camera_poses, stereo_camera_settings, load_camera_transform, camera_sensor_parameters


class StereoCameraTests(unittest.TestCase):
    def test_disabled_preserves_existing_camera(self):
        with patch.dict(os.environ, {'STEREO_RGB': '0'}):
            poses = front_camera_poses()
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0], ('front_cam', *front_camera_pose()))

    def test_physical_baseline_and_orientation_match_tf_configuration(self):
        with patch.dict(os.environ, {'STEREO_RGB': '1', 'STEREO_BASELINE': '.12', 'CAMERA_RIG': 'legacy'}):
            left, right = front_camera_poses()
        rotation = load_camera_transform()[:3, :3]
        delta = np.asarray(right[1])-left[1]
        np.testing.assert_allclose(delta, rotation @ [.12, 0, 0], atol=1e-12)
        self.assertEqual(left[2], right[2])
        self.assertAlmostEqual(np.linalg.norm(delta), .12)
        self.assertAlmostEqual(delta[1], -.12)

    def test_hardware_rig_preserves_sru_and_separates_sensor_geometry(self):
        with patch.dict(os.environ, {'STEREO_RGB': '1', 'STEREO_BASELINE': '.12', 'CAMERA_RIG': 'd455f_twin612'}):
            poses = front_camera_poses()
        self.assertEqual(poses[0], ('front_cam', *front_camera_pose()))
        self.assertEqual([p[0] for p in poses], ['front_cam', 'd455f_depth', 'twin612_left', 'twin612_right'])
        self.assertAlmostEqual(np.linalg.norm(np.asarray(poses[3][1])-poses[2][1]), .12)
        depth = camera_sensor_parameters('d455f_depth')
        thermal = camera_sensor_parameters('twin612_left')
        self.assertEqual((depth['width'], depth['height']), (640, 360))
        self.assertEqual((thermal['width'], thermal['height']), (640, 512))
        self.assertEqual(thermal['horizontal_aperture'], 7.68)
        self.assertEqual(thermal['vertical_aperture'], 6.144)
        self.assertAlmostEqual(np.degrees(2*np.arctan(depth['horizontal_aperture']/2/depth['focal_length'])), 87.)
        self.assertEqual(depth['max_range'], 6.)
        self.assertEqual(camera_sensor_parameters('front_cam')['focal_length'], 24.)

    def test_invalid_thermal_lens_rejected(self):
        for value in ('nan', '-1', '1000'):
            with patch.dict(os.environ, {'THERMAL_LENS_MM': value}):
                with self.assertRaises(ValueError):
                    camera_sensor_parameters('twin612_left')

    def test_invalid_baseline_rejected(self):
        for value in ('nan', '0', '-.1', '2'):
            with patch.dict(os.environ, {'STEREO_BASELINE': value}):
                with self.assertRaises(ValueError):
                    stereo_camera_settings()

    def test_invalid_enable_flag_rejected(self):
        with patch.dict(os.environ, {'STEREO_RGB': 'perhaps'}):
            with self.assertRaises(ValueError):
                stereo_camera_settings()

    def test_graph_right_eye_is_rgb_only_and_uses_sim_stamps(self):
        records = []
        class Controller:
            Keys = types.SimpleNamespace(CREATE_NODES='create', SET_VALUES='values', CONNECT='connect')
            @staticmethod
            def edit(graph, body):
                records.append((graph, body))
        omni = types.ModuleType('omni')
        graph = types.ModuleType('omni.graph')
        core = types.ModuleType('omni.graph.core')
        core.Controller = Controller
        core.GraphPipelineStage = types.SimpleNamespace(GRAPH_PIPELINE_STAGE_SIMULATION=1)
        omni.graph = graph
        graph.core = core
        with patch.dict(sys.modules, {'omni': omni, 'omni.graph': graph, 'omni.graph.core': core}):
            spec = importlib.util.spec_from_file_location('stereo_graph_contract', ROOT/'core/omnigraph.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.dict(os.environ, {'STEREO_RGB': '1', 'CAMERA_RIG': 'legacy'}):
                module.create_front_cam_omnigraph(0)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0][0]['graph_path'], records[1][0]['graph_path'])
        left, right = records[0][1], records[1][1]
        self.assertIn('DepthPublisher', dict(left['create']))
        self.assertIn('SemanticPublisher', dict(left['create']))
        self.assertNotIn('DepthPublisher', dict(right['create']))
        self.assertNotIn('SemanticPublisher', dict(right['create']))
        values = dict(right['values'])
        self.assertEqual(values['RGBPublisher.inputs:frameId'], 'robot0/front_cam_right')
        self.assertEqual(values['RGBPublisher.inputs:topicName'], 'robot0/front_cam_right/rgb')
        self.assertIs(values['RGBPublisher.inputs:useSystemTime'], False)
        self.assertIs(values['CameraInfoPublisher.inputs:useSystemTime'], False)
        self.assertEqual(values['IsaacCreateRenderProduct.inputs:width'], 640)
        self.assertEqual(values['IsaacCreateRenderProduct.inputs:height'], 360)

if __name__ == '__main__':
    unittest.main()
