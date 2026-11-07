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

import os
import time
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState, Image, CameraInfo
from core.robot_profiles import robot_namespace, robot_type, b2w_camera_pose
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped, Vector3, Quaternion
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from go2_interfaces.msg import Go2State
from std_msgs.msg import Header, String
import json
import math
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs_py import point_cloud2

from isaaclab.sensors import CameraCfg, Camera
import omni.kit.commands
import omni.replicator.core as rep
import isaaclab.sim as sim_utils

from pxr import UsdGeom, UsdPhysics, Gf, Sdf, UsdPhysics
from tf_transformations import quaternion_from_euler

UnitreeL1_translation = (0.293, 0.0, -0.08)
UnitreeL1_quat = quaternion_from_euler(0, 165 * 3.14159265 / 180, 0) # 165° y axis

ExtraLidar_translation = (0.15, 0.0, 0.18)
ExtraLidar_quat = quaternion_from_euler(0, 0, 0) 

def add_rtx_lidar(num_envs, robot_type, lidar_type, debug=False):
    if lidar_type == "UnitreeL1":
        trans = UnitreeL1_translation
        quat = UnitreeL1_quat
        config = "Unitree_L1"
    if lidar_type == "Extra":
        trans = ExtraLidar_translation
        quat = ExtraLidar_quat
        config = "Unitree_L1_old"
    
    annotator_lst = []
    for i in range(num_envs):
        if robot_type == "g1":
            lidar_sensor = LidarRtx(
                f"/World/envs/env_{i}/Robot/head_link/lidar_sensor",
                rotation_frequency=200,
                pulse_time=1,
                translation=(0.0, 0.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
                config_file_name="Unitree_L1",
            )

        else:
            _, lidar_sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=f"/World/envs/env_{i}/Robot/base/lidar_sensor",
                parent=None,
                translation=trans,
                orientation=Gf.Quatd(quat[3], quat[0], quat[1], quat[2]),
                config=config,
            )
            # if lidar_type == "Extra":
            #     attach_usd_to_sensor(lidar_sensor.GetPath(), "./lidar/os2_mesh.usd")

        lidar_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="UnitreeL1")
        if debug:
            # Create the debug draw pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([lidar_texture])

        # writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        # writer.initialize(topicName=f"{robot_namespace(i)}/point_cloud2", frameId=f"{robot_namespace(i)}/base_link")
        # writer.attach([lidar_texture])

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        annotator.attach(lidar_texture)

        annotator_info = {
            "type": lidar_type,
            "annotator_object": annotator
        }

        annotator_lst.append(annotator_info)
    
    return annotator_lst

def attach_usd_to_sensor(sensor_path: str, usd_path: str, visible=True):
    stage = omni.usd.get_context().get_stage()
    visual_path = f"{sensor_path}/visual"
    UsdGeom.Xform.Define(stage, Sdf.Path(visual_path))
    mesh_prim = stage.DefinePrim(Sdf.Path(f"{visual_path}/mesh"), "Xform")
    mesh_prim.GetReferences().AddReference(usd_path)
    
    UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr(False)
    
    # rt_vis_api = UsdGeom.Tokens
    mesh_prim.CreateAttribute("visibility:raytracing:camera", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:transmission", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:shadow", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:diffuse", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:glossy", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:specular", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:scatter", Sdf.ValueTypeNames.Token).Set("invisible")


    if visible:
        path = Sdf.Path(sensor_path)
        prim = stage.GetPrimAtPath(path)
        prim.GetAttribute("visibility").Set("inherited")

from core.camera_extrinsic import front_camera_pose, front_camera_poses, camera_sensor_parameters


def add_camera(num_envs):
    """Create the unchanged front camera and optional parallel right RGB eye."""
    cam_configs = front_camera_poses()

    for i in range(num_envs):
        for cam_name, pos, rot in cam_configs:
            params = camera_sensor_parameters(cam_name)
            if cam_name != 'front_cam':
                # Auxiliary cameras are USD cameras consumed by the existing
                # OmniGraph render product, avoiding an unused IsaacLab product.
                import torch
                from isaaclab.utils.math import convert_camera_frame_orientation_convention
                orientation = convert_camera_frame_orientation_convention(
                    torch.tensor([rot]), origin='ros', target='opengl')[0].tolist()
                lens = sim_utils.PinholeCameraCfg(
                    focal_length=params['focal_length'],
                    horizontal_aperture=params['horizontal_aperture'],
                    vertical_aperture=params['vertical_aperture'],
                    clipping_range=(params['min_range'], params['max_range']))
                path = f"/World/envs/env_{i}/Robot/base/{cam_name}"
                lens.func(path, lens, translation=pos, orientation=tuple(orientation))
                continue
            cameraCfg = CameraCfg(
                prim_path=f"/World/envs/env_{i}/Robot/base/{cam_name}",
                update_period=0.1,
                # Reduced from 480 × 640 to match OmniGraph render product (640 × 360).
                # 1280×720 @ ~5 Hz is too heavy for RTX; 640×360 cuts pixel count by 75%.
                height=params['height'],
                width=params['width'],
                # Only "rgb" here. Isaac Sim 4.5 has a bug: adding "semantic_segmentation"
                # to data_types forces the robot USD from instanceable → non-instanceable,
                # which deletes collision prims mid-simulation and invalidates the PhysX
                # tensors simulationView (causes hang/crash).
                # Semantic segmentation is published via OmniGraph (ROS2CameraHelper with
                # type="semantic_segmentation"), which reads directly from the RTX render
                # pipeline and is completely independent of this IsaacLab data_types list.
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=params['focal_length'],
                    focus_distance=400.0,
                    horizontal_aperture=params['horizontal_aperture'],
                    vertical_aperture=params['vertical_aperture'],
                    clipping_range=(params['min_range'], params['max_range']),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=pos,
                    rot=rot,
                    convention="ros",
                ),
            )

            Camera(cameraCfg)

def pub_robo_data_ros2(
    robot_type,
    num_envs,
    base_node,
    env,
    annotator_lst,
    base_commands=None,
    timestamp_seconds=None,
    requested_commands=None,
    applied_commands=None,
    terrain_results=None,
):
    # Single timestamp for the entire frame so TF and all sensor messages
    # are consistent — prevents RViz2 from extrapolating transforms.
    if timestamp_seconds is None:
        stamp = base_node.get_clock().now().to_msg()
    else:
        stamp = Time()
        stamp.sec = int(timestamp_seconds)
        stamp.nanosec = int((float(timestamp_seconds) - stamp.sec) * 1.0e9)

    if robot_type == 'b2w':
        clock = Clock()
        clock.clock = stamp
        base_node.clock_pub.publish(clock)

    for i in range(num_envs):
        base_node.publish_joints(
            env.unwrapped.scene["robot"].data.joint_names,
            env.unwrapped.scene["robot"].data.joint_pos[i],
            i,
            stamp,
        )
        base_node.publish_odom(
            env.unwrapped.scene["robot"].data.root_state_w[i, :3],
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            env.unwrapped.scene["robot"].data.root_lin_vel_b[i],
            i,
            stamp,
            base_ang_vel_b=env.unwrapped.scene["robot"].data.root_ang_vel_b[i],
        )
        if base_commands is not None:
            cmd = base_commands.get(str(i), [0.0, 0.0, 0.0])
            base_node.publish_cmd(cmd, i, stamp)
        if requested_commands is not None and applied_commands is not None:
            base_node.publish_control_commands(requested_commands[i], applied_commands[i], i, stamp)
        if terrain_results:
            result = terrain_results[i]
            details = result.as_dict()
            details['query_timestamp'] = details.pop('timestamp')
            details['timestamp'] = stamp.sec+stamp.nanosec*1e-9
            def clean(value):
                if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)): return [clean(v) for v in value]
                if isinstance(value,float) and not math.isfinite(value): return None
                return value
            msg=String();msg.data=json.dumps(clean(details))
            base_node.terrain_diagnostic_pub[i].publish(msg)
        base_node.publish_imu(
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            env.unwrapped.scene["robot"].data.projected_gravity_b[i, :],
            env.unwrapped.scene["robot"].data.root_ang_vel_b[i, :],
            i,
            stamp,
        )

        if robot_type == "go2":
            base_node.publish_robot_state(
                [
                    env.unwrapped.scene["contact_forces"].data.net_forces_w[i][4][2],
                    env.unwrapped.scene["contact_forces"].data.net_forces_w[i][8][2],
                    env.unwrapped.scene["contact_forces"].data.net_forces_w[i][14][2],
                    env.unwrapped.scene["contact_forces"].data.net_forces_w[i][18][2],
                ],
                i,
            )

        if robot_type == "b2w":
            base_node.publish_b2w_depth(env.unwrapped.scene["raycast_camera"], i, stamp)

        try:
            if not annotator_lst:
                continue
            for lidar_id in range(2):
                index_annotator = (i * 2) + lidar_id
                base_node.publish_lidar(annotator_lst[index_annotator], i, stamp)
        except Exception as e:
            print(f"Erro ao publicar LiDAR para o ambiente {i}: {e}")


class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__(f"{robot_type()}_driver_node")
        qos_profile = QoSProfile(depth=10)
        if robot_type() == 'b2w':
            self.clock_pub = self.create_publisher(Clock, '/clock', 1)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_L1_pub = []
        self.go2_lidar_extra_pub = []
        self.odom_pub = []
        self.imu_pub = []
        self.requested_command_pub = []
        self.applied_command_pub = []
        self.terrain_diagnostic_pub = []
        self.depth_pub = []
        self.camera_info_pub = []
        self.last_depth_time = {}

        for i in range(num_envs):
            if robot_type() == "b2w":
                self.depth_pub.append(self.create_publisher(Image, f"{robot_namespace(i)}/front_cam/depth", 1))
                self.camera_info_pub.append(self.create_publisher(CameraInfo, f"{robot_namespace(i)}/front_cam/camera_info", 1))
            self.terrain_diagnostic_pub.append(self.create_publisher(
                String, f'{robot_namespace(i)}/velocity_diagnostics', qos_profile))
            self.requested_command_pub.append(self.create_publisher(
                TwistStamped, f"{robot_namespace(i)}/requested_command", qos_profile))
            self.applied_command_pub.append(self.create_publisher(
                TwistStamped, f"{robot_namespace(i)}/controller_command", qos_profile))
            self.joint_pub.append(
                self.create_publisher(JointState, f"{robot_namespace(i)}/joint_states", qos_profile)
            )
            if robot_type() == 'go2':
                self.go2_state_pub.append(
                    self.create_publisher(Go2State, f"{robot_namespace(i)}/go2_states", qos_profile)
                )
            self.odom_pub.append(
                self.create_publisher(Odometry, f"{robot_namespace(i)}/odom", qos_profile)
            )
            self.imu_pub.append(
                self.create_publisher(Imu, f"{robot_namespace(i)}/imu", qos_profile)
            )
            if robot_type() == 'go2':
                self.go2_lidar_L1_pub.append(
                    self.create_publisher(
                        PointCloud2, f"{robot_namespace(i)}/point_cloud2_L1", qos_profile
                    )
                )
                self.go2_lidar_extra_pub.append(
                    self.create_publisher(
                        PointCloud2, f"{robot_namespace(i)}/point_cloud2_extra", qos_profile
                    )
                )

        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Driving command publishers — one per environment
        self.cmd_pub = []
        for i in range(num_envs):
            self.cmd_pub.append(
                self.create_publisher(Twist, f"{robot_namespace(i)}/base_command", qos_profile)
            )

        # Broadcast the same configurable base-to-camera transform used by
        # CameraCfg so rendered data and ROS TF cannot diverge.
        poses = [('front_cam', *b2w_camera_pose())] if robot_type() == 'b2w' else front_camera_poses()
        static_tfs = []
        for i in range(num_envs):
            for name, camera_position, camera_rotation in poses:
                qw, qx, qy, qz = camera_rotation
                cam_tf = TransformStamped()
                cam_tf.header.stamp = self.get_clock().now().to_msg()
                cam_tf.header.frame_id = f"{robot_namespace(i)}/base_link"
                cam_tf.child_frame_id = f"{robot_namespace(i)}/{name}"
                cam_tf.transform.translation.x = camera_position[0]
                cam_tf.transform.translation.y = camera_position[1]
                cam_tf.transform.translation.z = camera_position[2]
                cam_tf.transform.rotation.x = qx
                cam_tf.transform.rotation.y = qy
                cam_tf.transform.rotation.z = qz
                cam_tf.transform.rotation.w = qw
                static_tfs.append(cam_tf)
        self.static_broadcaster.sendTransform(static_tfs)

    def publish_b2w_depth(self, camera, index, stamp):
        now = stamp.sec + stamp.nanosec * 1e-9
        previous = self.last_depth_time.get(index)
        if previous is not None and 0 <= now - previous < 0.199:
            return
        self.last_depth_time[index] = now
        depth = camera.data.output["distance_to_image_plane"][index].detach().cpu().numpy().reshape(40, 64)
        depth = np.ascontiguousarray(depth, dtype='<f4')
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = f"{robot_namespace(index)}/front_cam"
        image.height, image.width = depth.shape
        image.encoding = '32FC1'
        image.is_bigendian = False
        image.step = image.width * 4
        image.data = depth.tobytes()
        self.depth_pub[index].publish(image)
        info = CameraInfo()
        info.header = image.header
        info.height, info.width = image.height, image.width
        info.distortion_model = 'plumb_bob'
        info.d = [0.0]*5
        fx, fy, cx, cy = 72.7025/3, 72.7025/3, 94.4457/3, 62.5424/3
        info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
        info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        info.p = [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]
        self.camera_info_pub[index].publish(info)

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num, stamp):
        joint_state = JointState()
        joint_state.header.stamp = stamp

        joint_state_names_formated = []
        for joint_name in joint_names_lst:
            joint_state_names_formated.append(f"{robot_namespace(robot_num)}/" + joint_name)

        joint_state_formated = []
        for joint_state_val in joint_state_lst:
            joint_state_formated.append(joint_state_val.item())

        joint_state.name = joint_state_names_formated
        joint_state.position = joint_state_formated
        self.joint_pub[robot_num].publish(joint_state)

    def publish_odom(self, base_pos, base_rot, base_lin_vel_b, robot_num, stamp, base_ang_vel_b=None):
        now = stamp

        odom_trans = TransformStamped()
        odom_trans.header.stamp = now
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = f"{robot_namespace(robot_num)}/base_link"
        odom_trans.transform.translation.x = base_pos[0].item()
        odom_trans.transform.translation.y = base_pos[1].item()
        odom_trans.transform.translation.z = base_pos[2].item()
        odom_trans.transform.rotation.x = base_rot[1].item()
        odom_trans.transform.rotation.y = base_rot[2].item()
        odom_trans.transform.rotation.z = base_rot[3].item()
        odom_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(odom_trans)

        if robot_type() == 'go2':
            UnitreeL1_trans = TransformStamped()
            UnitreeL1_trans.header.stamp = now
            UnitreeL1_trans.header.frame_id = f"{robot_namespace(robot_num)}/base_link"
            UnitreeL1_trans.child_frame_id = f"{robot_namespace(robot_num)}/UnitreeL1_link"
            UnitreeL1_trans.transform.translation = Vector3(x=UnitreeL1_translation[0], y=UnitreeL1_translation[1], z=UnitreeL1_translation[2])
            UnitreeL1_trans.transform.rotation = Quaternion(x=UnitreeL1_quat[0], y=UnitreeL1_quat[1], z=UnitreeL1_quat[2], w=UnitreeL1_quat[3])
            self.broadcaster.sendTransform(UnitreeL1_trans)

            lidar_trans = TransformStamped()
            lidar_trans.header.stamp = now
            lidar_trans.header.frame_id = f"{robot_namespace(robot_num)}/base_link"
            lidar_trans.child_frame_id = f"{robot_namespace(robot_num)}/lidar_link"
            lidar_trans.transform.translation = Vector3(x=ExtraLidar_translation[0], y=ExtraLidar_translation[1], z=ExtraLidar_translation[2])
            lidar_trans.transform.rotation = Quaternion(x=ExtraLidar_quat[0], y=ExtraLidar_quat[1], z=ExtraLidar_quat[2], w=ExtraLidar_quat[3])
            self.broadcaster.sendTransform(lidar_trans)
        odom_topic = Odometry()
        odom_topic.header.stamp = now
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = f"{robot_namespace(robot_num)}/base_link"
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        # Linear velocity in body frame (vx = forward speed = ego_speed)
        odom_topic.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom_topic.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom_topic.twist.twist.linear.z = base_lin_vel_b[2].item()
        if base_ang_vel_b is not None:
            odom_topic.twist.twist.angular.x = base_ang_vel_b[0].item()
            odom_topic.twist.twist.angular.y = base_ang_vel_b[1].item()
            odom_topic.twist.twist.angular.z = base_ang_vel_b[2].item()
        self.odom_pub[robot_num].publish(odom_topic)

    def publish_control_commands(self, requested, applied, robot_num, stamp):
        # Stamp with the resulting odometry frame. This command was held over
        # the just-completed control interval, not read again from the ROS buffer.
        for command, publisher in ((requested, self.requested_command_pub[robot_num]),
                                   (applied, self.applied_command_pub[robot_num])):
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = f"{robot_namespace(robot_num)}/base_link"
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = map(float, command)
            publisher.publish(msg)

    def publish_imu(self, base_rot, gravity_b, base_ang_vel, robot_num, stamp):
        imu_trans = Imu()
        imu_trans.header.stamp = stamp
        imu_trans.header.frame_id = f"{robot_namespace(robot_num)}/base_link"

        # projected_gravity_b is the unit gravity vector in body frame (points downward).
        # A real IMU accelerometer measures specific force = a_body - g_body.
        # When stationary: a_meas = -g_body = -9.81 * projected_gravity_b.
        # Result: upright robot → [0, 0, +9.81], matching real hardware.
        G = 9.81
        imu_trans.linear_acceleration.x = -G * gravity_b[0].item()
        imu_trans.linear_acceleration.y = -G * gravity_b[1].item()
        imu_trans.linear_acceleration.z = -G * gravity_b[2].item()

        imu_trans.angular_velocity.x = base_ang_vel[0].item()
        imu_trans.angular_velocity.y = base_ang_vel[1].item()
        imu_trans.angular_velocity.z = base_ang_vel[2].item()

        imu_trans.orientation.x = base_rot[1].item()
        imu_trans.orientation.y = base_rot[2].item()
        imu_trans.orientation.z = base_rot[3].item()
        imu_trans.orientation.w = base_rot[0].item()

        self.imu_pub[robot_num].publish(imu_trans)

    def publish_cmd(self, base_cmd, robot_num, stamp):
        """Publish the current high-level velocity command [vx, vy, wz] as Twist.

        Topic: robot{N}/base_command  (geometry_msgs/Twist)
        This is recorded by the data collector as 'driving_command'.
        """
        twist = Twist()
        twist.linear.x = float(base_cmd[0])
        twist.linear.y = float(base_cmd[1])
        twist.angular.z = float(base_cmd[2])
        self.cmd_pub[robot_num].publish(twist)

    def publish_robot_state(self, foot_force_lst, robot_num):

        go2_state = Go2State()
        go2_state.foot_force = [
            int(foot_force_lst[0].item()),
            int(foot_force_lst[1].item()),
            int(foot_force_lst[2].item()),
            int(foot_force_lst[3].item()),
        ]
        self.go2_state_pub[robot_num].publish(go2_state)

    def publish_lidar(self, data, robot_num, stamp):
        pts_raw = data["annotator_object"].get_data()["data"]

        if pts_raw is None or pts_raw.size == 0:
            return

        if data["type"] == "UnitreeL1":
            # Publish raw annotator data in the sensor's own frame.
            # TF (base_link → UnitreeL1_link) handles the mounting transform
            # for RViz2; Point-LIO uses extrinsic_T/R in the YAML for SLAM.
            frame_id       = f"{robot_namespace(robot_num)}/UnitreeL1_link"
            publisher      = self.go2_lidar_L1_pub[robot_num]
            scan_period    = 1.0 / 200.0   # Unitree L1: 200 Hz rotation
            n_scans        = 18            # must match yaml scan_line
        elif data["type"] == "Extra":
            frame_id       = f"{robot_namespace(robot_num)}/lidar_link"
            publisher      = self.go2_lidar_extra_pub[robot_num]
            scan_period    = 1.0 / 200.0
            n_scans        = 18
        else:
            return

        # --- extract x, y, z, intensity, ring, time --------------------------
        # RTX annotator returns a structured numpy array with named fields:
        # (x, y, z, d, v, intensity, emitterId, beamId, tick, proc, flags).
        # Fall back to plain float array if dtype has no names.
        if pts_raw.dtype.names:
            x   = pts_raw["x"].astype(np.float32)
            y   = pts_raw["y"].astype(np.float32)
            z   = pts_raw["z"].astype(np.float32)
            intensity = (
                pts_raw["intensity"].astype(np.float32)
                if "intensity" in pts_raw.dtype.names
                else np.ones(len(x), dtype=np.float32)
            )
            # Clamp ring to [0, n_scans-1].
            # Point-LIO indexes internal per-ring vectors with this value;
            # any ring >= N_SCANS causes an out-of-bounds access → segfault.
            # Isaac Sim beamId is not guaranteed to be in [0, scan_line-1].
            ring = (
                (pts_raw["beamId"] % n_scans).astype(np.uint16)
                if "beamId" in pts_raw.dtype.names
                else np.zeros(len(x), dtype=np.uint16)
            )
            # Use hardware tick to compute per-point relative timestamp.
            # tick is a monotonic counter; normalize to [0, scan_period] seconds.
            if "tick" in pts_raw.dtype.names:
                ticks      = pts_raw["tick"].astype(np.float64)
                tick_range = ticks.max() - ticks.min()
                if tick_range > 0:
                    # Normalize tick to [0, scan_period] in seconds.
                    # yaml timestamp_unit=0 (SEC) → time_unit_scale=1000,
                    # so Point-LIO curvature = time_sec * 1000 (ms).
                    time_f = ((ticks - ticks.min()) / tick_range * scan_period).astype(np.float32)
                else:
                    time_f = np.linspace(0.0, scan_period, len(x),
                                         endpoint=False, dtype=np.float32)
            else:
                time_f = np.linspace(0.0, scan_period, len(x),
                                     endpoint=False, dtype=np.float32)
        else:
            flat      = pts_raw.reshape(-1, pts_raw.shape[-1])
            x         = flat[:, 0].astype(np.float32)
            y         = flat[:, 1].astype(np.float32)
            z         = flat[:, 2].astype(np.float32)
            intensity = np.ones(len(x), dtype=np.float32)
            ring   = np.zeros(len(x), dtype=np.uint16)
            # Use linspace so time values are strictly increasing.
            # All-zero time causes time_compressing() to return [N], which
            # makes Point-LIO access feats_down_body->points[N] → segfault.
            time_f = np.linspace(0.0, scan_period, len(x),
                                 endpoint=False, dtype=np.float32)

        N = len(x)

        # PointXYZIRT memory layout expected by Point-LIO / FAST-LIO2:
        #   x(4) y(4) z(4) intensity(4) ring(2) pad(2) time(4) → 24 bytes/pt
        POINT_STEP = 24
        buf = np.zeros((N, POINT_STEP), dtype=np.uint8)
        buf[:, 0:4]   = x.view(np.uint8).reshape(N, 4)
        buf[:, 4:8]   = y.view(np.uint8).reshape(N, 4)
        buf[:, 8:12]  = z.view(np.uint8).reshape(N, 4)
        buf[:, 12:16] = intensity.view(np.uint8).reshape(N, 4)
        buf[:, 16:18] = ring.view(np.uint8).reshape(N, 2)
        buf[:, 20:24] = time_f.view(np.uint8).reshape(N, 4)

        fields = [
            PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring",      offset=16, datatype=PointField.UINT16,  count=1),
            PointField(name="time",      offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = PointCloud2()
        pc_msg.header.stamp    = stamp
        pc_msg.header.frame_id = frame_id
        pc_msg.height          = 1
        pc_msg.width           = N
        pc_msg.fields          = fields
        pc_msg.is_bigendian    = False
        pc_msg.point_step      = POINT_STEP
        pc_msg.row_step        = N * POINT_STEP
        pc_msg.is_dense        = True
        pc_msg.data            = buf.tobytes()

        publisher.publish(pc_msg)
