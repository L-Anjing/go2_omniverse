# Same scene, selectable robot, external navigation

The supported simulator entry is `commands/run_sim.sh` -> `main.py` ->
`core/omniverse_sim.py`. It loads the existing Hospital scene and a selected
robot. No goal selection or SRU navigation policy runs in this entry.

Stop the previous simulator before starting another instance:

```bash
cd /media/user/data1/carl/workspace/go2_omniverse
ROBOT_TYPE=b2w bash commands/run_sim.sh
# Original Go2 configuration:
ROBOT_TYPE=go2 bash commands/run_sim.sh
```

One robot type is selected per process; `ROBOT_AMOUNT` replicates that type.
Mixed Go2/B2W spawning within one process is not implemented. The existing
Go2 topics stay `/robot0/...`; B2W topics and frames use `/b2w0/...`.
`CMD_SOURCE=ros2` remains the default. B2W stops accepting a held velocity
when no new command arrives for one simulation second. `MAX_STEPS` may be
used for a finite smoke test; no simulator has been started by these edits.

| Interface | Go2 | B2W |
| --- | --- | --- |
| Velocity input, Twist | `/robot0/cmd_vel` | `/b2w0/cmd_vel` |
| Odometry, body-frame twist | `/robot0/odom` | `/b2w0/odom` |
| IMU | `/robot0/imu` | `/b2w0/imu` |
| Depth | `/robot0/front_cam/depth` | `/b2w0/front_cam/depth` |
| CameraInfo | `/robot0/front_cam/camera_info` | `/b2w0/front_cam/camera_info` |
| Base frame | `robot0/base_link` | `b2w0/base_link` |
| Optical frame | `robot0/front_cam` | `b2w0/front_cam` |

Both use world/odom frame `odom`. Odometry includes angular velocity.
B2W also publishes `/clock`; its deployment uses simulation time so slower
rendering does not cause wall-clock input timeouts between valid depth frames.
B2W publishes metric `32FC1` depth at 64x40, 5 Hz in simulation time,
using a static RayCasterCamera with the SRU training intrinsics and pose:
fx=fy=72.7025/3, cx=94.4457/3, cy=62.5424/3,
position `(0.387,0,0.28)` m and downward pitch 20 degrees.
Depth range is 11 m at the sensor; invalid/out-of-range filtering to
0.25--10 m and VAE encoding belong to deployment.
This static-scene depth sensor does not observe moving robots/objects.
B2W also mounts an RTX RGB camera under `Robot/base_link/front_cam`, using
the depth sensor's position, rotation, lens aperture and focal length directly.
RGB is 640x400 (the same 1.6 aspect ratio and nominal FOV as 64x40 depth),
published on `/b2w0/front_cam/rgb`. Its separate calibration topic is
`/b2w0/front_cam/rgb/camera_info`, so the depth CameraInfo is not overwritten.
Both use the optical frame `b2w0/front_cam`. The RGB camera is available in
the Isaac viewport camera selector and in RViz's Front RGB display. It is
observation-only; SRU still consumes raycast depth. RGB/render timing and
raycast depth are not a synchronized RGB-D pair. Pixel-perfect registration
has not been verified in the live renderer. Isaac Sim 4.5 reports that aperture
offsets are unsupported and ignores them for RGB; shared pose and nominal FOV
do not imply pixel-perfect depth/RGB registration. The depth camera retains
its training principal point. LiDAR is not added to B2W.
Go2 sensors are unchanged.

## Low Follow View

B2W adds a separate viewport camera at `Robot/base_link/follow_cam`.
It is mounted 0.6 m behind the torso's rear bound and 0.4 m above its top
bound, looking forward with 10 degrees downward pitch relative to the body.
It inherits body movement/rotation through USD parenting, with no per-frame
PhysX tracking callbacks. The GUI selects this view by default; headless runs
create the camera without selecting a viewport. Front RGB and depth are unchanged.

The near distance can crop the robot; seeing the entire robot may require a
larger setback. These optional environment variables adjust the view:

```bash
ROBOT_TYPE=b2w B2W_FOLLOW_DISTANCE=0.8 B2W_FOLLOW_HEIGHT=0.4 \
  B2W_FOLLOW_PITCH_DEG=10 bash commands/run_sim.sh
```

Distances are measured from the torso edges, not the base origin or wheels.
`B2W_FOLLOW_VIEW=0` disables automatic view selection; choose `follow_cam` from
the Isaac viewport camera selector instead. This camera publishes no ROS topic
and is not an input to SRU. Live framing is not verified by the offline tests.

## Local robot and locomotion ownership

Copied from the training assets into this repository:

- `assets/robots/b2w/b2w_rsl.usd`
- `assets/robots/b2w/config.py` (original articulation/actuator settings)
- `checkpoints/b2w/locomotion/policy_b2w_new_2.pt`

`configs/b2w_deploy.py` constructs exactly the 60-feature locomotion input:
linear velocity, angular velocity, gravity, requested body velocity,
16 relative joint positions, 16 joint velocities, 16 previous actions.
The locomotion model outputs 12 leg actions and 4 wheel actions, mapped
through position scale 0.5 and wheel velocity scale 5.0 at 50 Hz.
The model's recurrence is reset for environment resets.
This model is not the SRU navigation checkpoint and receives no depth/goal.

The production entry runs simulation and locomotion only. Navigation policy
inference, SRU evaluation and robot deployment belong to separate projects
connected through ROS2. `run_sru_b2w_hospital.sh` is retained only as a
compatibility alias to the simulator-only B2W entry; it starts no navigation.

## Scene differences corrected

The first experimental B2W entry replaced the terrain with the whole Hospital
USD under `/World/ground`, inherited a training HDR light and automatically
sampled goals. It was not equivalent to the established simulator.

Now both profiles use `core/hospital_scene.py`, extracted from the original
scene loader: Hospital at `/World/hospital`, independent support plane at
`/World/ground`, duplicate CollisionPlane/WorldGrid removal, original semantic
label processing, shared floor friction, original distant/dome lights, and
the same `HIDE_DEFAULT_GROUND` option. The scene USD itself is not edited.
The B2W profile only adds an invisible non-colliding merged mesh for static
depth raycasting; Isaac Lab 2.1 otherwise reads just the first child mesh.
Go2's low-level model and sensor configuration are preserved.

The original GPU selection, ROS setup, rendering preset and reset/spawn path
are shared by using the original shell/main entry. Depth intrinsics and robot
dynamics intentionally differ between profiles. Navigation starts separately
in `sru-robot-deployment`; see that repository's `README_local.md`.

Validation: local locomotion CPU inference/reset, copied-asset checksums,
USD composition inspection, ONNX export/inference in deployment, isolated
ROS input/output tests, shell/Python checks. Rendering, PhysX initialization,
depth appearance and physical navigation require a live simulator test.
The previous PhysX CUDA failure is not claimed fixed by these changes.
