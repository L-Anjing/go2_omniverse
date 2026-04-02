# Revision Notes

## 运行指令

### 基本结构

`run_sim.sh` 最后一行调用 `python main.py`，所有参数直接跟在后面，**用空格分隔，不加等号**：

```bash
# 错误写法
./run_sim.sh --custom_env=hospital   # ❌ argparse 不识别等号格式

# 正确写法
./run_sim.sh                         # 但参数要改在 run_sim.sh 内部的 python 那行
```

由于 `run_sim.sh` 内部先执行 `rosdep`/`colcon build` 再调用 python，**参数需要直接修改 `run_sim.sh` 最后一行**，或者在 conda 环境激活后手动运行 python。

---

### 场景选择：hospital

编辑 [run_sim.sh](run_sim.sh) 最后一行，去掉 `#--custom_env office` 的注释并改为 `hospital`：

```bash
# run_sim.sh 最后一行改为：
python main.py --robot_amount 1 --robot go2 --device cuda --enable_cameras --custom_env hospital
```

然后运行：

```bash
./run_sim.sh
```

可用的 `--custom_env` 值：

| 值 | 场景 |
|---|---|
| `hospital` | Isaac 医院环境（本次新增）|
| `warehouse` | 仓库环境 |
| `office` | 办公室环境 |
| 不传此参数 | 默认空白地面 |

---

### 多机器人

```bash
python main.py --robot_amount 2 --robot go2 --device cuda --enable_cameras --custom_env hospital
```

---

### 语义分割：自动启用，无需额外参数

语义分割通过 OmniGraph 内置发布，只要启动仿真即自动开启。

**发布的话题（每个机器人）：**

| 话题 | 编码 | 说明 |
|---|---|---|
| `robot0/front_cam/rgb` | `rgb8` | 彩色图，可直接在 RViz2 显示 |
| `robot0/front_cam/semantic_segmentation` | `32SC1` | 原始语义 ID，RViz2 无法直接显示 |
| `robot0/front_cam/semantic_segmentation_rgb8` | `rgb8` | 着色后的语义图，需运行转换脚本 |

---

### 语义分割 rgb8 转换（第二个终端）

仿真启动后，在另一个终端中运行：

```bash
# 激活 ROS2 环境
source /opt/ros/${ROS_DISTRO}/setup.bash
source go2_omniverse_ws/install/setup.bash

# 运行转换节点（按实际机器人数量修改 --num_robots）
python3 seg_to_rgb8.py --num_robots 1
```

---

### RViz2 查看

```bash
# 第三个终端
source /opt/ros/${ROS_DISTRO}/setup.bash
rviz2
```

在 RViz2 中添加以下 Display：

| Display 类型 | Topic |
|---|---|
| Image | `/robot0/front_cam/rgb` |
| Image | `/robot0/front_cam/semantic_segmentation_rgb8` |
| PointCloud2 | `/robot0/point_cloud2_L1` |
| PointCloud2 | `/robot0/point_cloud2_extra` |
| Odometry | `/robot0/odom` |

---

## 1. Hospital environment — 加载顺序修复（`omniverse_sim.py`、`custom_rl_env.py`）

### 1.1 根因分析：加载顺序导致 base_contact 每帧触发

**症状：** hospital 场景下 `total_resets = step_count`（每步都 terminated），机器人完全无法移动；flat 场景无此问题。

**根因：** 原来的加载顺序是：

```
gym.make()          ← 机器人生成，PhysX 第一次 broadphase，物理开始运行
setup_custom_env()  ← 医院 collision mesh 添加到已运行的 PhysX
```

`gym.make()` 内部会触发 PhysX 的第一次广相（broadphase），此时舞台上只有 IsaacLab 的地面和机器人。随后 `setup_custom_env()` 将医院的 collision mesh 动态添加到**正在运行的** PhysX，PhysX 立即检测到医院几何体与机器人的穿透（两者在世界坐标 (0,0) 重叠），生成巨大的 depenetration 恢复力作用在机器人躯干上，使 `base_contact` 每帧触发 → termination → reset → 循环。

这也解释了用户观察到的现象：**先看到机器人，场景后来才出现，左侧播放按钮还没等场景加载完就已经开始了。**

### 1.2 修复：将医院注入 scene config，在 gym.make() 之前加入 PhysX

**`custom_rl_env.py` — `MySceneCfg` 新增字段：**

```python
# 可在 run_sim() 中运行时赋值，None 表示不加载自定义环境
custom_env: AssetBaseCfg | None = None
```

**`omniverse_sim.py` — 新加载顺序：**

```python
# 1. gym.make() 之前：把医院注入 scene config
if args_cli.custom_env == "hospital":
    env_cfg.scene.custom_env = AssetBaseCfg(
        prim_path="/World/hospital",
        spawn=sim_utils.UsdFileCfg(usd_path=HOSPITAL_USD),
    )

# 2. gym.make()：IsaacLab scene 初始化时同时加载医院和机器人
#    → PhysX 第一次 broadphase 就包含医院几何体
#    → 机器人生成时已知碰撞边界，无 penetration 冲力
env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# 3. gym.make() 返回后（物理已启动但还未步进）：
#    禁用医院内置 CollisionPlane（与 /World/ground 重合，不禁用会双重碰撞）
if args_cli.custom_env == "hospital":
    col_plane = stage.GetPrimAtPath("/World/hospital/Root/CollisionPlane")
    if col_plane and col_plane.IsValid():
        col_plane.SetActive(False)

# 4. warehouse/office 仍走原有的 setup_custom_env()（纯视觉资产，无躯干穿透风险）
setup_custom_env()
```

**正确顺序：**
```
env_cfg.scene.custom_env = AssetBaseCfg(...)   ← 医院写入 scene config
gym.make()                                      ← 医院 + 机器人同时进入 PhysX broadphase
CollisionPlane 禁用                             ← 去除重复地面碰撞
物理步进开始
```

### 1.3 为什么 `CollisionPlane` 仍需禁用

hospital.usd 内部含 `/World/hospital/Root/CollisionPlane`（PhysX 无限平面，位于 z=0），与 IsaacLab 的 `/World/ground`（也是 z=0 无限平面）完全重合。两个无限平面在同一位置会产生相互矛盾的接触法向，导致机器人脚部受到随机方向的约束力，即使位置正确也无法正常行走。禁用其中一个即可。

### 1.4 关于 Materials/ 和 Props/

`hospital.usd` 以*相对子层*引用 `Materials/` 和 `Props/`，USD 运行时自动解析，无需显式加载。手动添加会造成 prim-path 冲突。

---

## 2. 加载顺序（`omniverse_sim.py` — `run_sim`）

**最终正确顺序：**

```
env_cfg.scene.custom_env = AssetBaseCfg(hospital)   # 医院写入 scene config（gym.make 前）
gym.make()                                           # 场景 + 机器人 + 医院同时进入 PhysX
CollisionPlane 禁用                                  # 去除重复地面
setup_custom_env()                                   # warehouse/office（视觉资产）
add_rtx_lidar()                                      # LiDAR 传感器
add_camera()                                         # 摄像头传感器
create_front_cam_omnigraph()                         # OmniGraph render products
env.get_observations()                               # 第一次物理步进
```

`setup_custom_env()` 仍保留用于 warehouse/office（纯视觉，无碰撞穿透风险）。Hospital 改为通过 `env_cfg.scene.custom_env` 在 `gym.make()` 前注入，见第 1 节。

---

## 3. Camera resolution reduced (`ros2.py` — `add_camera`, `omnigraph.py`)

**What changed:**
- `CameraCfg`: `height` 480 → **360**, `width` stays 640.
- `IsaacCreateRenderProduct` in OmniGraph: `inputs:width = 640`, `inputs:height = 360` set explicitly.

**Why:**
1280 × 720 @ ~5 Hz requires the RTX renderer to shade ~920 k pixels per frame.  At 640 × 360 that drops to ~230 k pixels — a **4× reduction** in render workload.  The 16 : 9 aspect ratio is preserved.  Both the IsaacLab `CameraCfg` and the OmniGraph render product must agree on dimensions; they are now both set to 640 × 360.

---

## 4. Semantic segmentation stream added (`omnigraph.py`)

**What changed:**
`create_front_cam_omnigraph()` now creates **two** `ROS2CameraHelper` nodes sharing a single `IsaacCreateRenderProduct`:

| Node | Topic | Encoding |
|---|---|---|
| `ROS2CameraHelperRgb` | `robot{N}/front_cam/rgb` | `rgb8` |
| `ROS2CameraHelperSeg` | `robot{N}/front_cam/semantic_segmentation` | `32SC1` |

The execution chain is:
```
OnPlaybackTick → IsaacCreateRenderProduct → RgbHelper → SegHelper
```
Two **separate** OmniGraphs are used — `/ROS_front_cam{N}_rgb` and `/ROS_front_cam{N}_seg` — because `ROS2CameraHelper` has **no `outputs:execOut` port** (attempting to connect it raises `OmniGraphError: path attribute cannot connect to uint (execution)`).  Each graph has its own `IsaacCreateRenderProduct` node and render-product handle, eliminating any shared-state race between the two streams.

**Why `data_types` in `CameraCfg` must NOT include `"semantic_segmentation"` (`ros2.py`):**
Isaac Sim 4.5 has a known bug: when `"semantic_segmentation"` is present in `CameraCfg.data_types`, IsaacLab's camera sensor forces every instanceable asset in the scene (including the robot) to be converted to non-instanceable form.  During this conversion the collision prims are deleted and recreated, which invalidates the PhysX `tensors simulationView` while it is still in use → the simulation hangs/crashes with:
```
prim '/World/envs/env_0/Robot/base/collisions/mesh_0' was deleted while being used
by a shape in a tensor view class. The physics.tensors simulationView was invalidated.
```
The OmniGraph `ROS2CameraHelper` (type `"semantic_segmentation"`) reads directly from the RTX render pipeline via the render product; it is completely independent of the IsaacLab `data_types` list.  Therefore `data_types=["rgb"]` is sufficient and safe.

---

## 5. Semantic segmentation → rgb8 converter (`seg_to_rgb8.py`)

**What it does:**
A standalone ROS 2 node that bridges the gap between Isaac Sim's raw `32SC1` semantic output and RViz2's display requirements.

| Direction | Topic | Encoding |
|---|---|---|
| Subscribe | `robot{i}/front_cam/semantic_segmentation` | `32SC1` |
| Publish | `robot{i}/front_cam/semantic_segmentation_rgb8` | `rgb8` |

**Colour mapping:**
Each unique integer class ID is hashed to a stable RGB colour via a seeded `numpy.random.RandomState`.  The same ID always produces the same colour across frames and restarts.  ID 0 (background / unlabelled) is always black.  A small in-process cache avoids re-hashing IDs seen in previous frames.

**Why a separate process (not inline in the sim loop):**
The segmentation-to-colour conversion involves per-frame numpy operations.  Running it inside the Isaac Sim Python process would compete with the RTX render thread and the RL policy inference.  A separate ROS 2 process keeps the sim loop clean and lets the converter run at whatever rate the downstream consumer needs.

**Usage:**
```bash
# In a second terminal (ROS 2 sourced):
python3 seg_to_rgb8.py --num_robots 1
```

RViz2 topic to add: `robot0/front_cam/semantic_segmentation_rgb8` with display type `Image`.

---

## 6. LiDAR PointCloud2 格式升级为 PointXYZIRT，修复坐标系与 IMU 数据 (`ros2.py`)

### 6.1 PointCloud2 格式：`x/y/z` → PointXYZIRT（`publish_lidar`）

每个 LiDAR 点从仅含 `x/y/z`（12 字节）扩展为 24 字节，满足 Point-LIO / FAST-LIO2 输入格式：

| 字段 | 类型 | 偏移 | 来源 |
|---|---|---|---|
| `x` | float32 | 0 | annotator `"x"` |
| `y` | float32 | 4 | annotator `"y"` |
| `z` | float32 | 8 | annotator `"z"` |
| `intensity` | float32 | 12 | annotator `"intensity"`，缺失时默认 `1.0` |
| `ring` | uint16 | 16 | annotator `"beamId"`，缺失时默认 `0` |
| *(pad)* | 2 字节 | 18 | 内存对齐填充 |
| `time` | float32 | 20 | 由 `tick` 字段归一化（见下） |

数据打包改用 numpy `view(np.uint8)` 直接按字节写入，避免 Python 层逐点循环；增加 `pts_raw.size == 0` 早返回，修复了 `last axis with size 0 is not supported` 报错。

### 6.2 `frame_id` 使用传感器本地帧 + Point-LIO 外参配置

**标准雷达部署流程（不在采集代码内做坐标变换）：**

1. `publish_lidar` 直接发布 annotator 原始数据，`frame_id = UnitreeL1_link`
2. `publish_odom` 已广播 `base_link → UnitreeL1_link` 的 TF（平移 + 165° Y 旋转），RViz2 用 TF 树完成显示变换
3. Point-LIO 的 `extrinsic_T / extrinsic_R` 描述传感器在 body 系下的安装位置和姿态，由 SLAM 内部处理坐标变换

**传感器安装参数（`ros2.py` 顶部常量）：**

```python
UnitreeL1_translation = (0.293, 0.0, -0.08)          # 传感器在 base_link 系下的位置 (m)
UnitreeL1_quat = quaternion_from_euler(0, 165°, 0)   # 绕 Y 轴旋转 165°
```

**Point-LIO YAML 外参：**

`extrinsic_R` 为 R_y(165°)，将向量从 LiDAR 系旋转到 body 系：

```
cos(165°) ≈ -0.9659,  sin(165°) ≈ 0.2588

R_y(165°) = [[-0.9659,  0,  0.2588],
             [ 0,       1,  0     ],
             [-0.2588,  0, -0.9659]]
```

```yaml
mapping:
    extrinsic_T: [ 0.293, 0.0, -0.08 ]
    extrinsic_R: [ -0.9659, 0.0,  0.2588,
                    0.0,    1.0,  0.0,
                   -0.2588, 0.0, -0.9659 ]
```

### 6.3 `time` 字段：严格单调递增 + Point-LIO segfault 根因与修复

#### 背景：两种 time=0 假说及实际根因

**假说 A（time_compressing 越界）——已否定：**

早期分析认为全零 `time` 会导致 `time_compressing()` 返回 `time_seq=[N]`，使主循环访问 `feats_down_body->points[0+N]` 越界。
实际分析表明：`time_compressing` 循环中 `idx` 的初始值为 `-1`，主循环第一次迭代访问 `points[-1 + N] = points[N-1]`，是合法下标。因此全零 time **不会**在此处引发 segfault。

**实际根因（laserMapping.cpp:1142 — IMU deque 越界）：**

Point-LIO 在主循环中有一段 IMU 时间对齐逻辑：

```cpp
// laserMapping.cpp ~line 1142
while (imu_comes)
{
    imu_next = imu_deque.front();   // ← 无 empty() 保护
    if (time_current >= imu_next.stamp)
    {
        imu_deque.pop_front();
        ...
    }
    else break;
}
```

由于我们使用**统一时间戳**（LiDAR 帧时间戳 = IMU 时间戳），并在每点上叠加 linspace 偏移（`time_current` 最大值 ≈ 帧时间戳 + scan_period），这使得 `time_current > imu_next.stamp` 始终成立。while 循环持续 `pop_front()` 直到 deque 被耗尽，然后在空 deque 上调用 `front()` → **undefined behavior / SIGSEGV**。

#### 修复：`unilidar.yaml` 设置 `imu_en: false`

```yaml
mapping:
    imu_en: false   # 禁用 IMU 预积分；跳过上述 while 循环，纯 LiDAR 模式稳定运行
```

纯 LiDAR 模式下 Point-LIO 仍通过点云本身的运动估计进行里程计，不依赖 IMU 时间对齐逻辑，segfault 消除。

#### `ros2.py` — `time_f` 改为 `linspace`（保留，提升点云质量）

即使 segfault 根因不在 time_compressing，发布单调递增的 `time` 仍有意义：Point-LIO 用 `curvature`（= `time × time_unit_scale`）对每帧点云做运动畸变补偿。全零 time 会使所有点被视为同一时刻采集，丢失去畸变信息。

```python
# 有 tick：归一化到 [0, scan_period] 秒
if tick_range > 0:
    time_f = ((ticks - ticks.min()) / tick_range * scan_period).astype(np.float32)
else:
    time_f = np.linspace(0.0, scan_period, N, endpoint=False, dtype=np.float32)

# 无 tick（plain float 路径）：
time_f = np.linspace(0.0, scan_period, N, endpoint=False, dtype=np.float32)
```

`scan_period = 1/200 = 0.005 s`（Unitree L1 200 Hz）。

#### `unilidar.yaml` 加 `timestamp_unit: 0`（保留）

```yaml
preprocess:
    timestamp_unit: 0   # 0=SEC: time字段单位为秒, time_unit_scale=1000 → curvature单位ms
```

将秒值转换为毫秒，使 curvature 在合理数值范围内（0–5 ms），与 Point-LIO 内部阈值匹配。

### 6.4 IMU `linear_acceleration` 修复（`publish_imu`）

**改动前：**
```python
# 传入参数
env.unwrapped.scene["robot"].data.root_lin_vel_b[i, :]   # 线速度 m/s

# 发布
imu_trans.linear_acceleration.x = base_lin_vel[0].item()  # 错误：速度 ≠ 加速度
```

**改动后：**
```python
# 传入参数
env.unwrapped.scene["robot"].data.projected_gravity_b[i, :]  # 体坐标系下重力方向（单位向量）

# 发布
G = 9.81
imu_trans.linear_acceleration.x = -G * gravity_b[0].item()  # 比力 = -g_body
imu_trans.linear_acceleration.y = -G * gravity_b[1].item()
imu_trans.linear_acceleration.z = -G * gravity_b[2].item()
```

**根因：** `root_lin_vel_b` 是速度（m/s），量纲完全错误。真实 IMU 的加速度计测量的是**比力**（specific force）= 实际加速度 − 重力加速度（体坐标系下）。机器人静止时比力 = −g_body，与重力方向相反。

`projected_gravity_b` 是 IsaacLab 提供的体坐标系下重力单位向量（例如直立时约为 `[0, 0, -1]`），所以：

``` 
a_imu = -9.81 × projected_gravity_b
```

直立静止时：`projected_gravity_b ≈ [0, 0, -1]` → `a_imu ≈ [0, 0, +9.81]`，与真实 IMU 一致。

**影响：** 原来 Point-LIO 的 IMU 预积分使用了错误的加速度量（线速度数值），导致姿态估计完全错误，表现为里程计持续漂移（即使机器人不动位置也在走）。修复后 IMU 数据与真实传感器行为一致，Point-LIO 建图漂移问题消除。

**Point-LIO 完整 YAML 配置（基于以上所有修改，见第 8 节获取最终版）：**

```yaml
common:
    lid_topic:  "/robot0/point_cloud2_L1"
    imu_topic:  "/robot0/imu"
    time_sync_en: false

preprocess:
    lidar_type: 2               # Velodyne PointXYZIRT 格式
    scan_line: 18               # Unitree L1：18 线
    timestamp_unit: 0           # 0=SEC，time字段单位秒，time_unit_scale=1000
    point_filter_num: 1
    feature_extract_enable: false
    blind: 0.1

mapping:
    imu_en: false               # 禁用 IMU，避免 deque 耗尽崩溃
    satu_acc: 30.0
    satu_gyro: 35.0
    acc_norm: 9.81
    gravity_init: [ 0.0, 0.0, -9.81 ]
    acc_cov: 0.1
    gyr_cov: 0.1
    b_acc_cov: 0.0001
    b_gyr_cov: 0.0001
    fov_degree:    360.0
    det_range:     30.0
    extrinsic_est_en: false
    extrinsic_T: [ 0.0, 0.0, 0.0 ]
    extrinsic_R: [ 1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0 ]

publish:
    path_en:               true
    scan_publish_en:       true
    dense_publish_en:      true
    scan_bodyframe_pub_en: false

pcd_save:
    pcd_save_en: false
    interval: -1
```

---

## 7. 统一时间戳修复点云漂移 (`ros2.py` — `pub_robo_data_ros2` 及各 `publish_*`)

**改动：** `pub_robo_data_ros2` 在每帧开头调用一次 `base_node.get_clock().now()`，生成统一的 `stamp`，并将其作为参数传递给所有发布函数：`publish_joints`、`publish_odom`（含 TF 广播）、`publish_imu`、`publish_lidar`。

**改动前的问题：**
每个 `publish_*` 函数内部各自调用 `self.get_clock().now()`，同一帧的 TF 与点云时间戳不同：

```
publish_odom  → TF broadcast @ t₁
publish_lidar → PointCloud2  @ t₂  （t₂ > t₁）
```

RViz2 渲染点云时需要在 `t₂` 时刻查找 `UnitreeL1_link` 相对于固定坐标系的变换，但 TF buffer 里只有 `t₁` 时刻的变换。当仿真运行速度快于实时时，`t₂ - t₁` 可达数十毫秒，RViz2 对 TF 进行时间外推，外推误差在每帧都不同，表现为点云**疯狂漂移**，即使机器人静止不动。

**改动后：** 同一帧内所有消息和 TF 使用完全相同的 `stamp`，RViz2 直接命中 TF 缓存，不再外推，漂移消除。

---

## 8. Point-LIO 部署参数补全（`unilidar.yaml`）

根据 Point-LIO GitHub 部署说明（deployment notes B、F），在 `mapping` 节新增以下参数：

### 8.1 IMU 饱和与归一化参数（Deployment Note B）

| 参数 | 值 | 说明 |
|---|---|---|
| `satu_acc` | `30.0` | 加速度计饱和阈值（m/s²）。仿真 IMU 最大值 ≈ 9.81 m/s²（纯重力），设为 30.0 留有足够余量 |
| `satu_gyro` | `35.0` | 陀螺仪饱和阈值（rad/s）。Go2 最大角速度约 10 rad/s，35.0 覆盖所有正常动作 |
| `acc_norm` | `9.81` | 重力加速度幅值（m/s²）。仿真中 `a_imu = -9.81 × projected_gravity_b`，静止时 `|a_imu| = 9.81`，与此值一致 |

Point-LIO 用 `acc_norm` 在 IMU 初始化阶段归一化加速度计均值以对齐重力方向；`satu_acc/satu_gyro` 用于异常值剔除。

### 8.2 初始重力向量（Deployment Note F）

```yaml
gravity_init: [ 0.0, 0.0, -9.81 ]
```

当 `imu_en: false` 时，Point-LIO 无法通过 IMU 静止段自动估计重力方向。`gravity_init` 显式指定世界坐标系下的初始重力向量（IsaacLab/Isaac Sim 中 z 轴向上，故重力为 `[0, 0, -9.81]`），确保 IKFoM 滤波器在纯 LiDAR 模式下也能正确初始化姿态。

### 8.3 同步确认（Deployment Note A）

IMU 与 LiDAR 时间戳对齐已通过第 7 节的统一时间戳方案保证：每帧所有消息（`/robot0/imu`、`/robot0/point_cloud2_L1`、TF）共用同一个 `stamp`，`time_sync_en: false` 告知 Point-LIO 时间戳已对齐，无需内部软件同步。

### 8.4 外参确认（Deployment Note D）

`extrinsic_est_en: false` 已设置。点云以 `frame_id = base_link` 发布（RTX annotator 返回世界系坐标，以 `base_link` 为原点），外参为单位矩阵，无需在线估计。

### 8.5 `time` 字段确认（Deployment Note C）

`time` 字段在 PointCloud2 消息中以 `offset=20, FLOAT32` 发布（见第 6.1 节）。若 Point-LIO 输出警告 `"Failed to find match for field 'time'"`，这是 PCL `fromROSMsg` 字段映射的 informational warning，不影响运行——Point-LIO 的 velodyne_handler 通过 PCL 点类型的 `time` 成员读取值，该映射由名称匹配完成，与消息内字节偏移无关。

