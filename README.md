# go2_omniverse

Isaac Lab / Isaac Sim 中的 Unitree Go2 仿真、传感器发布和大场景语义资产。

## 环境准备

```bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /media/user/data1/carl/workspace/go2_omniverse
```

首次运行 `commands/run_sim.sh` 会编译并 source：

- `IsaacSim-ros_workspaces/humble_ws`
- `go2_omniverse_ws`

`run_sim.sh` 会设置 `PYTHONNOUSERSITE=1`，确保 IsaacLab 使用
`env_isaaclab` 内匹配的 PyTorch/torchvision。不要让 `~/.local` 中其他 CUDA 或
PyTorch 版本覆盖该环境；启动时脚本会打印两个包的版本和实际路径。

也可以预先构建后设置 `SKIP_ROS_BUILD=true`。换机器时重点检查：

```bash
export ISAAC_ASSETS_ROOT=/absolute/path/to/Assets/Isaac/4.5
export CONDA_ENV_NAME=env_isaaclab
```

## 启动仿真

运行时固定使用 MoE-CTS 和单个 `front_cam`。推荐使用 ROS2 控制、RGB-D 和 labeled Hospital 场景：

```bash
cd /media/user/data1/carl/workspace/go2_omniverse
CMD_SOURCE=ros2 CONTROLLER=lab  \
 ./commands/run_sim.sh
```

headless 模式：

```bash
HEADLESS=1 CMD_SOURCE=ros2 \
CUSTOM_ENV=hospital ./commands/run_sim.sh
```

默认参数包括 `CONTROLLER=lab`、`CUSTOM_ENV=hospital`、`CMD_SOURCE=ros2`。
启动时打印实际场景、传感器、policy 路径与地形限速开关。当前只支持 Lab student，
相机固定为单个 `front_cam`。

## Big scene 资产

当前大场景位于 `assets/env/big_scene/`：

```text
hospital/hospital_base.usd
hospital/hospital_navigation.usda
office/office_base.usd
office/office_navigation.usda
warehouse/warehouse_base.usd
warehouse/warehouse_navigation.usda
```

Hospital 和 warehouse 的正式运行路径由 `core/omniverse_sim.py` 固定到 labeled
`*_navigation.usda`，防止采集时误加载未标注资产。open-door 变体用于离线资产处理和
验证，不能绕过正式 profile 的路径检查。

离线生成 Hospital 语义与开门 overlay：

```bash
python tools/usd_scene_pipeline.py navigation-overlay \
  assets/env/big_scene/hospital/hospital_base.usd \
  --output assets/env/big_scene/hospital/hospital_navigation.usda \
  --json assets/env/big_scene/hospital/hospital_navigation.json \
  --open-doors --all-semantics
```

## 语义标注约定

语义必须在 USD source / composed runtime stage 的实际 rendered Mesh 上生效：

| 对象 | USD semanticData | 对应语义类别 |
|---|---|---|
| floor / corridor / upper-floor support surface | `floor` | `navigable` |
| stair tread / step / stair landing | `stairs` | `terrain_transition` |
| railing / handrail / guardrail / fence | `wall` | `structure` |
| wall / door panel / door frame / jamb | `wall` | `structure` |
| ceiling | `ceiling` 或拆分楼板的 `wall` | `structure` |
| warning sign | `sign` | `sign` |

当前 `--open-doors` overlay 将识别到的 door panel 设为 inactive；门框保持 `wall`。
门洞后方实际可见的 floor 才是 `navigable`。楼梯踏步与栏杆应是独立 Prim/Mesh，不能在 stair root 上统一
覆盖所有子项，也不能用 traversability、depth、normal 或 morphology 后处理掩盖源头
语义错误。

## ROS2 话题

控制输入：

```text
/robot0/cmd_vel
```

主要传感器：

```text
/robot0/odom
/robot0/imu
/robot0/point_cloud2_L1
/robot0/front_cam/rgb
/robot0/front_cam/depth
/robot0/front_cam/camera_info
/robot0/front_cam/semantic_segmentation
/robot0/front_cam/semantic_segmentation_labels
```

`front_cam/depth` 始终发布，不提供关闭开关。

## Front camera extrinsic

相机安装外参统一在 [configs/robot_extrinsic.yaml](configs/robot_extrinsic.yaml) 中修改：

```yaml
# p^base = R_camera^base * p^camera + t_camera^base
robot_extrinsic:
  camera_to_robot_R: [ 0.0,  0.0,  1.0,
                     -1.0,  0.0,  0.0,
                      0.0, -1.0,  0.0]
  camera_to_robot_T: [0.32487, -0.00095, 0.05362]
```

`R` 为 row-major 旋转矩阵，`T` 为相机光心在 base_link 中的位置，单位米。
机体系 x 前、y 左、z 上；相机采用 ROS 光学坐标系 x 右、y 下、z 前。
因此这里的矩阵不是相机外壳坐标系的姿态，也不是 base 点到 camera 点的变换。
当前已恢复原始安装位置和水平朝向；此前的 **10° 下倾**配置完整保留在配置文件注释中。

渲染相机 `CameraCfg` 和 `base_link -> front_cam` 静态 TF 从同一配置加载。
修改配置后重启仿真即可；启动脚本无需添加参数，旧 `FRONT_CAM_EXTRINSIC`
环境变量不再作为覆盖来源。


键盘控制示例：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/robot0/cmd_vel
```

## Lab 控制器与权重

当前只使用本地 `go2_rl_lab` 的 MoE-CTS student。旧 `go2_rl_gym` 子仓库、
legacy 加载分支及旧训练实现已移除，`CONTROLLER=legacy` 不再支持。
默认使用 Lab，原 `bash commands/run_sim.sh` 启动形式继续可用。

原始权重为 `checkpoints/moe_cts_lab.pt`，更新后先导出：

```bash
python tools/export_lab_policy.py
CONTROLLER=lab CMD_SOURCE=ros2 bash commands/run_sim.sh
```

导出到 `checkpoints/moe_cts_lab/policy.pt` 和配套 `config.yaml`，不覆盖原始权重。
可通过 `POLICY_PATH`、`POLICY_CONFIG` 选择兼容导出。
接口仍为机体系 `[vx,vy,wz]`；45D 单帧、10 帧 term-wise 历史、12D action、20ms 控制周期。
地形信息不进入 policy observation，不改变 normalization、action scale 或关节 PD。

训练入口 `commands/run_train_stairs.sh` 转发到 Lab 的 `go2-mixed-moe-cts` 任务，
应在该仓库要求的独立训练环境使用；现有 Isaac Sim 4.5 部署环境不升级、不替换依赖。
本次未执行训练。

## USD Terrain-Aware Velocity Limiter

启用底层地形限速器，无需新增 ROS2 节点：

```bash
CONTROLLER=lab TERRAIN_SPEED_LIMIT=1 bash commands/run_sim.sh

# 明确关闭，保留 Lab 原有统一限幅行为
TERRAIN_SPEED_LIMIT=0 bash commands/run_sim.sh
```

功能默认关闭。关闭时 Lab 默认速度上限仍为 ±`[1.0,1.0,1.5]`，
可用 `MAX_VELOCITY_COMMAND="3.5 0.5 1.5"` 修改。
启用时，未显式设置该变量则使用地形配置中的平地上限，避免继续被 1 m/s 截断；
显式设置更低的全局上限仍然有效。不会将导航的低速请求自动放大。

地面接触摩擦默认调整为静摩擦 `0.9`、动摩擦 `0.8`（默认地面原为 `0.5 / 0.5`）。
Hospital、Warehouse 和自定义 USD 中已识别且可编辑的静态地板、楼梯、坡道同步应用该设置；
启动日志 `[ground friction]` 会报告覆盖数量及无法单独覆盖的实例代理数量。
设置只作用于本次仿真 Stage，不写回 USD 资产。可按需调整，例如：

```bash
GROUND_STATIC_FRICTION=0.9 GROUND_DYNAMIC_FRICTION=0.8 \
CONTROLLER=lab TERRAIN_SPEED_LIMIT=1 bash commands/run_sim.sh
```

要求 `0 <= 动摩擦 <= 静摩擦`。该参数是地面接触摩擦，独立于关节摩擦和速度限幅；
不会提高速度指令，也不保证实际速度一定增加。关闭地形限速器仍使用上述地面摩擦设置。

地形限速参数集中在 [configs/terrain_speed.yaml](configs/terrain_speed.yaml)，
可用 `TERRAIN_SPEED_CONFIG` 指定其他配置：

| 参数 | 初始值 | 含义 |
|---|---:|---|
| flat_max_vx / flat_max_vy | 3.5 / 0.5 m/s | 已确认开阔平地的指令范围 |
| stairs_max_vx / stairs_max_vy | 0.7 / 0.2 m/s | 楼梯及退出保持阶段的指令范围 |
| unknown_max_vx / unknown_max_vy | 0.5 / 0.2 m/s | 地形、净空或位姿不可靠时的保守范围 |
| max_yaw_rate | 1.5 rad/s | 角速度上限，与平移共用椭球预算 |
| accel_limit / decel_limit | 1.0 / 2.0 m/s² | 指令幅值变化率参数 |
| braking_decel_factor | 0.75 | 制动规划余量 |
| brake_margin | 0.5 m | 制动距离余量 |
| lookahead_min / lookahead_max | 1.0 / 8.0 m | 前瞻查询距离 |
| terrain_state_hysteresis | 0.3 s | 清除楼梯/确认平地的保持时间 |

以上是仿真调参值，不代表实体机器人安全极限。停止或更低速度请求、新发现的保守硬限幅
优先于变化率平滑；限速器不负责路径规划或绕障，UNKNOWN 的低速也不意味着区域可安全通行。

```text
原始 vx、vy、wz → 地形限速器 → 同一个 Lab student → Go2 关节控制
                   ↑
          USD 静态碰撞几何、语义、世界位姿与实际速度
```

初始化时扫描当前 World 的有效静态碰撞几何，按支撑高度区分重叠楼层；
包含实际仍参与物理的默认地面，不把“渲染隐藏”误当作“碰撞禁用”。
普通 Mesh 按真实三角形覆盖查询；楼梯名称只作线索，不单独确认台阶。
无可靠语义时，可利用多个共线、不同高度的水平踏面辅助确认。
场景几何或语义变化使索引失效并重建，机器人正常运动不重扫整个场景。

前瞻检查请求方向和实际残余运动方向，覆盖倒退、横移及转向后的运动；
状态包括 `FLAT`、`APPROACHING_STAIRS`、`ON_STAIRS`、`LEAVING_STAIRS`、`UNKNOWN`。
进入楼梯前减速，完全离开并满足滞回时间后再平滑恢复。
不切换模型，不因地形变化重置 policy history，不修改 GeoXMobility/SRU。

`[velocity robot0]` 将 request、limited、actual 与地形状态合并在同一行，默认每 **5 秒墙钟时间**输出一次。
request/limited 是刚完成控制周期内 policy 使用的指令，actual 是该周期结束时的机体系实测速度；
`t` 使用仿真秒。`VELOCITY_LOG_INTERVAL=10` 可进一步降低频率，设为 `0` 关闭周期日志。
不再额外输出重复的 `[terrain robot0]` 周期日志；启动、重置和错误事件仍保留。
仿真还以里程计相同时间戳发布 `/robot0/requested_command` 和 `/robot0/controller_command`
（`TwistStamped`），用于区分原始请求与地形限幅后的实际下发指令。

测试方法、实现边界与实测结果见 [地形限速说明](docs/terrain_speed.md)。
初次 Lab 适配和不带地形限速器的历史结果见 [Lab 迁移记录](docs/lab_controller.md)。
