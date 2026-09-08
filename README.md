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
CMD_SOURCE=ros2 \
CUSTOM_ENV=hospital ./commands/run_sim.sh
```

headless 模式：

```bash
HEADLESS=1 CMD_SOURCE=ros2 \
CUSTOM_ENV=hospital ./commands/run_sim.sh
```

默认参数包括 `CUSTOM_ENV=hospital`、`CMD_SOURCE=ros2`，
`EXPERIMENT_NAME=unitree_go2_fullscene_moe_cts`。启动时会打印实际场景、传感器与
checkpoint 配置。模型架构和相机数量不再提供运行时切换接口。

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

`front_cam` 的外参定义为从 robot base frame 到 camera frame 的齐次变换
`T_base_front_cam`：

```text
T_base_front_cam =[ [ R_base_front_cam  t_base_front_cam ],
                    [       0 0 0               1        ] ]

                   [ [ 0   0   1    0.32487],
                  =  [-1   0   0   -0.00095],
                     [0  -1   0     0.05362],
                     [0   0   0    1]         ]
```

其中平移单位为米。默认矩阵同时用于 IsaacLab `CameraCfg` 和 ROS2
`base_link -> front_cam` static TF，因此不会再出现渲染相机外参与 TF 分别修改后不一致的问题。

可以通过一行 16 个 row-major 数值覆盖外参：

```bash
export FRONT_CAM_EXTRINSIC="0 0 1 0.32487 -1 0 0 -0.00095 0 -1 0 0.05362 0 0 0 1"
./commands/run_sim.sh
```


键盘控制示例：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/robot0/cmd_vel
```

## 训练与部署

MoE-CTS 楼梯/全场景强化学习训练：

```bash
./commands/run_train_stairs.sh
```

训练固定使用 `ActorCriticMoECTS + MoECTS`。可以用 `CHECKPOINT` 继续训练，也可以用
`EXPERIMENT_NAME` 修改日志目录；这两个参数只选择权重和运行记录，不切换模型结构：

```bash
CHECKPOINT=logs/rsl_rl/unitree_go2_fullscene_moe_cts/seed_123/model_14000.pt \
  ./commands/run_train_stairs.sh
```

部署默认读取 `logs/rsl_rl/unitree_go2_fullscene_moe_cts/`，可通过 `LOAD_RUN` 和
`CHECKPOINT` 选择该架构下的 checkpoint。非 MoE-CTS checkpoint 会在加载阶段被拒绝。
