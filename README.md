# go2_omniverse

Isaac Lab / Isaac Sim 中的 Unitree Go2 仿真、传感器发布和大场景语义资产。

## 环境准备

```bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /media/user/data1/carl/workspace/go2_omniverse
```

首次运行 `scripts/run_sim.sh` 会编译并 source：

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

推荐使用 ROS2 控制、单目 RGB、深度和 labeled Hospital 场景：

```bash
cd /media/user/data1/carl/workspace/go2_omniverse
CMD_SOURCE=ros2 CAMERA_MODE=single ENABLE_DEPTH=true \
CUSTOM_ENV=hospital ./scripts/run_sim.sh
```

headless 模式：

```bash
HEADLESS=1 CMD_SOURCE=ros2 ENABLE_DEPTH=true \
CUSTOM_ENV=hospital ./scripts/run_sim.sh
```

部署模型可切换：

```bash
DEPLOY_ARCHITECTURE=cts ./scripts/run_sim.sh
DEPLOY_ARCHITECTURE=moe_cts ./scripts/run_sim.sh
```

默认参数包括 `CUSTOM_ENV=hospital`、`CMD_SOURCE=ros2`、`ENABLE_DEPTH=true`、
`CAMERA_MODE=single`。启动时会打印实际场景、传感器与 checkpoint 配置。

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

出现楼梯为 background 或栏杆为可通行区域时，排查顺序是：

1. 确认运行时加载的 USD 路径和 composed runtime stage；
2. 检查楼梯 rendered Mesh 的 semantic type/data；
3. 检查 semantic annotator 的 raw ID 与 `idToLabels`；
4. 核对消费端是否按 raw label 映射语义类别。

Hospital 的 `Geo_Roof_Costum1_34`、`Geo_Roof_Costum6_19` 楼板同时包含二层支撑面和
一层天花板。离线生成工具将其顶面提取为同级 `*_SupportSurface` Mesh，标为 `floor`；
原 Mesh 的底面和侧面标为 `wall`。分类限定于这两块已审核楼板，并核对顶面高度和朝向。
静态几何和标签检查通过不等于运行时图像已验证，仍需检查实际 annotator 输出。

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

双目模式 (`CAMERA_MODE=double`) 额外发布 `/robot0/front_cam_left/*` 和
`/robot0/front_cam_right/*`。Depth 话题仅在 `ENABLE_DEPTH=true` 时可用。

键盘控制示例：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/robot0/cmd_vel
```

## 训练与部署

楼梯/全场景 CTS 训练：

```bash
./scripts/run_train_stairs.sh
ARCHITECTURE=moe_cts ./scripts/run_train_stairs.sh
```

训练参数和 checkpoint 路径见 `scripts/run_train_stairs.sh`、`scripts/run_sim.sh`。
