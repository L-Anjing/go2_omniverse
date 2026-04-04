# go2_omniverse

**Forked from https://github.com/abizovnuralem/go2_omniverse**  

本仓库具有 **Isaac Lab / Omniverse 里的 Go2 仿真、低层 RL 推理和训练**。  


## 这个仓库目前已经具备的效果


- Go2 / G1 在 Isaac Lab 中的实时平衡与 locomotion 推理
- 键盘实时控制
- ROS2 实时控制接口
- RTX LiDAR、IMU、相机、语义分割话题发布
- Unitree L1 LiDAR 接入
- hospital / office / warehouse 等自定义场景加载
- 楼梯 locomotion 训练脚本与 checkpoint 推理


## 这个仓库负责什么

- 在 Isaac Lab 中加载 Go2 / G1 机器人和 hospital 等场景
- 运行 RSL-RL checkpoint 做低层 locomotion 控制
- 发布 ROS2 传感器和状态话题
- 接收 `robot0/cmd_vel` 作为高层速度命令
- 提供从零训练和楼梯训练脚本


### Hospital 资产路径

Hospital 场景使用本地 Isaac Sim 资产路径。  
如果你的机器路径不同，需要修改 [omniverse_sim.py](go2_omniverse/omniverse_sim.py) 里 `HOSPITAL_USD` 的实际位置。

当前默认路径是：

`/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5/Isaac/Environments/Hospital/hospital.usd`

## 主要入口

- [omniverse_sim.py](go2_omniverse/omniverse_sim.py)
  仿真主入口。创建 Isaac Lab 环境、加载 checkpoint、执行 `policy(obs) -> env.step(actions)`，并发布 ROS2 话题。
- [run_sim.sh](go2_omniverse/run_sim.sh)
  常用运行脚本。
- [train_stairs.py](go2_omniverse/train_stairs.py)
  楼梯 locomotion 训练入口。
- [run_train_stairs.sh](go2_omniverse/run_train_stairs.sh)
  训练启动脚本。
- [ros2.py](go2_omniverse/ros2.py)
  ROS2 发布接口，负责 odom、imu、lidar、camera、base_command 等话题。

## 快速部署与启动

### 1. Go2 仿真推理

最常用的启动方式：

```bash
cd go2_omniverse
bash run_sim.sh
```

这会走 [run_sim.sh](go2_omniverse/run_sim.sh)，默认加载当前脚本里配置的实验名、checkpoint 和 `CMD_SOURCE`。

### 2. G1 仿真推理

如果你保留了对应脚本，也可以单独启动 G1：

```bash
cd go2_omniverse
bash run_sim_g1.sh
```

### 3. 楼梯训练

```bash
cd go2_omniverse
bash run_train_stairs.sh
```

### 4. 直接指定模型与控制源

例如：

```bash
EXPERIMENT_NAME=unitree_go2_rough \
CHECKPOINT=model_7850.pt \
CMD_SOURCE=ros2 \
bash run_sim.sh
```

## 运行模式

### 1. 键盘遥控模式

适合手动探索、录传感器流、辅助建图。

```bash
cd go2_omniverse
CMD_SOURCE=keyboard bash run_sim.sh
```

或直接：

```bash
python main.py \
  --robot_amount 1 \
  --robot go2 \
  --custom_env hospital \
  --cmd_source keyboard
```

键位：

- `W/S` 前进/后退
- `A/D` 左移/右移
- `Q/E` 左转/右转

### 2. ROS2 控制模式

适合接导航栈。此模式下 `omniverse_sim.py` 只接收 `robot{i}/cmd_vel`，不会再响应键盘。

```bash
cd go2_omniverse
CMD_SOURCE=ros2 bash run_sim.sh
```

这个模式适合接 `3d-navi` 导航栈，或者其他会往 `robot0/cmd_vel` 发速度命令的 ROS2 上层。

## 默认控制接口

仿真内部保留这些原生话题：

- `robot0/odom`
- `robot0/imu`
- `robot0/point_cloud2_L1`
- `robot0/front_cam/rgb`
- `robot0/front_cam/camera_info`
- `robot0/front_cam/semantic_segmentation`
- `robot0/base_command`
- `robot0/cmd_vel`

说明：

- `robot0/cmd_vel` 是外部导航命令写入仿真的入口
- `robot0/base_command` 是当前真正送进 RL policy 的高层速度命令

## 模型推理

`omniverse_sim.py` 的推理流程是：

1. 解析实验目录和 checkpoint
2. 用 `OnPolicyRunner.load(...)` 载入 RSL-RL 权重
3. 获取 inference policy
4. 在循环里执行 `obs -> policy(obs) -> env.step(actions)`

常用环境变量：

- `EXPERIMENT_NAME`
- `LOAD_RUN`
- `CHECKPOINT`
- `CMD_SOURCE`

例如：

```bash
EXPERIMENT_NAME=unitree_go2_rough \
CHECKPOINT=model_7850.pt \
CMD_SOURCE=ros2 \
bash run_sim.sh
```

## 训练

楼梯训练入口：

```bash
cd go2_omniverse
bash run_train_stairs.sh
```

如果你要直接跑 Python：

```bash
python train_stairs.py --headless --num_envs 4096 --max_iterations 15000
```

训练日志默认在：

- `train_stairs_logs/`
- `logs/rsl_rl/`

## 这层最常见的问题

### 仿真能开，但机器人不动

- 确认 checkpoint 路径是否正确
- 确认 `CMD_SOURCE` 是否和当前使用方式一致
- 如果是导航模式，确认是否真的有消息发到 `robot0/cmd_vel`

### hospital 场景加载失败

- 检查 `HOSPITAL_USD` 路径
- 检查本地 Isaac Sim 资产目录是否完整

### 训练模型加载后行为异常

- 确认 checkpoint 对应的 observation / action 配置没有变
- 确认当前运行模式没有让键盘和 ROS2 同时写命令


