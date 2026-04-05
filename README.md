# go2_omniverse

**Forked from https://github.com/abizovnuralem/go2_omniverse**  
**Reference repo https://github.com/wty-yy/go2_rl_gym**

本仓库在原始 `go2_omniverse` 基础上，补齐了更完整的 Isaac Lab / Omniverse 仿真、ROS2 接口，以及参考 `go2_rl_gym` 思路实现的 Go2 楼梯训练与部署链路。

当前重点能力：

- Go2 / G1 在 Isaac Lab 中的实时 locomotion 推理
- 键盘控制与 ROS2 `cmd_vel` 控制
- RTX LiDAR、IMU、相机、语义分割数据发布
- hospital / office / warehouse 自定义场景加载
- Go2 楼梯 CTS 训练、checkpoint 保存与仿真部署

## 仓库结构

- [main.py](/media/user/data1/carl/workspace/go2_omniverse/main.py)
  仿真启动入口，内部调用 `core/omniverse_sim.py`
- [core/omniverse_sim.py](/media/user/data1/carl/workspace/go2_omniverse/core/omniverse_sim.py)
  Isaac Lab 仿真主逻辑，负责环境创建、checkpoint 加载、policy 推理与 ROS2 发布
- [core/ros2.py](/media/user/data1/carl/workspace/go2_omniverse/core/ros2.py)
  ROS2 传感器与状态话题发布
- [configs/custom_rl_env.py](/media/user/data1/carl/workspace/go2_omniverse/configs/custom_rl_env.py)
  Go2 / G1 / Go2StairDeploy 的环境配置
- [configs/agent_cfg.py](/media/user/data1/carl/workspace/go2_omniverse/configs/agent_cfg.py)
  RSL-RL 默认 agent 配置
- [train_stairs.py](/media/user/data1/carl/workspace/go2_omniverse/train_stairs.py)
  Go2 楼梯 CTS 训练入口
- [rsl_rl_cts](/media/user/data1/carl/workspace/go2_omniverse/rsl_rl_cts)
  CTS runner、actor-critic、rollout storage 等实现
- [scripts/run_sim.sh](/media/user/data1/carl/workspace/go2_omniverse/scripts/run_sim.sh)
  常用仿真启动脚本
- [scripts/run_train_stairs.sh](/media/user/data1/carl/workspace/go2_omniverse/scripts/run_train_stairs.sh)
  常用双卡训练脚本

## 当前训练与部署逻辑

### 1. 楼梯训练现在是 CTS，不是纯 PPO

[train_stairs.py](/media/user/data1/carl/workspace/go2_omniverse/train_stairs.py) 当前实现的是 **CTS: Concurrent Teacher-Student**。

- Teacher 使用 privileged observation
- Student 只使用部署可用的本体历史观测
- 训练时使用 `history_length=5`
- 默认实验目录是 `logs/rsl_rl/unitree_go2_stairs`

CTS 训练配置对齐了 `go2_rl_gym` 的一些经验：

- 4096 env 时 `num_steps_per_env=48`
- `base_height_target=0.38`
- 宽摩擦随机化 `0.2 ~ 2.0`
- 加入 `hip_deviation`、`joint_deviation_all`、`feet_regulation`、`foot_contact_force` 等楼梯相关奖励

### 2. 仿真会自动识别 CTS checkpoint

[core/omniverse_sim.py](/media/user/data1/carl/workspace/go2_omniverse/core/omniverse_sim.py) 会先检查 checkpoint 是否是 CTS 训练产物：

- 如果是普通 RSL-RL checkpoint，走 `UnitreeGo2CustomEnvCfg`
- 如果是 CTS checkpoint，自动切到 `Go2StairDeployCfg`

这样可以保证部署时 observation 维度和训练一致，避免 weight 维度不匹配。

### 3. CTS 部署时 student 不直接读 height scan

[configs/custom_rl_env.py](/media/user/data1/carl/workspace/go2_omniverse/configs/custom_rl_env.py) 里的 `Go2StairDeployCfg` 会输出 45D student observation：

- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos`
- `joint_vel`
- `actions`

也就是说，部署阶段 student 主要依赖**运动历史**推断楼梯，而不是直接读取 teacher 训练时的 privileged terrain 信息。

## 自定义场景

仓库内当前使用的场景资产路径：

- hospital: `assets/env/hospital_labeled.usd`
- office: `assets/env/office.usd`
- warehouse: `assets/env/warehouse.usd`

其中 hospital 的逻辑比较特殊：

- 场景在 `gym.make()` 之前注入，避免 PhysX 运行后再动态加碰撞体
- 会主动关闭 hospital 内置的 `CollisionPlane`
- 这样可以避免它和 Isaac Lab 的 `/World/ground` 重叠导致接触异常

如果你换机器，只需要确认这些仓库内的 USD 文件路径仍然有效。

## 快速启动

### 1. Go2 仿真

```bash
cd /media/user/data1/carl/workspace/go2_omniverse
./scripts/run_sim.sh
```

当前 [scripts/run_sim.sh](/media/user/data1/carl/workspace/go2_omniverse/scripts/run_sim.sh) 默认参数是：

- `EXPERIMENT_NAME=unitree_go2_stairs`
- `LOAD_RUN=.*`
- `CHECKPOINT=model_2800.pt`
- `CMD_SOURCE=keyboard`

也就是默认直接跑楼梯实验的 checkpoint，而不是旧的 `unitree_go2_rough`。

### 2. 指定 checkpoint / 控制源

```bash
EXPERIMENT_NAME=unitree_go2_stairs \
CHECKPOINT=model_2800.pt \
CMD_SOURCE=ros2 \
./scripts/run_sim.sh
```

### 3. 楼梯训练

```bash
cd /media/user/data1/carl/workspace/go2_omniverse
./scripts/run_train_stairs.sh
```

也可以直接指定环境数和训练轮数：

```bash
./scripts/run_train_stairs.sh 4096 15000
```

或者直接运行 Python：

```bash
python train_stairs.py --headless --num_envs 4096 --max_iterations 15000
```

## 日志与模型

训练脚本 [scripts/run_train_stairs.sh](/media/user/data1/carl/workspace/go2_omniverse/scripts/run_train_stairs.sh) 会输出：

- 日志：`logs/train_stairs_logs/train_gpu0.log`
- PID：`logs/train_stairs_logs/train_gpu0.pid`
- 日志：`logs/train_stairs_logs/train_gpu1.log`
- PID：`logs/train_stairs_logs/train_gpu1.pid`

模型权重默认保存在：

- `logs/rsl_rl/unitree_go2_stairs/`

实时看训练日志：

```bash
tail -f logs/train_stairs_logs/train_gpu0.log | grep -E '\[CTS|ERROR|Traceback'
tail -f logs/train_stairs_logs/train_gpu1.log | grep -E '\[CTS|ERROR|Traceback'
```

停止双卡训练：

```bash
kill $(cat logs/train_stairs_logs/train_gpu0.pid) $(cat logs/train_stairs_logs/train_gpu1.pid)
```

## 控制模式

### 1. 键盘控制

```bash
CMD_SOURCE=keyboard ./scripts/run_sim.sh
```

按键：

- `W/S` 前进 / 后退
- `A/D` 左移 / 右移
- `Q/E` 左转 / 右转

当前 [core/omniverse_sim.py](/media/user/data1/carl/workspace/go2_omniverse/core/omniverse_sim.py) 中键盘速度默认是：

- 线速度 `1.5 m/s`
- 角速度 `1.5 rad/s`

### 2. ROS2 控制

```bash
CMD_SOURCE=ros2 ./scripts/run_sim.sh
```

此时仿真订阅：

- `robot0/cmd_vel`

并将当前送入 policy 的高层命令维护在：

- `robot0/base_command`

## 默认 ROS2 话题

- `robot0/odom`
- `robot0/imu`
- `robot0/point_cloud2_L1`
- `robot0/front_cam/rgb`
- `robot0/front_cam/camera_info`
- `robot0/front_cam/semantic_segmentation`
- `robot0/base_command`
- `robot0/cmd_vel`

## 常见问题

### 1. 仿真能启动，但机器人不动

- 检查 `EXPERIMENT_NAME` 是否指向正确实验目录
- 检查 `CHECKPOINT` 是否真的存在于 `logs/rsl_rl/<experiment>/`
- 如果使用 ROS2，确认 `robot0/cmd_vel` 真的有消息

### 2. CTS checkpoint 加载失败

- 检查 checkpoint 是否来自 `train_stairs.py`
- 检查部署环境是否自动切到了 `Go2StairDeployCfg`
- 如果自己改过 observation，确认训练和部署的 obs 维度一致

### 3. hospital 场景碰撞异常

- 检查 `assets/env/hospital_labeled.usd` 是否存在
- 检查 hospital 的 `CollisionPlane` 是否已被关闭
- 不要在 `gym.make()` 之后再手工往场景里追加 hospital 碰撞体

### 4. 训练日志路径和旧 README 不一致

当前训练日志目录已经是：

- `logs/train_stairs_logs/`

不是旧版本里写的仓库根目录 `train_stairs_logs/`。
