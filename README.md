# go2_omniverse

Forked from https://github.com/abizovnuralem/go2_omniverse  
Reference repo https://github.com/wty-yy/go2_rl_gym
Reference repo https://github.com/ZiwenZhuang/parkour?tab=readme-ov-file

---

## 项目简介

本仓库主要实现以下功能：
- 在 Isaac Lab / Omniverse 中进行 Go2、G1 机器人的仿真与推理
- 训练并部署 Go2 机器人的 CTS 楼梯 / full-scene 运动控制策略

## 已实现功能

- Go2 / G1 仿真运行与 checkpoint 模型推理
- 键盘控制与 ROS2 `cmd_vel` 控制
- ROS2 传感器数据与机器人状态发布
- RTX LiDAR（适配 Point_LIO）、IMU、相机、语义分割数据输出
- `hospital` / `office` / `warehouse` 场景加载
- Go2 CTS 模型训练、日志保存与 checkpoint 部署

## 主要入口文件

- 仿真入口：`main.py`
- 仿真主逻辑：`core/omniverse_sim.py`
- ROS2 发布：`core/ros2.py`
- 环境配置：`configs/custom_rl_env.py`
- Go2 CTS 训练：`train_stairs.py`
- 常用仿真脚本：`scripts/run_sim.sh`
- 常用训练脚本：`scripts/run_train_stairs.sh`

## 使用方法

### 1. 启动 Go2 仿真

```bash
cd go2_omniverse
./scripts/run_sim.sh
```

默认加载配置：
- `EXPERIMENT_NAME=unitree_go2_fullscene_cts/seed_*`
- `CHECKPOINT=model_*.pt`
- `CMD_SOURCE=keyboard`

可自行修改参数，支持 ROS2 话题控制。

### 2. 启动双卡训练

```bash
./scripts/run_train_stairs.sh
```

自定义环境数量与训练轮数：
```bash
./scripts/run_train_stairs.sh 4096 150000
```

## 输出路径

### 训练日志
- `logs/train_stairs_logs/train_gpu0.log`
- `logs/train_stairs_logs/train_gpu1.log`

### 训练 PID
- `logs/train_stairs_logs/train_gpu0.pid`
- `logs/train_stairs_logs/train_gpu1.pid`

### Checkpoint 默认目录
- `logs/rsl_rl/unitree_go2_fullscene_cts/seed_*/`

## 常用命令

### 实时查看训练日志
```bash
tail -f logs/train_stairs_logs/train_gpu0.log | grep -E '\[CTS|ERROR|Traceback'
tail -f logs/train_stairs_logs/train_gpu1.log | grep -E '\[CTS|ERROR|Traceback'
```

### 停止双卡训练
```bash
kill -9 -$(cat logs/train_stairs_logs/train_gpu0.pid) -$(cat logs/train_stairs_logs/train_gpu1.pid)
```

## ROS2 话题

### 控制输入
- `robot0/cmd_vel`

### 常用输出
- `robot0/odom`
- `robot0/imu`
- `robot0/point_cloud2_L1`
- `robot0/front_cam/rgb`
- `robot0/front_cam/camera_info`
- `robot0/front_cam/semantic_segmentation`

## 补充说明

- Go2 的 CTS 部署会根据 checkpoint 自动匹配 `45D` 或 `48D` student observation
- `hospital`、`office`、`warehouse` 场景资产位于 `assets/env/`
- 详细训练迭代记录见 `docs/TRAIN_UPDATE.md`
- 详细仿真改动记录见 `docs/SIM_UPDATE.md`