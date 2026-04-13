# Omniverse 仿真更新

## 常用命令

```bash
./scripts/run_sim.sh

# 场景可切：hospital / warehouse / office
# 控制源可切：--cmd_source keyboard | ros2
```

## 版本记录（从上到下递增）

### v0.1
1. 基础仿真框架搭建，集成 IsaacLab env 与 ROS2 接口。
2. 支持多机器人、多场景。

### v0.2
1. 修复 hospital 加载顺序与碰撞问题。
2. 禁用内置 `CollisionPlane`，避免重复地面。

### v0.3
1. 相机分辨率从 `1280x720` 降到 `640x360`。
2. 降低 RTX 渲染负载。

### v0.4
1. 加入语义分割发流。
2. 发布原始 `32SC1` 与 `RGB8` 两种格式。

### v0.5
1. LiDAR 点云升级为 `PointXYZIRT`。
2. 修复点云打包性能。

### v0.6
1. LiDAR 坐标系标准化。
2. 支持 YAML 外参配置。

### v0.7
1. 统一同帧时间戳。
2. 修复 RViz2 点云时间外推误差。

### v0.8
1. IMU 线加速度改为正确的比力。
2. 修复 Point-LIO IMU 预积分输入。

### v0.9
1. 定位 Point-LIO segfault 根因是 IMU deque 耗尽。
2. `imu_en=false` 下纯 LiDAR 模式稳定。

### v0.10
1. 统一时间戳方案扩展到 IMU、joint、odom。
2. LiDAR 时间字段改为 `linspace`。

### v0.11
1. 新增语义标签 JSON 话题。
2. 新增 `camera_info` 话题。

### v0.12
1. 新增 `--cmd_source`。
2. 支持 `keyboard` 与 `ros2` 两种高层控制源。

### v0.13
1. hospital 资产切到工程内 `env/hospital_labeled.usd`。
2. checkpoint 加载支持从实验根目录自动解析。
3. hospital 与 warehouse/office 加载策略分离。

### v0.14
1. 修复 hospital 下 RayCaster 多网格崩溃。
2. 修复 CTS checkpoint 无法加载到部署环境的问题。
3. 新增 `Go2StairDeployCfg`，按 checkpoint 维度推断 CTS 网络参数。

### v0.15
1. 修复楼梯 CTS 部署动作缩放不一致。
2. CTS 楼梯部署速度指令按训练范围裁剪。

### v0.16
1. 修复 `hospital_labeled.usd` 下重复地面 prim 漏禁用。
2. 增加早期调试打印，用于区分物理发散和渲染异常。

### v0.17
1. 新增 `Go2FullSceneDeployCfg`。
2. CTS checkpoint 不再一律按 `45D` 处理，而是按 actor obs 维度自动选择 `45D` 或 `48D`。

### v0.18
1. `scripts/run_sim.sh` 默认实验目录切到 `unitree_go2_fullscene_cts/seed_123`。
2. full-scene 成为默认部署主路径。

### v0.19
1. viewer 跟随改成启动参数开关：`--viewer_follow {auto,on,off}`。
2. 新增 `--action_smoothing`、`--zero_cmd_stance_blend`、`--zero_cmd_threshold`。
3. 这些参数只用于缓解部署抖动，不替代训练修复。

### v0.20
1. 修复 hospital 加载两难：post-gym.make 导致碰撞体与机器人重叠（`base_contact` 每步触发）；pre-gym.make 导致 CollisionPlane 在 PhysX broad-phase 里崩溃。
2. 最终方案：新增 `_spawn_hospital_usd()` 自定义 spawner，在 USD 加载后、PhysX broad-phase 前立即停用 CollisionPlane，恢复 pre-gym.make 路径。
3. 补记 v0.17 漏记：48D CTS 指令速度上限自动收窄到 `lin=1.0 m/s / yaw=1.0 rad/s`。

### v0.21
1. CTS 训练侧数值稳定性加固：reward 改用 `nan_to_num` + `clamp` 双步裁剪；actor 输出非有限时自动回滚到上一个 good snapshot；`_fwd()` 输入全部净化；critic 输出套 tanh 有界；GAE 全路径 `_safe()`。
2. 兼容 IsaacLab `extras["log"]` 与旧版 `extras["episode"]` 两种 key；episode 指标即时转 CPU 防 OOM。
