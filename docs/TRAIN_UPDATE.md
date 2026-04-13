# Go2 训练更新

## 常用命令

```bash
./scripts/run_train_stairs.sh

tail -f logs/train_stairs_logs/train_gpu0.log
tail -f logs/train_stairs_logs/train_gpu1.log

tensorboard \
  --logdir /media/user/data1/carl/workspace/go2_omniverse/logs/rsl_rl/unitree_go2_fullscene_cts \
  --port 6006 --host 127.0.0.1
```

## 版本记录（从上到下递增）

### v2.8
1. 全面对照 go2_rl_gym/UPDATE.md + go2_config.py 完成迁移审计：obs缩放、263D privileged obs、全部惩罚 ×0.02、dynamic_sigma、指令课程、terrain curriculum 均已对齐，无遗漏。
2. 补上 go2_rl_gym 剩余两项：`dof_pos_limits=-0.04`（有效权重，防关节越限）和 `dof_power=-4e-7`（自定义 `_joint_power_l1`，补 τ·q̇ 能量项）。
3. 根本结论：model_17500.pt 步态差（后腿抬起/步态不稳）是训练时奖励配方残缺的历史遗留，当前 v2.8 配置已正确迁移，需重新训练才能改善步态。
4. 部署侧 `Go2FullSceneDeployCfg` 新增 `episode_length_s=100000` + `bad_orientation=None`，消除运动中机器人被传送回原点的问题。

### v2.7
1. 对照 TensorBoard 与 `go2_rl_gym`/`parkour` 后，确认问题不是 seed，而是策略收敛到“长时存活但错误步态”局部最优；`seed_123` 略优于 `seed_42`。
2. 补回参考仓库核心配方：新增 `lin_vel_z` / `base_height` reward curriculum，并给线速度/角速度 tracking 加上 `dynamic_sigma`。
3. 去掉会鼓励或过度干预青蛙跳的 shaping：关闭 `feet_air_time`、`leg_airborne_duration`、`front_rear_support_balance`，仅保留少量对称约束。
4. `bad_orientation` 放宽到 `1.2rad`，`max_init_terrain_level` 降到 `2`，先按 `parkour` 的思路让策略在更容易的起点把步态学顺。

### v0.1
1. 建立楼梯任务基础配置，先用 PPO 跑通训练链路。
2. 先验证地形、观测、奖励是否正常工作。

### v0.2
1. 发现楼梯场景下多项 L2 惩罚爆炸，训练信号被负项淹没。
2. 开始压低或移除高风险惩罚项，保留主任务正向奖励。

### v0.3
1. 处理策略更新过小问题。
2. 加入生存导向奖励，缓解早期不学习。

### v0.4
1. 重点排查 episode 过短与频繁失败。
2. 调整终止条件与楼梯相关奖励形态。

### v0.5
1. 继续稳定训练流程，清理无效 checkpoint 与过重配置。
2. 明确从平地模型 fine-tune 的训练路径。

### v0.6
1. 回退到平地预训练 checkpoint 继续微调。
2. 课程起始改为低台阶，减小冷启动难度。
3. 调整保存频率，避免无效 checkpoint 过多。

### v0.7
1. 训练环境基类改为 `UnitreeGo2RoughEnvCfg`。
2. 修复缺少 `reset_base` 导致 `ep_len=1` 的问题。

### v0.8
1. 解决训练中 `std` 数值不稳定导致的崩溃。
2. 修复 reward/梯度 NaN 传播问题。

### v0.9
1. 切换 CTS 训练链路。
2. 新增 critic 特权观测与 Teacher/Student 分流。
3. 同步加入 NaN 防护。

### v1.0
1. 定位早期“悬腿/幻影 trot”问题。
2. 修正 `_feet_regulation_reward` 的尺度与零命令门控。
3. 补充全关节默认位姿约束与接触力约束。
4. 收紧早期命令速度范围。

### v1.1
1. 定位 legged_gym 到 IsaacLab 的 reward scale 迁移错误。
2. Student obs 改回 `45D`，Teacher privileged obs 改为 `263D`。
3. 删除不合适的 `foot_contact_force` 与 `joint_deviation_all` 强惩罚。

### v1.2
1. 定位 `base_height_l2` 在楼梯上产生无界负奖励。
2. 修复 reward 截断逻辑，避免有限大负值直接累爆 episode。

### v1.3
1. 将 `base_height` 改成安全的 terrain-relative 版本。
2. 训练输出改为 `seed_<seed>` 子目录，避免双卡日志与 checkpoint 互相覆盖。
3. `CTSRunner` 补齐 `extras["log"]` 指标记录。
4. 单卡环境数限制到 `4096`，避免 `6144` 触发 OOM。

### v1.4
1. 定位 IsaacLab 默认惩罚过强导致的 dying strategy。
2. 将 `lin_vel_z_l2`、`dof_torques_l2`、`action_rate_l2` 压回可训练区间。
3. CTS 前向增加 `nan_to_num` 防御。

### v1.5
1. 楼梯相关 reward 统一回到同一套 effective 标尺。
2. `undesired_contacts`、`hip_deviation`、`feet_regulation`、`feet_air_time`、`feet_slide` 收到同量级。

### v1.6
1. 定位 `feet_regulation` 几何量实现不等价的问题。
2. 改回 terrain-relative 脚高，保留零命令门控。

### v1.7
1. 处理 CTS/PPO 数值稳定性问题。
2. 对 critic value、returns、advantages 与异常 batch 增加保护。

### v1.8
1. 指令课程改为参考 `go2_rl_gym` CTS 的训练壳子。
2. 地形课程改为按“实际推进 / 指令推进”升降级。
3. 默认训练预算拉到 `150000`。

### v1.9
1. 训练分布继续向 `go2_rl_gym` CTS 主配方对齐。
2. 对齐 student/teacher 观测缩放、部署缩放、指令课程与地形配方。
3. 去掉额外 `is_alive`、`foot_clearance`、`feet_slide`，保留必要 regularization。

### v2.0
1. 目标切到 full-scene：平地 + 楼梯共用一套可部署 CTS。
2. Student obs 从 `45D` 改为 `48D`，补回 `base_lin_vel`。
3. 部署侧新增 `Go2FullSceneDeployCfg`，按 checkpoint 自动区分 `45D` / `48D`。
4. 默认实验名改为 `unitree_go2_fullscene_cts`。

### v2.1
1. 收尾对齐 full-scene 工作流。
2. `train_stairs.py`、`scripts/run_train_stairs.sh`、TensorBoard 默认路径统一到 `unitree_go2_fullscene_cts`。

### v2.2
1. 定位 full-scene CTS 在运动期长期抬后腿的问题。
2. 新增 `_action_smoothness_penalty`、`_stand_still_joint_penalty`、`_front_rear_support_balance_penalty`。
3. `feet_regulation` 权重 `-0.001 -> -0.0005`。

### v2.3
1. 训练期新增 gait 诊断指标。
2. 记录 `rear_air_ratio`、`rear_air_ratio_moving`、`rear_air_ratio_standing`、`double_rear_air_ratio_moving`、`front_rear_contact_imbalance_moving`。

### v2.4
1. iter 1000 数据显示 `rear_air_ratio_moving` 从 0.55 单调升至 0.76，`front_rear_support_balance=-0.002` 惩罚量级仅 -0.0009/step，策略完全忽略。
2. 判断：只惩罚后腿不对称本身有缺陷——策略可把"悬空模式"从后腿转移到前腿来规避惩罚，治标不治本。
3. **根本修复**：新增 `_airborne_torque_penalty`（weight=-2e-4），对所有四腿对称施加"悬空时关节扭矩²"惩罚，自然驱动策略放松任何抬起的腿。参考 go2_rl_gym 的对称约束思路（hip_to_default 四腿统一，无前后区分）。
4. `hip_deviation` 权重从 -0.001 恢复到 go2_rl_gym 原始值 **-0.05**（之前误用了 ×0.02 的 effective 值，IsaacLab 不乘 dt，但髋关节偏差是位置量而不是速度量，IsaacLab 本身就应该用 go2_rl_gym 原始值）。
5. `front_rear_support_balance` 降为辅助项 -0.01，保留轻微前后对称监督但不再作为主要约束。

### v2.5
1. 训练 reward 去掉 `front_rear_support_balance`，不再显式区分前/后腿。
2. 保留对称主约束：`airborne_torque` + `hip_deviation` + 弱权重 `feet_regulation`。
3. 新增 `front_air_ratio_moving`、`double_front_air_ratio_moving`，专门监控前腿是否接管悬空 exploit。

### v2.6
1. TensorBoard（10500步）显示 `rear_air_ratio_moving=58.6%`、`front_air_ratio_moving=36.5%`、`front_rear_contact_imbalance=33.4%`，青蛙跳+单腿悬空严重。
2. 根因：`feet_air_time` 只在着地瞬间触发，后腿持续不落地则惩罚盲区；`airborne_torque`（-2e-4）量级不足。
3. 修复：新增 `_leg_airborne_duration_penalty`（weight=-0.5，threshold=0.3s，**四腿对称**），重启 `front_rear_support_balance`（-0.3），`airborne_torque` 提至 -5e-4，`feet_air_time` 改为 weight=0.05/threshold=0.25s，`feet_regulation` 提至 -0.001，`stand_still_joint_posture` 提至 -0.002。
4. 四腿对称设计防止策略把悬腿行为从后腿转移到前腿；`duration_threshold=0.3s` 允许正常迈步摆动相，不影响楼梯步高清障。
