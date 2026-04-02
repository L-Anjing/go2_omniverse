# Go2 楼梯爬行训练说明

## 快速启动

```bash
# 一键启动双卡
./run_train_stairs.sh

# 监控
tail -f train_stairs_logs/train_gpu0.log | grep TRAIN
tail -f train_stairs_logs/train_gpu1.log | grep TRAIN

# 停止
kill $(cat train_stairs_logs/train_gpu0.pid) $(cat train_stairs_logs/train_gpu1.pid)
```

---

## 训练策略

### 基础思路

从已有的 `model_7850.pt`（平地步行策略）出发，在楼梯地形上继续训练（fine-tune）。
不从头训练，保留平地步行能力，只在此基础上叠加爬楼梯知识。

参考仓库：`unitree_go2/`（本地 clone），借鉴其奖励设计和终止条件策略。

### 地形配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 地形类型 | 比例 | 说明 |
|----------|------|------|
| 平地 (flat) | 15% | 机器人熟悉的安全区域 |
| 随机粗糙 (random_rough) | 15% | 与预训练地形兼容 |
| HF 锥形坡 (hf_pyramid_slope) | 10% | 过渡地形 |
| **金字塔楼梯上行** | **30%** | 主要训练目标，step_height 0.02-0.22 m |
| **金字塔楼梯下行** | **20%** | 下楼能力，step_height 0.02-0.22 m |
| 方块障碍 (boxes) | 10% | 丰富多样性 |
| 踏步宽度 | 0.28 m | 与 hospital 场景一致 |
| 课程起始等级 | **0（共 10 级）** | 第 0 级 ≈ 2 cm，第 9 级 ≈ 22 cm（医院楼梯 17cm 在第 7 级） |

**参考 go2_rl_gym 仓库策略**：混合地形防止训练初期机器人在 100% 楼梯上全部摔倒，通过课程逐步增加难度。

### 课程机制

`terrain_levels_vel`：根据机器人前进速度自动调节难度等级。
- 超过阈值速度 → 升一级（更高台阶）
- 低于阈值或摔倒 → 降一级

---

## 奖励函数调参历程

### 问题一：`dof_torques_l2` 淹没正信号（第一版）

**现象**：reward 在 0 和 -100000 之间剧烈跳动，策略无法收敛。

**根因**：`dof_torques_l2 = -0.0002` 为平地调参。楼梯上需要大力矩对抗重力，
单步惩罚达 -50000，远超正向奖励信号，PPO 梯度被负信号主导。

### 问题二：所有 L2 项在楼梯上都爆炸（第二版）

**现象**：清零 `dof_torques_l2` 后仍有 reward=-20000（ep_len=131）。

**根因**：脚踩台阶边缘时关节加速度达 10^4~10^5 rad/s²；机器人翻滚时
X/Y 轴角速度达 80~100 rad/s。各 L2 项均会爆炸：

| 惩罚项 | 爆炸原因 |
|--------|---------|
| `dof_torques_l2` | 对抗重力需要大力矩 |
| `dof_acc_l2` | 脚踩台阶边缘冲击加速度极大 |
| `action_rate_l2` | 早期策略动作剧烈变化 |
| `lin_vel_z_l2` | 机器人从台阶摔落时 z 速度大 |
| `ang_vel_xy_l2` | 机器人侧翻时角速度大 |

**修复**：全部清零，只保留纯正向信号。

### 问题三：策略锁死，4000 iter 零进步（第三版）

**现象**：ep_len 从第 50 iter 起固定在 21~91 循环，reward 无增长趋势，
训练 4000 iter 后完全无改善。

**根因**：
1. `desired_kl = 0.008` 太紧，PPO 更新步幅极小，策略无法真正改变
2. 无生存奖励，机器人没有激励活得更久
3. `learning_rate = 3e-4` 配合紧 KL，实际更新量接近零

**修复**：放宽 KL 至 0.02，恢复 lr=1e-3，加入 `is_alive` 生存奖励。

### 问题四：`base_contact` 阈值过严，ep_len 卡死（第五版）

**现象**：即使清零所有 L2 项、放宽 KL，ep_len 仍卡在 21~51，
出现大量 ep_len=1 的立即死亡。训练 9000+ iter 完全无改善。

**根因**（对比 `unitree_go2` 参考仓库发现）：
1. **`base_contact` 阈值 = 1.0 N**：台阶边缘触碰机器人躯干即立即终止 episode，
   机器人根本没机会学爬楼梯。参考仓库同样 1.0 N，但其地形是鹅卵石不是台阶，
   躯干不会碰到地形。
2. **迭代次数严重不足**：参考仓库训练 50000 iter，我们只做 5000，远不够收敛。
3. **缺少动态抬脚奖励**：参考仓库核心奖励 `dynamic_foot_clearance_reward`
   根据 height_scan 自适应地形高度，指导机器人抬脚越过台阶。我们没有这个奖励，
   机器人只会拖脚走，无法越过台阶边缘。

**修复**：
```python
# 终止条件：放宽躯干接触阈值，改用姿态角作为主要失败信号
self.terminations.base_contact.params["threshold"] = 100.0  # 1.0 → 100.0 N
self.terminations.bad_orientation = DoneTerm(
    func=mdp.bad_orientation, params={"limit_angle": 0.87}  # ~50°
)

# 动态抬脚奖励（来自参考仓库）
self.rewards.foot_clearance = RewTerm(
    func=_dynamic_foot_clearance_reward, weight=0.4,
    params={
        "sensor_cfg": SceneEntityCfg("height_scanner"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT"),
        "base_margin": 0.08, "max_additional": 0.20, "std": 0.05, "tanh_mult": 4.0,
    },
)
```
迭代次数改为 15000，KL 恢复参考仓库标准值 0.01。

---

### 问题七：ep_len=1 的根本原因 — 缺少 reset_base 事件（第七版，已修复）

**现象**：无论 base_contact 阈值多高（100N、None）、是否有 bad_orientation，ep_len 始终为 1。

**根因**：`custom_rl_env.py` 的 `EventCfg` 只有 `physics_material`（startup），
完全缺少 `reset_base`（`reset_root_state_uniform`）事件。

- 官方 `LocomotionVelocityRoughEnvCfg` 用 `reset_root_state_uniform` 在每次 episode reset 时
  根据 **地形高度** 重新放置机器人，保证机器人落在地形表面上
- 我们的自定义 env 缺少此事件 → reset 后机器人回到 URDF init_state 默认高度 →
  在高低不平的楼梯地形上，机器人可能从数十厘米高空自由落体到台阶上 →
  冲击力 >> 100N → `base_contact` 立即终止 →  **ep_len=1 一直循环**
- 即使把 base_contact 设为 None，自由落体仍会让机器人弹跳 → `bad_orientation` 触发

**修复**：将 `Go2StairTrainCfg` 的基类从 `UnitreeGo2CustomEnvCfg`（部署环境）
改为 `UnitreeGo2RoughEnvCfg`（官方训练环境），获得完整的训练基础设施：

```python
# 之前（错误）
class Go2StairTrainCfg(UnitreeGo2CustomEnvCfg):  # 缺 reset_base

# 之后（正确）
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
class Go2StairTrainCfg(UnitreeGo2RoughEnvCfg):  # 有 reset_base, reset_robot_joints 等
```

**效果**：iter 50 时 ep_len 从 1 → **892**，reward 从 0.004 → **+23.4**。

---

### 问题五：策略彻底崩溃，ep_len=1 持续 15000 iter（第六版）

**现象**：第五版整个训练过程 ep_len 恒为 1，reward ≈ 0.01，完全无学习信号。

**根因**：
1. **从崩溃的 `model_9950.pt` 继续训练**：该 checkpoint 本身来自一次失败训练，
   加载到新奖励结构后策略权重与奖励信号完全不兼容，策略无法恢复。
2. **初始地形难度太高**：`max_init_terrain_level=2`，台阶高度从 0.10 m 起。
   从平地 fine-tune 来的策略面对 10 cm 台阶就立即倒地，机器人根本无法积累任何正向奖励。
3. **`save_interval=50` 产生 310 个无用 checkpoint**，全部占据磁盘空间。

**修复（第六版）**：
- 回退到平地预训练 checkpoint：`model_7850.pt`（来自 `unitree_go2_rough/`）
- 初始台阶高度改为 0.02 m（几乎是平地），`max_init_terrain_level=0`
- 台阶高度范围 `(0.02, 0.22)` — 课程从微小台阶逐步升至医院楼梯高度
- `save_interval=200`（每 200 iter 存一次，大幅减少 checkpoint 数量）
- 删除所有 310 个 `unitree_go2_stairs/model_*.pt`（全为失败训练产物）

---

### 问题八：RuntimeError: normal expects all elements of std >= 0.0（第八版修复）

**现象**：训练跑到约 iter 500~1100 时进程崩溃，报错：
```
RuntimeError: normal expects all elements of std >= 0.0
```

**根因**：RSL-RL 默认 `noise_std_type="scalar"`，策略噪声的标准差 `std` 是一个直接的
可训练参数（不是 `log_std`）。PPO 的梯度更新没有约束它必须为正，当策略收敛到某个
高奖励区域时，`std` 被梯度推成负数，`Normal(mean, negative_std)` 采样时崩溃。

**错误尝试**：曾试图改用 `noise_std_type="log"`（`exp(log_std)` 永远 > 0），
但这会改变 checkpoint 的 key（`std` → `log_std`），导致：
- 训练 checkpoint 与仿真不兼容
- `omniverse_sim.py` 加载预训练模型时报 `Missing key: std / Unexpected key: log_std`

**正确修复（仅影响训练，不改 checkpoint 格式）**：在 `runner.alg.update` 上加一层
monkey-patch，每次参数更新后把 `std` clamp 到 `≥ 0.01`：

```python
_orig_update = runner.alg.update
def _safe_update():
    result = _orig_update()
    with torch.no_grad():
        runner.alg.policy.std.clamp_(min=0.01)
    return result
runner.alg.update = _safe_update
```

- 训练 checkpoint 仍保存 `std` key（与仿真兼容）
- 仿真 `omniverse_sim.py` 加载 `model_7850.pt` 完全不受影响
- std 不再变负，训练可持续运行

---

## 当前奖励配置（第八版）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy_exp` | +2.0 | 主信号：跟踪前进速度指令 |
| `track_ang_vel_z_exp` | +0.75 | 航向控制 |
| `is_alive` | +0.5 | 生存奖励，驱动 ep_len 增长 |
| `feet_air_time` | +0.25 | 迈步节奏 |
| `foot_clearance` | +0.4 | 动态抬脚高度（参考仓库核心奖励） |
| 所有 L2 惩罚项 | **0.0** | 全清零，防止楼梯冲击爆炸 |

每步 reward 范围：**[0, 3.9]**，PPO 梯度干净无爆炸项。

### `dynamic_foot_clearance_reward` 说明

来源：`unitree_go2/source/.../mdp/rewards.py`，复制到 `train_stairs.py`。

工作原理：
1. 从 `height_scanner` 读取机器人周围地形高度
2. 计算相对于机器人基座的最高地形点
3. 动态设定目标抬脚高度 = `base_margin` + `clamp(max_rel_height, 0, max_additional)`
4. 惩罚脚部位置与目标高度的偏差 × 脚部水平速度（只在脚运动时奖励）

在楼梯上效果：台阶越高，目标抬脚高度越大，机器人被激励抬脚越过台阶边缘。

---

## 终止条件配置（第八版）

| 终止项 | 阈值 | 说明 |
|--------|------|------|
| `time_out` | 20s | 正常 episode 超时 |
| `base_contact` | **1 N**（官方默认） | 有了正确 reset_base 后恢复原值，不再误触发 |
| `bad_orientation` | **0.87 rad（50°）** | 严重倾斜时终止（新增，补充楼梯安全检测） |

---

## 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | **3e-4** | 降低以提高数值稳定性 |
| `desired_kl` | **0.01** | 参考仓库标准值（原 0.02） |
| `max_init_terrain_level` | **0** | 课程从最简单（2cm台阶）开始 |
| `save_interval` | **200 iter** | 每 200 次迭代保存（原 50 产生 310 个无用文件） |
| `max_iterations` | **15000** | 参考仓库 50000，fine-tune 取 15000 |

---

## GPU 策略（双 RTX 4090）

```
GPU 0 (CUDA_VISIBLE_DEVICES=0) — train seed=123 → train_stairs_logs/train_gpu0.log
GPU 1 (CUDA_VISIBLE_DEVICES=1) — train seed=42  → train_stairs_logs/train_gpu1.log
```

- RSL-RL 不支持单任务跨多 GPU（无 DDP），用两个独立进程 + 不同随机种子
- 两次训练结束后取 reward 更高的 checkpoint
- 如果同时运行 `omniverse_sim.py`（hospital 推理），则 GPU 0 留给仿真，只用 GPU 1 训练

**预期资源占用：**

| | VRAM | GPU Util | 功耗 |
|-|------|----------|------|
| 单个训练进程（4096 env） | ~9-10 GB | ~94% | ~200W |
| 两个训练进程同时 | ~9-10 GB × 2 | 双卡均 ~94% | ~200W × 2 |

---

## Checkpoint 位置

```
logs/rsl_rl/unitree_go2_stairs/model_<iter>.pt
```

当前接续训练的起点：`logs/rsl_rl/unitree_go2_rough/2024-04-06_02-37-07/model_7850.pt`（平地预训练）

seed=42 的 run_name：`hospital_stairs_v8_seed42`
seed=123 的 run_name：`hospital_stairs_v8_seed123`

---
## Tensorboard  
```  
nohup conda run -n env_isaaclab tensorboard \
  --logdir /media/user/data1/carl/workspace/go2_omniverse/logs/rsl_rl/unitree_go2_stairs \
  --port 6006 --host 0.0.0.0 \
  --reload_interval 15 \
  > /tmp/tensorboard.log 2>&1 &
echo "TensorBoard PID=$!"
 
```
## 训练结束后部署

将最优 `.pt` 文件路径更新到 `omniverse_sim.py` 的 checkpoint 参数，或替换
`logs/rsl_rl/unitree_go2_rough/2024-04-06_02-37-07/model_7850.pt`。

预期收敛指标：
- ep_len 从 1~3 增长到 500+ → 机器人能持续行走不摔倒
- reward 收敛到 +2 ~ +3.9 → 机器人成功执行前进指令并爬楼梯

---

## 问题排查历史（第八版新增）

### 问题9：reward=NaN，ETA 暴增到 100h+（iter 550 起）

**现象：** iter 550 起 reward 变 NaN，ep_len 锁死在 1000，ETA 从 10h 增长到 100h+。

**根本原因：** 标量 `noise_std_type="scalar"` 参数在 PPO 多轮迭代中梯度积累变为 NaN（不是负数），导致 Normal 分布采样 NaN 动作 → 整个网络权重被 NaN 污染。

**修复过程：**
1. ❌ 修改 `actor_critic.py` 中 `clamp(min=1e-6)` — std 本身是 NaN，clamp 对 NaN 无效
2. ❌ 在 `update_distribution` 添加 debug 打印 — 确认 std 不是负数而是 NaN
3. ✅ 改用 `noise_std_type="log"`：训练用 `log_std` 参数，`exp(log_std)` 永远是正数且不会变 NaN

**兼容性问题（关键）：** `log_std` 版 checkpoint 与仿真不兼容（仿真期望 `std` key）。
- **解决方案：** 训练保存到 `unitree_go2_stairs/`，仿真从 `unitree_go2_rough/model_7850.pt` 加载 → 完全不同目录，互不影响。
- 加载时在内存中转换：`log_std = log(std.clamp(min=1e-3))`，不修改磁盘文件。

### 问题10：NaN 梯度（25% optimizer steps 被跳过，reward 逐渐下降）

**现象：** 即使用了 `log_std`，仍有大量 NaN 梯度警告，reward 从 62 下降到 22。

**根本原因：** 环境返回的 **reward**（不只是 obs）中有 NaN，来自 height_scanner 在地形边界产生 NaN → 传入 advantage 计算 → NaN 梯度。

**修复：** 在 `env.step` 和 `env.reset` 的 wrapper 中同时过滤 reward：
```python
rew = torch.nan_to_num(rew, nan=0.0, posinf=100.0, neginf=-100.0)
```

**结果：** NaN 梯度率从 25% 降到 0.88%，reward 稳定在 42-48，训练过 iter 1050+ 无崩溃。
