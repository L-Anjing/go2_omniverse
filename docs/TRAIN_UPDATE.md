# Go2 训练更新
## 常用命令
```bash
# 启动训练
./scripts/run_train_stairs.sh

# 实时查看训练日志
tail -f logs/train_stairs_logs/train_gpu0.log
tail -f logs/train_stairs_logs/train_gpu1.log

# 启动TensorBoard可视化
tensorboard \
  --logdir /media/user/data1/carl/workspace/go2_omniverse/logs/rsl_rl/unitree_go2_fullscene_cts \
  --port 6006 --host 127.0.0.1
```

## 版本记录（从上到下，版本号递增）

### v0.1
1. 搭建楼梯任务基础配置，基于PPO算法完成训练链路跑通。
2. 验证地形、观测空间、奖励函数的基础功能有效性。

### v0.2
1. 定位楼梯场景下多项L2惩罚数值爆炸问题，训练信号被负向奖励淹没。
2. 调低/移除高风险惩罚项，保留主任务正向奖励，保障训练信号正常。

### v0.3
1. 解决策略更新幅度过小的训练问题。
2. 新增生存导向奖励，缓解机器人早期无学习行为的问题。

### v0.4
1. 重点排查回合时长过短、频繁失败的问题。
2. 调整训练终止条件与楼梯相关奖励的函数形态。

### v0.5
1. 持续稳定训练流程，清理无效检查点与冗余配置。
2. 确定基于平地模型微调的训练路径。

### v0.6
1. 回退至平地预训练检查点，开展楼梯任务微调。
2. 课程学习起始改为低难度台阶，降低训练冷启动难度。
3. 调整模型保存频率，减少无效检查点生成。

### v0.7
1. 训练环境基类切换为`UnitreeGo2RoughEnvCfg`。
2. 修复缺少`reset_base`导致回合长度`ep_len=1`的异常问题。

### v0.8
1. 解决训练中策略标准差`std`数值不稳定引发的训练崩溃。
2. 修复奖励/梯度NaN值传播导致的训练异常。

### v0.9
1. 切换至CTS训练链路。
2. 新增评论家网络特权观测，实现Teacher/Student模型分流。
3. 同步加入NaN值防护机制。

### v1.0
1. 定位训练早期**悬腿/幻影trot步态**问题。
2. 修正`_feet_regulation_reward`奖励尺度与零命令门控逻辑。
3. 补充全关节默认位姿约束与足端接触力约束。
4. 收紧早期指令速度范围，规范初始训练行为。

### v1.1
1. 定位legged_gym到IsaacLab的奖励缩放迁移错误。
2. Student观测维度改回`45D`，Teacher特权观测维度设为`263D`。
3. 删除不合适的强惩罚项：`foot_contact_force`、`joint_deviation_all`。

### v1.2
1. 定位`base_height_l2`在楼梯场景产生无界负奖励的问题。
2. 修复奖励截断逻辑，避免有限负值直接耗尽回合奖励。

### v1.3
1. 将`base_height`改为地形相对高度的安全计算方式。
2. 训练输出按`seed_<seed>`子目录分类，避免双卡日志/检查点覆盖。
3. `CTSRunner`补齐`extras["log"]`指标记录功能。
4. 单卡环境数限制为`4096`，避免`6144`触发内存溢出OOM。

### v1.4
1. 定位IsaacLab默认惩罚过强导致的**策略失效（dying strategy）**问题。
2. 将`lin_vel_z_l2`、`dof_torques_l2`、`action_rate_l2`惩罚调至可训练区间。
3. CTS前向推理增加`nan_to_num`数值防护。

### v1.5
1. 统一楼梯相关奖励至同一套有效标尺。
2. 对齐`undesired_contacts`、`hip_deviation`、`feet_regulation`、`feet_air_time`、`feet_slide`奖励量级。

### v1.6
1. 定位`feet_regulation`几何量实现不等价的问题。
2. 改回地形相对足端高度，保留零命令门控逻辑。

### v1.7
1. 处理CTS/PPO训练的数值稳定性问题。
2. 对评论家价值、收益、优势函数及异常批次增加防护机制。

### v1.8
1. 指令课程学习对齐`go2_rl_gym` CTS训练框架。
2. 地形课程学习按**实际推进/指令推进**比例升降级。
3. 默认训练总迭代数提升至`150000`。

### v1.9
1. 训练分布全面对齐`go2_rl_gym` CTS核心配置。
2. 对齐Student/Teacher观测缩放、部署缩放、指令/地形课程配方。
3. 移除冗余奖励：`is_alive`、`foot_clearance`、`feet_slide`，仅保留必要正则化项。

### v2.0
1. 训练目标切换为全场景：平地+楼梯共用一套可部署CTS策略。
2. Student观测维度从`45D`改为`48D`，补回`base_lin_vel`观测。
3. 部署侧新增`Go2FullSceneDeployCfg`，自动适配`45D/48D`检查点。
4. 默认实验名改为`unitree_go2_fullscene_cts`。

### v2.1
1. 完成全场景训练工作流收尾对齐。
2. 统一训练脚本、启动脚本、TensorBoard日志路径至`unitree_go2_fullscene_cts`。

### v2.2
1. 定位全场景CTS训练中机器人运动时长期抬后腿的问题。
2. 新增三项惩罚：`_action_smoothness_penalty`、`_stand_still_joint_penalty`、`_front_rear_support_balance_penalty`。
3. 调整`feet_regulation`权重：`-0.001 → -0.0005`。

### v2.3
1. 训练阶段新增步态诊断监控指标。
2. 记录核心步态指标：`rear_air_ratio`、`rear_air_ratio_moving`、`rear_air_ratio_standing`、`double_rear_air_ratio_moving`、`front_rear_contact_imbalance_moving`。

### v2.4
1. 迭代1000步数据显示：`rear_air_ratio_moving`从0.55升至0.76，前后支撑平衡惩罚量级过低，策略无响应。
2. 定位问题：仅惩罚后腿不对称存在缺陷，策略会转移悬空行为规避约束。
3. 根本修复：新增四腿对称`_airborne_torque_penalty`（权重-2e-4），惩罚悬空时关节扭矩平方。
4. 恢复`hip_deviation`权重至原始值`-0.05`，修正之前的缩放误用。
5. 下调`front_rear_support_balance`为辅助项（权重-0.01）。

### v2.5
1. 移除训练奖励中的`front_rear_support_balance`，取消前后腿显式区分。
2. 保留对称核心约束：`airborne_torque`+`hip_deviation`+弱权重`feet_regulation`。
3. 新增前腿悬空监控指标：`front_air_ratio_moving`、`double_front_air_ratio_moving`。

### v2.6
1. TensorBoard（10500步）显示：后腿悬空率58.6%、前腿36.5%，青蛙跳+单腿悬空问题严重。
2. 根因：`feet_air_time`惩罚存在盲区，`airborne_torque`惩罚量级不足。
3. 修复优化：新增四腿对称`_leg_airborne_duration_penalty`（权重-0.5，阈值0.3s）；重启`front_rear_support_balance`（-0.3）；提升`airborne_torque`至-5e-4；调整`feet_air_time`/`feet_regulation`/`stand_still_joint_posture`权重。
4. 四腿对称设计规避策略转移悬空行为，0.3s阈值不影响正常迈步与楼梯越障。

### v2.7
1. 对比TensorBoard与参考仓库，确认问题为策略收敛至**长时存活但错误步态**的局部最优，`seed_123`效果优于`seed_42`。
2. 补回参考仓库核心奖励：新增`lin_vel_z`/`base_height`奖励课程，为线速度/角速度跟踪添加`dynamic_sigma`。
3. 移除鼓励异常步态的塑形奖励：关闭`feet_air_time`、`leg_airborne_duration`、`front_rear_support_balance`，仅保留对称约束。
4. 放宽`bad_orientation`至`1.2rad`，降低`max_init_terrain_level`至`2`，优先让策略学习标准步态。

### v2.8
1. 完成配置迁移审计：对齐观测缩放、263D特权观测、全惩罚×0.02、动态标准差、指令/地形课程。
2. 补全缺失配置：`dof_pos_limits=-0.04`（防关节越限）、`dof_power=-4e-7`（自定义关节能量惩罚）。
3. 结论：历史模型步态差为奖励配方残缺导致，当前配置已对齐，需重新训练优化。
4. 部署侧`Go2FullSceneDeployCfg`新增`episode_length_s=100000`+`bad_orientation=None`，消除机器人回传问题。

### v2.9
1. TensorBoard分析（0~25500步）：前20k步训练稳定，21000步因指令课程突变导致地形等级/噪声飙升，训练崩溃。
2. 调整指令课程：第一档从20k迭代推迟至40k，第二档从50k迭代推迟至80k，给策略充足巩固时间。
3. 参数优化：`airborne_torque`权重-2e-4→-5e-4；`default_sigma`0.25→0.20；地形升级阈值size/2→size/

### v3.1
1. 指令课程优化生效，消除地形崩溃，seed_123 训练稳定
2. 修复 seed_42 地形误降级，改用固定距离判断降级条件（实际位移<1.5m才降级）
3. 补加腿部悬空时长惩罚（阈值0.5s，权重-0.3），解决后腿悬空率回升问题
4. `_airborne_torque_penalty`新增地形感知缩放：低难度地形保持全强度，高难度地形（≥level 4）缩放至0.15，允许楼梯越障时合理抬腿

### v3.2
1. 修复`leg_airborne_duration`惩罚从未生效的代码Bug：v2.7遗留的`None`赋值（1501行）覆盖了v3.1新增的`RewTerm`（1478行），导致TensorBoard无该tag、悬空抑制完全失效
2. 删除冗余的`None`覆盖，`leg_airborne_duration`（-0.3，阈值0.5s）现已真正启用
3. TensorBoard分析（3500步）：seed_123地形正常推进至5.8级，seed_42地形卡在~2级但不再下降；rear_air_ratio_moving（0.61/0.57）偏高主因是上述Bug，修复后预期下降

### v3.3
1. 对齐速度跟踪奖励权重至参考仓库：`track_lin_vel_xy` 1.5→1.0，`track_ang_vel_z` 0.75→0.5
2. 修复地形课程升降级逻辑：`move_up`阈值对齐参考仓库（tile/2=4m），`move_down`改为按指令速度等比判断（实际位移<指令速度×回合时长×50%才降级）
3. 指令课程改为三档渐进扩展（±0.5→±0.75@30k→±1.0@55k→±2.0@90k），消除两次大幅扩档导致的策略崩溃