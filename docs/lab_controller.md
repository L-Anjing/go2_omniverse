# Go2 RL Lab student 适配记录（历史验证）

此文记录最初的 Lab 适配与新旧对照。当前工程已移除旧版 Gym 和 legacy 运行分支，默认仅使用 Lab；
新的地形限速配置与启动方式以 README 和 terrain_speed.md 为准。导航、路径规划、世界模型和 ROS2 topic 保持不变。
本地参考仓库 `go2_rl_lab`，分支 `lab`，commit
`5702ff7704c53e0b92e69876dc8da25153d36a25`。未重新 clone 或升级 Isaac Sim。

## 审计与接口

| 项目 | 旧版当前默认权重 | 新版 Lab |
|---|---|---|
| 原始文件 | `checkpoints/go2/moe_cts/legacy/model_164000.pt` | `checkpoints/go2/moe_cts/source/moe_cts_lab.pt`，iteration 82500 |
| 部署格式 | 原始 PyTorch state_dict，现场构造 actor/student | 官方 exporter 生成 TorchScript student |
| 单帧 | 45D（另支持旧版 48D） | 45D |
| 历史 | 5 帧，按时间堆叠后 flatten | 10 帧，各观测项的时间块依次排列，450D |
| 角速度 | body angular velocity × 0.25 | 相同 |
| 重力 | projected_gravity_b | 相同，站立 `[0,0,-1]` |
| 速度指令 | `[vx,vy,wz] × [2,2,0.25]` | `[vx,vy,wz] × [0.5,1,1]` |
| 指令单位/坐标 | m/s、m/s、rad/s，机器人机体系 | 相同 |
| 输入限速 | ±[1,1,1.5] | 配置项，默认 ±[1,1,1.5]；配置允许最大 ±[4.2,1,3] |
| 关节位置 | q − q_default | 相同，scale=1 |
| 关节速度 | dq × 0.05 | (dq − dq_default) × 0.05，默认 dq=0 |
| previous action | 12D | 12D 已做 raw action clipping 的上一步动作；reset 清零 |
| action | 12D，q_target=q_default+0.25a | 相同；按官方 MuJoCo 部署先将 raw action 限制在 ±100 |
| 控制周期 | 0.005s physics × 4 = 20ms | 相同 |
| PD | Kp=20，Kd=0.5，旧 DCMotor | Kp=25，Kd=0.5，Go2HV torque-speed 曲线 |
| 电机 | 23.5 Nm DC 模型 | 同向 20.2 Nm，反向 23.4 Nm，13.5–30 rad/s 线性降到零 |
| delay/friction/armature | 旧配置保持 | 0 step / 0.01 / 0，匹配官方部署及 PhysX preset |

45D 顺序为：角速度 `[0:3]`、重力 `[3:6]`、缩放后指令 `[6:9]`、
相对关节位置 `[9:21]`、关节速度 `[21:33]`、previous action `[33:45]`。
各项按官方 observation manager 先 clip 原始观测到 ±100，再 scale。
TorchScript 接收 `[1,45]`、输出 `[1,12]`；内部 `obs_history` 为 `[1,450]`。
ONNX 官方接口接收 term-wise `[1,450]` 并由调用者维护历史；本次选择 TorchScript，未生成 ONNX。
部署模块不包含 teacher 或 critic，不读取 base linear velocity、height scan 或 privileged observation。

关节顺序为 FL、FR、RL、RR，每条腿按 hip、thigh、calf。
默认位置为 `[0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5]` rad。
读取与发送均用 `preserve_order=True` 显式映射，不依赖 USD articulation 的内部排列。

现有链路是 ROS2 Twist callback → `base_command` → observation manager → policy →
action manager `JointPositionAction` → `Articulation.set_joint_position_target` →
每个 physics step 的显式 PD actuator → PhysX joint effort。
没有另一个独立 ArticulationController 接管，也不增加 ROS2 节点。
动作按仿真时间每 20ms 更新；原运行循环没有墙钟实时保证，渲染慢会使观看速度降低。

## 来源与动力学限制

导出使用本地官方 `moe_cts/modules/moe.py` 网络与 `moe_cts/exporter.py`。
薄 student 包装与官方 `StudentMoEEncoder` 相同，仅组合 MoE 与 L2Norm，避免导入训练用
TensorDict/Isaac Lab 3 依赖。actor/student 所有参数均严格加载；teacher/critic/optimizer 不进入导出。
导出配置包含源 commit、原始 ckpt SHA256、部署配置 SHA256 和官方 metadata；同目录 `export_manifest.json` 记录导出产物哈希及加载/前向 smoke test。
ckpt 不包含原训练 `params/env.yaml`、`params/agent.yaml`，因此 activation、normalization、
controller 配置以本地分支为准，不能将其称为原始训练快照。

本地训练 command curriculum 的 flat 范围为 vx=[−4.5,4.2]、vy=[−1,1]、wz=[−3,3]；
stairs 的范围为 ±[1,1,1.5]。这不是部署端的地形判别规则。
新版 USD 是未拉取内容的 Git LFS 指针，保留现有 Isaac Sim 4.5 Go2 USD。
运行时导出的 `asset_audit.json` 保存实际质量、惯量、关节限制与默认位置，用于与新版 URDF 对照。
已对照全部 19 个刚体和 12 个运动关节：质量最大误差 4.81e-7 kg，
主惯量最大误差 1.41e-8 kg·m²，关节限位最大误差 5.06e-7 rad，均在 float32 精度范围内。
PhysX 与训练的 Newton MJWarp 不保证动力学数值等价。
训练 actuator 有 0–4 physics step 随机延迟；此处采用官方独立 MuJoCo 部署的零延迟，
可在导出的 deployment 配置中设置固定 `actuator_delay_steps` 进行额外实验。

## 启动与本地验证

```bash
python tools/export_lab_policy.py
CONTROLLER=lab CMD_SOURCE=ros2 ./commands/run_sim.sh
# legacy 已移除，此处旧版结果仅作历史记录

python tests/lab_policy/validate_offline.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  python -u tests/lab_policy/validate_sim.py --headless --device cuda:0
```

验证脚本没有 ROS2 节点，使用同一 `Go2LabDeployCfg` 和 `LabStudentPolicy`，
站立与各速度 case 分别 reset；两秒站立后用一秒 ramp 给速度。
平地 case 位于与楼梯分开的平行通道，使用无限平面避免高速驶出有限 terrain tile。
楼梯为 6 级、默认每级 10cm、踏面 30cm、平台 1.2m，分别测试上行、下行和连续过渡。
先测试低速，再递增至 4.2 m/s；跌倒/机体碰撞后跳过更高速度。
每步保存 command、body velocity、tracking error、position、roll/pitch、joint target/position、
torque、base contact force 到 NPZ，摘要在 `tests/lab_policy/sim_results/summary.json`。
超过 1 rad 倾角、base 接触力超过 20N 或出现非有限值时停止该 case。
正常脚接触地面不计为碰撞；此标准不能代替全面自碰撞或寿命评估。
所有验证文件保留本地 `tests/`，不发布。

## 2026-10-22 实测结果

现有环境 GPU PhysX，单机器人，seed=42，各 case 一次 rollout。
每次 reset 后站立 2s，再用 1s ramp 输入指令；指标取 t>5s 的 20ms 采样。

| 场景 | vx 指令 m/s | 平均实际 vx m/s | vx tracking RMSE m/s |
|---|---:|---:|---:|
| 站立 | 0 | 0.001 | 0.001 |
| 平地 | 0.5 | 0.432 | 0.070 |
| 平地 | 1.0 | 0.898 | 0.104 |
| 平地 | 1.5 | 1.454 | 0.053 |
| 平地 | 2.0 | 1.985 | 0.031 |
| 平地 | 2.5 | 2.537 | 0.047 |
| 平地 | 3.0 | 3.006 | 0.027 |
| 平地 | 3.5 | 3.632 | 0.136 |
| 平地 | 4.0 | 4.165 | 0.171 |
| 平地 | 4.2 | 4.029 | 0.795 |
| 10cm 楼梯上行 | 0.5 | 0.460 | 0.070 |
| 10cm 楼梯下行 | 0.5 | 0.398 | 0.109 |
| 平地—上楼—下楼—平地 | 0.5 | 0.425 | 0.093 |

另完成横移 vy=0.5（实际 0.422 m/s）、转向 wz=0.5（实际 0.408 rad/s）、
倒退 vx=−1（实际 −0.934 m/s）。共 16 个正式 case，无非有限输出、检测到的机体碰撞或跌倒；
三个楼梯 case 均到达目标出口。楼梯最大 |roll/pitch| 为 0.062/0.324 rad，
最大采样关节力矩 20.782 Nm。完整逐步记录和所有轴的误差位于本地
`tests/lab_policy/validation_report.md`、`validation_summary.json` 和各 NPZ。

**4.2 m/s 仍属实验范围：**测试未跌倒，但实际 vx 波动约 1.77–4.64 m/s，
tracking RMSE 明显高于 4.0 m/s。不能称为稳定精确跟踪 4.2 m/s。
默认限速仍为 `[1,1,1.5]`。独立平地仿真可显式设置：

```bash
CONTROLLER=lab CUSTOM_ENV=flat MAX_VELOCITY_COMMAND="4.0 1.0 1.5" ./commands/run_sim.sh
# 4.2 仅用于进一步实验；将上面的 4.0 改为 4.2。
```

限速参数只允许上层表达更大指令，不会把上层 0.5 m/s 自动放大。
未测试高速楼梯、20cm 台阶、Hospital 实景过渡或长期稳定性，不推断这些条件下同样通过。

原 `run_sim.sh` 新旧两版均在 `CUSTOM_ENV=flat`、ROS_DOMAIN_ID=97、RTX 传感器启用时
完成 100 个有限值控制步并以 0 退出。可用 `MAX_STEPS=100` 复现有界启动检查。
两版均出现既有 `Camera.__del__` 的 `_rep_registry` 析构异常；控制链路通过不等于相机数据正确性认证。
未修改相机实现。
