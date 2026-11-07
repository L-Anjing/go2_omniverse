# USD 地形速度限幅

## 2026-10-24 速度包络修订

`behavior_version: smooth` 启用按未知区域/楼梯距离制动和单一最终包络；
`legacy` 保留此前立即套用预算的速度算法。基础UNKNOWN预算仍为0.5m/s，
脚下未知或制动不足保留明确保护性覆盖，不把未知地形判成平地。

Lab v2托管采集自动设置 `NAV_SPEED_CONSTRAINTS=1`，从原命令接收节点消费
`/robot0/navigation_speed_constraints`。单独手动启动默认不要求该消息，原启动方式兼容。
启用后消息缺失/过期会停车；重用模拟器时3d-navi会先检查支持情况。
ROS timeline时间转换到控制时钟后再检查有效期，保留两种时间戳供审计。

`/robot0/velocity_diagnostics` 与odom同时间戳发布地形查询、Prim/包围盒、世界位姿、
前瞻交点、预算、制动距离、状态切换、保护标志及索引版本。
普通索引刷新保留上一条指令，避免凭空重启加速；policy history机制未改。

本地详细验证和全部修改文件清单见相邻工程
`3d-navi/docs/speed_revision_20261024.md`。12个独立地形/运动场景无检测到的机身碰撞/跌倒；
3条闭环路线进入目标半径，但原有稳定停止保持条件未通过，均如实记为超时。
这些结果不能作为正式导航成功或完整Hospital路线已验证的声明。

## 范围与启动

当前工程仅部署 Lab MoE-CTS student。旧 `go2_rl_gym` 子仓库、legacy 加载分支、
旧版 `rsl_rl_cts` 和 `train_stairs.py` 已移除。训练入口转发到本地 Lab 仓库；未执行训练，
未升级 Isaac Sim 4.5，未覆盖任何 checkpoint。历史权重文件不再被运行时选择。

```bash
# 默认 Lab；Go2 默认启用地形限速和平地 3.5 m/s 命令上限
bash commands/run_sim.sh

# 显式关闭 Go2 地形限速
TERRAIN_SPEED_LIMIT=0 bash commands/run_sim.sh

# 配置文件、显式全局上限（更低的全局限制仍会生效）
TERRAIN_SPEED_LIMIT=1 TERRAIN_SPEED_CONFIG=configs/terrain_speed.yaml \
  MAX_VELOCITY_COMMAND="3.0 0.4 1.0" bash commands/run_sim.sh
```

Go2 默认启用；B2W 不支持该 Go2 专用 limiter，默认关闭。
Go2 的全局命令上限默认 `[3.5,1.0,1.5]`，平地 limiter 上限为 3.5 m/s；楼梯与未知地形分别使用配置中的较低上限。更低的地形限幅仍会生效。
`TERRAIN_SPEED_LIMIT=0` 或配置 `enabled: false` 可绕过适配器。

## 执行链与文件

ROS2 callback 保留原始机体系 `vx,vy,wz` → 原 `constant_commands` →
`lab_commands` 中的可选适配器 → 45D observation 的原有 command term →
同一个 TorchScript student → 原 12D action 和显式 PD / PhysX。

| 文件 | 职责 |
|---|---|
| `configs/terrain_speed.yaml` | 集中管理速度、前瞻、滞回、足迹和索引参数 |
| `core/usd_terrain.py` | 从现有 Stage 的真实静态碰撞几何构建索引；USD notice 失效处理 |
| `core/terrain_index.py` | 分层支撑三角形、空间桶、Z 连续查询、几何台阶确认与净空检查 |
| `core/terrain_speed.py` | 与 Isaac/ROS 无关的状态机、制动包络、指令预算及平滑 |
| `core/terrain_runtime.py` | 每台机器人一个限速器，读取现有机器人状态，按控制时间缓存 |
| `configs/lab_deploy.py` | 在原 command observation term 接入；关闭时原有 clipping 保持 |
| `core/omniverse_sim.py` | Lab 加载、索引初始化、episode reset 和资源释放 |

没有新增 ROS2 节点、TF 系统、神经网络或学习任务。观测顺序、45D 单帧、10 帧历史、
command normalization、action scale、关节顺序和 PD 均保持 Lab 接口。
地形状态切换不重置 policy history；只有原有 episode reset 才重置它。

## 地形证据与 Z 区分

只索引 active、collision-enabled 的静态几何，跳过机器人和非静态 rigid body。
主程序扫描当前 `/World`，包含实际仍参与物理接触的默认地面，是否渲染可见不改变碰撞证据。
复用现有语义归一化、栏杆/门结构识别与 walkable triangle 几何判断。
使用 USD XformCache 读取组合后的世界坐标，不使用某个场景的绝对世界坐标。

- 平地：有地面证据的近水平碰撞支撑面。坡度过大或净空不足不认证高速平地。
- 楼梯：可信 stairs 语义和可行走几何，或至少三层共线、不同高度的水平踏面。
  路径中的 stairs/steps 仅作为几何确认候选，单独命名不产生 STAIRS 分类。
- 墙和其他碰撞体：检查机体高度范围内的障碍边界；无法提取支撑面的 Gprim 使用保守世界包围盒。
- UNKNOWN：没有兼容支撑面、未覆盖区域、未知支撑、坡面或不足以确认的净空。

空间桶内保留多个高度层，以机器人底部估计高度选择兼容的最高支撑面；不会把头顶楼层当作当前地面。
前瞻沿支撑高度逐点连续跟随，检查机体中心和外围足迹，支持上下楼梯、平台和两端入口。
三角形真实覆盖范围参与判断，不把整个楼梯包围盒当作可行走面。

USD 解析碰撞 Plane 不按有限 Mesh 的 transform scale 解释物理尺寸。
它在配置的 `plane_index_half_extent=100m` 世界范围内索引，超出该范围为 UNKNOWN；
普通 Mesh 的范围不会被放大。孔面、非水平面和障碍不会因名称或包围盒被填成高速平地。

索引仅初始化或静态几何/语义变更后重建。监听 resync、父级变换、points、拓扑、
语义和碰撞属性等变更；机器人正常运动不触发全场景重扫。构建失败使用空索引并降级 UNKNOWN。

## 前瞻、制动和平滑

使用完整 WXYZ quaternion 将机体系平面运动投影到世界 XY，包括楼梯上的 roll/pitch。
同时查询请求方向和实际残余运动方向，停车、倒退或转向时不会固定检查正前方。
纯转向且没有平移时，检查机体旋转足迹。

物理最小制动距离记录为 `max(0,(v_actual²-v_stairs²)/(2*decel_limit))`。
前瞻还考虑请求速度、机体半径、`brake_margin`，限制在配置的 1–8m。
制动规划使用 `decel_limit × braking_decel_factor`（初值 2×0.75=1.5 m/s²），
给跟踪误差留余量。前方楼梯入口距离决定允许速度包络，进入楼梯前降到楼梯指令范围。
对边界进行二分细化，避免 0.15m 粗采样产生阶跃限速。

线速度按方向相关的椭圆范围缩放，不逐轴改变请求方向；角速度单独限制，
并与平移共用椭球预算，避免满速平移和满速旋转同时出现。
低速请求不会放大，零请求直接保留。

`FLAT → APPROACHING_STAIRS → ON_STAIRS → LEAVING_STAIRS → FLAT` 是典型路径。
危险状态立即进入；清除楼梯前瞻、确认平地和机体完全离开时有滞回。
退出保持时间后以加速度约束恢复，不立即跳回高速。

变化率约束施加于指令幅值及 yaw。新的停止/低速请求、突发 UNKNOWN 或硬地形上限
优先于平滑，日志会用 `operator_or_hard_cap_over_slew` 标记相应情况。
改变请求方向时按新方向缩放；这不是对任意方向切换的严格三维 jerk 保证。
非有限指令置零；无效位姿、时间、速度或过大倾斜采用 UNKNOWN 范围。

## 日志和限制

```text
[velocity robot0] t=... body frame request=(...) limited=(...) actual(vx,vy,wz)=(...) speed_xy=... terrain=APPROACHING_STAIRS cap=... stairs_distance=... brake_distance=... reason=...
```

时间为仿真秒，速度为 m/s、角速度为 rad/s，距离为米。
`cap` 是请求方向的平面速度上限；组合 yaw 预算可能进一步降低输出。
状态变化时及每个配置的日志周期打印，正常查询不会每帧打印。

这是静态地形速度限制，不是路径规划、碰撞避让或安全停止认证。
UNKNOWN 的 0.5 m/s 也可能无法通过障碍、断崖或被错误标注的区域。
静态索引不保证检测运动物体；圆形足迹、层高容差和 AABB 净空判断有保守近似。
误判为 UNKNOWN 会降低速度，不会退回“未知即平地”。
高速测试仅在独立仿真中进行，不能把平地测试结果推广为高速楼梯或实体机器人安全极限。

## 本地验证

```bash
python -m unittest discover -s tests/terrain_speed -v
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  python -u tests/terrain_speed/validate_sim.py --headless --device cuda:0 \
  --output tests/terrain_speed/release_results
```

脚本不创建 ROS2 节点，不向硬件发命令。测试场景含独立高速通道、12m 接近跑道、
6 级 10cm 台阶、30cm 踏面、1.2m 顶部平台和下行台阶，以及 Z=3m 的重叠楼层。
各 case 固定 seed=42，重置后站立 2s；始终使用同一 Lab student。
记录 raw/limited/actual、状态、速度上限、楼梯距离、制动距离、状态切换时刻、
机体姿态、关节目标/实际位置、力矩和机体接触力到逐步 NPZ。

正式结果在 `tests/terrain_speed/release_results/summary.json`。
测试及报告保留在本地被忽略的 `tests/`，不提交或发布。

## 实测结果（2026-10-22）

24 项单元测试通过；8 个独立仿真 case 均完成，没有触发非有限值、机体接触力大于 20N
或倾角大于 1 rad 的停止条件。不是全面碰撞安全认证。

| 阶段 | 指令/限幅 | 平均实际 vx |
|---|---|---:|
| 独立开阔平地 | raw=3.5，limited≤3.5 | 3.609 m/s |
| 楼梯首级入口前 | limited=0.7 | 范围 0.589–0.711 m/s |
| 上行台阶 | limited=0.7 | 0.674 m/s |
| 下行台阶 | limited=0.7 | 0.572 m/s |
| 离开后恢复平地巡航 | limited 恢复至 3.5 | 3.627 m/s |
| 地图缺失/位姿故障 | limited≤0.5，UNKNOWN | 0.431 m/s |

完整过渡中，`APPROACHING_STAIRS` 从 x≈6.76m 开始，首级台阶起于 x=12m；
机体足迹接触台阶前已降速。尾部完全离开后保持约 0.32s，再逐渐恢复。
该过渡的指令最大采样减速度约 1.764 m/s²。
这些 x 坐标只是独立测试场景的数据，运行模块没有对应场景坐标常量。

停止、倒退、转向、横移，以及上下楼层 XY 重叠均完成测试。
原启动脚本地形功能开/关各完成 100 步并以 0 退出。
真实 Hospital 加载实际 USD 索引并完成 30 步静止检查，确认支撑后进入 FLAT；
本次未在 Hospital 内发高速指令，未对 warehouse 路线做动态验证。
Hospital 大索引初始化约分钟量级，日志会输出三角形/障碍数量，运行时使用缓存查询。
两种启动模式均保留既有相机 `_rep_registry` 析构告警，未修改传感器实现。

原始 checkpoint SHA256 已核对不变。完整逐步记录、曲线和检查摘要见本地
`tests/terrain_speed/validation_report.md`、`verified_metrics.json`、`approach_velocity.png`。

## 地面摩擦调整后的复测

默认地面由静/动摩擦 `0.5/0.5` 调整为 `0.9/0.8`，可通过
`GROUND_STATIC_FRICTION`、`GROUND_DYNAMIC_FRICTION` 覆盖。导入 USD 的已识别静态
地板、楼梯、坡道在 PhysX 初始化前绑定独立接触材质；保留原材质的恢复系数与组合方式，
不改写源 USD，不改关节摩擦。实例代理无法单独覆盖时输出跳过计数。

使用相同独立场景复测 8 个 case 全部完成，未触发上述跌倒/机身碰撞停止条件。
平地 raw=3.5 m/s 时平均实际 vx 为 3.621 m/s；上行台阶平均 0.683 m/s，
下行平均 0.555 m/s（limited=0.7 m/s）；完整过渡首次进入楼梯足迹区域时实际 vx 为
0.608 m/s。24 项限速器单元测试仍通过。结果与原摩擦设置接近，不能据此声称
摩擦调整带来明显提速。记录在本地 `tests/terrain_speed/friction_results/`。
通过原 `run_sim.sh` 对 Hospital 再做 30 步静止检查并以 0 退出：95 个静态可行走碰撞
几何绑定新摩擦材质，无实例代理被跳过；仍有上述既有相机析构告警。
