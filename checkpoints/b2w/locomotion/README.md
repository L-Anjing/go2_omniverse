# B2W 仿真运动策略

放置 `policy_b2w_new_2.pt`，由 `ROBOT_TYPE=b2w` 默认加载。该 TorchScript 策略使用 60D 状态，输出 12D 腿部动作与 4D 轮速动作。

权重需从已有服务器文件补齐。SRU 导航策略由独立部署工程管理。
