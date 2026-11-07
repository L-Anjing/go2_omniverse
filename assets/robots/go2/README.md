# Go2 机器人资产

当前 Go2 由 `configs/custom_rl_env.py` 中的 Isaac Lab `UNITREE_GO2_CFG` 提供模型，USD 路径为 `${ISAAC_ASSETS_ROOT}/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd`，本地没有另外复制 USD。

本目录为后续自定义 Go2 模型及其材质、网格预留。接入自定义模型时，应显式修改对应机器人配置，并核对关节命名、惯量和执行器参数。

运动权重统一放在 `checkpoints/go2/moe_cts/`。
