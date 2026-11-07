# 仿真资产与模型

## 目录约定

```text
assets/
├── robots/
│   ├── go2/                  # 本地自定义模型预留；当前使用 Isaac Lab 模型
│   ├── b2w/                  # config.py + b2w_rsl.usd（待服务器补齐）
│   ├── g1/                   # config.py + g1.usd
│   ├── copter/               # config.py + cf2x.usd
│   └── local_models.json     # 本地已有 USD 的大小与 SHA-256
└── env/big_scene/            # 共享场景、overlay 和引用资产
```

每种机器人的 `config.py` 管理其物理参数、执行器与 USD 入口。USD 以及依赖的网格、材质、纹理保留在同一机器人目录，保持 USD 内部相对引用；运行配置从自身文件位置解析 USD 路径。

运动权重放在 [checkpoints/](../checkpoints/README.md)，按机器人和策略分开。导航、SRU、检测器等应用模型在独立部署工程管理。

## 本机清单与服务器归并

| 机器人 | 模型状态 | 仿真策略状态 |
| --- | --- | --- |
| Go2 | 配置继承 `UNITREE_GO2_CFG`，USD 位于 `${ISAAC_ASSETS_ROOT}`；本机无另存 USD | `checkpoints/go2/moe_cts/` 目录已准备，待补齐 |
| B2W | 物理配置已在本地；`b2w_rsl.usd` 待从服务器补齐 | `checkpoints/b2w/locomotion/` 待补齐 |
| G1 人形 | 已有 `g1.usd` 和配置 | 预留 `checkpoints/g1/`，未接入统一启动入口 |
| Crazyflie | 已有 `cf2x.usd` 和配置 | 预留 `checkpoints/copter/`，未接入统一启动入口 |

本次保留已有两份 USD 的位置和内容，清单记录其 SHA-256，便于服务器归并时核对。服务器文件按相同目录放入；复制 USD 时一并复制它引用的文件，先比较同名文件版本。B2W USD 和全部权重保持 Git 忽略，目录结构与配置随代码分发。

共享场景保存在 `env/big_scene/<场景>/`，与机器人模型分开。场景加载和运行命令见 [主使用说明](../README.md)。
