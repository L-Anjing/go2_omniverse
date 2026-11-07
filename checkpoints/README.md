# 运动策略权重

按 `机器人 / 策略 / 阶段` 存放仿真运动策略。各目录的 README 纳入 Git，权重、导出配置、审计文件等内容由本目录 `.gitignore` 忽略。

```text
checkpoints/
├── go2/moe_cts/
│   ├── source/moe_cts_lab.pt       # 原始训练 checkpoint
│   └── export/
│       ├── policy.pt              # 仿真加载的 TorchScript student
│       └── config.yaml            # 同一次导出的配套配置
├── b2w/locomotion/
│   └── policy_b2w_new_2.pt         # B2W 仿真运动策略
├── g1/                            # 人形运动策略预留
└── copter/                        # 飞行器运动策略预留
```

Go2 默认导出：在仓库根目录运行 `python tools/export_lab_policy.py`。启动时可通过 `POLICY_PATH`、`POLICY_CONFIG` 指向其他兼容版本。

从服务器归并旧文件时：

| 旧位置 | 新位置 |
| --- | --- |
| `checkpoints/moe_cts_lab.pt` | `go2/moe_cts/source/moe_cts_lab.pt` |
| `checkpoints/moe_cts_lab/` 全部文件 | `go2/moe_cts/export/` |
| `checkpoints/b2w/policy_b2w_new_2.pt` | `b2w/locomotion/policy_b2w_new_2.pt` |

同一次导出的 `policy.pt`、`config.yaml` 和审计元数据一起归并；模型文件名相同不表示版本相同。归并前后可用 `sha256sum` 核对文件，存在同名文件时先核对内容。

SRU、检测器、世界模型等高层应用权重由各自部署工程管理。当前本机尚未放入运动策略权重；目录说明用于后续服务器归并。
