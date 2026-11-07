# Go2 MoE-CTS 原始权重

放置 `moe_cts_lab.pt`。这是 exporter 的输入，与仿真直接加载的 TorchScript 导出分开保存。

在仓库根目录运行 `python tools/export_lab_policy.py`，输出到同级 `../export/`。原始权重需从已有服务器文件补齐。
