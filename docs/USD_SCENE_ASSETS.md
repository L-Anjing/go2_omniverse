# USD 场景资产流水线

本工程不提交 Isaac Sim 场景资产本体。`assets/env/` 被忽略，因为其中包含本机路径软链接、大型 USD 和派生文件。可复现的脚本在 `tools/usd_scene_pipeline.py`，资产默认位于：

```text
/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5/Isaac/Environments
```

## 本机目录约定

```text
assets/env/raw/        Isaac 环境包的目录软链接，保留 Props/材质/子层依赖
assets/env/open_door/  只包含 USD 覆盖层和开门清单，不改原始场景
```

当前筛选的首批完整室内场景：

| ID | 原始 USD | 用途 |
| --- | --- | --- |
| `hospital` | `raw/Hospital/hospital.usd` | 医院、楼梯、房间门 |
| `office` | `raw/Office/office.usd` | 多层办公室、楼梯、电梯、房间门 |
| `simple_warehouse` | `raw/Simple_Warehouse/full_warehouse.usd` | 大面积货架仓储 |
| `digital_twin_warehouse` | `raw/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd` | 大型数字孪生仓储 |

Rivermark Plaza 是室外城市场景，且当前资产包有未解析引用；它不进入首批室内 tomogram 任务。

## 审计与开门

使用 IsaacLab Python，工具会自动注入 USD 运行时，无需启动 Isaac Sim：

```bash
PY=/home/carl/miniforge3/envs/env_isaaclab/bin/python

$PY tools/usd_scene_pipeline.py audit \
  assets/env/raw/Hospital/hospital.usd \
  --json /tmp/hospital_audit.json

$PY tools/usd_scene_pipeline.py audit-semantics \
  assets/env/hospital_labeled.usd \
  --json /tmp/hospital_semantic_audit.json

$PY tools/usd_scene_pipeline.py open-doors \
  assets/env/hospital_labeled.usd \
  --output assets/env/open_door/hospital_labeled_open_doors.usda \
  --json assets/env/open_door/hospital_labeled_open_doors.json
```

`audit-semantics` 会按叶子 Mesh 到父节点的顺序重新推断导航语义，并列出与 USD 已写标签冲突的 prim。新增或替换外部场景时，应先运行该检查。

开门操作仅对可确认的 `SM_Door*` 和 `GlassDoor*` 门板编写 `active = false` 覆盖；墙、门框、固定玻璃和源 USD 均不修改。对于 Hospital 仿真应优先使用 `hospital_labeled_open_doors.usda`，因为它继承现有语义标注。

## Tomogram

从 USD 到 PCT 点云再到 tomogram 的 profile 流程位于 [3d-navi/PCT_planner/tomography/SCENE_PROFILES.md](../../3d-navi/PCT_planner/tomography/SCENE_PROFILES.md)。构建大场景时必须先确认导航区域并按需要设置 XY/Z 裁剪，避免无意义的超大 GPU 栅格。
