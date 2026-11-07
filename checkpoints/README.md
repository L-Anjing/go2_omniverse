# Locomotion checkpoints

Physics locomotion policies are organized by robot, policy, and artifact stage.
Weights and audit metadata remain ignored by Git; README files document expected
files. SRU, navigation, person-detector, and world-model weights are maintained in
their independent deployment projects.

```text
checkpoints/
├── go2/moe_cts/
│   ├── source/moe_cts_lab.pt       # original training checkpoint
│   ├── export/                     # one audited TorchScript export
│   │   ├── policy.pt
│   │   ├── config.yaml
│   │   └── export_manifest.json
│   └── legacy/model_164000.pt      # preserved older Go2 checkpoint
├── b2w/locomotion/policy_b2w_new_2.pt
├── g1/                            # no locomotion checkpoint present
└── copter/                        # no locomotion checkpoint present
```

## Server inventory and migration

| Before | Current path | Size | SHA-256 | Role / consumer |
|---|---|---:|---|---|
| `checkpoints/moe_cts_lab.pt` | `go2/moe_cts/source/moe_cts_lab.pt` | 26,367,183 B | `48e2018b18bfbaa36cec5911a9295babbdc1a3748b97e47ce2adde7e4ca86e52` | Iteration 82,500 raw Go2 MoE-CTS checkpoint; `tools/export_lab_policy.py` input. |
| `checkpoints/moe_cts_lab/{policy.pt,config.yaml}` | `go2/moe_cts/export/` | 6,398,660 B / 2,895 B | Current policy SHA `6c0d1ea4a53b5678274123c431a93e0278030a56aabf7ecdc19ea893e9dc637a` (6,398,660 B); prior export SHA `7d13310a1384c875ee3e20806d5d11ee7bac7c6b1236de9e80f9d3b65c4501ae` retained under `legacy/` and output-equivalent. | Go2 runtime default in `commands/run_sim.sh`; config source path now resolves to `../source/moe_cts_lab.pt`. |
| `checkpoints/moe_cts_lab/config.yaml` (prior) | `go2/moe_cts/export/config.yaml` | 2,850 B | `deb9663241a1839c06c2786aeeb99792945c0acd5daee30258ddf654adb0e365` | Runtime deployment config; checkpoint reference is relative to the new source directory. |
| generated with current export | `go2/moe_cts/export/export_manifest.json` | 1,075 B | `b0561aa8f1781a27ad09ca294ea3c44288a82589d4239c43ca3ea7109f678c70` | Source/export hashes, pinned source commit, interface dimensions, and TorchScript smoke-test record. |
| `checkpoints/b2w/policy_b2w_new_2.pt` | `b2w/locomotion/policy_b2w_new_2.pt` | 2,024,914 B | `d2cece212a16ad53ebea4a8b096090b440b5d22ed6fdbc6800ddffddfb393a3b` | 60D recurrent state to 16D leg/wheel actions; `core/b2w_policy.py`. |
| `checkpoints/model_164000.pt` | `go2/moe_cts/legacy/model_164000.pt` | 22,662,338 B | `3eb56954fbc48806b243bab559d6a6d5b7253e4254cb9f5cb2070f3f37d39e36` | Older unique Go2 checkpoint, retained for historical audit; not selected by current entry point. |
| `checkpoints/moe_cts_lab/{policy.pt,config.yaml}` (prior export) | `go2/moe_cts/legacy/{policy_20261022.pt,config_20261022.yaml}` | 6,398,660 B / 2,895 B | Prior policy SHA shown above; prior config SHA `a29db77374754fd14310c0099301f04671ecf51f2a58bd63fa32f6c067999419` | Exact original pair retained for audit; not selected by runtime.

The newly generated Go2 export passed TorchScript load and a finite `[1,45] → [1,12]`
zero-input call. Across three 10-step input sequences it is bitwise output-equivalent
to the prior export. Details and hashes are in `go2/moe_cts/export/export_manifest.json`.

G1 and Copter have asset/config directories but no locomotion weights. Do not infer
that model assets or an entry-point configuration imply a validated locomotion loop.
The export requires the pinned `go2_rl_lab` checkout; no Isaac/PyTorch dependencies
were upgraded during this inventory.
