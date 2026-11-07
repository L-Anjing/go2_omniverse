# Go2 MoE-CTS runtime export

`policy.pt`, `config.yaml`, and `export_manifest.json` belong to the same export.
`commands/run_sim.sh` selects `policy.pt` and this directory's config by default.
The manifest records source and artifact SHA-256 values, pinned exporter source,
model dimensions, runtime versions, and a TorchScript smoke check. The current
45D single-frame input maintains 10 frames internally and returns 12 actions.
