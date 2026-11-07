# B2W robot asset

`b2w_rsl.usd` (44,033,387 bytes, SHA-256
`5801459641e40b9bd2b48dc76c566f483f5077ffd4097d364fb76fd81a71b0c2`) and
`config.py` are present in this server checkout. The USD stage opens in Isaac Sim
4.5; its external material reference is Isaac's built-in `OmniPBR.mdl`.

The articulation and actuator configuration is used by `configs/b2w_deploy.py`.
The unified entry exposes `ROBOT_TYPE=b2w` for the Hospital scene; this records
entry adaptation, not a verified B2W closed-loop run. Locomotion weights are at
`checkpoints/b2w/locomotion/policy_b2w_new_2.pt`. High-level navigation weights
belong to their deployment project.
