# B2W locomotion policy

`policy_b2w_new_2.pt` (2,024,914 bytes, SHA-256
`d2cece212a16ad53ebea4a8b096090b440b5d22ed6fdbc6800ddffddfb393a3b`) is the
server's recurrent TorchScript motion policy. `core/b2w_policy.py` consumes a 60D
state and validates its 16D output (12 leg actions and 4 wheel actions). The
unified entry selects it for `ROBOT_TYPE=b2w`, currently on Hospital only. This
is an asset/entry configuration status; a server closed-loop run is not yet
verified. SRU and other high-level policies remain external.
