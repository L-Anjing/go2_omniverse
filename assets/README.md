# Robot simulation assets

Robot configs and their referenced model resources live under `assets/robots/<robot>/`.
The robot USD stages open with the Isaac Sim 4.5 USD runtime. B2W, G1 and CF2X
reference the standard `OmniPBR.mdl`; the installed Isaac Sim tree contains it at
`/opt/isaac-sim-4.5/kit/mdl/core/Base/OmniPBR.mdl`. Go2 also references its relative
`Props/instanceable_meshes.usd` layer under `ISAAC_ASSETS_ROOT`. Keep these USD-relative
references intact; Go2 is deliberately not copied into this repository.

## Inventory on this server

| Robot / asset | Path | Size | SHA-256 | Use and code reference |
|---|---|---:|---|---|
| Go2 | `${ISAAC_ASSETS_ROOT}/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd` | 10,462 B | `d0a2c624674355af14d93a7512614fec115750d57a7fd5222c4d19234e48c11f` | Isaac Lab `UNITREE_GO2_CFG`, imported by `configs/custom_rl_env.py`; external Isaac asset, intentionally not copied into this repository. |
| B2W | `assets/robots/b2w/b2w_rsl.usd` | 44,033,387 B | `5801459641e40b9bd2b48dc76c566f483f5077ffd4097d364fb76fd81a71b0c2` | Asset and config exist; `assets/robots/b2w/config.py` and `configs/b2w_deploy.py`. Unified entry has a B2W option; Hospital launch only. |
| G1 | `assets/robots/g1/g1.usd` | 23,296,549 B | `f96da90e60841866b501fb3a8597f4ee04840c19d25af1b3d3ecf1fe98d1b753` | Asset and `assets/robots/g1/config.py` exist; not wired into the unified `run_sim.sh` robot selector. |
| Crazyflie CF2X | `assets/robots/copter/cf2x.usd` | 215,712 B | `d5e57dcdd28f929b018186aa1c35e8e8685c11ea6582f2eca2b450736ccb5ad` | Asset and `assets/robots/copter/config.py` exist; no unified launch entry or closed-loop verification. |

The USD assets are self-contained for mesh/material/texture content apart from
Isaac's built-in `OmniPBR.mdl` material. The directly referenced resources inspected on this server are:

| Referenced resource | Size | SHA-256 | Resolution |
|---|---:|---|---|
| `ISAAC_ASSETS_ROOT/Isaac/IsaacLab/Robots/Unitree/Go2/Props/instanceable_meshes.usd` | 18,802,187 B | `2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5` | Relative dependency of Go2 USD; remains under external Isaac asset root. |
| `/opt/isaac-sim-4.5/kit/mdl/core/Base/OmniPBR.mdl` | installed Kit material | `685a82c18b381f1cd29831490e4c356e61dbeac2d86803babc115cd6a4a845db` | Isaac Sim installation dependency, not copied into robot folders. |

## Robot configuration inventory

| Configuration | Size | SHA-256 | Referenced by / purpose |
|---|---:|---|---|
| `configs/custom_rl_env.py` | 14,339 B | `3feba5f85a9550250f5dcf4e9061b83740f4304489cb29da02f092b902626d49` | Imports Isaac `UNITREE_GO2_CFG`; Go2 training/deployment environment. |
| `assets/robots/b2w/config.py` | 2,425 B | `a6c576e522443c3071f39e2ed385529a0aac9141c4f72cca9b6f1555c17525e9` | B2W articulation and actuator config, imported by `configs/b2w_deploy.py`. |
| `configs/b2w_deploy.py` | 4,057 B | `137c8c08f3ef7f7d68ec7654e894219c6cf995795d96ca6c3d03ac2a8d28c392` | Unified B2W locomotion environment and RayCaster sensor. |
| `assets/robots/g1/config.py` | 6,522 B | `dadc152732a07ea2aabc59ae565f37a2f4ada3c9208b3a42e6b85d9f1af9154c` | G1 robot and actuator config; not selectable by the unified entry. |
| `assets/robots/copter/config.py` | 2,682 B | `ffaaccdc8be977456e04cf10c3b0d2c77e3e621532a5a68368999e222a0fedaf` | CF2X config; not selectable by the unified entry. |
| `assets/robots/local_models.json` | 1,189 B | `45d6b7ff779a39fb2ce9ed3f73ce2656dbc323ce085b443cc89bf4f1e74feee4` | Machine asset inventory and provenance. |
| `configs/robot_extrinsic.yaml` | 1,904 B | `a6e8619deeb7c48effddf6e9428ff48dd7359c1e5263f1e9cedafa7dcfbac9ab` | D455f/TWIN612 mount, intrinsics and stereo baseline. |
| `core/camera_extrinsic.py` | 7,462 B | `2fff7523a7ff21105d664e0082b6cd389e3b08c2f139fb18db33531adc4a0814` | Builds the sensor extrinsics from the rig config. |
| `core/omnigraph.py` | 7,380 B | `989d2b072229d00a983a52adade8ba95407f28d81566c9307299c0cad0fe6d7c` | Publishes D455f Depth and TWIN612 RGB/CameraInfo; left depth reference is optional evaluation-only. |

## Sensor configuration

The current D455f + TWIN612 layout is a simulation proxy: an independent D455f
Depth reference and two TWIN612 grayscale RGB cameras (640×512, 12 μm pixels,
25 mm lens, provisional 120 mm baseline). C++ stereo estimates sparse depth from
the two TWIN612 RGB streams and fuses it with D455f Depth. The optional left-eye
`depth_reference` is evaluation-only and is not an estimator or fusion input.
The SRU continues to use the original `front_cam` Depth and control interface.
Camera settings and mounts are in `configs/robot_extrinsic.yaml`; publication wiring
is in `core/omnigraph.py` and `core/camera_extrinsic.py`.

See [checkpoints](../checkpoints/README.md) for locomotion weights. SRU, navigation,
perception, and world-model application weights belong to their deployment projects.
