# Local B2W robot

`b2w_rsl.usd` and the articulation settings in `config.py` were copied from
the B2W SRU training assets. The configuration retains the upstream MIT notice.
No source checkout is read at runtime. The USD composes all 77 robot prims
locally; `OmniPBR.mdl` is a standard Isaac Sim material resolved by Kit.

USD SHA256: `5801459641e40b9bd2b48dc76c566f483f5077ffd4097d364fb76fd81a71b0c2`.

Locomotion lives in `checkpoints/b2w/`, with the runtime interface in
`configs/b2w_deploy.py`. No high-level navigation checkpoint belongs here.
