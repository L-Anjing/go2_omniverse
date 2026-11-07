# B2W 机器人模型

本目录管理 B2W 的 `config.py` 与模型入口 `b2w_rsl.usd`。物理配置沿用原始训练资产的关节、执行器设置与 MIT 声明。

当前本机仅有配置，USD 待从服务器补齐。服务器已有模型的记录 SHA-256 为 `5801459641e40b9bd2b48dc76c566f483f5077ffd4097d364fb76fd81a71b0c2`，归并时可核对；模型文件及其相对引用资源应一起放在本目录。`OmniPBR.mdl` 由 Isaac Sim 解析。

运动权重统一位于 `checkpoints/b2w/locomotion/policy_b2w_new_2.pt`，对应 `configs/b2w_deploy.py` 的仿真输入输出接口。高层导航权重由外部部署工程管理。
