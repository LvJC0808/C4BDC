# Legacy Archive

本目录保存 2026-04-23 之前的三模型 ensemble 实现，原因是 Path B 已确定为主线：
- LGB-only 能满足 30 min train / 1 min predict / <= 2 GB image 目标
- MASTER / StockMixer 在 4060 环境下不稳定且不再属于主执行路径

关联文档：
- `docs/superpowers/specs/2026-04-23-lgb-mainline-reproducibility-design.md`
- `docs/findings/2026-04-23-path-b-lgb-only.md`
- `docs/findings/2026-04-23-cross-platform-icir-divergence.md`

注意：
- 归档文件不参与默认 train/test 流程
- 如需对照实验，显式设置 legacy 环境或手动引用这些文件
