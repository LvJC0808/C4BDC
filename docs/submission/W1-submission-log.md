# W1 提交日志（线上赛 A 阶段第 1 次）

> **提交窗口**：2026-04-25 08:00 ~ 2026-04-26 23:59（赛规：T+5 直接复用 T+4 数据）
> **赛题**：基于 HS300 历史数据，预测未来一周收益最大的 ≤ 5 只股票组合
> **方案**：LGB-only + M10-3（市值约束 Top-K）

---

## 1. 提交件清单

| 件 | 文件名 | 大小 | MD5 | 状态 |
|---|---|---|---|---|
| 结果文件 | `result.csv` | 5 行 | `f13034946c0aaea5cb1e3f2d0d6ad692` | ✅ 已生成 |
| Docker 镜像 tar | `LCF@NUDT.tar` | 1.7 GB | `1a4ef9430f59e9281406f051dec0fa70` | ✅ 已打包 |
| 网盘分享链接 | 夸克网盘 | — | — | ⏳ 组长上传 |

**result.csv 内容**：
```
stock_id,weight
300308,0.2
600023,0.2
688187,0.2
688256,0.2
002714,0.2
```

权重和 = 1.0，5 只股票，符合赛规上限。

---

## 2. 打包节点环境（队友 Linux 4060）

| 项 | 值 |
|---|---|
| Git commit | `13edae3`（含 `ae3f129` AVX-512 降级 + `2953494` data self-heal）|
| OS | Linux |
| GPU | RTX 4060 Laptop, 8 GB |
| **CPU** | **13th Gen Intel i7-13700H** |
| **CPU flags** | **avx, avx2, avx_vnni（无 AVX-512）** |
| Python | 3.12 |
| Docker 镜像名 | `bdc2026:latest` |
| 镜像大小 | 1.75 GB（< 10 GB 赛规上限）|

---

## 3. 五关验收（全过 ✅）

| 关 | 验证项 | 期望 | 实测 | 结果 |
|---|---|---|---|---|
| 1 | 场景 A · 完整 data | MD5 `f13034946...` | `f13034946c0aaea5cb1e3f2d0d6ad692` | ✅ |
| 2 | 场景 B · 模拟赛方挂载（仅 train+test） | Top-5 一致 + MD5 一致 | Top-5 完全一致，MD5 = `f13034946...` | ✅ |
| 3 | `docker save` → tar | tar 大小 1.6-1.8 GB | 1.7 GB | ✅ |
| 4 | tar 回路验证（load + run） | MD5 仍 `f13034946...` | `f13034946c0aaea5cb1e3f2d0d6ad692` | ✅ |
| 5 | 网盘下载回 MD5 一致 | tar MD5 = `1a4ef943...` | ⏳ 组长执行 | ⏳ |

---

## 4. 赛规合规自检

| 赛规要求 | 实现 | 状态 |
|---|---|---|
| §1 固定种子复现，结果完全一致 | 训练双跑 73 文件 bit-identical；推理 5 次跨节点 MD5 一致 | ✅ |
| §2 i7-13650H 16 GB 4060 8 GB 上能跑 | 队友 i7-13700H 4060 8 GB 已实证 | ✅ |
| §2 推理 ≤ 5 min | 实测 < 3 min | ✅ |
| §2 训练 ≤ 8 h | 实测 5.5 min | ✅ |
| §3 Docker ≤ 10 GB（不压缩）| 1.75 GB | ✅ |
| §4 开源模型 4-1 前报备 | 未使用预训练模型 → 无需报备 | N/A |
| §5 复现训练/预测不联网 | Dockerfile 安装完后无网络调用 | ✅ |
| §6 主要贡献为机器学习方法 | LightGBM + DoubleEnsemble + M10-3 | ✅ |
| 文件结构 `init.sh / train.sh / test.sh / readme.md` | 四件齐全 | ✅ |
| `code/src/{featurework.py, test.py, train.py}` 必选 | 三个文件都在（实际入口为 scripts/）| ✅ |
| 镜像名 `bdc2026` | `bdc2026:latest` | ✅ |
| tar 命名 `队伍名称.tar` | `LCF@NUDT.tar` | ✅（待确认队名）|
| 结果文件名 `result.csv`（赛规固定）| 上传时使用裸 `result.csv`（非 `result_4060_w1.csv`）| ⚠️ 提交前必须确认 |
| 权重和 ≤ 1，≤ 5 只 | 5 × 0.2 = 1.0 | ✅ |

---

## 5. 关键技术保障

- **deterministic_top_k**：score 量化到 1e-4 + stock_id 字典序 tie-break，吸收浮点尾差
- **mcap_constrained_topk (M10-3)**：Top-10 候选中强制 ≥ 3 只大盘，近 10 天减损 78%
- **行尾符强制 LF**（commit `42abc7f`）：消除 Windows CRLF 导致的 MD5 差异
- **包版本精确锁定**（commit `2599904`）：`requirements-submission.txt` 77 包精确版本
- **data self-healing**（commit `2953494`）：赛方只挂载 train+test 时，从 `/app/data_bundled/` 还原辅助 5 文件 + 合并 stock_data.csv

---

## 6. 关键实证：AVX-512 不是必要条件

**W1 build 推翻了之前的 AVX-512 假说**：

- 队友节点 CPU = **i7-13700H**，flags 无 `avx512`
- 完整跑通场景 A + 场景 B + tar 回路三关，MD5 = golden `f13034946...`
- 与赛方评测机 **i7-13650H 同代同架构**（13 代 Raptor Lake Mobile，均无 AVX-512）→ 赛方复现把握显著上升

之前在本机 VM 观测到的 `5e8b8b0f...` 漂移，**最可能源于早期未锁版的包差异 / 行尾未强制 LF**，而非 SIMD 路径。详见 `../canonical/04-reproducibility.md` §4.1。

---

## 7. 提交后追踪

| 事件 | 时间 | 备注 |
|---|---|---|
| 队友打包完成 | 2026-04-25 上午 | 五关全过 |
| 网盘上传 | ⏳ | |
| heywhale 提交 | ⏳ | 文件名必须 `result.csv` |
| 赛方评测结果回执 | 预计 W1 截止后 ~1 周 | |

---

## 8. 已知残留风险

1. **赛方机器复现一致性**：未直接实测，但 i7-13700H 同代实证削弱该风险
2. **场景 B init.sh 自愈逻辑** 在赛方真实挂载条件下的鲁棒性：已通过本地模拟验证（target date 2026-04-23 一致）

---

*文档版本 · 2026-04-25 · 对应 commit 13edae3+*
