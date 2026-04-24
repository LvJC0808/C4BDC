# 关键决策记录 · Decision Log (SSOT)

> **SSOT（Single Source of Truth）** · 本文是所有重大技术/方法论决策的唯一真理源。
> 新证据推翻已有决策时，必须先改本文，再同步其他 canonical/ 文档。
> 最后更新：2026-04-24 · Phase 4

---

## DR-001 · 模型路线：LGB-only（非三模型 ensemble）

**决策**：主线使用 LightGBM + DoubleEnsemble 单模型，seeds=[42, 2024, 7] 平均。

**依据**：
1. v2a 三模型 ensemble holdout grid 搜出 `{lgb:0.1, master:0.9, mixer:0.0}` → 伪集成
2. MASTER attention 在 4060/5090/Windows 5 次推理输出不一致，分差 0.17
3. 赛规第 5 条「两次推理 MD5 一致」硬约束 → MASTER 不合规
4. 训练 6.3h → 8min，节省 47× 算力换长 rolling（87d → 213d）

**状态**：✅ 已锁定。W2+ 不考虑重新加入 Transformer 系模型。

---

## DR-002 · 标签处理：rank-gauss（非 zscore）

**决策**：训练标签与特征均使用 rank-gauss 高斯化（rank / N → Φ⁻¹）。

**依据**：
- v1 阶段曾选 zscore，v2 切到 rank-gauss 后 213 天 rolling 提升 +0.18 pp
- 对极端 outlier 更鲁棒（涨停/跌停日不污染模型）

**已知反方证据**：
- 队伍 B（外部队伍）实证 zscore 更强
- **W2 待验证**：在当前 213 天框架上重新 AB（见 DR-010）

**状态**：✅ 已锁定为 W1 提交配置。

---

## DR-003 · 配权策略：equal（非 linear）

**决策**：推理阶段 Top-5 等权 0.2。

**依据**：
- v1 阶段选 linear，v2 实测发现单窗口方差极大（T=04-16 linear -2.32% vs equal +0.13%）
- 213 天 rolling AB：equal 稳健性显著优于 linear/softmax
- Top-1 winner-take-all 在坏 regime 下损失放大

**状态**：✅ 已锁定。`code/src/ensemble/allocation.py` 保留 linear/softmax 接口供实验，但生产走 equal。

---

## DR-004 · Deterministic Top-K 选股机制

**决策**：
1. 对 LightGBM 输出 score 量化到 1e-4 精度
2. 相同量化值按 `stock_id` 字典序 tie-break
3. 保证跨平台 result.csv bit-identical

**依据**：
- 赛规第 5 条「两次推理 MD5 一致」硬约束
- LightGBM 跨平台浮点尾差会改变选股顺序（实测）
- 跨 4060 Linux / 4060 Windows / 5090 Linux 各 5/5 次 MD5 一致

**意外 bonus**：该机制还吸收了"训练权重 MD5 不同但 result.csv 相同"的情况（详见 `04-reproducibility.md`）。

**状态**：✅ 已锁定。核心实现 `code/src/ensemble/portfolio.py::deterministic_top_k`。

---

## DR-005 · Risk Containment 补丁：M10-3（市值硬约束 Top-K）

**决策**：W1 提交启用 `MCAP_MIN_LARGE=3`——即从 Top-10 候选中强制挑出至少 3 只大盘股（log_mktcap ≥ HS300 中位数）组成 Top-5。

**依据**（213 天 rolling AB）：
| 档位 | 213d | 近 30d | 近 10d | Δ30 vs HS300 |
|---|---|---|---|---|
| Baseline (MCAP=0) | +2.91% | +0.17% | +0.14% | +0.30% |
| M10-2 | +2.92% | +0.03% | +0.21% | +0.16% |
| **M10-3** | **+2.95%** | **+0.25%** | **+0.25%** | **+0.38%** |

M10-3 在 213d / 近 30d / 近 10d 三个关键窗口**全部最优或并列最优**，是 dominated solution。

**状态**：✅ 已锁定为 W1 提交配置。

---

## DR-006 · Holiday-Forward-Fill：**不必做**（赛方自动处理）

**决策**：**放弃** label holiday-forward-fill 路线，不列入任何优先级。

**依据**：
- 赛方在评分时自动用 T+4 替代缺失的 T+5（参见 CLAUDE.md §"Competition Requirements"）
- 我们这边做 forward-fill 反而会与赛方评分口径错位

**历史背景**（清理冲突）：
- `docs/reports/2026-04-23-stage-report.md` 曾列为 🔴 P0
- `docs/reports/2026-04-24-stage-report.md` 曾列在「仍在排查」清单
- `docs/superpowers/plans/2026-04-24-optimization-roadmap.md` M3 曾列入
- `docs/reports/2026-04-24-path-to-now-v2.md` L187 首次明确"不用做"，但 L335 又误列入路线图

**归档规则**：上述文件已移入 `docs/archive/`，本条为 SSOT 最终结论。**任何新文档/PPT 不得再将 holiday-fill 列为待办**。

**状态**：🔒 DEAD（永久关闭）。

---

## DR-007 · W1 提交主线完备状态（W1 提交前状态）

**已完成**：
- ✅ 213 天 rolling 验证（+2.41% / 5d, t=+12.05）
- ✅ 跨平台 MD5 一致（Linux/Windows 4060 × 5 次）
- ✅ 本机训练双跑 73 文件 MD5 一致
- ✅ 本机推理双跑 result.csv MD5 一致
- ✅ Top-5 持仓锁定：`300308, 600023, 688187, 688256, 002714`
- ✅ result.csv MD5：`f13034946c0aaea5cb1e3f2d0d6ad692`
- ✅ readme.md 赛规必选文件就位
- ✅ docker-compose.yml 调试行已清理
- ✅ 18 项 pytest 全绿
- ✅ test.sh 启用 MCAP_MIN_LARGE=3

**待完成（阻断 W1 提交的未闭环项）**：
- ❌ **Docker `docker buildx build` 未实测**（预计 15-25 分钟）
- ❌ **Docker `docker compose up` 本机 dry-run 未实测**
- ❌ **Docker `docker save -o <team>.tar` 未导出**
- ❌ **.tar 文件体积未验证 ≤ 10 GB**
- ❌ **队友 4060 跨机验证回执未到**
- ❌ **.dockerignore 补完（过滤 docs/ / test/ / CLAUDE.md）**

**结论**：**主线代码与数据已 ready，但提交包工程链未闭环**。不能单独说"W1 已 ready"。

**状态**：⏳ 进行中，预计 T+4h 全闭环。

---

## DR-008 · 赛方 Baseline 权威信息

**官方代码路径**：`/root/shared-nvme/C4BD/THU-BDC2026-main/`（勿改）

**权重 MD5**：`d9c56a6a332c0fef08157ca7661a2acb`（与我们仓库 `model/60_158+39/best_model.pth` 一致）

**赛方 baseline 真实表现**（6 天 rolling 2026-04-01 ~ 04-09）：
- mean = **−0.881% / 5d**
- 固定永远押 5-8 只金融蓝筹（600919/601658/601169/601939/601328/601916/601816）

**M10-3 vs 赛方 baseline（6 天）**：+0.755% vs −0.881% = **+1.64 pp** (5/6 胜率)

**跑赢标准**：**vs 赛方 baseline**（不是 vs HS300 等权）。

**状态**：✅ 锁定。详见 `05-baseline.md`。

---

## DR-009 · 失败实验存档（警示后续）

**M1 · beta60 + log_mktcap 作为特征**（2026-04-23）
- 假设：模型看不到 style 暴露 → 加特征可解 regime shift
- 结果：所有窗口变差 -0.23 ~ -0.38 pp
- 根因：两因子已在中性化阶段用过，再作为输入造成信号重复冲突
- 教训：中性化用过的因子不能再直接做特征

**StockTransformer Ensemble 集成**（v2a）
- 结果：holdout 权重 {0.1, 0.9, 0.0} 实为 MASTER 单模型
- 根因：MASTER CUDA 非确定性 → 赛规不合规 → 舍弃
- 教训：任何 GPU 非确定性组件不进主线

**动态 IC Dynamic Weights**（Phase-A 实验）
- 结果：87 天上拖累 15 bp（+1.01% → +0.86%）
- 根因：样本不足时 rolling-IC 过度自适应
- 教训：dynamic > static 需要 150+ 天样本才安全

---

## DR-010 · W2-W4 路线图（按 ROI 排序）

| 优先级 | 动作 | 成本 | 预期 | 风险 |
|---|---|---|---|---|
| P0 | 板块均值特征（对标 2024 一等奖） | 重训 8min + AB 1h | +10-30 bp 或证伪 | 与中性化冲突（参考 DR-009） |
| P1 | SHAP 可解释性分析 | 1-2h | 答辩刚需 | 低 |
| P1 | 市场 regime 特征（指数动量+北向） | 2-3h | style 切换信号 | 中（参考 M1 教训） |
| P2 | Bootstrap CI 8 seeds | 训练 8×8min + 分析 | 7/18 报备置信区间 | 低 |
| P2 | zscore vs rank-gauss 重新 AB（见 DR-002） | 训练 8min + AB 1h | 标签稳健性验证 | 低 |

**已删除**（不再列入路线图）：
- ~~Holiday-forward-fill~~（DR-006）
- ~~Transformer 重新引入~~（DR-001）
- ~~Linear 配权~~（DR-003）

---

## 变更记录

| 日期 | 变更 | 触发 |
|---|---|---|
| 2026-04-24 | 初始创建，10 条决策固化 | 文档收束行动 |
| 2026-04-24 | DR-006 holiday-fill DEAD 结论写入 | 解决多文档冲突 |
