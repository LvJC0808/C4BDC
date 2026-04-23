# THU-BDC2026 上下文交接（2026-04-23 会话结束时）

> 本文件是 Claude Code 会话上下文的完整快照。下一个会话应**第一时间读取此文件**。

---

## 一、项目是什么

清华大学大数据竞赛 2026（THU-BDC2026）。赛题：基于沪深 300 成分股历史数据，预测未来一周收益最大的股票组合（≤5 只，权重和 ≤1）。

### 硬约束（赛方）

| 约束 | 限制 |
|---|---|
| 评测硬件 | i7-13650H / 16GB RAM / **RTX 4060 8GB** / 50GB 存储 |
| 训练时间 | ≤ 8 小时 |
| 推理时间 | ≤ 5 分钟 |
| Docker 镜像 | ≤ 10GB（不压缩） |
| 复现性 | 固定随机种子，**从训练开始复现**，结果完全一致 |
| 离线 | 复现时不得联网 |
| 数据截止 | 2026-04-01 前公开资源 |
| 报备截止 | **2026-07-18** 前发邮件到 data@tsinghua.edu.cn |

### 开发环境

- 主力开发机：RTX 5090 32GB（`/root/shared-nvme/bigdata/THU-BDC2026`）
- 队友机器：有真实 RTX 4060 可用（Linux + Windows 各一台）
- venv：`/root/shared-nvme/bigdata/THU-BDC2026/.venv`（**必须用绝对路径**，否则被系统 python3.12 抢）
- 分支：`feat/ensemble-v1`

---

## 二、方案演进时间线

### Phase 0：Baseline（赛方提供）
- StockTransformer 单模型，60 天窗口，158+39 特征

### Phase 1：三模型集成（我方原方案，Apr 20–22）
- LightGBM + DoubleEnsemble / MASTER (AAAI 2024) / StockMixer (AAAI 2024)
- 3 fold × 3 seed × walk-forward CV + 5d embargo
- rank-blend + holdout grid search 出权重 → **{lgb:0.1, master:0.9, mixer:0.0}**
- 87 天 rolling backtest：mean 5d +1.00%，t=+4.78
- 训练 6.3h on 5090（已经接近 8h 上限）

### Phase A：集成稳健化尝试（Apr 22）
- tradability filter + transaction cost → +1.00%（几乎无损）
- dynamic rolling-IC weights → -15 bp（反而更差）

### Phase B：ICIR + KL-shrink 权重学习（Apr 23 上午，本会话）
- 目标：把 {0.1, 0.9, 0.0} 替换成鲁棒权重
- 实现了完整代码：`rank_normalize_daily` / `compute_blend_ic_series` / `icir_objective` / `optimize_weights_icir` / `select_lambda_loo` / `bootstrap_weights`
- **6 个单元测试全部通过**
- 集成进 `pipeline.py cmd_train`
- 但**全量训练在 5090 上跑了 5+ 小时都没跑完**（MASTER 太慢）

### 🔑 Phase B 关键转折：Path B 发现（Apr 23 中午）
- 砍掉 MASTER/Mixer，只跑 LGB+DE → **训练 9 分钟**
- 82 天 rolling backtest：**mean 5d +1.446%**（比三模型 +1.00% 高 45 bp）
- 胜率 **84%**（vs 76%），t=**+7.24**（vs +4.78）
- **LGB-only 全面碾压三模型集成**

### 🔑 Phase B 第二发现：MASTER 跨平台不可复现（Apr 23 下午）
- 用队友 Win/Linux 4060 的 val_scores 做 ICIR 重算
- **MASTER 分数跨平台差异 mean 0.17 / max 0.95**（完全不同！）
- mixer 只有 1e-6，lgb 只有 1e-3
- 同一套 ICIR 代码在 Linux 上搜出 {0.28, 0.42, 0.30}，在 Windows 上搜出 {0, 0.98, 0.015} → **完全反向**
- 根因：MASTER 用了 `scaled_dot_product_attention`，backward 走 Memory Efficient 非确定路径
- **结论：MASTER 违反赛规复现要求，LGB-only 是唯一合规选项**

### Phase C：LGB-only 主线化 + 跨平台 Top-5 一致（Apr 23 晚，本会话）
- 写了完整 spec + plan（8 个 Task）
- 用户**已手动执行了 Task 1–7 的全部代码改动**
- 当前状态：**代码就位但未 commit**

---

## 三、当前 git 状态（极其重要）

### 最新 commit
```
8e82bf9 spec: replace simulated 4060 tests with real-hardware T1-T8 matrix
```

### 未提交的变更（用户已手动完成 Task 1–7）
```
 D CLAUDE.md                           # 已改名为 AGENTS.md
 M Dockerfile                          # 移除 torch
 M code/src/ensemble/portfolio.py      # 新增 deterministic_top_k()
 M code/src/models/__init__.py         # 只 export LGB
 M code/src/pipeline.py                # MODEL_NAMES=("lgb",)
 M code/src/verify_reproducibility.py  # 改写为 Top-5 一致性校验
 M pyproject.toml                      # torch 移到 optional
 M readme.md                           # 全文改为 LGB-only
 M test.sh                             # 改调 predict_lgb_only.py
 M train.sh                            # 改调 train_lgb_only.py
 M uv.lock
?? AGENTS.md                           # CLAUDE.md 改名
?? docs/findings/2026-04-23-4060-validation.md
?? docs/superpowers/plans/2026-04-23-lgb-mainline-reproducibility-plan.md
?? legacy/                             # master.py + stockmixer.py + pipeline_ensemble.py
?? model_lgb_only/                     # LGB-only 训练产出（gitignored）
?? scripts/predict_lgb_only.py         # 新推理入口
?? tests/test_portfolio_deterministic_topk.py
?? tests/test_verify_reproducibility.py
```

### ⚠️ 需要做的：
1. **跑 pytest 确认 Task 1–7 单测通过**（pip install pytest 上次网络超时没装上）
2. **git add + commit 全部变更**
3. **跑端到端 smoke test**（train → predict → verify 全链路）

---

## 四、关键文件索引

### Spec & Plan
| 文件 | 内容 |
|---|---|
| `docs/superpowers/specs/2026-04-23-lgb-mainline-reproducibility-design.md` | LGB 主线化 + 跨平台 Top-5 一致 设计文档（已 commit） |
| `docs/superpowers/plans/2026-04-23-lgb-mainline-reproducibility-plan.md` | 8 个 Task 的实施计划（未 commit） |
| `docs/superpowers/specs/2026-04-23-ensemble-robustification-design.md` | ICIR 集成稳健化设计文档（已 commit） |
| `docs/superpowers/plans/2026-04-23-ensemble-robustification-plan.md` | ICIR 集成 5 个 Task 的实施计划（已 commit） |

### Findings
| 文件 | 内容 |
|---|---|
| `docs/findings/2026-04-23-path-b-lgb-only.md` | Path B 发现：LGB-only +1.45% > 三模型 +1.00% |
| `docs/findings/2026-04-23-cross-platform-icir-divergence.md` | MASTER CUDA 跨平台非确定性，ICIR 权重反向 |
| `docs/findings/2026-04-23-4060-validation.md` | 4060 真机验证模板（T1–T8 待填） |

### 核心脚本
| 文件 | 用途 |
|---|---|
| `scripts/train_lgb_only.py` | **主线训练入口**（LGB+DE，9 min on 5090） |
| `scripts/predict_lgb_only.py` | **主线推理入口**（load refit → deterministic_top_k → result.csv） |
| `scripts/rolling_backtest_lgb_only.py` | LGB-only 回测 wrapper（monkey-patch MODEL_NAMES） |
| `scripts/recompute_icir_from_mates.py` | 用队友 OOF 重算 ICIR 权重 |
| `train.sh` / `test.sh` | 赛方入口，改为调 train_lgb_only / predict_lgb_only |
| `init.sh` | Docker 内依赖自检 |

### 模型产出
| 路径 | 内容 |
|---|---|
| `model_lgb_only/lgb/seed_{42,2024,7}_refit/` | LGB-only refit checkpoint（gitignored） |
| `model_lgb_only/ensemble_config.json` | `{lgb:1.0, master:0, mixer:0}` |
| `model/` | 旧三模型产出（备份在 `/root/shared-nvme/bigdata/backups/`） |

### ICIR 代码（已 commit，供未来 GBDT 集成用）
| 文件 | 函数 |
|---|---|
| `code/src/ensemble/blender.py` | `rank_normalize_daily`, `compute_blend_ic_series`, `icir_objective`, `optimize_weights_icir`, `select_lambda_loo`, `bootstrap_weights` |
| `tests/test_ensemble_blender_icir.py` | 6 个单测 |

### 测试
| 文件 | 内容 |
|---|---|
| `tests/test_portfolio_deterministic_topk.py` | deterministic_top_k 测试（2 个） |
| `tests/test_verify_reproducibility.py` | verify_topk_consistency 测试（3 个） |
| `tests/test_ensemble_blender_icir.py` | ICIR 相关测试（6 个） |
| `test/rolling_backtest_lgb_only.csv` | 82 天 LGB-only 回测原始数据 |

---

## 五、备份

| 内容 | 位置 |
|---|---|
| git tag（三模型 → LGB 切换前） | `backup-pre-pathB-20260423_110704` |
| 旧 model/ 目录完整副本 | `/root/shared-nvme/bigdata/backups/model-20260423_110704/` (190M) |
| 旧 model/ 压缩包 | `/root/shared-nvme/bigdata/backups/bdc2026-model-20260423_110704.tar.gz` (15M) |
| data 备份 | `data.tar.gz` (18M) + `data_cp1/` (47M) |

回滚命令：
```bash
git checkout backup-pre-pathB-20260423_110704
rm -rf model && cp -r /root/shared-nvme/bigdata/backups/model-20260423_110704 model
```

---

## 六、commit 历史（关键节点）

```
8e82bf9 spec: 4060 真机 T1-T8 测试矩阵
228ca63 spec: LGB-only 主线 + 跨平台 Top-5 一致
7874d74 findings: MASTER CUDA 跨平台非确定性
104a9e8 findings: Path B LGB-only 大胜三模型
1c30620 exp: LGB-only 回测 +1.446%
b04a93f feat: backtest --ensemble_method flag
0e22926 feat: ICIR 集成进 pipeline
0f59f73 feat: LOO lambda + bootstrap CI
44bd9ee feat: SLSQP 多起点 ICIR 优化器
a078439 feat: rank_normalize + blend IC + ICIR objective
5476d61 plan: ICIR 集成实施计划
792fe67 spec: ICIR 集成设计文档
c1c1184 docs: CLAUDE.md gate-skip rules（已被用户移除）
```

---

## 七、下一个会话必须做的事（按优先级）

### 🔴 立即
1. **`pip install pytest` + 跑 11 个单测**（上次网络超时没装上）
2. **git add + commit** 所有 Task 1–7 变更（当前 11 个 Modified + Untracked）
3. **端到端 smoke**：`scripts/train_lgb_only.py` → `scripts/predict_lgb_only.py` → 检查 `output/result.csv`

### 🟠 本周
4. **4060 真机实测 T1–T8**（spec 第 8 节有完整命令脚本）
5. **Docker 打包 dry-run**（`docker buildx build` → `docker compose up`）
6. **7/18 报备邮件准备**（baostock / LightGBM / TA-Lib MD5）

### 🟡 提升
7. **扩大回测窗口**（82 天 → 150+ 天）
8. **多超参 LGB 集成实验**（用已有 ICIR 代码）
9. **PPT 制作**（`scripts/make_stage_report_ppt.py` 写了一半）

---

## 八、关键数字速查

| 指标 | LGB-only | 旧三模型 | 等权 HS300 |
|---|---|---|---|
| Mean 5d return (82d) | **+1.446%** | +1.00% | +0.11% |
| 胜率 | **84.15%** | 75.86% | 60.92% |
| t-stat vs avg | **+7.24** | +4.78 | — |
| 训练时间 (5090) | **9 min** | 6.3h | — |
| GPU 需求 | **0** | 必需 | — |
| 镜像估算 | **~1.5 GB** | ~4 GB | — |
| 跨平台 score diff | **1e-3** | MASTER: 0.17 | — |

---

## 九、注意事项 & 坑

1. **永远用 `.venv/bin/python` 绝对路径**。`python -m` 在 nohup / 后台任务里会被 `/usr/bin/python3.12` 抢走。
2. **CLAUDE.md 已被用户改名为 AGENTS.md**。新版移除了所有 Working Discipline 规则，保留第一性原理指导。
3. **`code/src/pipeline.py` 已改为 `MODEL_NAMES = ("lgb",)`**，但文件里还有大量旧 import try/except。如果跑旧 predict，会因为没有 `model/master/` 报错。
4. **`model_lgb_only/` 在 `.gitignore` 里**（`1c30620` commit 加的），不会被 git add。
5. **队友的 seeds 是 `[42, 2024]`**（2 个），我方主线是 `[42, 2024, 7]`（3 个）。跨机对比时注意对齐。
6. **`init.sh` 里还 import torch**（`import lightgbm, torch, ...`），Dockerfile 瘦身后这行会报错。需要改成只检查 lightgbm。
7. **`uv.lock` 还包含 torch**，`uv sync --frozen` 可能会拉 torch。需要 `uv lock` 重新生成或用 `--no-install-package torch`。
8. **temp/icir_result_linux.json 和 temp/icir_result_windows.json** 是 ICIR 重算结果，在 temp 目录里不被 git 追踪。

---

## 十、这个文件怎么用

新会话开始时：
```
读取 /root/shared-nvme/bigdata/THU-BDC2026/docs/handoff/2026-04-23-session-handoff.md
```

然后按第七节的优先级继续工作。
