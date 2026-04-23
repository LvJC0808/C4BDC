# LGB-Only 主线 + 跨平台 Top-5 一致可复现性 设计文档

> 2026-04-23 | 分支 `feat/ensemble-v1` | 前置 findings: `2026-04-23-path-b-lgb-only.md`, `2026-04-23-cross-platform-icir-divergence.md`

## 1. 目标

把 Path B LGB-only 提升为项目主线，同时把"跨平台 Top-5 一致"做成可验证可审核的工程产物。

- 训练 ≤ 30 min，推理 ≤ 1 min，镜像 ≤ 2 GB
- 同机两次 predict Top-5 100% 一致
- 跨机 Top-5 交集 ≥ 4/5，权重 max abs diff ≤ 0.01
- 回测 mean 5d ≥ +1.3%
- MASTER/Mixer 代码归档到 `legacy/`，不删除但不在主路径上

## 2. 架构

```
train.sh → scripts/train_lgb_only.py → model/lgb/seed_*_refit/ + ensemble_config.json
test.sh  → scripts/predict_lgb_only.py → deterministic_top_k → output/result.csv
verify   → code/src/verify_reproducibility.py → Top-5 一致性校验 → model/repro_check.log
```

三个解耦单元：train（只 LGB+DE）、predict（load+score+tie-break）、verify（Top-5 比对）。

## 3. 确定性 Tie-Breaker

LGB 跨平台 score 有 1e-3 噪声。两股分数差 < 1e-3 时 `nlargest(5)` 结果平台依赖。

```python
def deterministic_top_k(df, k=5, quantize=1e-4):
    df = df.copy()
    df["score_q"] = np.round(df["score"] / quantize) * quantize
    df = df.sort_values(by=["score_q", "stock_id"], ascending=[False, True])
    return df.head(k)
```

- quantize=1e-4：比 LGB 噪声（1e-3）小一个量级，保留绝大部分 alpha
- 并列时按 stock_id 字典序（确定性且跨平台一致）
- Top-5 内等权 0.2 each 为默认

## 4. Verify 工具

```python
def verify_topk_consistency(result_a, result_b, tol_weight=0.01):
    a, b = pd.read_csv(result_a), pd.read_csv(result_b)
    same_stocks = set(a["stock_id"]) == set(b["stock_id"])
    merged = a.merge(b, on="stock_id", suffixes=("_a", "_b"))
    max_wdiff = (merged["weight_a"] - merged["weight_b"]).abs().max()
    return {"pass": same_stocks and max_wdiff <= tol_weight, ...}
```

支持双模式：`--single`（同机双跑）、`--cross machine_a/result.csv machine_b/result.csv`。

## 5. Legacy 归档

```
legacy/
├── README.md              # 归档原因 + 链到 findings 文档
├── master.py
├── stockmixer.py
└── pipeline_ensemble.py   # pipeline.py 3-model 集成段抽出
```

## 6. Dockerfile 瘦身

移除 torch + cu128，保留 TA-Lib 源码编译 + lightgbm + pandas/numpy/scipy/sklearn/pyarrow。
pyproject.toml 中 torch 移到 `[project.optional-dependencies] legacy`。

## 7. 代码改动清单

| 文件 | 动作 |
|---|---|
| `train.sh` | 改调 `scripts/train_lgb_only.py` |
| `test.sh` | 改调 `scripts/predict_lgb_only.py`（新建） |
| `scripts/predict_lgb_only.py` | 新建：load refit → score → tie-break → result.csv |
| `code/src/ensemble/portfolio.py` | 加 `deterministic_top_k` |
| `code/src/verify_reproducibility.py` | 改写为 Top-5 一致性校验 |
| `code/src/pipeline.py` | MODEL_NAMES=("lgb",)，删 master/mixer 分支 |
| `code/src/models/__init__.py` | 不再 export master/stockmixer |
| `Dockerfile` | 移除 torch |
| `pyproject.toml` | torch 移到 optional |
| `readme.md` | 改写算法章节 |
| `legacy/` | 新建，移入 master/stockmixer + README |

## 8. 4060 实测方案

### 8.1 在 5090 上模拟 4060 约束

无需真 4060 硬件即可验证最关键的约束：

```bash
# 模拟 8 GB 显存限制（LGB 不用 GPU，这步实际验证特征工程是否碰 GPU）
CUDA_VISIBLE_DEVICES="" python scripts/train_lgb_only.py

# 模拟 16 GB RAM（用 cgroups 限制内存）
systemd-run --scope -p MemoryMax=16G -p MemorySwapMax=0 \
    python scripts/train_lgb_only.py

# 模拟 CPU 性能（i7-13650H 约 6 P-core + 8 E-core）
# 限制到 6 线程模拟 4060 机器
LGB_NUM_THREADS=6 taskset -c 0-5 python scripts/train_lgb_only.py
```

三者组合的全约束模拟命令：

```bash
CUDA_VISIBLE_DEVICES="" LGB_NUM_THREADS=6 \
systemd-run --scope -p MemoryMax=16G -p MemorySwapMax=0 \
taskset -c 0-5 \
/root/shared-nvme/bigdata/THU-BDC2026/.venv/bin/python scripts/train_lgb_only.py
```

验证点：
1. 训练完成，无 OOM
2. 耗时 ≤ 8h（实际应 ≤ 30 min）
3. 产出 checkpoint 可正常 predict

### 8.2 Docker 内端到端（最接近赛方环境）

```bash
# 构建镜像
docker buildx build --platform linux/amd64 -t bdc2026 .

# 模拟赛方流程
docker compose up   # 依次跑 init.sh → train.sh → test.sh

# 挂载 data/、output/、temp/，和赛方 docker-compose.yml 完全一致
# 验证 output/result.csv 存在且格式正确
```

Docker 内自动获得：
- 固定 OS（debian bookworm）
- 固定 Python/包版本（uv.lock 锁定）
- 无 GPU（镜像内没 torch，LGB 用 CPU）

这是**最接近赛方复现环境的测试**，优先级高于真 4060 硬件。

### 8.3 队友 4060 真机验证

如果队友有 4060 台式机（赛规硬件 i7-13650H / 16GB / 4060 8GB）：

```bash
# 1. 传输镜像
scp bdc2026.tar mate@192.168.x.x:~/
# 在 mate 机器上
docker load -i bdc2026.tar
docker compose up

# 2. 收集产出
scp mate@192.168.x.x:~/output/result.csv ./test/result_4060.csv

# 3. 比对
python code/src/verify_reproducibility.py \
    --cross output/result.csv test/result_4060.csv
```

验证点：
1. Top-5 一致（tie-break 生效）
2. 权重 diff ≤ 0.01
3. 训练 ≤ 8h，推理 ≤ 5 min

### 8.4 无真 4060 时的替代

如果无法获得真 4060：
- 8.1 + 8.2 组合已覆盖 90% 风险（LGB CPU-only，无 CUDA 依赖）
- 队友已有 Linux/Windows 4060 产出的 val_scores 和 refit checkpoint，可直接跑 predict 比对
- 剩余 10% 风险：不同 CPU 微架构的浮点 reduction 差异，被 tie-break 量化消化

## 9. 测试矩阵

| 测试 | 通过条件 |
|---|---|
| `test_deterministic_top_k` | 分数差 < 1e-4 时选出相同 Top-5 |
| `test_verify_reproducibility` | 人造数据正确判 pass/fail |
| 端到端 smoke | 同机两次 predict Top-5 一致 |
| 跨机比对 | 三份 result.csv Top-5 交集 ≥ 4/5 |
| 4060 模拟 | 16GB RAM + 6 线程 + 无 GPU 下 train+predict 成功 |
| Docker 端到端 | docker compose up 产出 result.csv |

## 10. 验收标准

| 指标 | 目标 |
|---|---|
| 训练时长 | ≤ 30 min |
| 推理时长 | ≤ 1 min |
| 镜像大小 | ≤ 2 GB |
| 同机 Top-5 一致 | 100% |
| 跨机 Top-5 交集 | ≥ 4/5 |
| 跨机权重 max diff | ≤ 0.01 |
| 回测 mean 5d | ≥ +1.3% |

## 11. 风险与回退

| 风险 | 缓解 |
|---|---|
| quantize 抹 alpha | config 参数化，backtest AB 选最优 |
| 删 torch 后 build.py 报错 | 先 grep；lazy import 保护 |
| LGB 跨平台差异 > 1e-3 | 强化模式：num_threads=1 + force_col_wise |
| seeds 不对齐 | 锁 SEEDS=[42,2024] 与队友一致 |
| 赛方无 TA-Lib | Dockerfile 编译安装（已有） |

回退：`ENSEMBLE_METHOD=legacy` 仍可跑旧三模型（需自装 torch）。
