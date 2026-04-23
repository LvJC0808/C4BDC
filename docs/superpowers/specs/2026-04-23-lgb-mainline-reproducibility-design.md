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

## 8. 4060 真机实测方案

我们已确认有真实 RTX 4060 硬件。以下 T1–T8 测试矩阵按优先级执行，产出数据写入 `docs/findings/2026-04-23-4060-validation.md`。

### 8.0 4060 机器初始化检查

```bash
# 硬件自检
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
# 期望：NVIDIA GeForce RTX 4060, driver ≥ 530, memory.total ≥ 8000 MiB

lscpu | grep -E "Model name|CPU\(s\):|Thread"
# 期望：i7-13650H 或同级 (≥ 10 cores)

free -g                 # 期望 total ≥ 16 GB
df -h /                 # 期望剩余 ≥ 50 GB

python3 --version       # 期望 3.12.x
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh

cd /path/to/THU-BDC2026
uv sync                 # 5–10 min
```

### 8.1 测试矩阵

| ID | 场景 | 通过条件 |
|---|---|---|
| T1 | venv train | ≤ 30 min, 无 OOM, 产出 checkpoint |
| T2 | venv predict | ≤ 1 min, result.csv 合规 |
| T3 | 同机复现 | Top-5 一致, 权重 diff ≤ 1e-6 |
| T4 | Docker 端到端 | 镜像 ≤ 2 GB, 总时长 ≤ 8h + 5min |
| T5 | 跨机 Top-5 一致 | 4060 vs 5090 交集 ≥ 4/5, w diff ≤ 0.01 |
| T6 | LGB bit-level | `LGB_NUM_THREADS=1` 下两次 checkpoint MD5 一致 |
| T7 | 显存旁路监控 | GPU mem peak ≤ 50 MiB（证明不碰 GPU） |
| T8 | legacy 三模型对照 | 预期 OOM 或超时，佐证 Path B 必要性 |

### 8.2 T1：venv train + 资源监控

```bash
nvidia-smi dmon -s um -o DT > /tmp/4060_gpu_train.log &
GPU_MON=$!

/usr/bin/time -v .venv/bin/python scripts/train_lgb_only.py 2>&1 \
    | tee temp/4060_train.log

kill $GPU_MON

grep -E "Maximum resident|Elapsed.*wall" temp/4060_train.log
awk '{print $4}' /tmp/4060_gpu_train.log | sort -nr | head -1
```

### 8.3 T2：predict + 格式校验

```bash
/usr/bin/time -v .venv/bin/python scripts/predict_lgb_only.py 2>&1 \
    | tee temp/4060_predict.log

.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('output/result.csv')
assert len(df) <= 5
assert df['weight'].sum() <= 1.0 + 1e-6
assert list(df.columns) == ['stock_id', 'weight']
print('OK', df.to_dict('records'))
"
```

### 8.4 T3：同机复现

```bash
# Run 1
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv temp/result_run1.csv

# 清缓存保证不泄漏
rm -rf temp/*.parquet model_lgb_only

# Run 2
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv temp/result_run2.csv

.venv/bin/python code/src/verify_reproducibility.py \
    --a temp/result_run1.csv --b temp/result_run2.csv
```

### 8.5 T4：Docker 端到端

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
docker images bdc2026 --format "{{.Size}}"       # ≤ 2 GB

docker compose up 2>&1 | tee temp/4060_docker.log
ls -la output/result.csv
```

### 8.6 T5：跨机 Top-5 一致（最关键）

在 5090 上跑一次并保留产出：

```bash
# 5090
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv /tmp/result_5090.csv
```

传到 4060 再跑：

```bash
# 4060
scp user@5090-host:/tmp/result_5090.csv /tmp/result_5090.csv
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv /tmp/result_4060.csv

.venv/bin/python code/src/verify_reproducibility.py \
    --cross /tmp/result_5090.csv /tmp/result_4060.csv
```

### 8.7 T6：极端 bit-level 复现

```bash
LGB_NUM_THREADS=1 .venv/bin/python scripts/train_lgb_only.py
md5sum model_lgb_only/lgb/seed_42_refit/sub_*.txt > /tmp/md5_run1.txt

rm -rf model_lgb_only temp/*.parquet

LGB_NUM_THREADS=1 .venv/bin/python scripts/train_lgb_only.py
md5sum model_lgb_only/lgb/seed_42_refit/sub_*.txt > /tmp/md5_run2.txt

diff /tmp/md5_run1.txt /tmp/md5_run2.txt         # 期望无输出
```

如果 diff 有输出，说明还有非确定性源头（典型：TA-Lib 并行），需要加 `OMP_NUM_THREADS=1` 进一步排查。

### 8.8 T7：训练时 GPU 显存监控

```bash
.venv/bin/python scripts/train_lgb_only.py &
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 5
done > /tmp/gpu_mem_samples.txt
sort -nr /tmp/gpu_mem_samples.txt | head -3
# 期望 peak ≤ 50 MiB；偏高说明有意外的 CUDA 初始化
```

### 8.9 T8：legacy 三模型对照（故意失败）

目的是产生"MASTER 在 4060 上跑不完"的实测证据，不是让它跑通。

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

timeout 3h ENSEMBLE_METHOD=legacy .venv/bin/python -m code.src.pipeline train \
    --data_path ./data --model_dir ./model --temp_dir ./temp \
    2>&1 | tee temp/4060_legacy.log

grep -iE "out of memory|killed|cuda error" temp/4060_legacy.log
tail -30 temp/4060_legacy.log
```

### 8.10 决策树

- **T1–T4 全绿** → LGB-only 主提交方案确定，直接打包镜像提交
- **T3 失败** → 有非确定性源头，跑 T6；如果 T6 也失败，排查 TA-Lib / numpy / scipy 并行
- **T5 失败（Top-5 不一致）** → quantize 从 1e-4 调到 1e-3
- **T5 权重 diff > 0.01** → 改用 `num_threads=1` 强化训练（慢 3×，仍远低于 8h）
- **T7 > 100 MiB** → grep 排查意外的 `import torch` / `import cupy`
- **T8 能跑完** → 意外，反向更新 Path B 结论

### 8.11 数据收集模板

跑完后把下表填入 `docs/findings/2026-04-23-4060-validation.md`：

| Test | Status | 实测值 | 赛规上限 |
|---|---|---|---|
| T1 train wall time | ✅/❌ | ?? min | 8 h |
| T1 RAM peak | ✅/❌ | ?? GB | 16 GB |
| T1 GPU mem peak | ✅/❌ | ?? MiB | 8 GB |
| T2 predict wall time | ✅/❌ | ?? s | 5 min |
| T3 same-machine Top-5 | ✅/❌ | — | 100% |
| T3 same-machine w diff | ✅/❌ | ?? | ≤ 1e-6 |
| T4 docker image size | ✅/❌ | ?? GB | 10 GB |
| T4 docker end-to-end | ✅/❌ | ?? min | 8h+5min |
| T5 cross-machine Top-5 交集 | ✅/❌ | ?/5 | ≥ 4 |
| T5 cross-machine w diff | ✅/❌ | ?? | ≤ 0.01 |
| T6 bit-level md5 same | ✅/❌ | — | yes |
| T7 GPU mem during train | ✅/❌ | ?? MiB | ≈ 0 |
| T8 legacy status | 记录 | OOM / 超时 / 其他 | — |

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
