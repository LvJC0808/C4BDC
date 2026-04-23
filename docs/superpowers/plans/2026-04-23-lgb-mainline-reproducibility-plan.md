# LGB Mainline Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前三模型集成主线切到 LGB-only 主线，并把同机/跨机 Top-5 一致性做成可验证、可审计、可交付的工程产物。

**Architecture:** 保留 `scripts/train_lgb_only.py` 作为训练入口，新建 `scripts/predict_lgb_only.py` 作为唯一推理入口，预测阶段通过 `deterministic_top_k()` 先做量化分桶再按 `stock_id` 排序打破平台差异。主流程只保留 LGB 所需代码；原 MASTER / StockMixer / 三模型 ensemble 逻辑整体归档到 `legacy/`，验证链路由 `code/src/verify_reproducibility.py` 统一输出 Top-5 / 权重差异报告。

**Tech Stack:** Python 3.12 / pandas / numpy / lightgbm / pytest / uv / Docker

**Spec:** `docs/superpowers/specs/2026-04-23-lgb-mainline-reproducibility-design.md`

---

## File Map

- `train.sh`
  责任：竞赛训练入口，改为只调用 `scripts/train_lgb_only.py`。
- `test.sh`
  责任：竞赛推理入口，改为只调用 `scripts/predict_lgb_only.py`。
- `scripts/train_lgb_only.py`
  责任：LGB-only 训练、refit、输出 `model_lgb_only/ensemble_config.json`。
- `scripts/predict_lgb_only.py`
  责任：加载 refit checkpoint，预测最新交易日，执行确定性 Top-K，输出 `output/result.csv`。
- `code/src/ensemble/portfolio.py`
  责任：新增 `deterministic_top_k()`，让 Top-K 选择平台无关。
- `code/src/verify_reproducibility.py`
  责任：从 “双次 predict 比 MD5” 改为 “Top-5 集合 / 权重差异 / 量化阈值” 校验器。
- `tests/test_portfolio_deterministic_topk.py`
  责任：验证量化排序和同分数 tie-break 行为。
- `tests/test_verify_reproducibility.py`
  责任：验证 single-run / cross-machine 两种校验模式。
- `code/src/pipeline.py`
  责任：主线只保留 LGB 相关 train / predict 逻辑，不再 import / dispatch MASTER、StockMixer。
- `code/src/models/__init__.py`
  责任：只 export LGB 相关类型。
- `legacy/README.md`
  责任：解释归档原因、列出被归档组件、链接 findings/spec。
- `legacy/master.py`
  责任：保存原 `code/src/models/master.py`。
- `legacy/stockmixer.py`
  责任：保存原 `code/src/models/stockmixer.py`。
- `legacy/pipeline_ensemble.py`
  责任：保存当前三模型 `pipeline.py` 逻辑快照，避免历史分支知识丢失。
- `Dockerfile`
  责任：移除 torch 相关依赖，控制镜像尺寸。
- `pyproject.toml`
  责任：把 `torch` 从主依赖移到 `optional-dependencies.legacy`。
- `readme.md`
  责任：文档改为 LGB-only 主线 + reproducibility 验证说明。
- `docs/findings/2026-04-23-4060-validation.md`
  责任：记录 4060 真机 T1-T8 实测结果。

## Task 1: 确定性 Top-K 选择器

**Files:**
- Create: `tests/test_portfolio_deterministic_topk.py`
- Modify: `code/src/ensemble/portfolio.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_portfolio_deterministic_topk.py
import pandas as pd

from code.src.ensemble.portfolio import deterministic_top_k


def test_deterministic_top_k_uses_stock_id_tie_break_after_quantize():
    df = pd.DataFrame(
        {
            "stock_id": ["sz000002", "sz000001", "sz000003", "sz000004"],
            "score": [0.81234, 0.81231, 0.79001, 0.78000],
        }
    )

    out = deterministic_top_k(df, k=2, quantize=1e-4)

    assert out["stock_id"].tolist() == ["sz000001", "sz000002"]
    assert out["score_q"].tolist() == [0.8123, 0.8123]


def test_deterministic_top_k_preserves_stronger_bucket_before_tie_break():
    df = pd.DataFrame(
        {
            "stock_id": ["sz000003", "sz000001", "sz000002"],
            "score": [0.70019, 0.70011, 0.70009],
        }
    )

    out = deterministic_top_k(df, k=2, quantize=1e-4)

    assert out["stock_id"].tolist() == ["sz000003", "sz000001"]
    assert out["score_q"].tolist() == [0.7002, 0.7001]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_portfolio_deterministic_topk.py -v`
Expected: FAIL with `ImportError: cannot import name 'deterministic_top_k'`

- [ ] **Step 3: 实现最小代码**

在 `code/src/ensemble/portfolio.py` 追加：

```python
def deterministic_top_k(
    df: pd.DataFrame,
    k: int = 5,
    score_col: str = "score",
    id_col: str = "stock_id",
    quantize: float = 1e-4,
) -> pd.DataFrame:
    """Stable Top-K selection across platforms with score quantization + lexical tie-break."""
    out = df.copy()
    out["score_q"] = np.round(out[score_col] / quantize) * quantize
    out = out.sort_values(by=["score_q", id_col], ascending=[False, True], kind="mergesort")
    return out.head(k).reset_index(drop=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_portfolio_deterministic_topk.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add code/src/ensemble/portfolio.py tests/test_portfolio_deterministic_topk.py
git commit -m "feat(portfolio): add deterministic top-k selection"
```

---

## Task 2: LGB-only 推理入口和竞赛脚本切换

**Files:**
- Create: `scripts/predict_lgb_only.py`
- Modify: `train.sh`
- Modify: `test.sh`

- [ ] **Step 1: 写失败测试**

在仓库根目录执行文件存在性和 CLI 冒烟检查：

```bash
test -f scripts/predict_lgb_only.py
python -m py_compile scripts/predict_lgb_only.py
```

期望：第一条失败，因为文件不存在。

- [ ] **Step 2: 新建最小实现**

创建 `scripts/predict_lgb_only.py`：

```python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from code.src import config
from code.src.features.build import build_feature_sets
from code.src.pipeline import _ensure_dirs, _load_model, _lgb_xy, _predict_scores, _sorted_union_dates, set_global_seed
from code.src.ensemble.portfolio import deterministic_top_k

DATA = "./data"
MODEL_DIR = "./model_lgb_only"
TEMP = "./temp"
OUTPUT = "./output/result.csv"


def main() -> None:
    set_global_seed(config.SEED)
    _ensure_dirs(TEMP, os.path.dirname(OUTPUT))

    with open(os.path.join(MODEL_DIR, "ensemble_config.json")) as f:
        cfg = json.load(f)

    fsets = build_feature_sets(DATA, TEMP, use_cache=True)
    fset = fsets["lgb"]
    panel = fset["panel"].copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    target_date = _sorted_union_dates(fsets).iloc[-1]
    today = panel[panel["datetime"] == target_date].copy()

    seed_scores = []
    for seed in cfg["seeds"]:
        mdl = _load_model("lgb", os.path.join(MODEL_DIR, "lgb", f"seed_{seed}_refit"))
        seed_scores.append(_predict_scores("lgb", mdl, today, fset["feature_cols"], None))

    scores = pd.concat(seed_scores, ignore_index=True)
    scores = scores.groupby(["instrument", "datetime"])["score"].mean().reset_index()
    ranked = deterministic_top_k(
        scores.rename(columns={"instrument": "stock_id"})[["stock_id", "score"]],
        k=int(cfg.get("top_k", 5)),
        quantize=1e-4,
    )
    ranked["weight"] = 1.0 / len(ranked)
    ranked[["stock_id", "weight"]].to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 切换竞赛入口脚本**

把 `train.sh` 改成：

```bash
#!/bin/bash
set -e
cd /app
python scripts/train_lgb_only.py
```

把 `test.sh` 改成：

```bash
#!/bin/bash
set -e
cd /app
python scripts/predict_lgb_only.py
```

- [ ] **Step 4: 跑语法和入口校验**

Run: `python -m py_compile scripts/predict_lgb_only.py`
Expected: no output

Run: `bash -n train.sh && bash -n test.sh`
Expected: no output

- [ ] **Step 5: Commit**

```bash
git add scripts/predict_lgb_only.py train.sh test.sh
git commit -m "feat(cli): switch competition entrypoints to lgb-only scripts"
```

---

## Task 3: 可审计的 reproducibility verifier

**Files:**
- Create: `tests/test_verify_reproducibility.py`
- Modify: `code/src/verify_reproducibility.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_verify_reproducibility.py
from pathlib import Path

import pandas as pd

from code.src.verify_reproducibility import verify_topk_consistency


def _write_result(path: Path, rows):
    pd.DataFrame(rows, columns=["stock_id", "weight"]).to_csv(path, index=False)


def test_verify_topk_consistency_accepts_small_weight_noise(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(a, [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)])
    _write_result(b, [("000001", 0.201), ("000002", 0.199), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)])

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=5)

    assert result["pass"] is True
    assert result["same_stocks"] is True
    assert result["intersection_size"] == 5


def test_verify_topk_consistency_accepts_cross_machine_overlap_four_of_five(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(a, [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)])
    _write_result(b, [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000006", 0.2)])

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=4)

    assert result["pass"] is True
    assert result["same_stocks"] is False
    assert result["intersection_size"] == 4


def test_verify_topk_consistency_rejects_insufficient_overlap(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write_result(a, [("000001", 0.2), ("000002", 0.2), ("000003", 0.2), ("000004", 0.2), ("000005", 0.2)])
    _write_result(b, [("000001", 0.2), ("000002", 0.2), ("000006", 0.2), ("000007", 0.2), ("000008", 0.2)])

    result = verify_topk_consistency(a, b, tol_weight=0.01, min_intersection=4)

    assert result["pass"] is False
    assert result["intersection_size"] == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_verify_reproducibility.py -v`
Expected: FAIL with `ImportError: cannot import name 'verify_topk_consistency'`

- [ ] **Step 3: 重写验证器**

把 `code/src/verify_reproducibility.py` 改成：

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def verify_topk_consistency(
    result_a,
    result_b,
    tol_weight: float = 0.01,
    min_intersection: int = 5,
) -> dict:
    a = pd.read_csv(result_a)
    b = pd.read_csv(result_b)
    set_a = set(a["stock_id"])
    set_b = set(b["stock_id"])
    merged = a.merge(b, on="stock_id", how="inner", suffixes=("_a", "_b"))
    intersection_size = len(set_a & set_b)
    max_wdiff = float((merged["weight_a"] - merged["weight_b"]).abs().max()) if not merged.empty else float("inf")
    return {
        "pass": intersection_size >= min_intersection and max_wdiff <= tol_weight,
        "same_stocks": set_a == set_b,
        "intersection_size": intersection_size,
        "only_a": sorted(set_a - set_b),
        "only_b": sorted(set_b - set_a),
        "max_weight_diff": max_wdiff,
        "tol_weight": tol_weight,
        "min_intersection": min_intersection,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--single", nargs=2, metavar=("RUN1", "RUN2"))
    ap.add_argument("--cross", nargs=2, metavar=("MACHINE_A", "MACHINE_B"))
    ap.add_argument("--log", default="model/repro_check.log")
    ap.add_argument("--tol-weight", type=float, default=0.01)
    args = ap.parse_args()

    left, right = args.single or args.cross or (args.a, args.b)
    if not left or not right:
        raise SystemExit("must provide --single, --cross, or --a/--b")

    min_intersection = 5 if args.single else 4 if args.cross else 5
    result = verify_topk_consistency(
        left,
        right,
        tol_weight=args.tol_weight,
        min_intersection=min_intersection,
    )
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] {json.dumps(result, ensure_ascii=False)}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_verify_reproducibility.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add code/src/verify_reproducibility.py tests/test_verify_reproducibility.py
git commit -m "feat(repro): verify top-k consistency across runs and machines"
```

---

## Task 4: Legacy 归档当前三模型实现

**Files:**
- Create: `legacy/README.md`
- Create: `legacy/master.py`
- Create: `legacy/stockmixer.py`
- Create: `legacy/pipeline_ensemble.py`

- [ ] **Step 1: 创建归档说明**

创建 `legacy/README.md`：

```markdown
# Legacy Archive

本目录保存 2026-04-23 之前的三模型 ensemble 实现，原因是 Path B 已确定为主线：
- LGB-only 能满足 30 min train / 1 min predict / <= 2 GB image 目标
- MASTER / StockMixer 在 4060 环境下不稳定且不再属于主执行路径

关联文档：
- `docs/superpowers/specs/2026-04-23-lgb-mainline-reproducibility-design.md`
- `docs/findings/2026-04-23-path-b-lgb-only.md`
- `docs/findings/2026-04-23-cross-platform-icir-divergence.md`
```

- [ ] **Step 2: 先归档旧模型实现，再动主线**

Run:

```bash
mkdir -p legacy
cp code/src/models/master.py legacy/master.py
cp code/src/models/stockmixer.py legacy/stockmixer.py
cp code/src/pipeline.py legacy/pipeline_ensemble.py
```

Expected: three files created under `legacy/` before any mainline refactor

- [ ] **Step 3: 校验归档文件存在**

Run: `test -f legacy/README.md && test -f legacy/master.py && test -f legacy/stockmixer.py && test -f legacy/pipeline_ensemble.py`
Expected: no output

- [ ] **Step 4: 补充目录可读性**

在 `legacy/README.md` 末尾追加：

```markdown
注意：
- 归档文件不参与默认 train/test 流程
- 如需对照实验，显式设置 legacy 环境或手动引用这些文件
```

- [ ] **Step 5: Commit**

```bash
git add legacy/README.md legacy/master.py legacy/stockmixer.py legacy/pipeline_ensemble.py
git commit -m "chore(legacy): archive ensemble-era models and pipeline snapshot"
```

---

## Task 5: 主线 pipeline 切到 LGB-only

**Files:**
- Modify: `code/src/pipeline.py`
- Modify: `code/src/models/__init__.py`

- [ ] **Step 1: 写失败测试**

```bash
python - <<'PY'
from code.src import pipeline
assert pipeline.MODEL_NAMES == ("lgb",)
PY
```

Expected: FAIL with `AssertionError` because current tuple is `("lgb", "master", "mixer")`.

- [ ] **Step 2: 删掉主线上的非 LGB 模型常量和导出**

把 `code/src/pipeline.py` 里的：

```python
MODEL_NAMES = ("lgb", "master", "mixer")
```

改成：

```python
MODEL_NAMES = ("lgb",)
```

把 `code/src/models/__init__.py` 改成：

```python
from .lgb_de import LGBModel, DoubleEnsembleModel

__all__ = [
    "LGBModel",
    "DoubleEnsembleModel",
]
```

- [ ] **Step 3: 删掉 predict/train 中对 master / mixer 的主线路径依赖**

在 `code/src/pipeline.py` 中保留：

```python
from .models.lgb_de import DoubleEnsembleModel
from .ensemble.portfolio import deterministic_top_k
```

删除：

```python
from .models.master import MasterTrainer
from .models.stockmixer import MixerTrainer
```

并把最终选股改成：

```python
today_df = blended_today[blended_today["datetime"] == target_date].copy()
portfolio = deterministic_top_k(
    today_df.rename(columns={"instrument": "stock_id", "final_score": "score"})[["stock_id", "score"]],
    k=top_k,
    quantize=1e-4,
)
portfolio["weight"] = 1.0 / len(portfolio)
portfolio[["stock_id", "weight"]].to_csv(args.output_path, index=False)
```

- [ ] **Step 4: 跑最小校验**

Run: `python - <<'PY'\nfrom code.src import pipeline\nprint(pipeline.MODEL_NAMES)\nPY`
Expected: `('lgb',)`

Run: `python - <<'PY'\nfrom code.src.models import __all__\nprint(__all__)\nPY`
Expected: `['LGBModel', 'DoubleEnsembleModel']`

- [ ] **Step 5: Commit**

```bash
git add code/src/pipeline.py code/src/models/__init__.py
git commit -m "refactor(pipeline): make lgb-only the mainline model path"
```

---

## Task 6: 依赖与镜像瘦身

**Files:**
- Modify: `Dockerfile`
- Modify: `pyproject.toml`

- [ ] **Step 1: 写失败检查**

Run: `rg -n 'torch|cu128' Dockerfile pyproject.toml`
Expected: current output contains `torch>=2.6.0` and `pytorch-cu128`

- [ ] **Step 2: 修改 Python 依赖**

把 `pyproject.toml` 主依赖改成：

```toml
dependencies = [
    "akshare>=1.18.28",
    "baostock>=0.8.9",
    "docker>=7.1.0",
    "joblib>=1.5.2",
    "lightgbm>=4.6",
    "numpy>=1.26",
    "pandas>=2.3.2",
    "pyarrow>=16.0",
    "scipy>=1.11",
    "scikit-learn>=1.7.2",
    "seaborn>=0.13.2",
    "ta-lib>=0.6.8",
    "tensorboard>=2.20.0",
    "tensorboardx>=2.6.4",
    "tqdm>=4.67.1",
]

[project.optional-dependencies]
legacy = [
    "torch>=2.6.0",
]
```

- [ ] **Step 3: 修改镜像构建**

把 `Dockerfile` 中依赖安装段保持为：

```dockerfile
# Install dependencies
RUN uv sync --frozen --no-dev
```

并确认文件里没有任何 `torch` / `cu128` / PyTorch index 相关内容。

- [ ] **Step 4: 跑文本校验**

Run: `rg -n 'torch|cu128|pytorch-cu128' Dockerfile pyproject.toml`
Expected: only `pyproject.toml` 的 `[project.optional-dependencies] legacy` 中保留 `torch` 一处

- [ ] **Step 5: Commit**

```bash
git add Dockerfile pyproject.toml
git commit -m "build: remove torch from mainline runtime dependencies"
```

---

## Task 7: README 和 4060 验证记录模板

**Files:**
- Modify: `readme.md`
- Create: `docs/findings/2026-04-23-4060-validation.md`

- [ ] **Step 1: 改写 README 目标与入口**

把 `readme.md` 的算法和脚本部分改成：

```markdown
## 算法主线

当前主线为 LGB + DoubleEnsemble。
训练入口：`bash train.sh` -> `python scripts/train_lgb_only.py`
推理入口：`bash test.sh` -> `python scripts/predict_lgb_only.py`

输出采用确定性 Top-5：
- score 先按 `1e-4` 量化
- 同桶内按 `stock_id` 字典序稳定排序
- 默认 5 只等权，各 `0.2`
```

- [ ] **Step 2: 补 reproducibility 文档段落**

在 `readme.md` 新增：

```markdown
## 可复现性校验

- 同机双跑：`python code/src/verify_reproducibility.py --single temp/result_run1.csv temp/result_run2.csv`
- 跨机对比：`python code/src/verify_reproducibility.py --cross /tmp/result_5090.csv /tmp/result_4060.csv`
- 通过标准：Top-5 交集 >= 4/5，且共有股票权重最大绝对差 <= 0.01
```

- [ ] **Step 3: 新建 4060 实测模板**

创建 `docs/findings/2026-04-23-4060-validation.md`：

```markdown
# 2026-04-23 4060 Validation

## Environment
- GPU:
- Driver:
- CPU:
- RAM:
- Python:

## Results
| Test | Status | Evidence | Notes |
|---|---|---|---|
| T1 venv train | | | |
| T2 venv predict | | | |
| T3 same-machine reproducibility | | | |
| T4 docker e2e | | | |
| T5 cross-machine top-5 consistency | | | |
| T6 bit-level LGB reproducibility | | | |
| T7 gpu memory bypass | | | |
| T8 legacy baseline control | | | |
```

- [ ] **Step 4: 跑最小校验**

Run: `test -f docs/findings/2026-04-23-4060-validation.md && rg -n '确定性 Top-5|可复现性校验' readme.md`
Expected: one file exists and two README headings are found

- [ ] **Step 5: Commit**

```bash
git add readme.md docs/findings/2026-04-23-4060-validation.md
git commit -m "docs: document lgb-only mainline and 4060 validation workflow"
```

---

## Task 8: 端到端验收

**Files:**
- Modify: `docs/findings/2026-04-23-4060-validation.md`

- [ ] **Step 1: 运行单元测试**

Run: `uv run pytest tests/test_portfolio_deterministic_topk.py tests/test_verify_reproducibility.py tests/test_ensemble_blender_icir.py -v`
Expected: all selected tests pass

- [ ] **Step 2: 跑主线脚本的静态校验**

Run: `python -m py_compile scripts/train_lgb_only.py scripts/predict_lgb_only.py code/src/pipeline.py code/src/verify_reproducibility.py`
Expected: no output

- [ ] **Step 3: 记录验证结果**

把 `docs/findings/2026-04-23-4060-validation.md` 的结果表至少填成：

```markdown
| Test | Status | Evidence | Notes |
|---|---|---|---|
| T1 venv train | pending | `temp/4060_train.log` | waiting for 4060 run |
| T2 venv predict | pending | `temp/4060_predict.log` | waiting for 4060 run |
| T3 same-machine reproducibility | pending | `model/repro_check.log` | waiting for double-run |
| T4 docker e2e | pending | `temp/4060_docker.log` | waiting for image build |
| T5 cross-machine top-5 consistency | pending | `/tmp/result_5090.csv`, `/tmp/result_4060.csv` | waiting for 5090 artifact |
| T6 bit-level LGB reproducibility | pending | `/tmp/md5_run1.txt`, `/tmp/md5_run2.txt` | requires `LGB_NUM_THREADS=1` |
| T7 gpu memory bypass | pending | `/tmp/gpu_mem_samples.txt` | should stay <= 50 MiB |
| T8 legacy baseline control | pending | `temp/4060_legacy.log` | expected fail/timeout |
```

- [ ] **Step 4: 提交验收状态**

```bash
git add docs/findings/2026-04-23-4060-validation.md
git commit -m "docs(validation): add acceptance checklist for lgb reproducibility rollout"
```

- [ ] **Step 5: 人工验收**

Run:

```bash
git status --short
```

Expected: clean working tree or only intentional follow-up changes
