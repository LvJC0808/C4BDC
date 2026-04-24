# 给 Linux 4060 队友的 W1 提交验证任务

> **背景**：5090 主开发机已完成 W1 防守补丁（M10-3：mcap_constrained_topk + 强制 ≥3 大盘股），并 push 到 `origin/feat/ensemble-v1`（最新 commit `a95a704`）。
> **你的任务**：在 Linux 4060 上拉最新代码 → 跑 `test.sh` → 把输出 `result.csv` 发回主开发机 diff，确认跨平台一致性。
> **预计时间**：10-15 分钟（无需 GPU，纯 CPU 推理够用）。

---

## 0. 先读（30 秒）

```bash
cd ~/THU-BDC2026   # 或你本机的 repo 路径
git fetch origin
git checkout feat/ensemble-v1
git pull
cat docs/reference/baseline-authoritative.md | head -40   # 知道赛方 baseline 是什么
cat docs/reports/2026-04-24-w1-ab-decision.md             # 知道为什么选 M10-3
```

---

## 1. 环境自检

```bash
source .venv/bin/activate
python -c "import lightgbm, pandas, numpy; print('lgb', lightgbm.__version__, 'pd', pandas.__version__)"
# 期望：lgb 4.6.0  pd 2.2.x（pandas 不能是 3.x）

# 模型权重 MD5（必须与主开发机一致）
md5sum model_lgb_only/lgb/seed_42_refit/model.pkl 2>/dev/null || \
  ls model_lgb_only/lgb/  # 至少有 seed_42_refit, seed_2024_refit, seed_7_refit
```

---

## 2. 关键验证：跑 W1 提交推理

```bash
# 用 M10-3 配置（test.sh 已 hardcode MCAP_MIN_LARGE=3）
MCAP_MIN_LARGE=3 python scripts/predict_lgb_only.py 2>&1 | tail -10
echo "--- result.csv ---"
cat output/result.csv
```

**期望输出**（与主开发机 2026-04-24 一致）：
```
[lgb-only] mcap_constrained Top-5 (min_large=3, cand_k=10, q=0.5)
[lgb-only] target date = 2026-04-23
[lgb-only] wrote ./output/result.csv (5 rows)

stock_id,weight
300308,0.2
600023,0.2
688187,0.2
688256,0.2
002714,0.2
```

**判定**：
- ✅ 5 只股票一字不差 → 跨平台一致，可以提交
- ⚠️ 有 1-2 只不同 → 立刻停手，把 `output/result.csv` 和 stderr 完整日志发回主开发机
- ❌ 报错 → 见 §4 排错

---

## 3. （可选）跑 Baseline 对照

如果你想确认 mcap 约束确实改变了选股：

```bash
MCAP_MIN_LARGE=0 python scripts/predict_lgb_only.py 2>&1 | tail -3
cp output/result.csv output/result_baseline.csv
MCAP_MIN_LARGE=3 python scripts/predict_lgb_only.py 2>&1 | tail -3
diff output/result.csv output/result_baseline.csv
```

期望：差 1 只股票（`002714` ↔ `600741`）。

---

## 4. 排错

| 症状 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: lightgbm` | 没 activate venv | `source .venv/bin/activate` |
| `feature_sets cache 不一致` | data 目录的 stock_data.csv 旧 | `bash scripts/fetch_today.sh`（需代理） |
| `target date != 2026-04-23` | 数据没更新到最新 | 同上 |
| `[lgb-only] mcap constraint skipped` | 缺 log_mktcap 列 | 删 `temp/feature_cache_*` 重跑特征 |

---

## 5. 完成回执

跑完后把这条信息发回主开发机：

```
4060 W1 验证：
- result.csv: 300308 / 600023 / 688187 / 688256 / 002714  （diff 主机：✅ 一致 / ⚠️ 不同）
- stderr 关键行：[lgb-only] mcap_constrained Top-5 (min_large=3, ...)
- target date: 2026-04-23
- 耗时：X 秒
```

---

## ⚠️ 不要做的事

- ❌ 不要重训模型（4-6 小时浪费算力，权重已 ready）
- ❌ 不要改 `MCAP_MIN_LARGE` 试别的值（AB 已决策）
- ❌ 不要改 code/ 下任何文件（除非主开发机指派）
- ❌ 不要 commit / push（你的任务只是验证，主开发机统一管理代码）
