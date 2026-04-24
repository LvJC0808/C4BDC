# 给 Windows 4060 队友的 W1 提交验证任务

> **背景**：5090 主开发机已完成 W1 防守补丁（M10-3：mcap_constrained_topk + 强制 ≥3 大盘股），并 push 到 `origin/feat/ensemble-v1`（最新 commit `a95a704`）。
> **你的任务**：在 Windows 4060 上拉最新代码 → 跑 `predict_lgb_only.py` → 把输出 `result.csv` 发回主开发机 diff，确认跨平台一致性。
> **预计时间**：10-15 分钟（无需 GPU，纯 CPU 推理够用）。

---

## 0. 先读（30 秒）

PowerShell：
```powershell
cd C:\path\to\THU-BDC2026   # 你本机的 repo 路径
git fetch origin
git checkout feat/ensemble-v1
git pull
type docs\reference\baseline-authoritative.md | Select-Object -First 40
type docs\reports\2026-04-24-w1-ab-decision.md
```

---

## 1. 环境自检

```powershell
# 激活 venv（如果用的是 uv/.venv）
.\.venv\Scripts\Activate.ps1
# 若 venv 名字不同请相应调整

python -c "import lightgbm, pandas, numpy; print('lgb', lightgbm.__version__, 'pd', pandas.__version__)"
# 期望：lgb 4.6.0  pd 2.2.x（pandas 不能是 3.x）

# 模型权重检查
dir model_lgb_only\lgb\
# 期望看到 seed_42_refit, seed_2024_refit, seed_7_refit 三个文件夹
```

**如果之前从未在这台 Windows 跑过**：先确认 `data/stock_data.csv` 已同步到 2026-04-23（约 166k 行），否则需先从主开发机同步 data 目录或跑 `python get_stock_data.py`。

---

## 2. 关键验证：跑 W1 提交推理

PowerShell（注意 Windows 设环境变量的语法不同）：

```powershell
$env:MCAP_MIN_LARGE = "3"
$env:MCAP_CAND_K = "10"
$env:MCAP_LARGE_Q = "0.5"
python scripts\predict_lgb_only.py 2>&1 | Select-Object -Last 10
type output\result.csv
```

或 cmd：
```cmd
set MCAP_MIN_LARGE=3
set MCAP_CAND_K=10
set MCAP_LARGE_Q=0.5
python scripts\predict_lgb_only.py
type output\result.csv
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
- ⚠️ 有 1-2 只不同 → 立刻停手，把 `output\result.csv` 和 stderr 完整日志发回主开发机
- ❌ 报错 → 见 §4 排错

---

## 3. （可选）跑 Baseline 对照

```powershell
$env:MCAP_MIN_LARGE = "0"
python scripts\predict_lgb_only.py 2>&1 | Select-Object -Last 3
copy output\result.csv output\result_baseline.csv
$env:MCAP_MIN_LARGE = "3"
python scripts\predict_lgb_only.py 2>&1 | Select-Object -Last 3
fc output\result.csv output\result_baseline.csv
```

期望：差 1 只股票（`002714` ↔ `600741`）。

---

## 4. 排错

| 症状 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: lightgbm` | 没激活 venv | `.\.venv\Scripts\Activate.ps1` |
| `OSError: [WinError 126] cannot find module` | LightGBM Windows 缺 OpenMP | `conda install libomp` 或装 vcredist |
| `feature_sets cache 不一致` | data 目录的 stock_data.csv 旧 | 从主开发机同步 data\ |
| `target date != 2026-04-23` | 数据没更新到最新 | 同上 |
| `[lgb-only] mcap constraint skipped` | 缺 log_mktcap 列 | 删 `temp\feature_cache_*` 重跑特征 |
| 行尾 CRLF/LF 警告 | Windows 默认 CRLF | 不影响功能，可忽略；或 `git config core.autocrlf true` |

---

## 5. Windows 特别注意事项

1. **路径分隔符**：Python 内 `os.path.join` 自动处理，但你写命令时用 `\`
2. **venv 激活**：如果 PowerShell 报 `cannot be loaded because running scripts is disabled`，先：
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
3. **DataFrame groupby 跨平台**：之前已验证 `deterministic_top_k` Windows 5/5 一致；本次新增的 `mcap_constrained_topk` 内部用相同的 quantize+id_sort 机制，理论上也一致——但**这是首次 Windows 验证**，请认真对比 result.csv

---

## 6. 完成回执

跑完后把这条信息发回主开发机：

```
Windows 4060 W1 验证：
- OS: Windows X / Python 版本
- result.csv: 300308 / 600023 / 688187 / 688256 / 002714  （diff 主机：✅ 一致 / ⚠️ 不同）
- stderr 关键行：[lgb-only] mcap_constrained Top-5 (min_large=3, ...)
- target date: 2026-04-23
- 耗时：X 秒
```

---

## ⚠️ 不要做的事

- ❌ 不要重训模型（4-6 小时浪费算力，权重已 ready）
- ❌ 不要改 `MCAP_MIN_LARGE` 试别的值（AB 已决策）
- ❌ 不要改 code\ 下任何文件（除非主开发机指派）
- ❌ 不要 commit / push（你的任务只是验证，主开发机统一管理代码）
- ❌ 不要在 Windows 上跑 train.sh（4-6 小时且无意义）
