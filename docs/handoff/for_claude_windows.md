# 给 Windows 4060 机器上 Claude Code 的任务交接

> 你（Claude）正在一台装有 **真实 RTX 4060 (8GB) 的 Windows 机器** 上启动。队伍的 5090 主开发机已完成 LGB-only 主线切换并 push 到 `origin/feat/ensemble-v1`。你现在的任务是**在 4060 Windows 真机上完成 T1–T8 验证矩阵**，结果填回 `docs/findings/2026-04-23-4060-validation.md` 并 commit。

**⚠️ 本文件与 `for_claude_linux.md` 的区别**：Windows 不是赛方评测环境（赛方跑的是 Linux Docker），但 Windows 4060 是跨平台一致性验证里**最重要的对照点**——因为我们已经发现 MASTER 模型在 Windows vs Linux 上 score 差 0.17，LGB 差 1e-3。Windows 通过 T5 = LGB 跨平台真正可复现。

---

## 0. 先读这两份文档（必读）

```powershell
type docs\handoff\2026-04-23-session-handoff.md
type docs\superpowers\specs\2026-04-23-lgb-mainline-reproducibility-design.md
```

**关键背景一句话**：主线已从"3 模型 ensemble +1.00%"切换到"LGB-only +1.446%"，训练从 6.3h 降到 9 min。你的任务是**证明 LGB-only 在 Windows + 4060 上与 Linux 5090 产出一致的 Top-5**。

---

## 1. 环境初始化

### 1.1 前置工具

```powershell
# 在 PowerShell 中
nvidia-smi                          # 确认 RTX 4060
python --version                    # 期望 3.12.x
where uv                            # 没有就装：
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Git
git --version
```

### 1.2 TA-Lib 的 Windows 安装（本项目历史坑）

TA-Lib 的 C 库在 Windows 上装起来比 Linux 麻烦。**如果 `uv sync` 报 TA-Lib 编译失败**：

- 方案 A（推荐）：用 conda —— `conda install -c conda-forge ta-lib`，然后在同环境下跑 uv
- 方案 B：下载官方 whl `TA_Lib-0.4.xx-cp312-cp312-win_amd64.whl` 放到项目根，然后 `uv pip install ./TA_Lib-*.whl`

**不要**自己去编译 TA-Lib C 源码，会吃掉几小时。

### 1.3 克隆与依赖

```powershell
cd <workspace>
git clone https://github.com/LvJC0808/C4BDC.git THU-BDC2026   # 或已有 repo 就 git pull
cd THU-BDC2026
git fetch origin
git checkout feat/ensemble-v1
git pull

uv sync                            # 已去除 torch，应 ~1.5GB
.venv\Scripts\python.exe -m pytest tests\ -v
# 期望 11 passed；ICIR bootstrap 测试慢（~17min），正常
```

> ⚠️ **Windows 下永远用 `.venv\Scripts\python.exe` 绝对路径**。不要用 `python`、`py`，会抢到系统 Python。

---

## 2. 获取 5090 参考产出

队友通过任意渠道给你 `result_5090.csv`，放到 `C:\Temp\result_5090.csv`（或仓库根的 `temp\result_5090.csv`）。

没收到就先问主开发机要，不要自己造。

---

## 3. 执行测试矩阵（T1–T8）

**PowerShell 语法注意**：`bash` 命令需要改写成 PowerShell 等价形式。下面每项已给出 Windows 版本。

### T1 — 训练入口 + 资源监控

```powershell
# 后台启动 GPU 监控
Start-Process nvidia-smi -ArgumentList "dmon -s um -o DT" -RedirectStandardOutput C:\Temp\4060_gpu_train.log -NoNewWindow

$sw = [Diagnostics.Stopwatch]::StartNew()
.venv\Scripts\python.exe scripts\train_lgb_only.py 2>&1 | Tee-Object -FilePath temp\4060_train.log
$sw.Stop(); "Elapsed: $($sw.Elapsed)"

Stop-Process -Name nvidia-smi -Force -ErrorAction SilentlyContinue
```

**通过标准**：训练 ≤ 30 min，无 OOM，`model_lgb_only\lgb\seed_{42,2024,7}_refit\` 产出齐全。

观测项：耗时（$sw.Elapsed）、任务管理器峰值内存、GPU 峰值显存（从 `4060_gpu_train.log` 第 4 列最大值取）。

### T2 — 推理入口 + 输出校验

```powershell
.venv\Scripts\python.exe scripts\predict_lgb_only.py 2>&1 | Tee-Object -FilePath temp\4060_predict.log

.venv\Scripts\python.exe -c @"
import pandas as pd
df = pd.read_csv('output/result.csv')
assert len(df) <= 5, f'too many rows: {len(df)}'
assert df['weight'].sum() <= 1.0 + 1e-6
assert list(df.columns) == ['stock_id', 'weight']
print('OK', df.to_dict('records'))
"@

Copy-Item output\result.csv C:\Temp\result_4060_windows.csv
```

### T3 — 同机复现

```powershell
Remove-Item -Recurse -Force model_lgb_only, temp\*.parquet -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\train_lgb_only.py
.venv\Scripts\python.exe scripts\predict_lgb_only.py
Copy-Item output\result.csv temp\result_run1.csv

Remove-Item -Recurse -Force model_lgb_only, temp\*.parquet -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\train_lgb_only.py
.venv\Scripts\python.exe scripts\predict_lgb_only.py
Copy-Item output\result.csv temp\result_run2.csv

.venv\Scripts\python.exe code\src\verify_reproducibility.py `
    --single temp\result_run1.csv temp\result_run2.csv
```

**通过标准**：`"pass": true`，weight diff ≤ 1e-6。

### T5 — 跨机 Top-5 一致性 🔴 **最关键**

```powershell
.venv\Scripts\python.exe code\src\verify_reproducibility.py `
    --cross C:\Temp\result_5090.csv C:\Temp\result_4060_windows.csv
```

**通过标准**：`"pass": true`，交集 ≥ 4/5，max weight diff ≤ 0.01。

**这一项决定 LGB-only 方案能不能用**。

**如果不通过**：立刻停下来，**不要**自己改代码"对齐"。把两边 result.csv 压缩回传给 5090 主开发机联合排查。典型嫌疑：
- Windows 上的 LightGBM prebuild wheel 使用不同 BLAS
- TA-Lib Windows 版的浮点路径
- OpenMP 调度差异

### T6 — Bit-level 复现

```powershell
Remove-Item -Recurse -Force model_lgb_only, temp\*.parquet -ErrorAction SilentlyContinue
$env:LGB_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"
.venv\Scripts\python.exe scripts\train_lgb_only.py
Get-FileHash model_lgb_only\lgb\seed_42_refit\sub_*.txt -Algorithm MD5 | Out-File C:\Temp\md5_run1.txt

Remove-Item -Recurse -Force model_lgb_only, temp\*.parquet -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts\train_lgb_only.py
Get-FileHash model_lgb_only\lgb\seed_42_refit\sub_*.txt -Algorithm MD5 | Out-File C:\Temp\md5_run2.txt

Compare-Object (Get-Content C:\Temp\md5_run1.txt) (Get-Content C:\Temp\md5_run2.txt)
# 期望无输出
```

### T7 — GPU 显存旁路（证明 LGB 不碰 GPU）

```powershell
Remove-Item -Recurse -Force model_lgb_only, temp\*.parquet -ErrorAction SilentlyContinue

$job = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    .venv\Scripts\python.exe scripts\train_lgb_only.py
}

while ($job.State -eq 'Running') {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Out-File -Append C:\Temp\gpu_mem_samples.txt
    Start-Sleep -Seconds 5
}
Receive-Job $job; Remove-Job $job

Get-Content C:\Temp\gpu_mem_samples.txt | Sort-Object {[int]$_} -Descending | Select-Object -First 3
```

**通过标准**：peak ≤ 50 MiB。

### T4 — Docker 端到端（可选）

如果 Windows 上装有 Docker Desktop + WSL2：

```powershell
docker buildx build --platform linux/amd64 -t bdc2026 .
docker images bdc2026 --format "{{.Size}}"          # 期望 ≤ 2 GB
docker compose up 2>&1 | Tee-Object -FilePath temp\4060_docker.log
```

> ⚠️ **已知坑**：`init.sh` 里可能还 `import torch`（见 handoff 第九节 #6）。如果 `docker compose up` 在 init 阶段报 `ModuleNotFoundError: torch`，**不要**自己加回 torch；改 `init.sh` 去掉 torch 检查，告诉主开发机同步。

Windows 原生（无 Docker）就跳过 T4，标 "Skipped (Windows native)"。

### T8 — Legacy 三模型对照（跳过）

Windows 版 PyTorch CUDA 安装在 4060 上更麻烦，直接标 "Skipped, see Linux validation" 即可。

---

## 4. 产出与回传

1. **填表**：把 T1–T8 每一项的实际值写进 `docs\findings\2026-04-23-4060-validation.md`，明确标注 `Platform: Windows`。
2. **附件**：新建 `docs\findings\2026-04-23-4060-logs-windows\`，放入 `4060_train.log` / `4060_predict.log` / `4060_gpu_train.log` / `verify_*.json`。
3. **commit + push**：
   ```powershell
   git add docs\findings\2026-04-23-4060-validation.md docs\findings\2026-04-23-4060-logs-windows\
   git commit -m "findings(4060-windows): T1-T8 real-hardware validation results"
   git push origin feat/ensemble-v1
   ```
4. **通知主开发机**：T5 是否通过、训练实际耗时、Windows 特有问题。

> 💡 如果 Linux 4060 那台机器也在跑同样的测试，**避免 force-push**，多人协作用 `git pull --rebase` 再 push。

---

## 5. Windows 特有的坑

| 现象 | 原因 | 解法 |
|---|---|---|
| `uv sync` TA-Lib 编译失败 | Windows 缺 C 编译器/TA-Lib C 库 | 见 §1.2 |
| `.venv\Scripts\python.exe` 被 Anaconda 抢 | PATH 顺序问题 | 永远用绝对路径 |
| 脚本里 `./model_lgb_only` 路径 | Python 里混用 `/` 和 `\` 大部分情况 OK，但 `os.path.join` 产物在 log 里会是反斜杠 | 不影响功能，不要自己替换 |
| 换行符 CRLF vs LF | git autocrlf 默认 true | 不要动，不会影响 Python/LightGBM |
| PowerShell 里反引号 ` 是续行符 | 跟 bash 不同 | 多行命令用 ` 结尾，不要用 `\` |
| 中文路径 | TA-Lib / LightGBM 对非 ASCII 路径不友好 | **仓库必须放在全英文路径下**（如 `C:\dev\THU-BDC2026`） |

---

## 6. 绝对不要做的事

1. ❌ 不要切换分支、rebase、改别的文件（只动 `docs\findings\` 再 commit）。
2. ❌ 不要因为 T5 不一致就"对齐"代码 —— 这是赛规层面的观测数据，必须原样上报。
3. ❌ 不要跑老的 `code\src\train.py` / `predict.py`（赛方 baseline，不是我们主线）。
4. ❌ 不要用 `python` / `py` —— 永远 `.venv\Scripts\python.exe`。
5. ❌ 不要把 `model_lgb_only\` 或 `data\*.csv` commit 进去，它们在 `.gitignore`。
6. ❌ 不要在中文路径下放仓库。

---

## 7. 遇到任何问题

- **优先查** `docs\handoff\2026-04-23-session-handoff.md` 第九节"注意事项 & 坑"
- **卡住就停** —— 把现象（命令 + 报错）写进 findings 文档，push 后让主开发机接手。不要自己猜。
