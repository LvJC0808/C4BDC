# Windows 训练指南（4060）

> 本指南给**已经从队友那拿到预先抓好的数据包（`bdc2026-data.tar.gz`）**的 Windows 用户。
> **无需联网抓 baostock**，只需要跑训练 + 推理。
>
> 预计时间：环境配置 ~1 h（首次）+ 训练 6-7 h + 推理 3 min

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 x64 |
| 显卡 | NVIDIA RTX 4060（8 GB 显存）或更高 |
| 驱动 | NVIDIA 驱动 ≥ 535（支持 CUDA 12.1） |
| 磁盘 | 至少 10 GB 空闲 |
| 网络 | 仅装依赖时需要，训练时可离线 |

验证驱动：
```powershell
nvidia-smi
```
应能看到 4060 型号和 CUDA Version ≥ 12.1。

---

## 2. 一次性环境准备

### 2.1 安装 Python 3.12

1. 从 https://www.python.org/downloads/windows/ 下载 **Python 3.12.x Windows installer (64-bit)**
2. 安装时勾选 **Add Python to PATH**
3. 验证：
   ```powershell
   python --version       # 应输出 Python 3.12.x
   ```

### 2.2 安装 uv（Python 包管理器，比 pip 快）

PowerShell 里执行：
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

关闭 PowerShell 重新打开，验证：
```powershell
uv --version
```

### 2.3 安装 Git

https://git-scm.com/download/win 下载安装（默认选项即可）。

---

## 3. 克隆代码

```powershell
cd C:\   # 或你想放的目录
git clone https://github.com/LvJC0808/C4BDC.git
cd C4BDC
git checkout feat/ensemble-v1
```

---

## 4. 装 Python 依赖

```powershell
# 创建虚拟环境
uv venv
.venv\Scripts\activate

# 装 PyTorch CUDA 12.1 版（Windows 原生）
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 装 TA-Lib（Windows 预编译 wheel）
# Python 3.12 x64 用下面这行；其它版本去 https://github.com/cgohlke/talib-build/releases 自己找对应
uv pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.3/ta_lib-0.6.3-cp312-cp312-win_amd64.whl

# 装剩余包
uv pip install numpy pandas scipy scikit-learn pyarrow tqdm baostock lightgbm
```

验证：
```powershell
python -c "import torch, lightgbm, talib, pyarrow, pandas; print('OK, cuda=', torch.cuda.is_available())"
```
应输出 `OK, cuda= True`。

---

## 5. 放数据

把队友传给你的 `bdc2026-data.tar.gz` 放到 `C4BDC\` 根目录，然后：

```powershell
# Windows 10+ 自带 tar
tar -xzf bdc2026-data.tar.gz
```

验证 6 个 CSV：
```powershell
dir data\*.csv
```
应看到：
```
stock_data.csv       (~85 MB)
industry_map.csv
stock_basic.csv
hs300_history.csv
csi300_index.csv
trade_calendar.csv
```

验证 `stock_data.csv` 列数（必须 16 列）：
```powershell
Get-Content data\stock_data.csv -TotalCount 1
```
应看到列头含 `peTTM,pbMRQ,psTTM,pcfNcfTTM` 四列。

（可选）校验 MD5：
```powershell
# 对比队友发给你的 bdc2026-data.md5 文件
Get-FileHash data\stock_data.csv -Algorithm MD5
```

---

## 6. 训练（约 6-7 h）

在 `C4BDC` 根目录：

```powershell
# 2 seeds 省时间（4060 友好）；改成 "42,2024,7" 可做完整 3 seeds
$env:SEEDS = "42,2024"

# 创建输出目录
mkdir model, temp, output -Force | Out-Null

# 启动训练，日志同时写到 train.log 和屏幕
python -u code\src\pipeline.py train `
    --data_path .\data --model_dir .\model --temp_dir .\temp `
    2>&1 | Tee-Object -FilePath train.log
```

### 监控

另开一个 PowerShell 窗口：
```powershell
# 看最新进度
Get-Content train.log -Tail 20 -Wait

# 看显存占用
nvidia-smi
```

训练流程约 6-7 h：
- 特征工程：~3 min（CPU）
- LGB+DE（2 seeds × 3 folds + refit）：~20 min
- MASTER（2 seeds × 3 folds + refit）：~3.5-4 h
- StockMixer（2 seeds × 3 folds + refit）：~1 h
- 集成网格搜索 + 写 config：<1 min

训练完成标志：
```
[train] wrote .\model\ensemble_config.json
```

### 如果中途崩溃

- **查日志**：`Get-Content train.log -Tail 50`
- **显存不够（OOM）**：减小 batch，或只跑 `SEEDS=42`
- **不要半途 Ctrl+C**：没有断点续训，需要重跑

---

## 7. 推理（约 3 min）

```powershell
python code\src\pipeline.py predict `
    --data_path .\data --model_dir .\model --temp_dir .\temp `
    --output_path .\output\result.csv
```

查看结果：
```powershell
Get-Content output\result.csv
```
应输出 ≤ 5 行 `stock_id,weight`，权重和 ≤ 1。

---

## 8. 自测得分

```powershell
python test\score_self.py
```

输出如 `预测股票的加权收益率得分: 0.0XXX`。

> 注：`score_self.py` 需要 `data\test.csv`（最近 5 天行情）。baseline 的 `test.csv` 是旧的，要算真实得分需手动从 `stock_data.csv` 切一份最后 5 天：
>
> ```powershell
> python -c "import pandas as pd; df=pd.read_csv('data/stock_data.csv'); df['日期']=pd.to_datetime(df['日期']); last5=sorted(df['日期'].unique())[-5:]; df[df['日期'].isin(last5)].to_csv('data/test.csv', index=False)"
> ```

---

## 9. 常见问题

### Q1: `ImportError: DLL load failed while importing _talib`
A: TA-Lib wheel 和 Python 版本不匹配。去 https://github.com/cgohlke/talib-build/releases 找对应 `cp312-win_amd64` 版本（Python 3.12 64-bit）。

### Q2: `UnicodeDecodeError` 读 CSV 时报错
A: Windows 默认编码是 GBK。在 `pd.read_csv(...)` 处加 `encoding='utf-8-sig'`。或联系队友确认。

### Q3: `torch.use_deterministic_algorithms` 相关 warning
A: `warn_only=True` 已设置，只会 warning 不崩。可以忽略。

### Q4: `CUDA out of memory`
A: 降低 SEEDS 到 `42`，或临时改 `code/src/config.py` 里 `MASTER_CONFIG["d_model"]` 从 256 → 128。

### Q5: 训练超 8 h 了怎么办
A: 赛题只在提交 Docker 镜像评测时限时 8h。本地训练超时是可以接受的，只是告诉我们要减 seeds / fold / epochs。

### Q6: 完全离线环境（无网）怎么装依赖
A: 在有网的机器用 `uv pip download ...` 下载 wheel 到本地目录，再拷过去离线 `uv pip install --no-index --find-links=<dir> ...`。

---

## 10. 对照命令表（Linux → Windows）

| Linux (bash) | Windows (PowerShell) |
|---|---|
| `cd /path/to/repo` | `cd C:\path\to\repo` |
| `source .venv/bin/activate` | `.venv\Scripts\activate` |
| `./scripts/fetch_then_train.sh` | 按本文档 6~7 节手动跑 |
| `tail -f train.log` | `Get-Content train.log -Tail 20 -Wait` |
| `export SEEDS=42,2024` | `$env:SEEDS = "42,2024"` |
| `md5sum file.csv` | `Get-FileHash file.csv -Algorithm MD5` |
| `ls -la` | `dir` 或 `Get-ChildItem` |

---

## 11. 完成后提交什么给队长

训练完后把以下内容传回（或走 git push）：

- `model\ensemble_config.json`（集成权重）
- `model\lgb\`、`model\master\`、`model\mixer\` 下的所有 `seed_*_refit\` 子目录
- `train.log`（训练日志，便于复查）
- `output\result.csv`（推理结果）

**不需要传** `temp\` 缓存（太大，能自动重建）。
