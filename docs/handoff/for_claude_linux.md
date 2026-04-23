# 给 Linux 4060 机器上 Claude Code 的任务交接

> 你（Claude）正在一台装有 **真实 RTX 4060 (8GB) 的 Linux 机器** 上启动。队伍的 5090 主开发机已完成 LGB-only 主线切换并 push 到 `origin/feat/ensemble-v1`。你现在的任务是**在 4060 真机上完成 T1–T8 验证矩阵**，结果填回 `docs/findings/2026-04-23-4060-validation.md` 并 commit。

---

## 0. 先读这两份文档（必读）

```bash
cat docs/handoff/2026-04-23-session-handoff.md       # 整体上下文
cat docs/superpowers/specs/2026-04-23-lgb-mainline-reproducibility-design.md  # 第 8 节有所有命令
```

**关键背景一句话**：主线已从"3 模型 ensemble +1.00%"切换到"LGB-only +1.446%"，训练从 6.3h 降到 9 min。MASTER 模型因 CUDA 非确定性（跨平台分数差 0.17）已被彻底废弃。你的任务是**证明 LGB-only 在 4060 真机上满足赛规复现要求**。

---

## 1. 环境初始化

```bash
cd <THU-BDC2026 repo root>
git fetch origin
git checkout feat/ensemble-v1
git pull

# 硬件自检（写进 findings 文档的 Environment 表）
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
lscpu | grep -E "Model name|CPU\(s\):"
free -g
python3 --version
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（已去除 torch，应 ~1.5GB）
uv sync

# 跑单测确认主线代码没坏
.venv/bin/python -m pytest tests/ -v
# 期望 11 passed（ICIR bootstrap 那个测试慢，~17 分钟，正常）
```

> ⚠️ **永远用 `.venv/bin/python` 绝对路径**。否则 `nohup` / 后台任务会被 `/usr/bin/python3.12` 抢走，是本项目的历史坑。

---

## 2. 获取 5090 参考产出

队友会通过 scp 或 U 盘给你：

| 文件 | 放到 |
| --- | --- |
| `result_5090.csv` | `/tmp/result_5090.csv` |
| （可选）`model_lgb_only_5090.tar.gz` | 仅用于"跳过训练直接验证推理"时解压到 `model_lgb_only/` |

没收到就先问主开发机要，不要自己瞎造。

---

## 3. 执行测试矩阵（T1–T8）

**每一项完成后立刻把观测值写进 `docs/findings/2026-04-23-4060-validation.md` 的表格**，不要攒到最后。

### T1 — 训练入口 + 资源监控

```bash
nvidia-smi dmon -s um -o DT > /tmp/4060_gpu_train.log &
GPU_MON=$!

/usr/bin/time -v .venv/bin/python scripts/train_lgb_only.py 2>&1 \
    | tee temp/4060_train.log

kill $GPU_MON

grep -E "Maximum resident|Elapsed.*wall" temp/4060_train.log
awk 'NR>2 {print $4}' /tmp/4060_gpu_train.log | sort -nr | head -1
```

**通过标准**：训练 ≤ 30 min，无 OOM，`model_lgb_only/lgb/seed_{42,2024,7}_refit/` 产出齐全。

### T2 — 推理入口 + 输出校验

```bash
/usr/bin/time -v .venv/bin/python scripts/predict_lgb_only.py 2>&1 \
    | tee temp/4060_predict.log

.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('output/result.csv')
assert len(df) <= 5, f'too many rows: {len(df)}'
assert df['weight'].sum() <= 1.0 + 1e-6, f'weight sum {df.weight.sum()}'
assert list(df.columns) == ['stock_id', 'weight'], df.columns.tolist()
print('OK', df.to_dict('records'))
"
cp output/result.csv /tmp/result_4060.csv
```

**通过标准**：≤ 1 min，schema 合规。

### T3 — 同机复现

```bash
rm -rf temp/*.parquet model_lgb_only
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv temp/result_run1.csv

rm -rf temp/*.parquet model_lgb_only
.venv/bin/python scripts/train_lgb_only.py
.venv/bin/python scripts/predict_lgb_only.py
cp output/result.csv temp/result_run2.csv

.venv/bin/python code/src/verify_reproducibility.py \
    --single temp/result_run1.csv temp/result_run2.csv
```

**通过标准**：`"pass": true`，Top-5 完全一致，weight diff ≤ 1e-6。

### T5 — 跨机 Top-5 一致性 🔴 **最关键**

```bash
.venv/bin/python code/src/verify_reproducibility.py \
    --cross /tmp/result_5090.csv /tmp/result_4060.csv
```

**通过标准**：`"pass": true`，交集 ≥ 4/5，max weight diff ≤ 0.01。

**如果不通过**：立刻停下来，**不要**自己改代码"对齐"。把两边 result.csv 和 model_lgb_only/ 的 checkpoint 压缩回传给 5090 主开发机，联合排查 LGB 的非确定性残余源头（典型嫌疑：TA-Lib 并行、LightGBM thread count、OpenMP 调度）。

### T6 — Bit-level 复现

```bash
rm -rf model_lgb_only temp/*.parquet
LGB_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_lgb_only.py
md5sum model_lgb_only/lgb/seed_42_refit/sub_*.txt > /tmp/md5_run1.txt

rm -rf model_lgb_only temp/*.parquet
LGB_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/train_lgb_only.py
md5sum model_lgb_only/lgb/seed_42_refit/sub_*.txt > /tmp/md5_run2.txt

diff /tmp/md5_run1.txt /tmp/md5_run2.txt   # 期望无输出
```

### T7 — GPU 显存旁路（证明 LGB 不碰 GPU）

```bash
rm -rf model_lgb_only temp/*.parquet
.venv/bin/python scripts/train_lgb_only.py &
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/gpu_mem_samples.txt
    sleep 5
done
wait $TRAIN_PID
sort -nr /tmp/gpu_mem_samples.txt | head -3
```

**通过标准**：peak ≤ 50 MiB。偏高说明有意外的 CUDA 初始化，告诉主开发机。

### T4 — Docker 端到端

```bash
docker buildx build --platform linux/amd64 -t bdc2026 .
docker images bdc2026 --format "{{.Size}}"       # 期望 ≤ 2 GB
docker compose up 2>&1 | tee temp/4060_docker.log
ls -la output/result.csv
```

> ⚠️ **已知坑**：`init.sh` 里可能还 `import torch`（见 handoff 第九节 #6）。如果 `docker compose up` 在 init 阶段报 `ModuleNotFoundError: torch`，**不要**自己加回 torch；改 `init.sh` 去掉 torch 检查（只留 `lightgbm`），然后告诉主开发机在仓库里同步修掉。

### T8 — Legacy 三模型对照（可选，低优先）

如果时间允许，尝试 `legacy/pipeline_ensemble.py` 在 4060 上跑三模型，**预期 OOM 或超时**，作为"Path B 必要性"的补充证据。不通过是预期结果，不要花时间优化它。

---

## 4. 产出与回传

1. **填表**：把 T1–T8 每一项的实际值（耗时、内存、GPU 峰值、pass/fail、备注）写进 `docs/findings/2026-04-23-4060-validation.md`。
2. **附件**：在 `docs/findings/` 下新建 `2026-04-23-4060-logs/`，放入 `4060_train.log` / `4060_predict.log` / `4060_gpu_train.log` / `4060_docker.log` / `verify_*.json`。
3. **commit + push**：
   ```bash
   git add docs/findings/2026-04-23-4060-validation.md docs/findings/2026-04-23-4060-logs/
   git commit -m "findings(4060-linux): T1-T8 real-hardware validation results"
   git push origin feat/ensemble-v1
   ```
4. **通知主开发机**：贴最关键的一行 —— T5 是否通过、镜像实际大小、训练实际耗时。

---

## 5. 绝对不要做的事

1. ❌ 不要切换分支、不要 rebase、不要改别的文件（只动 `docs/findings/` 和 commit）。
2. ❌ 不要因为 T5 不一致就"对齐"代码 —— 这是赛规层面的观测数据，必须原样上报。
3. ❌ 不要跑老的 `code/src/train.py` / `predict.py`（那是赛方 baseline，不是我们的主线）。
4. ❌ 不要用 `python` 或 `python3` —— 永远 `.venv/bin/python`。
5. ❌ 不要把 `model_lgb_only/` 或 `data/*.csv` commit 进去，它们在 `.gitignore` 里。

---

## 6. 遇到任何问题

- **优先查** `docs/handoff/2026-04-23-session-handoff.md` 第九节"注意事项 & 坑"
- **卡住就停** —— 把现象（命令 + 报错 + py-spy dump）记录到 findings 里，push 后让主开发机接手。不要自己猜。
