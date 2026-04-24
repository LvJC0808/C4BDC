# 复现性证据 · Reproducibility (SSOT)

> 所有跨平台 / 跨时点 / 跨权重的 MD5 证据集中存放地。
> 赛规第 1 条/第 5 条合规性证明。
> 最后更新：2026-04-24 · Phase 4

---

## 1. 赛规要求

> §1) 在固定随机种子点的前提下，从训练过程开始复现，选手提交项目代码运行生成的结果和提交的结果完全一致；
> §5) 复现训练和预测时不得联网；

**解读**：赛方会在他们机器上，用我们的代码 + 他们的数据，从 train 开始跑一遍，最终 result.csv 必须和我们提交的**完全一致**。

---

## 2. 三层复现性证明

### 2.1 训练复现性（Phase 4 硬证）

**双跑测试**（5090 主机，两次独立 `bash train.sh`）：

| Run | 耗时 | 文件数 | MD5 状态 |
|---|---|---|---|
| Run 1 | 5.5 min | 73 | - |
| Run 2 | 5.5 min | 73 | **与 Run 1 bit-identical** |

**覆盖文件类型**：
- `.txt` LightGBM booster（9 个 fold + 3 个 refit × 3 seeds = 36 个）
- `.npy` feature weights（DoubleEnsemble FR 权重）
- `.json` meta（模型元数据）
- `ensemble_config.json`（组合配置）

**结论**：训练完全 deterministic。

### 2.2 推理 MD5 跨平台一致

**实验**：同一训练好的权重，在不同硬件上跑推理 5 次。

| 硬件 | OS | Python | pandas | 5 次 MD5 一致 | 行尾符 | MD5 |
|---|---|---|---|---|---|---|
| RTX 5090 32 GB | Linux | 3.12 | 2.3.3 | ✅ 5/5 | LF | `f13034946c0aaea5cb1e3f2d0d6ad692` |
| RTX 4060 8 GB | Linux | 3.12 | 2.3.3 | ✅ 5/5 | LF | `f13034946c0aaea5cb1e3f2d0d6ad692` |
| RTX 4060 8 GB | Windows | 3.12 | 2.3.3 | ✅ 5/5 | CRLF → LF | `f13034946c0aaea5cb1e3f2d0d6ad692` |

**行尾符差异已修复**（Phase 4 commit `42abc7f`）：`result.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")` 强制 LF 输出。

### 2.3 跨权重稳健（Bonus）

**意外发现**（Phase 4 训练复现验证中）：
- 不同时刻训练的权重 MD5 **不同**（因训练数据时间戳、refit_epochs 微差）
- 但推理得到的 **`result.csv` 完全字节级一致**

**机制**：`deterministic_top_k`（quantize 1e-4 + stock_id 字典序 tie-break）吸收了权重层的浮点微扰。

| 权重来源 | 权重 `seed_42/sub_0.txt` MD5 | result.csv MD5 |
|---|---|---|
| Phase 3 snapshot | `c517168b73fe4bcb5ecbdb1f1dd2434e` | `f13034946c0aaea5cb1e3f2d0d6ad692` |
| Phase 4 重训 | `addf80edd40911bfbf559d85b61b93fc` | `f13034946c0aaea5cb1e3f2d0d6ad692` ✅ |

**含义**：这是**强于赛规要求**的稳健性——赛方重训后即使权重不 bit-identical，result.csv 仍一致。

---

## 3. Deterministic 关键工程措施

### 3.1 LightGBM 参数
```python
LGB_PARAMS = {
    "deterministic": True,       # 官方 deterministic 开关
    "force_row_wise": True,       # 行优先，消除列优先的线程顺序差异
    "num_threads": 4,             # 固定线程数
}
```

### 3.2 全局 seed
```python
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch 系（legacy，本次不用）
```

在每次 `_fit_lgb` / `_fit_lgb_refit` / `_predict_scores` 前调用。

### 3.3 Deterministic Top-K
```python
def deterministic_top_k(df, k=5, score_col="score", id_col="stock_id",
                        quantize=1e-4):
    df["score_q"] = (df[score_col] / quantize).round() * quantize
    df = df.sort_values(by=["score_q", id_col],
                        ascending=[False, True], kind="mergesort")
    return df.head(k).reset_index(drop=True)
```

**核心效应**：消除 LightGBM 跨平台浮点尾差对选股的影响。

### 3.4 输出行尾符
```python
result.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")
```
强制 LF，消除 Windows 默认 CRLF 导致的 MD5 差异。

---

## 4. 已知剩余风险

### 4.1 CPU 指令集（AVX-512）

**现象**：
- 主机 5090 + 队友 4060 × 2 节点：**都有 AVX-512 → golden MD5** `f13034946...`
- 你本机 VM（i9-13900HX）：**无 AVX-512 → 不同 MD5** `5e8b8b0f...`

**解读**：
- LightGBM 在不同 SIMD 路径（AVX-512 vs AVX2）下浮点累加顺序可能不同
- 罕见情况下差异能放大到 `1e-2` 级，改变 Top-K 选股
- **我们 deterministic_top_k 的 quantize=1e-4 无法吸收这种大差异**

**缓解**：
- 赛方评测机是数据中心 Xeon → **几乎必有 AVX-512**
- 我们在 AVX-512 Linux 机器上打 tar → 赛方机器跑出的 result.csv 会是 `f130349...`
- 详见 `docs/handoff/2026-04-25-w1-build-linux.md` §2.3

### 4.2 Python 包版本漂移

**历史遗留**：`uv sync --frozen` 在 Docker 内可能解析到与主机 `.venv` 不完全一致的次版本（Phase 4 验证时发现 pandas 2.3.2 vs 2.3.3 + numpy 2.2.6 vs 2.3.3）。

**最终解法**（commit `2599904`）：
- Docker 改用 `uv pip install -r requirements-submission.txt`
- `requirements-submission.txt` 是主机 `.venv` `uv pip freeze` 出来的 **精确版本号**
- 保证镜像内包版本 = 主机版本

**状态**：✅ Phase 4 已解决。

### 4.3 赛方 data 挂载不确定性

**风险**：赛方挂载 `/app/data` 可能只给 `train.csv + test.csv`，缺少我们的辅助 5 文件。

**缓解**（commit `2953494`）：
- Dockerfile `COPY` 一份 `data/` → `/app/data_bundled/`
- `init.sh` 启动时自检 `/app/data/` 完整性，缺失则从 bundled 恢复 + train/test 合并

**状态**：✅ Phase 4 已解决。

---

## 5. 完整复现性验证矩阵（W1 提交前截图）

| 验证项 | 耗时 | 结果 |
|---|---|---|
| 训练双跑 MD5 一致 | 11 min | ✅ 73 files bit-identical |
| 推理同权重双跑 MD5 一致 | 4 min | ✅ bit-identical |
| 跨平台 Linux × Linux MD5 一致 | 手工 | ✅ bit-identical |
| 跨平台 Linux × Windows MD5 一致 | 手工 | ✅（去 CRLF 后） |
| Docker 镜像 MD5 一致（load 前后） | 5 min | ✅（commit 2599904 后） |

---

## 6. 与赛方 baseline 对比

**赛方 baseline 权重 MD5**：`d9c56a6a332c0fef08157ca7661a2acb`
**我们本仓库内副本 MD5**：`d9c56a6a332c0fef08157ca7661a2acb` ✅

同一文件，赛规"baseline 超额"计算公平。详见 `05-baseline.md`。

---

## 7. 跨机同步工具

| 产物 | 位置 | MD5 | 用途 |
|---|---|---|---|
| `w1-handoff-20260424.tar.gz` | `/root/shared-nvme/` | `f14904c1...` | 给队友的 data + weights 包 |
| `w1-data-20260424-final.tar.gz` | `/root/shared-nvme/` | `9b3eb781...` | 更新版 data+weights（修正老版） |
| `LCF@NUDT.tar` | 待打包 | 待定 | 赛方提交用 Docker image tar |
| `requirements-submission.txt` | 仓库根 | 版本锁定 | Docker 精确安装清单 |

---

## 8. 验证清单（每次 W* 提交前必做）

### 8.1 打包节点（Linux AVX-512 队友）

- [ ] `git pull` 确认最新 commit
- [ ] `md5sum data/stock_data.csv` = `cf3e0526f3d832b2ea2e3f1dc22c52e9`
- [ ] `md5sum model_lgb_only/lgb/seed_42_refit/sub_0.txt` = `c517168b...`
- [ ] CPU 有 `avx512f`
- [ ] `docker buildx build` 成功，镜像 < 10 GB
- [ ] 本机场景 A（完整 data）→ result.csv MD5 = `f13034946c0aaea5cb1e3f2d0d6ad692`
- [ ] 本机场景 B（train+test 模拟赛方）→ Top-5 一致
- [ ] `docker save` → `<team>.tar`，记录 tar MD5
- [ ] `docker load -i <team>.tar` → compose 再跑一遍 → MD5 仍一致

### 8.2 提交节点

- [ ] `tar` 上传夸克网盘，**永久有效 + 不加提取码**
- [ ] 隐身窗口访问分享链接，确认能直接下载
- [ ] 下载回本地，md5sum 与打包节点一致
- [ ] 竞赛平台提交 `result.csv` + 网盘链接

**任一项失败**：不提交，排查。
