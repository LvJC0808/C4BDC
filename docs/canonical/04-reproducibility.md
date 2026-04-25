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

| 硬件 | OS | CPU | AVX-512 | Python | 跑数 | 行尾 | MD5 |
|---|---|---|---|---|---|---|---|
| RTX 5090 32 GB | Linux | Xeon Gold 6530 | ✅ | 3.12 | ✅ 5/5 | LF | `f13034946...` |
| RTX 4060 8 GB | Linux | **i7-13700H** | **❌（avx, avx2, avx_vnni）** | 3.12 | ✅ 5/5 | LF | `f13034946...` |
| RTX 4060 8 GB | Windows | （未记录） | — | 3.12 | ✅ 5/5 | CRLF → LF | `f13034946...` |
| RTX 4060 8 GB | **WSL Ubuntu** | **i9-13900HX** | **❌（avx, avx2, avx_vnni）** | 3.12 | ✅ 场景 A + 场景 B | LF | `f13034946...` |

**关键实证（W1 build, 2026-04-25）**：
- **Linux 4060 队友（i7-13700H，无 AVX-512）**：场景 A + 场景 B + tar 回路三关全过
- **WSL Ubuntu（i9-13900HX，无 AVX-512）**：场景 A + 场景 B 均产 golden MD5；**此机曾在 2026-04 上旬观测到漂移 `5e8b8b0f...`**，今日同机复现 golden — 直接证明当年漂移来自包版本/LF 等工程问题（已被 commit `2599904` + `42abc7f` 修复），**不是 CPU/SIMD 原因**
- 两个"13 代 Intel Mobile 无 AVX-512"节点实证复现 → 赛方评测机 i7-13650H（同代同家族）复现把握非常高

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

### 4.1 跨节点 MD5 漂移（原因未完全定位）

**已观测的现象（项目内可查）**：
- 主机 5090（Xeon Gold 6530）+ Linux 队友（i7-13700H）+ Windows 队友（CPU 型号未记录）+ **WSL Ubuntu（i9-13900HX）**：四节点 MD5 均为 golden `f13034946...`
- 早期本机 VM（i9-13900HX）曾观测到不同 MD5 `5e8b8b0f...`（具体 commit 与镜像构建条件未完整记录）
- **2026-04-25 同一台 i9-13900HX 于 WSL 下重跑，产出 golden MD5** → 直接推翻"CPU 导致漂移"的假说

**未验证的因果假说**（保留备查，**不作为已证事实**）：
- LightGBM 在不同 SIMD 路径（AVX-512 vs AVX2）下浮点累加顺序可能不同
- 包版本次级差异（pandas/numpy/lightgbm build）可能导致浮点尾差
- 早期未强制 LF 行尾 / Docker 镜像 `uv sync` 解析到不同次版本

**说明**：以上假说**未在本项目内做过控制变量实验**。当时 VM 的 MD5 差异可能由上述任一或多个因素叠加产生；在包版本锁定（commit `2599904`）与 LF 强制（commit `42abc7f`）之后，三节点实测一致，但未回头在 VM 上重测。

**新进展（W1 build 实证, 2026-04-25）**：
- Linux 4060 队友节点（i7-13700H，无 AVX-512）完整跑通场景 A + 场景 B + tar 回路验证，全部 MD5 = `f13034946c0aaea5cb1e3f2d0d6ad692` ✅
- 这意味着在最新 commit（镜像锁版 + LF + deterministic_top_k）下，**不依赖 AVX-512 也能跨节点一致**
- "AVX-512 假说"基本被推翻；当年 VM 的 MD5 差异**最可能来自包版本/LF 等已修复因素**，而非 SIMD 路径
- 赛方 i7-13650H 与已验证的 i7-13700H 同代同架构，复现把握**显著上升**（但仍不是直接实测）

**缓解（当前采用）**：
- 以**已复现 golden MD5** 的队友节点（Linux i7-13700H）作为最终打包节点
- 赛规 PDF 明确赛方评测机为 i7-13650H，与 Linux 队友同为 13 代 Intel 移动端
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
| `w1-data-20260425-fix.tar.gz` | `/root/shared-nvme/` | `11d30eb5af3b3a676c60b897d5cae119` | 场景 B 修复包（仅 6 csv，19 MB）|
| **`LCF@NUDT.tar`** | **队友 D:\C4大数据\** | **`1a4ef9430f59e9281406f051dec0fa70`** | **W1 提交镜像 tar（1.7 GB，✅ 已打包）** |
| `requirements-submission.txt` | 仓库根 | 版本锁定 | Docker 精确安装清单 |

---

## 8. 验证清单（每次 W* 提交前必做）

### 8.1 打包节点（已验证 MD5 一致的 Linux 队友节点）

- [ ] `git pull` 确认最新 commit
- [ ] `md5sum data/stock_data.csv` = `cf3e0526f3d832b2ea2e3f1dc22c52e9`
- [ ] `md5sum model_lgb_only/lgb/seed_42_refit/sub_0.txt` = `c517168b...`
- [ ] 记录 CPU 型号与 `avx*` flags（仅存档，不作阻断）
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
