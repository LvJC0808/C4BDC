# W1 提交前发现的关键问题 · Critical Findings (2026-04-25 晚)

> 在做最后的提交前合规性 review 时发现了**潜在致命兼容性问题**，需主开发者优先确认与处置。
> 编写时间：2026-04-25 提交窗口内
> 上下文：队友 + WSL 双节点已成功打 tar，正准备提交，但发现以下问题尚未在评测口径下验证。

---

## 1. 核心发现：赛方挂载的 data 字段格式与我们不一致

### 1.1 证据

**赛方 baseline 仓库**（`/root/shared-nvme/C4BD/THU-BDC2026-main/`）的 data/ 模板：

| 文件 | 列结构 |
|---|---|
| `data/stock_data.csv` | **12 列**：`股票代码,日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌额,换手率,涨跌幅` |
| `data/train.csv` | **同 12 列** |
| `data/test.csv` | **同 12 列** |

**我们仓库**的 data/：

| 文件 | 列结构 |
|---|---|
| `data/stock_data.csv` | **16 列** = 12 基础 + `peTTM, pbMRQ, psTTM, pcfNcfTTM`（4 估值因子）|
| `data/train.csv` | 同 16 列 |
| `data/test.csv` | 同 16 列 |

**日期格式也不同**：
- 赛方：`2024-01-02 00:00:00`
- 我们：`2024/1/2`

### 1.2 影响

我们的 174 列特征里包含 **16 个估值因子**（基于 4 个估值列做 raw / z-score 20d / z-score 60d / industry rank 四种变换），实现在 `code/src/features/valuation.py`。

**评测时若赛方挂载的 stock_data.csv 不含 peTTM/pbMRQ/psTTM/pcfNcfTTM**：
- `valuation.py` 读列时直接 `KeyError` → 推理崩溃 → 提交失败 → W1 没成绩

---

## 2. 真正的评测入口（赛方 docker-compose.yml）

### 2.1 关键证据

赛方 baseline `/root/shared-nvme/C4BD/THU-BDC2026-main/docker-compose.yml`：

```yaml
services:
  app:
    image: bdc2026:latest
    pull_policy: never
    volumes:
      - ./data:/app/data
      - ./test/output:/app/output
      - ./temp:/app/temp
    command: /bin/bash /app/data/run.sh   # ← 入口是赛方 data 里的 run.sh
    runtime: nvidia
    ...
```

赛方 `data/run.sh`：

```bash
/bin/bash /app/init.sh
# /bin/bash /app/train.sh
/bin/bash /app/test.sh
```

### 2.2 含义

- 赛方真正的入口是 **`/app/data/run.sh`**，不是 `test.sh`
- run.sh 是**赛方挂载进来的**（在他们的 data/ 里），不是我们镜像里的
- 赛方期望流程：`run.sh` → 调 `init.sh`（初始化 + 自愈）→ 调 `test.sh`（推理）

### 2.3 好消息

赛方会调 `init.sh` —— 我们的数据自愈逻辑会被触发。这部分**符合预期**，无需改动。

---

## 3. 致命点：赛方挂载会覆盖 /app/data，但提供的字段比我们模型期望的少

### 3.1 评测时刻的实际情况推测

赛方按 baseline 模板挂载 `/app/data/`，包含：
- `run.sh`（评测入口）
- `train.csv`（**12 列**，可能含最新交易日数据）
- `test.csv`（**12 列**）
- 可能也提供 `stock_data.csv`（**12 列**）

**完全不会**包含：peTTM / pbMRQ / psTTM / pcfNcfTTM

### 3.2 我们 init.sh 现状（问题所在）

`init.sh` 自愈逻辑：

```bash
# 仅当 stock_data.csv 不存在时才合并 train+test
if [ ! -f stock_data.csv ]; then
    merge train.csv + test.csv -> stock_data.csv  # 但这两份也是 12 列！
fi

# 5 个辅助 CSV 缺失就从 data_bundled/ 还原
for f in industry_map ...; do
    if [ ! -f $f ]; then cp /app/data_bundled/$f .; fi
done
```

**漏洞**：
- 自愈合并 train+test 出来的 stock_data.csv 仍是 **12 列，无估值**
- 即使赛方挂了 stock_data.csv，也是 **12 列，无估值**
- 现有 init.sh 不会强制替换为 bundled 版本

→ `valuation.py` 读 `peTTM` → `KeyError` → 提交失败

---

## 4. 反证：为什么我们之前的"场景 B"测试都通过了

### 4.1 场景 B 的实际测试逻辑

场景 B 命令：

```bash
mkdir -p /tmp/judge_data
cp data/train.csv data/test.csv /tmp/judge_data/   # ← 关键：用的是我们 16 列的 train/test
docker run -v /tmp/judge_data:/app/data ... bash init.sh && bash test.sh
```

**问题**：我们用的是**自己仓库里的 train.csv + test.csv**（16 列带估值）模拟赛方挂载，所以 init.sh 合并出来的 stock_data.csv 也带估值，自然不会触发 valuation.py 的 KeyError。

### 4.2 真正应该模拟的场景 B'

应该用赛方 baseline 模板的 12 列 train/test 模拟：

```bash
cp /root/shared-nvme/C4BD/THU-BDC2026-main/data/train.csv /tmp/judge_data/
cp /root/shared-nvme/C4BD/THU-BDC2026-main/data/test.csv /tmp/judge_data/
# 然后跑容器，看会不会挂
```

**这个测试我们从未做过**。

---

## 5. 解决方案候选

### 5.1 方案 A · init.sh 强制覆盖为 bundled stock_data.csv（最稳）

```bash
# init.sh 修订思路：
# 1. 不管赛方挂载什么，都用 bundled 的 stock_data.csv 替换
# 2. 5 个辅助 CSV 也都用 bundled 版本

cp /app/data_bundled/stock_data.csv /app/data/stock_data.csv
for f in industry_map hs300_history csi300_index stock_basic trade_calendar; do
    cp /app/data_bundled/${f}.csv /app/data/${f}.csv
done
```

**收益**：保证不挂，字段稳定，与训练时格式严格一致 → result.csv MD5 必然 = `f13034946...`
**牺牲**：赛方若挂了更新的 2026-04-24 / 04-25 数据，我们看不到，但 W1 目标日期是 04-23 后预测下一周（04-29 ~ 05-08，含五一假期），数据截止 04-23 已足够。

### 5.2 方案 B · init.sh 智能合并（保留赛方最新数据）

```bash
# 检测赛方 csv 列数：
# 若 12 列 → 用 bundled 替换
# 若 16 列 → 直接用赛方
```

**收益**：理论上最优
**风险**：增加 init.sh 复杂度，新增故障点；且实际赛方几乎肯定是 12 列，分支 B 永远走不到

### 5.3 方案 C · 让模型不依赖估值列（重训）

**收益**：根因解决
**牺牲**：训练 5.5 min，权重 MD5 改变，需重新走 build + 5 节点验证 + tar 回路；W1 截止时间紧

### 5.4 推荐：方案 A

最简单、零风险、零重训。预计 init.sh 改 5 行。

---

## 6. 还需要主开发者确认的事

### 6.1 赛方 baseline 是 2026-03-25 的旧版本

`@/root/shared-nvme/C4BD/THU-BDC2026-main/data/` 文件日期 `Mar 25 16:18`。这是赛方初始 baseline，**不一定等于赛方最终评测时挂载的内容**。

赛方在 W1 评测时（2026-04-26 提交截止后），**可能**：
- 挂载更新的 12 列 csv（含 2026-04-23 数据）
- 挂载更新的带估值列的 csv（不太可能，baseline 模板没暗示这个）
- 仍用 03-25 那份旧数据（最坏情况，但训练数据截止 04-01 对不上）

**我们不知道，赛规也没明确**。需要：
- (A) 联系组委会问清楚，或
- (B) 按最坏假设（12 列）做防御 → 即方案 5.4

### 6.2 真正的赛方 docker-compose.yml 可能不是 baseline 这份

赛规第 49 行说："使用下发的 docker-compose.yml 文件运行"——「下发」意味着赛方在评测时会**单独发**一份。baseline 仓库这份只是参考。

但 baseline 这份是**目前我们能找到的唯一证据**，应据此推断。

---

## 7. 时间窗紧迫度

- W1 截止：**2026-04-26 23:59**
- 距离截止：**~30 小时**（编写时刻 2026-04-25 ~12:00）

**保守路线**（方案 A）耗时估算：
- 改 init.sh：5 min
- 真场景 B' 测试（用 12 列 csv 模拟赛方挂载）：5 min
- 重 build：15 min
- 重 save tar：3 min
- 通知队友 + 重传：30 min（含网盘上传）
- 队友本地验证：10 min
- **总计 ~70 min**，时间充裕

---

## 8. 建议的处置顺序

1. ✅ **主开发者立刻拍板**：方案 A 或方案 B 或重训方案 C
2. 改 init.sh（按选定方案）
3. 用 baseline 12 列 csv 真场景 B' 验证 → 必须 result.csv MD5 = `f13034946...`
4. 重 build → 重 save tar → 记录新 MD5
5. tar 回路验证
6. 通知队友：用新 tar 替换提交
7. 更新 canonical 04 §4 残留风险表，登记此次发现
8. 更新 W1-submission-log §8，新增"赛方字段差异"条目

---

## 9. 一句话总结给主开发者

> **当前 LCF@NUDT.tar 在赛方真实评测时（12 列 csv）大概率会因 valuation.py KeyError 崩溃**。需立刻在 init.sh 加上"用 bundled 替换 data/" 的强制逻辑，重 build 重传。预计 70 分钟可完成，距 W1 截止仍有 ~30 小时，时间足够。

---

*本文档是发现风险时的实时记录，可能存在误判。请主开发者用第一性原理重新核对所有证据后再决策。*

---

## 10. 处置结论（2026-04-25 当晚已完成）

### 10.1 采用方案

**方案 A 改进版**：在 `init.sh` 中加 schema guard，检测 `/app/data/stock_data.csv` 是否含 `peTTM, pbMRQ, psTTM, pcfNcfTTM` 4 列；缺则强制从 `/app/data_bundled/stock_data.csv` 替换。

实现见 commit `32f57ac · Guard judge data schema in init`。

### 10.2 真场景 B' 验收（核心）

用 baseline 仓库的 12 列 `train.csv + test.csv` 模拟赛方挂载，跑 init.sh + test.sh：

容器日志命中预期分支：
```
[init] stock_data.csv missing required columns
[init] merged stock_data.csv still incompatible; falling back to bundled copy
[init] restoring stock_data.csv from /app/data_bundled/
[init] deps OK
[lgb-only] wrote ./output/result.csv (5 rows)
```

**未出现** `KeyError: 'peTTM'`。

### 10.3 全套验收回执（WSL i9-13900HX, 2026-04-25 晚）

```
=== W1 schema-fix Build Report · WSL ===
1. Git commit:           32f57ac ✅
2. CPU:                  i9-13900HX
3. CPU flags:            avx, avx2, avx_vnni（无 AVX-512）
4. Docker version:       29.4.1
5. baseline 12-col 检测: ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']  ✅
6. 场景 A MD5:           f13034946c0aaea5cb1e3f2d0d6ad692  ✅（init.sh 改动无副作用）
7. 场景 B' MD5:          f13034946c0aaea5cb1e3f2d0d6ad692  ✅（fallback 触发后 schema 一致）
8. 新 tar 大小:          526 MB（OCI 格式）
9. 新 tar MD5:           707133307dad3f7a221e74e507deede0
10. tar 回路 MD5:        f13034946c0aaea5cb1e3f2d0d6ad692  ✅
========================================
```

### 10.4 最终提交件

| 件 | MD5 | 状态 |
|---|---|---|
| `LCF@NUDT.tar` (526 MB) | `707133307dad3f7a221e74e507deede0` | ✅ 提交件 |
| `result.csv` | `f13034946c0aaea5cb1e3f2d0d6ad692` | ✅ 提交件 |
| ~~旧 tar `1a4ef943...`（1.7 GB legacy）~~ | — | ❌ 作废（schema 未修）|
| ~~旧 tar `1bb239f9...`（526 MB OCI 旧版）~~ | — | ❌ 作废（schema 未修）|

### 10.5 残留风险（已大幅降低）

- **OCI 格式 vs Legacy**：新 tar 是 Docker 26+ 默认 OCI 格式（526 MB）。赛方若用 Docker < 25 可能 load 失败。但 OCI 是 Docker 官方主推格式，赛方使用现代 Docker 概率高。
- **赛方真实评测数据是 12 列还是 16 列**：尚未直接确认。若赛方挂 16 列，schema OK 直通；若挂 12 列，触发 fallback；两条路径都已验证 result.csv 一致。**风险已闭环。**

### 10.6 关键经验

之前所有「场景 B」测试都用了我们仓库 16 列 csv 假装赛方挂载，**根本测不出 schema 风险**。本次发现的根因是**测试用例与真实评测口径不一致**——测试覆盖了"我们可控"的场景而非"对手可能给"的场景。教训记入 canonical 07-decisions（待写）。
