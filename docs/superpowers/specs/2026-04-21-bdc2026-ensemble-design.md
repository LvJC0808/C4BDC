# BDC2026 跑赢 Baseline 方案设计

日期：2026-04-21
分支：`feat/ensemble-v1`

## 0. 目标与约束

**目标**：在 2026 中国高校计算机大赛—大数据挑战赛（沪深 300 股价收益预测）中，
产出一个在 T+1 开盘买入、T+5 开盘卖出的 Top-K 投资组合，总收益率严格优于官方
baseline（[THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026)），从而获得
排名资格。

**硬约束**：

- 机器：i7-13650H / 16 GB / RTX 4060 8 GB / 50 GB 存储
- 训练时间 ≤ 8 h，预测时间 ≤ 5 min
- 复现训练/预测时不得联网
- Docker 镜像（含代码/数据/模型/依赖）总和 ≤ 10 G，不得压缩
- 开源数据/模型需 2026-04-01 前已公开，7/18 前邮件报备 MD5
- 代码与固定随机种子须完全可复现
- result.csv：stock_id,weight；≤5 只；权重和 ≤ 1（现金补足）

## 1. 核心思路

方案代号 **A4+ / B2 / D3 / E4 / F3 / Z1**：

| 维度 | 设计 |
|---|---|
| 模型集成（A4+） | LightGBM+DoubleEnsemble + MASTER（AAAI 2024）+ StockMixer（AAAI 2024） |
| 特征工程（B2/Z1） | Alpha158 + Alpha360 + 估值因子 + 行业/市值/Beta 中性化 + 横截面 rank-gauss |
| 组合构建（D3） | Top-K 等权 + 基于分数分散度/滚动 IC/模型一致性的置信度自适应仓位 |
| 验证策略（E4） | Walk-forward 3 段 + 5 天 embargo + 3 种子 + 最后一周 holdout |
| 交付（F3） | 保留 baseline，新增模块化 `pipeline.py` 总入口；git 多分支管理 |
| 数据扩展（Z1） | baostock 全量：行情+估值+行业+历史成分股+指数+交易日 |

## 2. 整体架构

```
app/
├── code/src/
│   ├── config.py
│   ├── features/
│   │   ├── alpha158.py        # 复用 baseline utils
│   │   ├── alpha360.py        # Qlib 风格 60×6
│   │   ├── neutralize.py      # 行业/市值/Beta 中性化 + rank-gauss
│   │   └── build.py           # 统一入口 (date, stock, X, y)
│   ├── data_prep/
│   │   └── fetch_all.py       # 离线 baostock 爬取
│   ├── models/
│   │   ├── lgb_de.py          # LightGBM + DoubleEnsemble
│   │   ├── master.py          # MASTER (基于 SJTU-DMTai)
│   │   └── stockmixer.py      # StockMixer (基于 SJTU-DMTai)
│   ├── cv/
│   │   └── walk_forward.py    # 3 段 + embargo=5 + 3 seed + holdout
│   ├── ensemble/
│   │   ├── blender.py         # rank 平均 / holdout grid-search
│   │   └── portfolio.py       # Top-K + 置信度自适应仓位
│   ├── pipeline.py            # 总入口 train / predict
│   ├── train.py               # wrapper → pipeline train
│   ├── predict.py             # wrapper → pipeline predict
│   ├── test.py                # 赛规必选，等同 predict.py
│   ├── featurework.py         # 赛规必选，等同 features/build.py 入口
│   └── verify_reproducibility.py
├── data/                      # 挂载；离线 baostock 产物
│   ├── stock_data.csv
│   ├── industry_map.csv
│   ├── stock_basic.csv
│   ├── hs300_history.csv
│   ├── csi300_index.csv
│   └── trade_calendar.csv
├── model/                     # docker 内训练产物
├── output/result.csv          # 挂载
├── temp/                      # 挂载，特征缓存
├── init.sh train.sh test.sh readme.md
├── Dockerfile
└── docker-compose.yml
```

**git 策略**：

- `main` = baseline 当前状态
- `feat/ensemble-v1` = 本次主分支
- 子分支按章节独立 PR：`feat/features-v2` → `feat/cv` → `feat/lgb-de` →
  `feat/master` → `feat/stockmixer` → `feat/ensemble-portfolio`
- `docs/superpowers/specs/experiments.md` 记录每次实验分数与 commit hash

## 3. 数据准备（Z1 扩展）

### 3.1 baostock 抓取清单

| 文件 | 字段 | 用途 |
|---|---|---|
| stock_data.csv | date,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM | 主行情 + 估值 |
| industry_map.csv | stock_id, sw_industry | 中性化 / MASTER embed |
| stock_basic.csv | stock_id, ipoDate | 剔除次新 |
| hs300_history.csv | date, stock_id | 动态成分股，防幸存者偏差 |
| csi300_index.csv | sh.000300 日线 | MASTER 市场特征 + Beta |
| trade_calendar.csv | date, is_trading | 对齐 |

**抓取方式**：`code/src/data_prep/fetch_all.py`，本地一次性执行（联网），产物
打包进 docker。Docker 内运行时**不**调用 baostock。

**总数据量**：~250 MB。

### 3.2 报备清单（7/18 前邮件到 data@tsinghua.edu.cn）

| 资源 | URL | MD5 |
|---|---|---|
| baostock | https://github.com/baostock/baostock | [库 + 数据快照 md5] |
| Qlib (Alpha158/360) | https://github.com/microsoft/qlib | [commit md5] |
| MASTER | https://github.com/SJTU-DMTai/MASTER | [commit md5] |
| StockMixer | https://github.com/SJTU-DMTai/StockMixer | [commit md5] |
| LightGBM | https://github.com/microsoft/LightGBM | [版本 md5] |
| TA-Lib | https://github.com/TA-Lib/ta-lib | [版本 md5] |
| 本队爬取数据 CSV | （不上传文件） | 逐文件 md5 |

## 4. 特征工程

### 4.1 原始特征

- **Alpha158**（baseline 已有）：技术指标因子集
- **Alpha360**（新增）：过去 60 日 × 6 字段（O/H/L/C/V/VWAP）除以 close_t 归一化
- **估值因子**（新增）：peTTM / pbMRQ / psTTM / pcfNcfTTM 及其
  20/60 日 z-score、行业内 rank
- **Beta（60 日）**：个股 vs CSI300 回归系数
- **市场特征**（MASTER 专用，63 维）：CSI300 的涨跌幅/振幅/换手/成交额
  + rolling mean/std/动量等（论文附录 A）

### 4.2 中性化（共享）

对每个交易日 t 横截面独立执行：

1. 行业组内减均值（按申万一级）
2. 对 log(流通市值或成交额代理) 和 Beta 做线性回归，取残差
3. 3σ 截尾
4. 横截面 rank → Φ⁻¹ 得 rank-gauss（均值 0、方差 1）

标签（T+1→T+5 开盘收益）同样做 rank-gauss。

### 4.3 样本池

候选 = 当日 HS300 成分股 ∩ 上市 ≥250 日 ∩ 近 20 日非连续停牌。训练/推理一致。

### 4.4 缺失值

- 停牌 ≤5 日：前值 ffill；>5 日：当日剔除
- 涨跌停：保留
- 新股不足 60 日：剔除

### 4.5 缓存

- `temp/features/<hash>.parquet`，hash = (特征集名, 起止, 中性化 flag)
- 首次 Alpha158+360 ~10 min；后续 <30 s

### 4.6 泄漏防控

- 所有特征只用到 t 日收盘前信息
- 中性化系数每日独立拟合，不跨日
- 标签跨 5 日 → CV 中 embargo ≥5

## 5. 模型

### 5.1 Model-1 · LightGBM + DoubleEnsemble

- 输入：Alpha158 + 估值 + Beta，中性化+rank-gauss
- objective = `regression_l1`，外部监控 Rank IC
- 超参：`num_leaves=64, lr=0.02, feature_fraction=0.7, bagging=0.7, min_data=200,
  deterministic=True, force_row_wise=True`
- DoubleEnsemble：3 轮样本/特征重加权子模型平均
- 早停：patience=50 on 验证集 final_score
- 规模：3 段 × 3 seed = 9 模型，~15 min/模型，合计 ~2.3 h

### 5.2 Model-2 · MASTER（论文 AAAI 2024）

- 基于 SJTU-DMTai 官方代码移植
- 输入：Alpha158 中性化 + market 63 维
- 结构：Feature Encoder → Market-Guided Gating (FiLM) → Intra-Stock Attn →
  Inter-Stock Attn → Temporal Agg → MLP Head
- 超参：T=8, d_model=256, n_head=4, dropout=0.5, β=5
- 损失：`0.5·MSE + 0.5·(1 − BatchRankIC)` + λ=0.1 Top-K aware 项
- AdamW，lr=1e-4，cosine decay，40 epochs，batch=1 天
- 规模：3 段 × 3 seed = 9 模型，~30 min/模型，合计 ~4.5 h

### 5.3 Model-3 · StockMixer（论文 AAAI 2024）

- 基于 SJTU-DMTai 官方代码移植
- 输入：Alpha360 原始 6 字段 × T=16 groups
- 三路并行：Time-Mixer / Feature-Mixer / Stock-Mixer（含 Market 聚合）
- 超参：hidden=128, scale=3, lookback=16, market_scale=3
- 损失：同 MASTER
- 规模：9 模型，~8 min/模型，合计 ~1.2 h

### 5.4 训练预算

| 项 | 累计 |
|---|---|
| 特征工程一次性 | 0.2 h |
| LightGBM+DE ×9 | 2.3 h |
| MASTER ×9 | 4.5 h |
| StockMixer ×9 | 1.2 h |
| **合计** | **~8.2 h** |

略超 8 h 预算；**维持不裁剪**，后续再通过 bf16/torch.compile 或降 seed/段 优化。

## 6. 交叉验证（E4）

```
全量：[D_start, D_end]；最后 1 周 holdout

段1: train[D_start .. D_end−150]  embargo 5  val[D_end−144 .. D_end−125]
段2: train[D_start .. D_end−100]  embargo 5  val[D_end− 94 .. D_end− 75]
段3: train[D_start .. D_end− 50]  embargo 5  val[D_end− 44 .. D_end− 25]
holdout: [D_end−20 .. D_end−5]
refit: 全量数据，epoch 数 = CV 每段中位数 × 1.05
```

- 每段独立重训；每段 3 个 seed = `[42, 2024, 7]`
- 以 final_score（评测一致）为 early-stop 唯一指标，patience=10
- holdout 只用于集成权重、仓位斜率、Top-K 的最终网格搜索

## 7. 集成与组合构建（D3）

### 7.1 分数融合

1. 每模型每 seed/段 内部横截面 rank 归一 [0,1]
2. 同模型 9 预测取均值 → 每模型一向量
3. 跨模型加权 rank 平均，初始 `LGB=0.4, MASTER=0.4, Mixer=0.2`
4. holdout 单纯形网格 step=0.1 搜索最优权重，写 `model/ensemble_config.json`

### 7.2 Top-K 选股

按 final_score 降序取 Top-K，K ∈ {3,4,5}，由 holdout 选择。

### 7.3 置信度自适应仓位

```
c_disp  = norm(Top5 mean rank − cross-section median)
c_ic    = clip(rolling_RankIC_last_20d / 0.05, 0, 1)
c_agree = |三模型 Top5 交集| / 5
c       = 0.4·c_disp + 0.4·c_ic + 0.2·c_agree
total_position = 0.5 + α·0.5·c    α ∈ {0.3,0.5,0.7}（holdout 选）
w_i = total_position / K（Top-K 内部等权）
cash = 1 − total_position（隐式，result.csv 不写现金行）
```

## 8. 工程交付

### 8.1 Dockerfile

- base: `python:3.10-slim`
- 装 TA-Lib C 库 + Python 依赖
- 依赖：`torch`（CPU wheel，运行时检测 GPU） / `lightgbm` / `numpy` / `pandas` /
  `scipy` / `scikit-learn` / `pyarrow` / `talib` / `tqdm`
- 镜像大小估计 ~3.5 G，加模型 ~1 G + 数据 ~0.25 G = **<5 G** ✅

### 8.2 init.sh / train.sh / test.sh

```bash
# init.sh
set -e
mkdir -p /app/model /app/output /app/temp
python -c "import lightgbm, torch, talib, numpy, pandas; print('OK')"
```

```bash
# train.sh
set -e
cd /app/code/src
python pipeline.py train \
  --data_path /app/data --model_dir /app/model --temp_dir /app/temp
```

```bash
# test.sh
set -e
cd /app/code/src
python pipeline.py predict \
  --data_path /app/data --model_dir /app/model --temp_dir /app/temp \
  --output_path /app/output/result.csv
```

### 8.3 随机性固定

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
```

LightGBM 显式 `deterministic=True, force_row_wise=True`。

### 8.4 可复现性自检

`verify_reproducibility.py`：连续两次训练比较权重 md5；连续两次推理比较
result.csv md5；结果写 `model/repro_check.log`。

### 8.5 推理预算

| 步骤 | 时间 |
|---|---|
| 特征（增量） | <30 s |
| LGB ×9 | <10 s |
| MASTER ×9 | <60 s |
| Mixer ×9 | <20 s |
| 聚合+选股 | <5 s |
| **总** | **<3 min** ✅ |

### 8.6 readme.md 骨架（赛规格式）

包含：环境配置、数据来源、预训练模型说明（本方案无预训练）、算法整体思路、
创新点（MASTER+Mixer+GBDT+动态仓位）、网络结构、损失函数、数据扩增
（无，改用中性化+rank-gauss）、模型集成、训练流程（注释 train.py）、推理流程
（注释 test.py）、其他注意事项（验证划分、embargo、种子）。

### 8.7 代理（离线爬取阶段）

本地爬取 baostock 及克隆 GitHub 代码时可使用：

```
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
```

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 训练总时长略超 8 h | bf16 + torch.compile；或降 seed/段；或 MASTER 早停更激进 |
| MASTER/StockMixer 移植 bug | 优先保留官方代码接口，只包一层 data adapter |
| 中性化导致模型看不到风格因子 | 保留未中性化版本作为 LGB 的额外列 |
| holdout 过拟合 | 网格仅搜 3 个维度；记录 CV 均值交叉验证 |
| 行情 csv 体积 / IO 慢 | parquet 缓存，增量追加 |
| 复现种子漂移 | 强制 cudnn deterministic；LightGBM deterministic 模式 |

## 10. 交付清单

- [ ] 离线 `fetch_all.py` 生成 6 个 CSV
- [ ] `feat/ensemble-v1` 主分支完成
- [ ] 可复现自检通过
- [ ] `docker build -t bdc2026 .`
- [ ] `docker save bdc2026 -o 队伍名称.tar`，上传夸克网盘无提取码
- [ ] 7/18 前邮件报备 data@tsinghua.edu.cn
- [ ] 平台提交 result.csv
