# 赛方 Baseline · 权威记录

> **创建于 2026-04-24，务必长期保留。** 后续所有"跑赢 baseline"的判断都以本文档为准。

## 一、赛方 baseline 的本体

- **代码**：`/root/shared-nvme/C4BD/THU-BDC2026-main/`（赛方下发原版，不要修改）
- **关键文件**：
  - `code/src/model.py` — StockTransformer
  - `code/src/utils.py` — 特征工程（158+39 列）
  - `code/src/predict.py` — 推理入口
  - `model/60_158+39/best_model.pth` — 训练好的权重
- **权重 MD5**：`d9c56a6a332c0fef08157ca7661a2acb`
- **我们仓库内的副本**：`/root/shared-nvme/bigdata/THU-BDC2026/model/60_158+39/best_model.pth`（MD5 完全相同，同一份权重）

## 二、我们如何跑赛方 baseline

**脚本**：`scripts/rolling_backtest_baseline.py`
- `from model import StockTransformer`（来自我们仓库的 `code/src/model.py`，与赛方一致）
- 加载 `model/60_158+39/best_model.pth`（与赛方 MD5 一致）
- 自己 reimplement 推理逻辑（不调用赛方 `predict.py`，但用相同模型与权重）

**赛方代码差异（不影响 baseline 推理结果）**：
| 文件 | 一致性 |
|---|---|
| `code/src/model.py` | ✅ 一致 |
| `code/src/utils.py` | ✅ 一致 |
| `code/src/predict.py` | ⚠️ 不同（我们不用它） |
| `code/src/train.py` | ⚠️ 不同（我们没重训权重） |
| `code/src/config.py` | ⚠️ 不同（仅影响训练，推理不影响） |
| `model/60_158+39/best_model.pth` | ✅ MD5 一致 |

## 三、赛方 baseline 真实表现（截至 2026-04-24）

### 6 天 rolling（2026-04-01 ~ 2026-04-09）

来源：`test/rolling_backtest_baseline_apr.csv`

| 日期 | Top-5 | 5d 收益 |
|---|---|---|
| 04-01 | 600919, 601658, 601939, 601328, 601169 | −1.04% |
| 04-02 | 600919, 601658, 601169, 601939, 601328 | −2.43% |
| 04-03 | 600919, 601658, 601169, 601939, 601816 | −1.43% |
| 04-07 | 600919, 601658, 601169, 601916, 601816 | −0.72% |
| 04-08 | 600919, 601658, 601169, 601916, 601816 | −0.51% |
| 04-09 | 600919, 601169, 601658, 601916, 601816 | +0.84% |

**统计**：
- mean = **−0.881% / 5d**
- t-stat ≈ −1.99
- 胜率 = 17% (1/6)

### 行为特征（结构性事实）

1. **永远固定 5-8 只金融/基建蓝筹**：
   - 600919 江苏银行
   - 601658 邮储银行
   - 601169 北京银行
   - 601939 建设银行
   - 601328 交通银行
   - 601916 浙商银行
   - 601816 京沪高铁
   - 601328 / 601398（工商银行偶现）
2. **相邻日期换手 ≈ 0**（5 只里 4 只重叠）
3. **本质**：固定单一风格 anchor（大盘金融），不是动态选股

## 四、跑赢 baseline 的硬门槛

**赛规**：必须跑赢 baseline 才能参与排名。

**我们的成绩**（vs 赛方 baseline 6 天重叠）：

| 档位 | 我们 mean | 赛方 mean | Δ | 胜率 |
|---|---|---|---|---|
| Baseline-LGB (MCAP=0) | +0.101% | −0.881% | **+0.98pp** | 4/6 (67%) |
| **M10-3 (W1 提交)** | **+0.755%** | −0.881% | **+1.64pp** | 5/6 (83%) |

**结论**：W1 提交的 M10-3 在 6 天测试上**确认跑赢赛方 baseline**。

## 五、重要注意事项 ⚠️

1. **赛方 baseline 不是 HS300 等权**。HS300 在 2026 Apr 大盘轮动中近 10 天上涨 +1.37%，赛方 baseline 反而 −0.88%。**判断"跑赢"必须用赛方 baseline，不要用 HS300**。
2. **样本只有 6 天**（04-01 ~ 04-09）。我们没有 213 天连续的赛方 baseline rolling，因为每跑一天要 GPU 推理 + 人工，成本高。
3. **赛方未来可能换股**。若赛方在 4 月底突然换风格（理论上不会，因为权重锁死），我们的 +1.64pp 优势可能缩小。
4. **如何加跑新日期**：
   ```bash
   python scripts/rolling_backtest_baseline.py \
     --eval_start <YYYY-MM-DD> --eval_end <YYYY-MM-DD> \
     --model_dir ./model/60_158+39 \
     --out test/rolling_backtest_baseline_<tag>.csv
   ```

## 六、产物索引

- `test/rolling_backtest_baseline_apr.csv` — 4 月 6 天赛方 baseline rolling 结果
- `scripts/rolling_backtest_baseline.py` — 赛方 baseline rolling wrapper
- `scripts/score_self_compare.py` — 单窗口 baseline vs LGB 对比
- 赛方代码权威副本：`/root/shared-nvme/C4BD/THU-BDC2026-main/`（**勿改**）
