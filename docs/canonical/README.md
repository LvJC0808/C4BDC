# Canonical 文档目录

> **SSOT（Single Source of Truth）**—— 本目录是项目现状的唯一权威来源。
> 所有新决策必须先改这里，再同步他处。
> 最后更新：2026-04-24 · Phase 4

---

## 快速查找

| 你想知道 | 打开 |
|---|---|
| 项目现状一览 | `01-project-overview.md` |
| 架构与算法细节 | `02-methodology.md` |
| 所有回测 / AB 数据 | `03-results.md` |
| 跨平台 MD5 / 复现证据 | `04-reproducibility.md` |
| 赛方 baseline 真实表现 | `05-baseline.md` |
| W1 后续做什么 | `06-roadmap.md` |
| 为什么不做 X / 为什么撤回 Y | `07-decisions.md` |

---

## 文档层级

```
docs/
├── README.md                          ← 文档总入口（指回这里）
│
├── canonical/                         ← ★ 本目录 · SSOT
│   ├── README.md                      ← 本文
│   ├── 01-project-overview.md
│   ├── 02-methodology.md
│   ├── 03-results.md
│   ├── 04-reproducibility.md
│   ├── 05-baseline.md
│   ├── 06-roadmap.md
│   └── 07-decisions.md
│
├── submission/                        ← 提交记录（每次 W* 一份）
│   ├── W1-submission-log.md
│   └── W1-verification-checklist.md
│
├── handoff/                           ← 跨节点交接
│   └── 2026-04-25-w1-build-linux.md
│
├── slides/                            ← 答辩 PPT
│   ├── April_End_Report.pptx
│   └── 2026-04-24-w1-defense.pptx
│
└── archive/                           ← 历史存档（只读）
    ├── v1/                            (path_to_now.md 副本)
    ├── v2a/                           (docs/report.md)
    ├── findings/
    ├── old-reports/
    ├── old-handoffs/
    └── old-specs-plans/
```

---

## Phase 编号对照（历史命名 vs 当前）

| 当前 | 旧命名（archive 中仍保留） | 时间 | 关键产出 |
|---|---|---|---|
| Phase 1 · Baseline Alignment | v1 · StockTransformer | 03/15 – 04/10 | 本地 0.0642 |
| Phase 2 · Ensemble Foundation | v2a · 三模型 Ensemble | 04/14 – 04/21 | 87d +1.01%, t=+4.78 |
| Phase 3 · Deterministic Core | v2b · LGB-only 主线 | 04/22 – 04/23 | 213d +2.41%, t=+12.05 |
| Phase 4 · Risk Containment | v2c · W1 防守补丁 | 04/24 | +2.95%, M10-3 |

**W1-W4 是赛方提交批次**（独立于 Phase 编号），不作为阶段名。

---

## 使用本目录的规则

### ✅ 应该做的

1. **新事实发现** → 更新对应 canonical 文件 + 在 07-decisions 留一笔
2. **查询项目现状** → 先读 01，再定向到其他文件
3. **引用数据** → 从 03-results 取，不要从 archive 取
4. **新决策** → 先 07-decisions 立项，再同步 02/03

### ❌ 不应该做的

1. **在 archive 里修改** → 历史文档只读
2. **创建重复的"stage-report" / "path-to-now" 类文件** → 统一进 canonical
3. **把 PPT 内容抄到 canonical** → PPT 只是 canonical 的演绎版
4. **重开已关闭决策（DR-001/003/006/009）** → 除非有新证据

---

## 冲突解决

若 canonical 文档之间出现冲突：
1. 优先以 `07-decisions.md` 为准（决策最权威）
2. 其次以最近更新日期为准
3. 发现矛盾立刻在本 README 开 issue 段

---

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-04-24 | canonical/ 初始创建，7 份 SSOT 文档就位 |
| 2026-04-24 | 统一 Phase 1-4 编号，废弃 v1/v2a/v2b/v2c |
| 2026-04-24 | Holiday-forward-fill 永久关闭（DR-006） |
