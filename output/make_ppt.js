// BDC2026 精简答辩 PPT v2 —— 大字号 + 高冲击力配色
// 调色：深墨蓝主色 + 金色强调 + 红色数据高光；Sandwich 深/浅交替
const pptxgen = require("pptxgenjs");
const pptx = new pptxgen();

pptx.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pptx.title = "BDC2026 阶段性汇报";

// ---------- Palette ----------
const INK    = "0A1628";
const NAVY   = "17305C";
const GOLD   = "E8B14A";
const RED    = "E63946";
const CREAM  = "FBF7EF";
const PAPER  = "FFFFFF";
const MUTED  = "5B6B7F";
const LINE   = "C9D2DE";
const ICE    = "CADCFC";

const H_FONT = "Georgia";
const B_FONT = "Calibri";
const M_FONT = "Consolas";

function darkTitleSlide(title, subtitle) {
  const s = pptx.addSlide();
  s.background = { color: INK };
  s.addShape("rect", { x: 0, y: 0, w: 0.35, h: 7.5, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText(title, {
    x: 0.75, y: 0.35, w: 12.3, h: 0.95,
    fontFace: H_FONT, fontSize: 40, bold: true, color: PAPER,
  });
  if (subtitle) {
    s.addText(subtitle, {
      x: 0.75, y: 1.25, w: 12.3, h: 0.55,
      fontFace: B_FONT, fontSize: 20, italic: true, color: GOLD,
    });
  }
  return s;
}

function lightSlide(title, subtitle) {
  const s = pptx.addSlide();
  s.background = { color: CREAM };
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 0.28, fill: { color: INK }, line: { color: INK } });
  s.addShape("rect", { x: 0, y: 0, w: 1.6, h: 0.28, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText(title, {
    x: 0.6, y: 0.42, w: 12.3, h: 0.85,
    fontFace: H_FONT, fontSize: 40, bold: true, color: INK,
  });
  if (subtitle) {
    s.addText(subtitle, {
      x: 0.6, y: 1.25, w: 12.3, h: 0.45,
      fontFace: B_FONT, fontSize: 18, italic: true, color: MUTED,
    });
  }
  return s;
}

function footer(s, pageNum, dark) {
  const c = dark ? ICE : MUTED;
  s.addText("BDC2026 · feat/ensemble-v1 · 2026-04-22", {
    x: 0.6, y: 7.12, w: 9, h: 0.3,
    fontFace: B_FONT, fontSize: 12, color: c,
  });
  s.addText(String(pageNum), {
    x: 12.4, y: 7.12, w: 0.5, h: 0.3,
    fontFace: H_FONT, fontSize: 14, bold: true, color: dark ? GOLD : INK, align: "right",
  });
}

// ============================================================
// 1. 封面
// ============================================================
{
  const s = pptx.addSlide();
  s.background = { color: INK };
  s.addShape("rect", { x: 0, y: 0, w: 0.7, h: 7.5, fill: { color: GOLD }, line: { color: GOLD } });
  s.addShape("rect", { x: 0.7, y: 6.5, w: 12.63, h: 0.08, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText("沪深 300 Top-K 组合", {
    x: 1.2, y: 1.9, w: 11.5, h: 1.2,
    fontFace: H_FONT, fontSize: 56, bold: true, color: PAPER,
  });
  s.addText("收益预测系统", {
    x: 1.2, y: 3.05, w: 11.5, h: 1.0,
    fontFace: H_FONT, fontSize: 56, bold: true, color: GOLD,
  });
  s.addText("LightGBM · MASTER · StockMixer   |   三模型集成", {
    x: 1.2, y: 4.25, w: 11.5, h: 0.6,
    fontFace: B_FONT, fontSize: 26, color: ICE,
  });
  s.addText("2026 中国高校计算机大赛 — 大数据挑战赛", {
    x: 1.2, y: 6.7, w: 11.5, h: 0.4,
    fontFace: B_FONT, fontSize: 18, italic: true, color: GOLD,
  });
  s.addText("阶段性汇报  ·  2026 / 04", {
    x: 1.2, y: 7.05, w: 11.5, h: 0.35,
    fontFace: B_FONT, fontSize: 16, color: ICE,
  });
}

// ============================================================
// 2. 核心结论
// ============================================================
{
  const s = darkTitleSlide("Key Results · 核心结论", "87 交易日严格滚动回测 · 训练 / 推理 / 镜像全部在硬约束内");
  const stats = [
    { v: "+1.01%", l: "组合 5 日开盘-开盘平均收益" },
    { v: "75.86%", l: "单日胜率（收益 > 0）" },
    { v: "p<0.0001", l: "vs 等权 HS300\nt = +4.78, N = 87" },
  ];
  stats.forEach((st, i) => {
    const x = 0.75 + i * 4.15;
    s.addShape("roundRect", { x, y: 2.1, w: 3.85, h: 2.6, fill: { color: NAVY }, line: { color: GOLD, width: 1.5 }, rectRadius: 0.1 });
    s.addText(st.v, { x, y: 2.3, w: 3.85, h: 1.5, fontFace: H_FONT, fontSize: 56, bold: true, color: GOLD, align: "center", valign: "middle" });
    s.addText(st.l, { x: x + 0.15, y: 3.75, w: 3.55, h: 0.9, fontFace: B_FONT, fontSize: 18, color: PAPER, align: "center" });
  });
  s.addShape("rect", { x: 0.75, y: 5.05, w: 0.12, h: 0.55, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText([
    { text: "工程预算对账   ", options: { bold: true, color: GOLD, fontSize: 20 } },
    { text: "训练 6.3 h / 8 h · 推理 < 3 min / 5 min · 镜像 ~4 GB / 10 GB · 两次推理 MD5 一致",
      options: { fontSize: 18, color: PAPER } },
  ], { x: 1.0, y: 5.0, w: 12.0, h: 0.65, fontFace: B_FONT });

  s.addShape("rect", { x: 0.75, y: 5.8, w: 0.12, h: 0.55, fill: { color: RED }, line: { color: RED } });
  s.addText([
    { text: "Phase-A 优化   ", options: { bold: true, color: GOLD, fontSize: 20 } },
    { text: "涨停过滤 + 6 bp 手续费后 mean 仅降 1 bp，显著性保持 p<0.0001，推荐作为正式提交配置。",
      options: { fontSize: 18, color: PAPER } },
  ], { x: 1.0, y: 5.75, w: 12.0, h: 0.65, fontFace: B_FONT });

  s.addText("关键启示：学习横截面相对排名，不依赖市场趋势 — 本窗口动量策略全程亏损 (−0.76% / 5d)。", {
    x: 0.75, y: 6.55, w: 12.0, h: 0.5, fontFace: B_FONT, fontSize: 16, italic: true, color: ICE,
  });
  footer(s, 2, true);
}

// ============================================================
// 3. 赛题与约束
// ============================================================
{
  const s = lightSlide("Task & Constraints · 赛题与约束", "CSI 300 · T+1 开盘买入 / T+5 开盘卖出 · Top-K ≤ 5");
  const rows = [
    [
      { text: "维度", options: { bold: true, color: PAPER, fill: INK, fontSize: 18 } },
      { text: "要求", options: { bold: true, color: PAPER, fill: INK, fontSize: 18 } },
    ],
    ["预测目标", "沪深 300 成分股 T+1 开盘买入、T+5 开盘卖出的 Top-K 组合收益"],
    ["输出",    "result.csv：stock_id, weight；≤ 5 只；权重和 ≤ 1（现金补足）"],
    ["硬件",    "i7-13650H   /   16 GB   /   RTX 4060 8 GB"],
    ["训练时长", "≤ 8 小时"],
    ["推理时长", "≤ 5 分钟"],
    ["运行时",  "离线（无网络） · Docker 镜像 ≤ 10 GB"],
    ["可复现",  "固定随机种子，两次训练 / 推理权重与输出 MD5 一致"],
    ["数据截止", "2026-04-01 前公开资源"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.95, w: 12.1, colW: [2.6, 9.5],
    fontFace: B_FONT, fontSize: 18, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE },
    rowH: 0.55, valign: "middle", fill: { color: PAPER },
  });
  footer(s, 3);
}

// ============================================================
// 4. 整体架构
// ============================================================
{
  const s = lightSlide("System Architecture · 整体架构", "单一入口 pipeline.py { train | predict }");

  const upper = [
    { x: 0.6,  w: 4.0, title: "① 数据层",    body: "baostock 离线 ~85 MB\n行情 · 估值 · 行业\n成分股史 · 指数" },
    { x: 4.85, w: 4.0, title: "② 特征工程",  body: "Alpha158 · Alpha360\n估值 + 市场 63 维 + Beta60\n中性化 + rank-gauss" },
    { x: 9.1,  w: 3.6, title: "③ CV 切分",   body: "Walk-Forward 3 段\nembargo = 5 天\nholdout 20 天 · seeds [42/2024/7]" },
  ];
  upper.forEach(b => {
    s.addShape("roundRect", { x: b.x, y: 1.85, w: b.w, h: 1.95, fill: { color: PAPER }, line: { color: NAVY, width: 1.5 }, rectRadius: 0.08 });
    s.addShape("rect", { x: b.x, y: 1.85, w: 0.14, h: 1.95, fill: { color: GOLD }, line: { color: GOLD } });
    s.addText(b.title, { x: b.x + 0.25, y: 1.95, w: b.w - 0.35, h: 0.48, fontFace: H_FONT, fontSize: 20, bold: true, color: INK });
    s.addText(b.body,  { x: b.x + 0.25, y: 2.45, w: b.w - 0.35, h: 1.35, fontFace: B_FONT, fontSize: 15, color: INK });
  });
  s.addShape("downArrow", { x: 6.45, y: 3.88, w: 0.45, h: 0.45, fill: { color: GOLD }, line: { color: GOLD } });

  const models = [
    { x: 0.6,  w: 4.0, t: "LightGBM + DoubleEnsemble", b: "Alpha158 + 估值\nSR+FR 3 轮重加权" },
    { x: 4.85, w: 4.0, t: "MASTER (AAAI 2024)",        b: "Alpha158 + 市场 63 维\nFiLM 门控 · 股内/股间 attn" },
    { x: 9.1,  w: 3.6, t: "StockMixer (AAAI 2024)",    b: "Alpha360 原始 OHLCV\nTime/Feature/Stock Mixer" },
  ];
  models.forEach(b => {
    s.addShape("roundRect", { x: b.x, y: 4.45, w: b.w, h: 1.55, fill: { color: INK }, line: { color: INK }, rectRadius: 0.08 });
    s.addText(b.t, { x: b.x + 0.2, y: 4.52, w: b.w - 0.35, h: 0.45, fontFace: H_FONT, fontSize: 18, bold: true, color: GOLD });
    s.addText(b.b, { x: b.x + 0.2, y: 4.98, w: b.w - 0.35, h: 1.0,  fontFace: B_FONT, fontSize: 14, color: PAPER });
  });
  s.addShape("downArrow", { x: 6.45, y: 6.08, w: 0.45, h: 0.45, fill: { color: RED }, line: { color: RED } });

  s.addShape("roundRect", { x: 2.0, y: 6.58, w: 9.33, h: 0.55, fill: { color: RED }, line: { color: RED }, rectRadius: 0.08 });
  s.addText("④ Rank-Blend + 置信度自适应仓位 → output / result.csv", {
    x: 2.0, y: 6.58, w: 9.33, h: 0.55, fontFace: H_FONT, fontSize: 18, bold: true, color: PAPER, align: "center", valign: "middle",
  });
  footer(s, 4);
}

// ============================================================
// 5. 三模型集成
// ============================================================
{
  const s = lightSlide("Ensemble · 三模型集成", "横截面 rank 归一 → holdout 单纯形网格搜索最优权重");

  const rows = [
    [
      { text: "模型", options: { bold: true, color: PAPER, fill: INK, fontSize: 18 } },
      { text: "输入特征", options: { bold: true, color: PAPER, fill: INK, fontSize: 18 } },
      { text: "结构要点", options: { bold: true, color: PAPER, fill: INK, fontSize: 18 } },
      { text: "权重", options: { bold: true, color: PAPER, fill: INK, fontSize: 18, align: "center" } },
    ],
    ["LightGBM + DoubleEnsemble", "Alpha158 + 估值（中性化）", "L1 回归 · SR + FR 3 轮级联",
      { text: "0.1", options: { align: "center", bold: true, color: MUTED } }],
    ["MASTER", "Alpha158 + 市场 63 维", "FiLM 市场门控 → 股内 / 股间 attn",
      { text: "0.9", options: { align: "center", bold: true, color: RED, fontSize: 22 } }],
    ["StockMixer", "Alpha360 原始 OHLCV", "Time / Feature / Stock Mixer",
      { text: "0.0", options: { align: "center", bold: true, color: MUTED } }],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.95, w: 12.1, colW: [3.3, 3.4, 4.2, 1.2],
    fontFace: B_FONT, fontSize: 16, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.62, valign: "middle", fill: { color: PAPER },
  });

  s.addShape("roundRect", { x: 0.6, y: 4.85, w: 12.1, h: 2.1, fill: { color: INK }, line: { color: INK }, rectRadius: 0.08 });
  s.addShape("rect", { x: 0.6, y: 4.85, w: 0.14, h: 2.1, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText("置信度自适应仓位 (D3)", {
    x: 0.9, y: 4.95, w: 11.7, h: 0.45, fontFace: H_FONT, fontSize: 22, bold: true, color: GOLD,
  });
  s.addText(
    "c = 0.4·c_disp  +  0.4·c_ic  +  0.2·c_agree\n" +
    "total_position = 0.5 + α · 0.5 · c         (α = 0.7)",
    { x: 0.9, y: 5.4, w: 11.7, h: 0.9, fontFace: M_FONT, fontSize: 18, bold: true, color: PAPER }
  );
  s.addText(
    "c_disp：Top-K 分散度   |   c_ic：近 20 日 rolling RankIC   |   c_agree：三模型 Top-K 交集",
    { x: 0.9, y: 6.3, w: 11.7, h: 0.55, fontFace: B_FONT, fontSize: 15, italic: true, color: ICE }
  );
  footer(s, 5);
}

// ============================================================
// 6. 实验结果
// ============================================================
{
  const s = lightSlide("Experimental Results · 滚动回测 87 天", "2025-11-03 ~ 2026-03-13 · 训练数据截止远早于回测窗口，无泄漏");

  const chartData = [{
    name: "平均 5 日收益 (%)",
    labels: ["我方集成", "等权 HS300", "5 日动量 Top-K"],
    values: [1.01, 0.11, -0.76],
  }];
  s.addChart(pptx.ChartType.bar, chartData, {
    x: 0.6, y: 1.95, w: 6.3, h: 4.4, barDir: "col",
    showTitle: true, title: "Mean 5-Day Return (%)",
    titleFontSize: 18, titleColor: INK, titleFontFace: H_FONT, titleBold: true,
    showValue: true, dataLabelFontSize: 16, dataLabelColor: INK, dataLabelFontBold: true,
    dataLabelFormatCode: "+0.00;-0.00",
    chartColors: [RED, NAVY, "9AA7B8"],
    catAxisLabelFontSize: 16, catAxisLabelFontFace: B_FONT, catAxisLabelColor: INK,
    valAxisLabelFontSize: 14, valAxisLabelColor: MUTED,
    showLegend: false,
  });

  const rows = [
    [
      { text: "指标", options: { bold: true, color: PAPER, fill: INK, fontSize: 16 } },
      { text: "我方", options: { bold: true, color: PAPER, fill: INK, fontSize: 16, align: "center" } },
      { text: "HS300", options: { bold: true, color: PAPER, fill: INK, fontSize: 16, align: "center" } },
      { text: "动量", options: { bold: true, color: PAPER, fill: INK, fontSize: 16, align: "center" } },
    ],
    ["平均 5 日收益",
      { text: "+1.01%", options: { bold: true, color: RED, align: "center" } },
      { text: "+0.11%", options: { align: "center" } },
      { text: "−0.76%", options: { align: "center" } }],
    ["中位数", { text: "+0.79%", options: { align: "center" } }, { text: "+0.31%", options: { align: "center" } }, { text: "−1.00%", options: { align: "center" } }],
    ["标准差", { text: "1.54%", options: { align: "center" } }, { text: "1.45%", options: { align: "center" } }, { text: "6.28%", options: { align: "center" } }],
    ["胜率 (>0)",
      { text: "75.86%", options: { bold: true, color: RED, align: "center" } },
      { text: "60.92%", options: { align: "center" } },
      { text: "45.98%", options: { align: "center" } }],
    ["vs 等权胜天",
      { text: "70.11%", options: { bold: true, color: RED, align: "center" } },
      { text: "—", options: { align: "center" } },
      { text: "—", options: { align: "center" } }],
  ];
  s.addTable(rows, {
    x: 7.15, y: 1.95, w: 5.55, colW: [1.85, 1.4, 1.2, 1.1],
    fontFace: B_FONT, fontSize: 15, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.5, valign: "middle", fill: { color: PAPER },
  });

  s.addShape("roundRect", { x: 7.15, y: 5.15, w: 5.55, h: 1.25, fill: { color: INK }, line: { color: INK }, rectRadius: 0.08 });
  s.addText("显著性检验", { x: 7.35, y: 5.22, w: 5.2, h: 0.4, fontFace: H_FONT, fontSize: 18, bold: true, color: GOLD });
  s.addText(
    "vs 等权：  t = +4.78,  p < 0.0001\n" +
    "vs 动量：  t = +2.67,  p = 0.009",
    { x: 7.35, y: 5.6, w: 5.2, h: 0.85, fontFace: M_FONT, fontSize: 16, bold: true, color: PAPER }
  );

  s.addText(
    "最佳日 +5.01% (02-13)  ·  最差日 −2.90% (01-27)  ·  平均置信度 0.77，平均仓位 ≈ 0.77  ·  尾部偏正",
    { x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontFace: B_FONT, fontSize: 15, italic: true, color: MUTED, align: "center" }
  );
  footer(s, 6);
}

// ============================================================
// 7. 工程交付 & Phase-A
// ============================================================
{
  const s = lightSlide("Delivery & Phase-A Ablation", "预算全部达标 · 推荐 Tradable + Cost 为正式提交配置");

  s.addText("预算对账", { x: 0.6, y: 1.95, w: 6.0, h: 0.5, fontFace: H_FONT, fontSize: 22, bold: true, color: INK });
  const budget = [
    [
      { text: "阶段", options: { bold: true, color: PAPER, fill: INK, fontSize: 16 } },
      { text: "预算", options: { bold: true, color: PAPER, fill: INK, fontSize: 16, align: "center" } },
      { text: "实测", options: { bold: true, color: PAPER, fill: INK, fontSize: 16, align: "center" } },
    ],
    ["特征工程（一次性）", { text: "0.2 h", options: { align: "center" } },
      { text: "0.04 h", options: { color: RED, bold: true, align: "center" } }],
    ["训练（3×3×3 + refit）", { text: "8 h", options: { align: "center" } },
      { text: "6.3 h", options: { color: RED, bold: true, align: "center" } }],
    ["推理", { text: "5 min", options: { align: "center" } },
      { text: "< 3 min", options: { color: RED, bold: true, align: "center" } }],
    ["镜像大小", { text: "10 GB", options: { align: "center" } },
      { text: "~ 4 GB", options: { color: RED, bold: true, align: "center" } }],
  ];
  s.addTable(budget, {
    x: 0.6, y: 2.5, w: 6.0, colW: [2.8, 1.6, 1.6],
    fontFace: B_FONT, fontSize: 16, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.55, valign: "middle", fill: { color: PAPER },
  });

  s.addText("Phase-A 对比", { x: 6.8, y: 1.95, w: 6.0, h: 0.5, fontFace: H_FONT, fontSize: 22, bold: true, color: INK });
  const phaseA = [
    [
      { text: "配置", options: { bold: true, color: PAPER, fill: INK, fontSize: 15 } },
      { text: "Mean", options: { bold: true, color: PAPER, fill: INK, fontSize: 15, align: "center" } },
      { text: "胜率", options: { bold: true, color: PAPER, fill: INK, fontSize: 15, align: "center" } },
      { text: "p 值", options: { bold: true, color: PAPER, fill: INK, fontSize: 15, align: "center" } },
    ],
    ["Baseline",
      { text: "+1.01%", options: { align: "center" } },
      { text: "75.86%", options: { align: "center" } },
      { text: "<0.0001", options: { align: "center" } }],
    [{ text: "+Tradable +Cost ✓", options: { bold: true, color: RED } },
      { text: "+1.00%", options: { bold: true, color: RED, align: "center" } },
      { text: "75.61%", options: { align: "center" } },
      { text: "<0.0001", options: { align: "center" } }],
    ["+Dynamic weights",
      { text: "+0.86%", options: { align: "center" } },
      { text: "67.07%", options: { align: "center" } },
      { text: "0.0001", options: { align: "center" } }],
  ];
  s.addTable(phaseA, {
    x: 6.8, y: 2.5, w: 6.0, colW: [2.5, 1.2, 1.15, 1.15],
    fontFace: B_FONT, fontSize: 15, color: INK,
    border: { type: "solid", pt: 0.75, color: LINE }, rowH: 0.6, valign: "middle", fill: { color: PAPER },
  });

  s.addShape("roundRect", { x: 0.6, y: 5.5, w: 12.1, h: 1.4, fill: { color: INK }, line: { color: INK }, rectRadius: 0.08 });
  s.addShape("rect", { x: 0.6, y: 5.5, w: 0.14, h: 1.4, fill: { color: GOLD }, line: { color: GOLD } });
  s.addText("Docker 交付", { x: 0.9, y: 5.58, w: 11.7, h: 0.4, fontFace: H_FONT, fontSize: 20, bold: true, color: GOLD });
  s.addText(
    "python:3.12-slim + TA-Lib + uv   |   init.sh / train.sh / test.sh / pipeline.py   |   verify_reproducibility.py 两次 MD5 比对",
    { x: 0.9, y: 6.0, w: 11.7, h: 0.9, fontFace: B_FONT, fontSize: 16, color: PAPER }
  );
  footer(s, 7);
}

// ============================================================
// 8. 结论
// ============================================================
{
  const s = darkTitleSlide("Conclusion · 结论与下一步", "pipeline 端到端跑通 · 显著跑赢 baseline · Docker 可直接交付");

  const items = [
    { k: "跑赢 Baseline",  v: "82 天扣除摩擦 · p < 0.0001 · 超额 ~87 bp / 5 日" },
    { k: "硬约束全达标",   v: "训练 6.3 h · 推理 < 3 min · 镜像 ~4 GB · MD5 一致" },
    { k: "推荐提交配置",   v: "Tradable 过滤 + 6 bp 手续费（近零损耗，贴近真实成交）" },
    { k: "下一步",         v: "回测窗口扩至 150–200 天 · 4060 联调 · Docker 上传" },
  ];
  items.forEach((it, i) => {
    const y = 2.1 + i * 1.1;
    s.addShape("ellipse", { x: 0.75, y: y + 0.05, w: 0.85, h: 0.85,
      fill: { color: GOLD }, line: { color: GOLD } });
    s.addText(String(i + 1), { x: 0.75, y: y + 0.05, w: 0.85, h: 0.85,
      fontFace: H_FONT, fontSize: 30, bold: true, color: INK, align: "center", valign: "middle" });
    s.addText(it.k, { x: 1.85, y: y, w: 11.0, h: 0.48,
      fontFace: H_FONT, fontSize: 22, bold: true, color: PAPER });
    s.addText(it.v, { x: 1.85, y: y + 0.48, w: 11.0, h: 0.55,
      fontFace: B_FONT, fontSize: 18, color: ICE });
  });

  s.addText("Thanks  ·  Q & A", {
    x: 0.6, y: 6.7, w: 12.1, h: 0.5,
    fontFace: H_FONT, fontSize: 22, italic: true, bold: true, color: GOLD, align: "right",
  });
  footer(s, 8, true);
}

pptx.writeFile({ fileName: "/root/shared-nvme/bigdata/THU-BDC2026/output/BDC2026_report.pptx" })
  .then(f => console.log("saved:", f));
