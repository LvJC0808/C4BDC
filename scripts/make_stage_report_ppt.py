"""阶段性报告 PPT 生成器 — THU-BDC2026 LGB-only 主线

设计：蓝白 editorial minimalism，≥18pt，正式字体
- 主色 #0B2545  次色 #1E4D8C  点缀 #3D8BFF
- 背景 #F5F7FA  分隔线 #D9E2EC  正文 #1A1F2C  次文 #6B7A8F
- 警示 #D97706  绿 #0E7C3A  红 #B92828

字体：
- 中文标题 Noto Serif CJK SC / 英文标题 Georgia
- 中文正文 Noto Sans CJK SC  / 英文正文 Georgia
- 数据与代码 Consolas
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ---------------- palette ----------------
NAVY   = RGBColor(0x0B, 0x25, 0x45)
COBALT = RGBColor(0x1E, 0x4D, 0x8C)
SKY    = RGBColor(0x3D, 0x8B, 0xFF)
CLOUD  = RGBColor(0xF5, 0xF7, 0xFA)
LINE   = RGBColor(0xD9, 0xE2, 0xEC)
INK    = RGBColor(0x1A, 0x1F, 0x2C)
MUTE   = RGBColor(0x6B, 0x7A, 0x8F)
AMBER  = RGBColor(0xD9, 0x77, 0x06)
WIN    = RGBColor(0x0E, 0x7C, 0x3A)
LOSE   = RGBColor(0xB9, 0x28, 0x28)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

# ---------------- fonts ----------------
# 西文（latin）与中文（ea/东亚）分轨指定
CJK_SERIF = "Noto Serif CJK SC"
CJK_SANS  = "Noto Sans CJK SC"
EN_SERIF  = "Georgia"
EN_SANS   = "Georgia"
MONO      = "Consolas"

# 字体 role 包装
SERIF = {"latin": EN_SERIF, "ea": CJK_SERIF}   # 标题
SANS  = {"latin": EN_SANS,  "ea": CJK_SANS}    # 正文
CODE  = {"latin": MONO,     "ea": CJK_SANS}    # 数据/代码

# ---------------- geometry ----------------
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)

# ---------------- helpers ----------------
def _set_run_font(run, font, size, color, bold=False, italic=False):
    f = run.font
    f.name = font["latin"]
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    # 通过 XML 把 ea（东亚）和 cs（复杂脚本）也指定
    rPr = run._r.get_or_add_rPr()
    # 清理旧的 ea 节点
    for child in rPr.findall(qn("a:ea")):
        rPr.remove(child)
    for child in rPr.findall(qn("a:cs")):
        rPr.remove(child)
    ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", font["ea"])
    cs = etree.SubElement(rPr, qn("a:cs"))
    cs.set("typeface", font["latin"])

def add_rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(0.5)
    s.shadow.inherit = False
    return s

def add_line(slide, x1, y1, x2, y2, color=LINE, weight=0.75):
    s = slide.shapes.add_connector(1, x1, y1, x2, y2)
    s.line.color.rgb = color; s.line.width = Pt(weight)
    return s

def add_text(slide, x, y, w, h, text, *,
             font=SANS, size=18, color=INK, bold=False, italic=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text.split('\n') if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        _set_run_font(r, font, size, color, bold, italic)
    return tb

def page_frame(slide, page_num, total, section, title_cn, title_en=None):
    add_rect(slide, 0, 0, W, H, CLOUD)
    add_rect(slide, 0, 0, W, Inches(0.08), NAVY)
    add_text(slide, MARGIN, Inches(0.26), Inches(7), Inches(0.35),
             section.upper(), font=CODE, size=18, color=COBALT, bold=True)
    add_text(slide, W - MARGIN - Inches(2.2), Inches(0.26), Inches(2.2), Inches(0.35),
             f"{page_num:02d} / {total:02d}",
             font=CODE, size=18, color=MUTE, align=PP_ALIGN.RIGHT)
    add_text(slide, MARGIN, Inches(0.7), W - 2 * MARGIN, Inches(0.8),
             title_cn, font=SERIF, size=32, color=NAVY, bold=True)
    if title_en:
        add_text(slide, MARGIN, Inches(1.25), W - 2 * MARGIN, Inches(0.4),
                 title_en, font=SERIF, size=18, color=MUTE, italic=True)
    add_line(slide, MARGIN, H - Inches(0.45), W - MARGIN, H - Inches(0.45), LINE)
    add_text(slide, MARGIN, H - Inches(0.38), Inches(7), Inches(0.3),
             "THU-BDC2026 · LGB-only Mainline", font=SANS, size=18, color=MUTE)
    add_text(slide, W - MARGIN - Inches(3), H - Inches(0.38), Inches(3), Inches(0.3),
             "2026-04-24", font=CODE, size=18, color=MUTE, align=PP_ALIGN.RIGHT)

# ============================================================
#                           SLIDES
# ============================================================

def slide_cover(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, W, H, CLOUD)
    add_rect(s, 0, 0, Inches(4.8), H, NAVY)

    add_text(s, Inches(0.6), Inches(0.7), Inches(4), Inches(0.4),
             "STAGE REPORT / 01", font=CODE, size=18, color=SKY, bold=True)
    add_text(s, Inches(0.6), Inches(1.35), Inches(4), Inches(4.2),
             "LGB-only\n主线收敛\n与赛方\nBaseline\n对比",
             font=SERIF, size=44, color=WHITE, bold=True, spacing=1.1)

    add_text(s, Inches(5.2), Inches(1.35), Inches(7.5), Inches(0.5),
             "THU-BDC2026", font=SERIF, size=22, color=NAVY, bold=True)
    add_text(s, Inches(5.2), Inches(1.85), Inches(7.5), Inches(0.4),
             "清华大学大数据竞赛 2026", font=SERIF, size=18, color=COBALT, italic=True)
    add_text(s, Inches(5.2), Inches(2.3), Inches(7.5), Inches(0.4),
             "Hushen-300 Weekly Portfolio Selection",
             font=SERIF, size=18, color=MUTE, italic=True)
    add_line(s, Inches(5.2), Inches(2.85), Inches(9.5), Inches(2.85), COBALT, 1.5)

    nums = [
        ("+1.446%",  "LGB · 82-day rolling mean"),
        ("84.15%",   "Daily win rate"),
        ("t = +7.24","Significance vs zero"),
        ("9 min",    "5090 training time"),
        ("4 / 5",    "Linux 4060 cross-machine"),
        ("5 / 5",    "Windows 4060 cross-machine"),
    ]
    for i, (big, cap) in enumerate(nums):
        row, col = divmod(i, 2)
        x = Inches(5.2 + col * 3.7)
        y = Inches(3.2 + row * 1.2)
        add_text(s, x, y, Inches(3.5), Inches(0.6), big,
                 font=CODE, size=28, color=NAVY, bold=True)
        add_text(s, x, y + Inches(0.65), Inches(3.5), Inches(0.35),
                 cap, font=SERIF, size=18, color=MUTE, italic=True)

    add_text(s, Inches(0.6), H - Inches(1.25), Inches(4), Inches(0.35),
             "2026 · 04 · 24", font=CODE, size=18, color=SKY)
    add_text(s, Inches(0.6), H - Inches(0.85), Inches(4), Inches(0.35),
             "branch  feat/ensemble-v1", font=CODE, size=18, color=LINE)
    add_text(s, Inches(0.6), H - Inches(0.5), Inches(4), Inches(0.35),
             "5090 + 4060 (Linux / Windows)",
             font=SERIF, size=18, color=LINE, italic=True)


def slide_tldr(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 2, total, "Executive Summary", "一句话结论",
               "Three independent protocols, same conclusion")

    add_text(s, MARGIN, Inches(1.95), W - 2 * MARGIN, Inches(1.8),
             "在赛方评分脚本、82 天独立 rolling backtest、跨窗口\n"
             "score_self 三种口径下，均以显著 t-stat 跑赢赛方 Transformer\n"
             "baseline。训练 6.3 h → 9 min；镜像 4 GB → 1.5 GB。",
             font=SERIF, size=22, color=NAVY, spacing=1.5)

    cards = [
        ("01", "82-day rolling",     "+1.446%",  "Baseline  −0.686%",  "+ 213 bp"),
        ("02", "score_self  03-03",  "+3.485%",  "Baseline  −0.944%",  "+ 443 bp"),
        ("03", "score_self  03-06",  "+1.761%",  "Baseline  +1.325%",  "+ 44 bp"),
    ]
    col_w = (W - 2 * MARGIN - Inches(0.5)) / 3
    y0 = Inches(4.7)
    for i, (tag, label, big, sub, delta) in enumerate(cards):
        x = MARGIN + i * (col_w + Inches(0.25))
        add_rect(s, x, y0, col_w, Inches(2.2), WHITE, line=LINE)
        add_text(s, x + Inches(0.3), y0 + Inches(0.18), Inches(0.8), Inches(0.35),
                 tag, font=CODE, size=18, color=SKY, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(0.55), col_w - Inches(0.6), Inches(0.35),
                 label, font=SERIF, size=18, color=MUTE, italic=True)
        add_text(s, x + Inches(0.3), y0 + Inches(0.95), col_w - Inches(0.6), Inches(0.55),
                 big, font=CODE, size=30, color=NAVY, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(1.55), col_w - Inches(0.6), Inches(0.35),
                 sub, font=CODE, size=18, color=INK)
        add_text(s, x + col_w - Inches(1.8), y0 + Inches(1.68), Inches(1.5), Inches(0.35),
                 delta, font=CODE, size=20, color=AMBER, bold=True, align=PP_ALIGN.RIGHT)


def slide_timeline(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 3, total, "Evolution", "方案演进 · Apr 20 → Apr 24",
               "From three-model ensemble to LGB-only mainline")

    phases = [
        ("P0", "Apr 20",     "赛方 Baseline",     "StockTransformer · 60d × 197 feat",
         "−0.69%   win 31%   t = −4.34"),
        ("P1", "Apr 20-22",  "三模型集成",        "LGB + MASTER + StockMixer",
         "+1.00%   win 76%   train 6.3 h"),
        ("PA", "Apr 22",     "ICIR 稳健化尝试",   "KL-shrink + bootstrap",
         "训练 5h+ 未完成 → 触发 Path B"),
        ("PB", "Apr 23 AM",  "Path B 发现",       "仅 LGB + DoubleEnsemble",
         "+1.446%   win 84%   train 9 min"),
        ("P2", "Apr 23 PM",  "MASTER 跨平台非确定",  "Win / Linux diff = 0.17",
         "ICIR 权重反向 → 违反赛规"),
        ("PC", "Apr 23-24",  "LGB-only 主线化",   "deterministic_top_k · 三平台真机",
         "跨机 Top-5 ≥ 4/5   镜像 ~1.5 GB"),
    ]
    y = Inches(1.95)
    row_h = Inches(0.85)
    for tag, date, name, detail, result in phases:
        add_rect(s, MARGIN, y, Inches(0.7), Inches(0.7), NAVY)
        add_text(s, MARGIN, y, Inches(0.7), Inches(0.7), tag,
                 font=CODE, size=18, color=WHITE, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

        add_text(s, MARGIN + Inches(0.9), y + Inches(0.02),
                 Inches(1.5), Inches(0.35), date,
                 font=CODE, size=18, color=MUTE, bold=True)
        add_text(s, MARGIN + Inches(0.9), y + Inches(0.35),
                 Inches(3.8), Inches(0.4), name,
                 font=SERIF, size=20, color=NAVY, bold=True)

        add_text(s, MARGIN + Inches(4.9), y + Inches(0.05),
                 Inches(4.0), Inches(0.35), detail,
                 font=SERIF, size=18, color=INK, italic=True)
        add_text(s, MARGIN + Inches(4.9), y + Inches(0.4),
                 Inches(7.4), Inches(0.35), result,
                 font=CODE, size=18, color=COBALT)
        y = y + row_h

def slide_compare(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 4, total, "Core Metrics", "核心性能对比 · 82-day rolling",
               "LGB-only vs StockTransformer vs HS300 equal-weight")

    headers = ["Metric", "LGB-only", "Baseline", "HS300 EW"]
    rows = [
        ("Mean 5d return",  "+1.446%",  "−0.686%",  "+0.110%"),
        ("Median",          "+0.944%",  "−0.783%",  "—"),
        ("Std",             "1.836%",   "1.360%",   "—"),
        ("Win rate (>0)",   "84.15%",   "31.08%",   "60.92%"),
        ("t-stat vs 0",     "+7.245",   "−4.335",   "—"),
        ("N (days)",        "82",       "74",       "82"),
    ]
    col_w = [Inches(3.0), Inches(3.0), Inches(3.0), Inches(3.1)]
    col_x = [MARGIN]
    for w in col_w[:-1]: col_x.append(col_x[-1] + w)

    y = Inches(2.0)
    for i, h in enumerate(headers):
        add_text(s, col_x[i], y, col_w[i], Inches(0.45), h,
                 font=CODE, size=18, color=MUTE, bold=True)
    add_line(s, MARGIN, y + Inches(0.52), W - MARGIN, y + Inches(0.52), COBALT, 1.2)
    y = y + Inches(0.65)

    for r, row in enumerate(rows):
        for i, cell in enumerate(row):
            if i == 0:
                add_text(s, col_x[i], y, col_w[i], Inches(0.5), cell,
                         font=SERIF, size=20, color=INK, bold=True)
            else:
                color = INK
                if i == 1 and (cell.startswith("+") or cell.startswith("8")):
                    color = WIN
                elif i == 2 and cell.startswith("−"):
                    color = LOSE
                size = 22 if i in (1, 2) else 20
                add_text(s, col_x[i], y, col_w[i], Inches(0.5), cell,
                         font=CODE, size=size, color=color,
                         bold=(i in (1, 2)))
        add_line(s, MARGIN, y + Inches(0.58), W - MARGIN, y + Inches(0.58), LINE, 0.4)
        y = y + Inches(0.65)

    y = y + Inches(0.15)
    add_rect(s, MARGIN, y, W - 2 * MARGIN, Inches(0.9), NAVY)
    add_text(s, MARGIN + Inches(0.35), y + Inches(0.15),
             W - 2 * MARGIN - Inches(0.7), Inches(0.35),
             "t-STAT DELTA", font=CODE, size=18, color=SKY, bold=True)
    add_text(s, MARGIN + Inches(0.35), y + Inches(0.45),
             W - 2 * MARGIN - Inches(0.7), Inches(0.45),
             "LGB +7.24 vs Baseline −4.34 — 差距 11.6 σ",
             font=SERIF, size=20, color=WHITE, bold=True)


def slide_scoreself(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 5, total, "Score_self", "赛方评分脚本单窗口验证",
               "score = Σ weight × (open_T+5 − open_T+1) / open_T+1")

    blocks = [
        ("T = 2026-03-03", "hold  03-04 → 03-10",
         "+3.485%", "−0.944%", "+ 443 bp",
         "主线数据",
         "03-04~03-10 市场偏弱，\n主动选股贡献显著"),
        ("T = 2026-03-06", "hold  03-09 → 03-13",
         "+1.761%", "+1.325%", "+ 44 bp",
         "C4BD 队友数据",
         "03-09~03-13 市场普涨，\nTop-5 重叠 4/5"),
    ]
    col_w = (W - 2 * MARGIN - Inches(0.5)) / 2
    y0 = Inches(2.1)
    for i, (T, window, ours, base, delta, source, note) in enumerate(blocks):
        x = MARGIN + i * (col_w + Inches(0.5))
        add_rect(s, x, y0, col_w, Inches(4.8), WHITE, line=LINE)
        add_rect(s, x, y0, col_w, Inches(0.95), NAVY)
        add_text(s, x + Inches(0.35), y0 + Inches(0.14), col_w - Inches(0.7), Inches(0.4),
                 T, font=CODE, size=20, color=WHITE, bold=True)
        add_text(s, x + Inches(0.35), y0 + Inches(0.55), col_w - Inches(0.7), Inches(0.35),
                 window, font=CODE, size=18, color=SKY)

        add_text(s, x + Inches(0.35), y0 + Inches(1.15), col_w - Inches(0.7), Inches(0.35),
                 "LGB-ONLY", font=CODE, size=18, color=MUTE, bold=True)
        add_text(s, x + Inches(0.35), y0 + Inches(1.5), col_w - Inches(0.7), Inches(0.7),
                 ours, font=CODE, size=36, color=WIN, bold=True)

        add_line(s, x + Inches(0.35), y0 + Inches(2.4),
                 x + col_w - Inches(0.35), y0 + Inches(2.4), LINE)
        add_text(s, x + Inches(0.35), y0 + Inches(2.5), col_w - Inches(0.7), Inches(0.35),
                 "BASELINE", font=CODE, size=18, color=MUTE, bold=True)
        color = LOSE if "−" in base else WIN
        add_text(s, x + Inches(0.35), y0 + Inches(2.85), col_w - Inches(0.7), Inches(0.7),
                 base, font=CODE, size=30, color=color, bold=True)

        add_line(s, x + Inches(0.35), y0 + Inches(3.7),
                 x + col_w - Inches(0.35), y0 + Inches(3.7), LINE)
        add_text(s, x + Inches(0.35), y0 + Inches(3.82), Inches(0.5), Inches(0.4),
                 "Δ", font=CODE, size=20, color=MUTE, bold=True)
        add_text(s, x + Inches(0.9), y0 + Inches(3.8), col_w - Inches(1.25), Inches(0.4),
                 delta, font=CODE, size=22, color=AMBER, bold=True)
        add_text(s, x + Inches(0.35), y0 + Inches(4.3), col_w - Inches(0.7), Inches(0.4),
                 note, font=SERIF, size=18, color=INK, italic=True, spacing=1.3)


def slide_decisions(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 6, total, "Technical Decisions", "关键技术决策",
               "Why drop MASTER, and how cross-machine consistency works")

    col_w = (W - 2 * MARGIN - Inches(0.5)) / 2

    x = MARGIN; y = Inches(1.95)
    add_text(s, x, y, col_w, Inches(0.5),
             "01  为何移除 MASTER / Mixer",
             font=SERIF, size=22, color=NAVY, bold=True)
    add_line(s, x, y + Inches(0.55), x + col_w, y + Inches(0.55), COBALT, 1.2)

    reasons = [
        ("性能", "LGB-only 比三模型集成高 + 45 bp"),
        ("速度", "训练 6.3 h → 9 min (×42 加速)"),
        ("合规", "MASTER attention backward 非确定\n跨平台 score 差 0.17，违反赛规"),
        ("部署", "去 torch，镜像 ~4 GB → ~1.5 GB"),
    ]
    y2 = y + Inches(0.85)
    for label, detail in reasons:
        add_text(s, x, y2, Inches(0.8), Inches(0.5),
                 label, font=CODE, size=18, color=SKY, bold=True)
        add_text(s, x + Inches(0.9), y2, col_w - Inches(0.9), Inches(1.0),
                 detail, font=SERIF, size=18, color=INK, spacing=1.4)
        y2 = y2 + Inches(0.85)

    x2 = MARGIN + col_w + Inches(0.5)
    add_text(s, x2, y, col_w, Inches(0.5),
             "02  Deterministic Top-K",
             font=SERIF, size=22, color=NAVY, bold=True)
    add_line(s, x2, y + Inches(0.55), x2 + col_w, y + Inches(0.55), COBALT, 1.2)
    add_text(s, x2, y + Inches(0.85), col_w, Inches(0.7),
             "LGB 跨平台 score 有 1e-3 级噪声 (OpenMP / BLAS)",
             font=SERIF, size=18, color=INK, italic=True, spacing=1.3)

    code_y = y + Inches(1.6)
    add_rect(s, x2, code_y, col_w, Inches(2.1), NAVY)
    lines = [
        "# deterministic_top_k",
        "q = round(s / 1e-4) * 1e-4",
        "sort by (q DESC,",
        "         stock_id ASC,",
        "         mergesort)",
        "take head(5)",
    ]
    for j, ln in enumerate(lines):
        c = SKY if ln.startswith("#") else WHITE
        add_text(s, x2 + Inches(0.35), code_y + Inches(0.15 + j * 0.3),
                 col_w - Inches(0.7), Inches(0.35),
                 ln, font=CODE, size=18, color=c,
                 bold=ln.startswith("#"))
    add_text(s, x2, code_y + Inches(2.3), col_w, Inches(0.8),
             "同桶内 stock_id 字典序打破同分\n→ Top-5 跨机稳定",
             font=SERIF, size=18, color=INK, spacing=1.4, italic=True)

def slide_constraints(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 7, total, "Engineering Budget", "赛题硬约束余量",
               "All dimensions well within limits")

    data = [
        ("Train time",      "≤ 8 h",      "6.3 h",   "9 min",    "98%"),
        ("Inference",       "≤ 5 min",    "~1 min",  "~1 min",   "—"),
        ("Docker image",    "≤ 10 GB",    "~4 GB",   "~1.5 GB",  "85%"),
        ("4060 GPU mem",    "8 GB",       "tight",   "0 MiB",    "—"),
        ("Reproducibility", "bit-level",  "—",       "Linux ✓",  "—"),
        ("Cross-plat diff", "≤ 1e-4",     "0.17",    "1e-3",     "—"),
    ]
    headers = ["Dimension", "Limit", "Old 3-model", "LGB-only", "Headroom"]
    col_w = [Inches(2.6), Inches(2.2), Inches(2.4), Inches(2.8), Inches(2.1)]
    col_x = [MARGIN]
    for w in col_w[:-1]: col_x.append(col_x[-1] + w)

    y = Inches(2.0)
    for i, h in enumerate(headers):
        add_text(s, col_x[i], y, col_w[i], Inches(0.4), h,
                 font=CODE, size=18, color=MUTE, bold=True)
    add_line(s, MARGIN, y + Inches(0.5), W - MARGIN, y + Inches(0.5), COBALT, 1.2)
    y = y + Inches(0.65)

    for row in data:
        for i, cell in enumerate(row):
            if i == 0:
                add_text(s, col_x[i], y, col_w[i], Inches(0.55), cell,
                         font=SERIF, size=20, color=INK, bold=True)
            elif i == 3:
                add_text(s, col_x[i], y, col_w[i], Inches(0.55), cell,
                         font=CODE, size=22, color=WIN, bold=True)
            elif i == 4:
                color = AMBER if cell != "—" else MUTE
                add_text(s, col_x[i], y, col_w[i], Inches(0.55), cell,
                         font=CODE, size=20, color=color, bold=True)
            else:
                add_text(s, col_x[i], y, col_w[i], Inches(0.55), cell,
                         font=CODE, size=18, color=MUTE)
        add_line(s, MARGIN, y + Inches(0.65), W - MARGIN, y + Inches(0.65), LINE, 0.4)
        y = y + Inches(0.7)


def slide_repro(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 8, total, "Cross-machine Verify", "三平台真机 T1–T8 验证",
               "5090 · Linux 4060 · Windows 4060")

    cards = [
        ("5090", "Ubuntu · anchor",
         [("T1 train",   "9 min"),
          ("T2 predict", "~1 min"),
          ("T3 same",    "100%"),
          ("T5 role",    "anchor"),
          ("Docker",     "pending")]),
        ("Linux 4060", "WSL2 · i7-13700H",
         [("T1 train",   "7.5 min"),
          ("T2 predict", "13 s"),
          ("T3 same",    "100%"),
          ("T5 cross",   "4 / 5 ✓"),
          ("T6 MD5",     "IDENTICAL")]),
        ("Windows 4060", "Windows 11 · venv",
         [("T1 train",   "~9 min"),
          ("T2 predict", "2–3 min"),
          ("T3 same",    "100%"),
          ("T5 cross",   "5 / 5 ✓"),
          ("T6 MD5",     "differ*")]),
    ]
    col_w = (W - 2 * MARGIN - Inches(0.5)) / 3
    y0 = Inches(2.05)
    for i, (name, sub, rows) in enumerate(cards):
        x = MARGIN + i * (col_w + Inches(0.25))
        add_rect(s, x, y0, col_w, Inches(4.4), WHITE, line=LINE)
        add_rect(s, x, y0, col_w, Inches(0.95), NAVY)
        add_text(s, x + Inches(0.35), y0 + Inches(0.14), col_w - Inches(0.7), Inches(0.45),
                 name, font=SERIF, size=22, color=WHITE, bold=True)
        add_text(s, x + Inches(0.35), y0 + Inches(0.56), col_w - Inches(0.7), Inches(0.35),
                 sub, font=SERIF, size=18, color=SKY, italic=True)
        ry = y0 + Inches(1.15)
        for lbl, val in rows:
            add_text(s, x + Inches(0.35), ry, Inches(1.8), Inches(0.4),
                     lbl, font=CODE, size=18, color=MUTE)
            color = INK
            if "✓" in val or "IDENTICAL" in val: color = WIN
            elif "differ" in val: color = AMBER
            elif "pending" in val: color = MUTE
            add_text(s, x + Inches(2.15), ry, col_w - Inches(2.5), Inches(0.4),
                     val, font=CODE, size=20, color=color, bold=True,
                     align=PP_ALIGN.RIGHT)
            add_line(s, x + Inches(0.35), ry + Inches(0.5),
                     x + col_w - Inches(0.35), ry + Inches(0.5), LINE, 0.3)
            ry = ry + Inches(0.63)

    add_text(s, MARGIN, Inches(6.65), W - 2 * MARGIN, Inches(0.35),
             "* Windows MSVC 浮点语义差异；生产链路 (Linux Docker) 已证实 bit-level 一致",
             font=SERIF, size=18, color=MUTE, italic=True)


def slide_risks(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    page_frame(s, 9, total, "Risk / Next", "剩余风险与下一步",
               "Each item tagged with effort / payoff")

    col_w = (W - 2 * MARGIN - Inches(0.4)) / 2

    x = MARGIN; y = Inches(2.0)
    add_text(s, x, y, col_w, Inches(0.5),
             "剩余风险",
             font=SERIF, size=22, color=NAVY, bold=True)
    add_line(s, x, y + Inches(0.55), x + col_w, y + Inches(0.55), COBALT, 1.2)

    risks = [
        ("H", "W1 五一 label holiday-forward-fill 未对齐",  LOSE),
        ("M", "数据仅到 03-13，W1 T 日 = 04-24 (灰区)",     AMBER),
        ("L", "init.sh 仍 import torch",                    AMBER),
        ("L", "回测窗口 82 天偏窄",                         MUTE),
        ("L", "单一 seed 族 {42,2024,7}",                   MUTE),
    ]
    y2 = y + Inches(0.8)
    for lvl, desc, color in risks:
        add_rect(s, x, y2 + Inches(0.06), Inches(0.5), Inches(0.5), color)
        add_text(s, x, y2 + Inches(0.06), Inches(0.5), Inches(0.5), lvl,
                 font=CODE, size=20, color=WHITE, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, x + Inches(0.7), y2 + Inches(0.1), col_w - Inches(0.7), Inches(0.55),
                 desc, font=SERIF, size=18, color=INK)
        y2 = y2 + Inches(0.75)

    x2 = MARGIN + col_w + Inches(0.4)
    add_text(s, x2, y, col_w, Inches(0.5),
             "改进清单 · 投入 / 回报",
             font=SERIF, size=22, color=NAVY, bold=True)
    add_line(s, x2, y + Inches(0.55), x2 + col_w, y + Inches(0.55), COBALT, 1.2)

    items = [
        ("1", "Label holiday-forward-fill",           "S / fix"),
        ("2", "init.sh 去 torch + Docker dry-run",    "S / safe"),
        ("3", "rank-weighted + regime gate",          "S / + 10-30 bp"),
        ("4", "更新数据至 04-24 (待确认)",             "S / freshness"),
        ("5", "LGB 超参集成 × ICIR",                   "S / + 5-15 bp"),
        ("6", "扩回测 250d + multi-seed bootstrap",   "S / narrative"),
    ]
    y3 = y + Inches(0.8)
    for num, desc, eff in items:
        add_text(s, x2, y3, Inches(0.5), Inches(0.45),
                 num, font=CODE, size=22, color=SKY, bold=True)
        add_text(s, x2 + Inches(0.55), y3 + Inches(0.05),
                 col_w - Inches(2.3), Inches(0.4),
                 desc, font=SERIF, size=18, color=INK)
        add_text(s, x2 + col_w - Inches(1.85), y3 + Inches(0.05),
                 Inches(1.85), Inches(0.4),
                 eff, font=CODE, size=18, color=COBALT, bold=True,
                 align=PP_ALIGN.RIGHT)
        add_line(s, x2, y3 + Inches(0.55), x2 + col_w, y3 + Inches(0.55), LINE, 0.3)
        y3 = y3 + Inches(0.65)


def slide_closing(prs, total):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, W, H, NAVY)
    add_line(s, Inches(0.6), Inches(1.2), Inches(3.0), Inches(1.2), SKY, 1.8)

    add_text(s, Inches(0.6), Inches(1.35), Inches(12), Inches(0.45),
             f"{total:02d} / {total:02d}   ·   CLOSING",
             font=CODE, size=18, color=SKY, bold=True)

    add_text(s, Inches(0.6), Inches(2.0), Inches(12), Inches(3.0),
             "主线收敛。\n跑赢 baseline 已完成。\n下一步，赛规正确性。",
             font=SERIF, size=52, color=WHITE, bold=True, spacing=1.15)

    add_line(s, Inches(0.6), Inches(5.45), Inches(12.3), Inches(5.45), COBALT, 0.8)

    items = [
        ("完成", "LGB-only 主线 · Path B · 三平台 T5 ≥ 4/5"),
        ("进行", "Label 对齐 · 组合优化 · Docker 实测"),
        ("待定", "数据更新 · W1 规则 · 7/18 报备"),
    ]
    col_w = (W - Inches(1.2) - Inches(0.8)) / 3
    for i, (lbl, detail) in enumerate(items):
        x = Inches(0.6) + i * (col_w + Inches(0.4))
        add_text(s, x, Inches(5.75), col_w, Inches(0.4),
                 lbl, font=CODE, size=18, color=SKY, bold=True)
        add_text(s, x, Inches(6.15), col_w, Inches(0.9),
                 detail, font=SERIF, size=18, color=CLOUD, spacing=1.4)

    add_text(s, Inches(0.6), H - Inches(0.55), Inches(7), Inches(0.3),
             "feat/ensemble-v1   ·   2026-04-24",
             font=CODE, size=18, color=SKY)


# ============================================================
#                            main
# ============================================================
def main():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    TOTAL = 10
    slide_cover(prs, TOTAL)
    slide_tldr(prs, TOTAL)
    slide_timeline(prs, TOTAL)
    slide_compare(prs, TOTAL)
    slide_scoreself(prs, TOTAL)
    slide_decisions(prs, TOTAL)
    slide_constraints(prs, TOTAL)
    slide_repro(prs, TOTAL)
    slide_risks(prs, TOTAL)
    slide_closing(prs, TOTAL)
    out = Path("docs/reports/2026-04-23-stage-report.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"wrote {out}  ({TOTAL} slides)")


if __name__ == "__main__":
    main()
