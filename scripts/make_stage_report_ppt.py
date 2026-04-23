"""阶段性报告 PPT 生成器。

设计语言：editorial + 瑞士现代主义 + 终端暗色调。
- 背景 #0F1012（深炭黑）
- 正文 #E8E6E0（米白）
- 次文 #8A8880（灰）
- signal #D4A017（金）标"发现 / 关键数字"
- 规则 #2A2B2F（深灰分隔线）
"""
from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree
