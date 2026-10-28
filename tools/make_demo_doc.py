"""Rebuild the two sample PDFs: the Chinese demo document and the English fixture.

    python tools/make_demo_doc.py     # needs reportlab; the service does not

The content is invented and organisation-neutral on purpose: the repository is a
generic PDF assistant, so no real policy or third-party material ships with it.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "demo-policy.pdf"
W, H = A4
LEFT, RIGHT = 64, W - 64
TOP, BOTTOM = H - 70, 60
FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

TITLE = "采购与合同管理办法（示例）"
SUBTITLE = "本文件为开源项目演示而虚构，不对应任何真实组织、制度或供应商"

CHAPTERS = [
    (
        "第一章 总则",
        [
            ("第一条", "目的与适用范围", "本办法适用于示例组织总部及各分支机构开展的全部采购活动，包括货物采购、工程采购和服务采购。"),
            ("第二条", "定义", "本办法所称采购，是指以合同方式有偿取得货物、工程和服务的行为；所称供应商，是指通过准入评审并签订框架协议的法人或其他组织。"),
            ("第三条", "职责分工", "采购部门负责供应商准入与询比价，财务部门负责预算控制与付款，使用部门负责需求提出与验收。"),
            ("第四条", "基本原则", "采购活动应当遵循公开、公平、可追溯的原则；同一采购事项不得拆分以规避审批权限。"),
        ],
    ),
    (
        "第二章 供应商准入与评估",
        [
            ("第五条", "准入材料", "供应商申请准入应当提交营业执照、近两年审计报告、同类项目业绩清单、无重大违法记录承诺书和廉洁承诺书。"),
            ("第六条", "分级管理", "供应商按年度采购金额与合作年限分为 A、B、C 三级；A 级供应商应当保持不少于三家备选。"),
            ("第七条", "复评周期", "A 级供应商每年复评一次，B 级供应商每两年复评一次，C 级供应商每三年复评一次；复评不合格的作降级处理。"),
            ("第八条", "暂停合作", "出现围标串标、提供虚假材料、连续两次交付延期超过十五日等情形的，暂停合作资格不少于六个月。"),
            ("第九条", "退出机制", "连续两次复评不合格或者被列入失信名单的供应商，终止框架协议并且三年内不再准入。"),
        ],
    ),
    (
        "第三章 采购申请与审批",
        [
            ("第十条", "申请内容", "使用部门在系统中提交采购申请，注明用途、预算科目、预计金额和交付时间。"),
            ("第十一条", "预算约束", "无预算或者超预算的采购申请，应当先履行预算调整程序；未完成调整的不得进入审批环节。"),
            ("第十二条", "单一来源", "只能从唯一供应商处采购、发生不可预见的紧急情况、需要保持技术一致性的，可以采用单一来源方式，但应当附情况说明并经分管负责人批准。"),
            ("第十三条", "紧急采购", "因生产安全事故、重大设备故障等紧急情况可以先行采购，但应当在五个工作日内补办审批手续。"),
            ("第十四条", "拆分禁止", "同一采购事项在一个自然年度内多次采购，累计金额达到下一档权限的，按累计金额确定审批层级。"),
        ],
    ),
    (
        "第四章 发票校验与付款",
        [
            ("第十六条", "三单匹配", "发票到达后五个工作日内完成三单匹配，即采购申请、验收单与发票三者一致；不一致的退回供应商重开。"),
            ("第十七条", "付款账期", "验收合格并完成发票校验后三十日内付款；框架协议另有约定的从其约定，但最长不超过六十日。"),
            ("第十八条", "质保金", "合同金额一百万元以上的，预留百分之五作为质保金；质保期满且无质量问题的，三十日内退还。"),
            ("第十九条", "预付款", "原则上不支付预付款；确需支付的，预付比例不得超过合同金额的百分之二十，并要求供应商提供等额保函。"),
        ],
    ),
    (
        "第五章 档案与监督",
        [
            ("第二十条", "档案保存", "采购档案自合同履行完毕之日起保存十年，包含申请单、审批记录、比价记录、合同、验收单和付款凭证。"),
            ("第二十一条", "采购台账", "采购部门按月编制采购台账，报财务部门核对预算执行情况。"),
            ("第二十二条", "审计抽查", "内部审计部门每年对采购事项抽查不少于百分之二十，重点检查拆分采购与单一来源情形。"),
            ("第二十三条", "违规处理", "违反本办法的责任人视情节给予通报、调离岗位或者移送处理；造成损失的依法追偿。"),
        ],
    ),
    (
        "附则",
        [
            ("第二十四条", "争议解决", "因采购合同发生争议的，应当先行协商；协商不成的，提交合同约定的仲裁机构仲裁。"),
            ("第二十五条", "生效日期", "本办法自 2026 年 3 月 1 日起施行，由采购部门负责解释。"),
        ],
    ),
]

FLOW = ["提交采购申请", "预算校验", "分级审批", "询比价 / 招标", "合同签订", "到货验收", "发票校验", "资金支付"]

APPROVAL = [
    ["金额区间", "审批人", "是否集体决策", "附加要求"],
    ["50 万元以下", "部门负责人", "否", "至少两家报价"],
    ["50 万元至 200 万元", "分管负责人", "否", "至少三家报价"],
    ["200 万元至 500 万元", "总经理", "是（采购决策会）", "需法务出具意见"],
    ["500 万元以上", "董事会", "是（董事会决议）", "需第三方评估报告"],
]

CONTENTS = [
    ("第一章 总则", 2),
    ("第二章 供应商准入与评估", 3),
    ("第三章 采购申请与审批（含审批权限矩阵）", 4),
    ("采购流程示意", 5),
    ("第四章 发票校验与付款", 6),
    ("第五章 档案与监督", 7),
    ("附则", 8),
]


def wrap(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        current += char
        if len(current) >= width:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


class Doc:
    def __init__(self, path: Path):
        self.canvas = canvas.Canvas(str(path), pagesize=A4)
        self.canvas.setTitle(TITLE)
        self.y = TOP

    def new_page(self):
        self.canvas.showPage()
        self.text = None
        self.y = TOP

    def heading(self, text: str, size: int = 16):
        if self.y < BOTTOM + 60:
            self.new_page()
        self.canvas.setFont(FONT, size)
        self.canvas.drawString(LEFT, self.y, text)
        self.y -= size + 14
        self.canvas.setLineWidth(0.6)
        self.canvas.line(LEFT, self.y + 8, RIGHT, self.y + 8)

    def body(self, text: str, size: int = 11, indent: int = 0, leading: int = 19):
        self.canvas.setFont(FONT, size)
        for line in wrap(text, 38 - indent // 12):
            if self.y < BOTTOM:
                self.new_page()
                self.canvas.setFont(FONT, size)
            self.canvas.drawString(LEFT + indent, self.y, line)
            self.y -= leading

    def space(self, amount: int = 8):
        self.y -= amount

    def table(self, rows: list[list[str]], col_widths: list[float], row_height: float = 26):
        x = LEFT
        top = self.y
        self.canvas.setLineWidth(0.8)
        for r, row in enumerate(rows):
            y = top - r * row_height
            self.canvas.rect(LEFT, y - row_height, sum(col_widths), row_height, stroke=1, fill=0)
            cx = x
            for c, cell in enumerate(row):
                self.canvas.setFont(FONT, 9)
                self.canvas.drawString(cx + 6, y - row_height + 8, cell)
                if c:
                    self.canvas.line(cx, y - row_height, cx, y)
                cx += col_widths[c]
        self.y = top - len(rows) * row_height - 12

    def flow(self, labels: list[str]):
        """Two rows of boxes joined by arrows -- vector art, no raster image."""
        box_w, box_h, gap = 96, 46, 24
        self.canvas.setLineWidth(1)
        positions = []
        for row in range(2):
            count = 4
            total = count * box_w + (count - 1) * gap
            start_x = (W - total) / 2
            y = self.y - row * (box_h + 56)
            for col in range(count):
                x = start_x + col * (box_w + gap)
                positions.append((x, y, box_w, box_h))
        control = {1, 2, 6}  # 预算校验 / 分级审批 / 发票校验
        for i, (x, y, w, h) in enumerate(positions[: len(labels)]):
            self.canvas.rect(x, y - h, w, h, stroke=1, fill=0)
            self.canvas.rect(x + 3, y - h + 3, w - 6, h - 6, stroke=1, fill=0)
            self.canvas.setFont(FONT, 9)
            self.canvas.drawCentredString(x + w / 2, y - h / 2 - 3, labels[i])
            if i in control:
                self.canvas.circle(x + w - 8, y - 8, 3, stroke=1, fill=1)
            if i % 4 < 3 and i + 1 < len(labels):
                self.canvas.line(x + w, y - h / 2, x + w + gap, y - h / 2)
                self.canvas.line(x + w + gap - 5, y - h / 2 + 3, x + w + gap, y - h / 2)
                self.canvas.line(x + w + gap - 5, y - h / 2 - 3, x + w + gap, y - h / 2)
            elif i + 1 < len(labels):
                nxt = positions[i + 1]
                self.canvas.line(x + w / 2, y - h, x + w / 2, y - h - 26)
                self.canvas.line(nxt[0] + nxt[2] / 2, nxt[1], nxt[0] + nxt[2] / 2, y - h - 26)
                self.canvas.line(x + w / 2, y - h - 26, nxt[0] + nxt[2] / 2, y - h - 26)
        self.y = self.y - 2 * (box_h + 56) - 10

    def save(self):
        self.canvas.save()


def build(path: Path = OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(path)
    c = doc.canvas

    c.setFont(FONT, 24)
    c.drawString(LEFT, H - 190, TITLE)
    c.setFont(FONT, 12)
    c.drawString(LEFT, H - 226, SUBTITLE)
    c.setFont(FONT, 10)
    c.drawString(LEFT, H - 252, "版本 v1.0　发布 2026-03-01　文档编号 DEMO-PROC-001")
    doc.y = H - 300
    doc.heading("目录", 14)
    c.setFont(FONT, 11)
    for name, page in CONTENTS:
        c.drawString(LEFT + 8, doc.y, name)
        c.drawRightString(RIGHT - 8, doc.y, str(page))
        doc.y -= 22
    doc.y -= 24
    doc.body("说明：全文为虚构示例，仅用于验证文档解析、切分与检索效果，不含任何真实机构的信息。", size=9)

    for chapter, clauses in CHAPTERS:
        doc.new_page()
        doc.heading(chapter)
        for no, name, text in clauses:
            doc.body(f"{no}　{name}", size=11)
            doc.body(text)
            doc.space(4)
        if chapter.startswith("第三章"):
            doc.heading("审批权限矩阵", 13)
            doc.table(APPROVAL, [120, 96, 118, 133])
            doc.body("第十五条　流程环节：采购流程依次为申请、预算校验、分级审批、询比价或招标、合同签订、到货验收、发票校验与资金支付，共八个环节。")
        if chapter.startswith("第三章"):
            doc.new_page()
            doc.heading("采购流程示意")
            doc.body("下图自左向右、自上而下表示一笔采购从申请到付款的完整路径，任一环节未通过则退回申请环节重新提交。", size=10)
            doc.space(6)
            doc.flow(FLOW)
            doc.body("图示说明：八个环节中有三个关键控制点——预算校验、分级审批与发票校验，任一控制点未通过，流程回到起始环节。", size=10)

    doc.save()
    return path


def build_fixture(path: Path | None = None) -> Path:
    """Rebuild the 3-page English fixture the unit tests run against.

    Page 1 is prose, page 2 is a pipe-separated table, page 3 is blank -- the
    three cases `load_pdf` has to tell apart. Text is deliberately dull.
    """
    path = path or OUT.with_name("sample-3pages.pdf")
    c = canvas.Canvas(str(path), pagesize=(595, 842))
    c.setFont("Helvetica", 10)
    y = 780
    for i in range(1, 26):
        c.drawString(56, y, f"Section 1.{i}: This clause describes the approval flow for supplier pricing.")
        y -= 24
    c.showPage()
    c.setFont("Helvetica", 10)
    y = 780
    c.drawString(56, y, "Item | Category | Approval Required | Owner")
    y -= 24
    for i in range(1, 25):
        c.drawString(56, y, f"Row{i:02d} | Direct Materials | Committee | Operations | Checked")
        y -= 24
    c.showPage()
    c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    print(build())
    print(build_fixture())
