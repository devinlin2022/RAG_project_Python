"""Table rail tests: pipe text, space-aligned text, and the refusal to invent tables."""

from rag.ingest.table import split_markdown_table, tables_to_markdown, text_tables

PIPE_PAGE = """Item | Category | Approval Required | Owner
Row01 | Direct Materials | CRMC | Capital Management | Checked
Row02 | Direct Materials | CRMC | Capital Management | Checked
Row03 | Indirect | IT | IT Manager | Pending
"""

SPACE_PAGE = """项目        预算        负责人      状态
服务器      120000      张工        已批准
带宽        35000       李工        待审核
实施服务    80000       王工        已批准
"""

PROSE = """第一条 合同目的
甲方委托乙方提供智能文档问答系统的开发与部署服务，乙方接受委托并按本合同约定完成交付。
第二条 服务范围
乙方应完成文档解析与切分模块的开发和测试，并提供不少于两次的现场培训。
"""


class TestPipeTables:
    def test_header_detected_and_padded(self):
        blocks = text_tables(PIPE_PAGE, "[表 · 第 1 页]")
        assert len(blocks) == 1
        lines = blocks[0].splitlines()
        assert lines[1].startswith("| Item | Category | Approval Required | Owner")
        assert lines[1].endswith("|  |"), "短表头应补齐到最大列数"

    def test_rows_become_markdown_rows(self):
        block = text_tables(PIPE_PAGE, "[表]")[0]
        assert block.count("\n| Row") == 3
        assert "| Row03 | Indirect | IT | IT Manager | Pending |" in block


class TestSpaceTables:
    def test_aligned_columns_parsed(self):
        blocks = text_tables(SPACE_PAGE, "[表]")
        assert len(blocks) == 1
        assert "| 项目 | 预算 | 负责人 | 状态 |" in blocks[0]
        assert "| 服务器 | 120000 | 张工 | 已批准 |" in blocks[0]


class TestRefusals:
    def test_prose_is_not_a_table(self):
        assert text_tables(PROSE, "[表]") == []

    def test_two_column_lines_are_not_a_table(self):
        assert text_tables("甲方：某某公司\n乙方：某某公司\n", "[表]") == []


class TestRuledTables:
    def test_pdf_without_ruled_tables_falls_back_to_text(self, tmp_path):
        import subprocess
        import sys

        pdf = tmp_path / "pipe.pdf"
        script = f"""
import pathlib
content = '''BT /F1 11 Tf 40 800 Td (Item | Category | Owner) Tj 0 -18 Td (Row01 | Hardware | Zhang) Tj 0 -18 Td (Row02 | Service | Li) Tj ET'''
objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    b"<< /Length " + str(len(content)).encode() + b" >>\\nstream\\n" + content.encode() + b"\\nendstream",
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
]
out = bytearray(b"%PDF-1.4\\n")
offsets = []
for i, body in enumerate(objects, start=1):
    offsets.append(len(out))
    out += str(i).encode() + b" 0 obj\\n" + body + b"\\nendobj\\n"
xref = len(out)
out += b"xref\\n0 " + str(len(objects) + 1).encode() + b"\\n0000000000 65535 f \\n"
for off in offsets:
    out += ("%010d 00000 n \\n" % off).encode()
out += b"trailer\\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\\nstartxref\\n" + str(xref).encode() + b"\\n%%EOF\\n"
pathlib.Path({str(pdf)!r}).write_bytes(bytes(out))
"""
        subprocess.run([sys.executable, "-c", script], check=True)
        blocks = tables_to_markdown(str(pdf), 1, page_text=PIPE_PAGE)
        assert blocks, "有文本表格时必须回退到文本解析"
        assert "Item" in blocks[0]


class TestSplit:
    def test_header_repeats_on_every_fragment(self):
        rows = "\n".join(f"| Row{i:02d} | 内容 {i} |" for i in range(40))
        block = f"[表 · 第 1 页]\n| 编号 | 内容 |\n| --- | --- |\n{rows}"
        fragments = split_markdown_table(block, 200)
        assert len(fragments) > 1
        assert all("| 编号 | 内容 |" in f for f in fragments)
        assert all(f.startswith("[表 · 第 1 页]") for f in fragments)

    def test_short_table_is_not_split(self):
        block = "[表]\n| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert split_markdown_table(block, 500) == [block]
