"""Tiny Markdown-to-PDF writer for Bilibili analyzer archives.

This intentionally avoids a heavyweight rendering dependency. The output is a
simple text PDF that preserves the report content and supports CJK text through
the common PDF CJK font mapping used by mainstream readers.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


def write_markdown_pdf(markdown: str, output_path: str | Path, *, title: str = "") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _markdown_to_lines(markdown, title=title)
    pages = _paginate(lines, lines_per_page=46)
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog_id = add(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add(b"<< /Type /Pages /Kids [] /Count 0 >>")
    font_id = add(
        b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light "
        b"/Encoding /UniGB-UCS2-H /DescendantFonts [4 0 R] >>"
    )
    cid_font_id = add(
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> "
        b"/FontDescriptor 5 0 R >>"
    )
    descriptor_id = add(
        b"<< /Type /FontDescriptor /FontName /STSong-Light /Flags 4 "
        b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 880 "
        b"/Descent -120 /CapHeight 700 /StemV 80 >>"
    )
    assert catalog_id == 1 and pages_id == 2 and font_id == 3
    assert cid_font_id == 4 and descriptor_id == 5

    page_ids: list[int] = []
    for page_index, page_lines in enumerate(pages, start=1):
        content = _page_stream(page_lines, page_index=page_index, page_count=len(pages))
        stream_id = add(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
        page_id = add(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    objects[1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
        f"/Count {len(page_ids)} >>"
    ).encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(pdf))
    return str(path)


def _markdown_to_lines(markdown: str, *, title: str = "") -> list[str]:
    lines: list[str] = []
    if title:
        lines.extend([title, ""])
    for raw in (markdown or "").splitlines():
        text = raw.rstrip()
        text = re.sub(r"^#{1,6}\s*", "", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        prefix = ""
        if text.startswith(("- ", "* ")):
            prefix, text = "• ", text[2:]
        if not text:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            text,
            width=46,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )
        for index, part in enumerate(wrapped or [""]):
            lines.append((prefix if index == 0 else "  ") + part)
    return lines or [""]


def _paginate(lines: list[str], *, lines_per_page: int) -> list[list[str]]:
    return [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)] or [[""]]


def _page_stream(lines: list[str], *, page_index: int, page_count: int) -> bytes:
    chunks = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
    for line in lines:
        chunks.append(f"<{_utf16_hex(line)}> Tj")
        chunks.append("T*")
    chunks.extend(["/F1 9 Tf", "0 -18 Td", f"<{_utf16_hex(f'Page {page_index}/{page_count}')}> Tj", "ET"])
    return "\n".join(chunks).encode("ascii")


def _utf16_hex(text: str) -> str:
    return text.encode("utf-16-be", errors="replace").hex().upper()

