"""Transcript chunking and LLM viewpoint analysis helpers."""

from __future__ import annotations

import json
from typing import Any

from .bilibili import format_ts

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "owner": {"type": "string"},
                "theme": {"type": "string"},
            },
        },
        "main_thesis": {"type": "string"},
        "viewpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "timestamps": {"type": "array", "items": {"type": "string"}},
                    "type": {"type": "string"},
                },
                "required": ["claim", "evidence", "timestamps"],
            },
        },
        "opposing_or_controversial_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "position": {"type": "string"},
                    "timestamps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["point", "position", "timestamps"],
            },
        },
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "timestamps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fact", "timestamps"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["video", "main_thesis", "viewpoints", "summary"],
}


def transcript_to_text(segments: list[dict[str, Any]], max_chars: int | None = None) -> str:
    lines = []
    total = 0
    for seg in segments:
        line = f"[{format_ts(seg.get('start', 0))}-{format_ts(seg.get('end', 0))}] {seg.get('text', '')}".strip()
        if not line:
            continue
        if max_chars and total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def chunk_segments(segments: list[dict[str, Any]], chunk_seconds: int = 240) -> list[list[dict[str, Any]]]:
    if not segments:
        return []
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chunk_start = float(segments[0].get("start") or 0)
    for seg in segments:
        start = float(seg.get("start") or 0)
        if current and start - chunk_start >= chunk_seconds:
            chunks.append(current)
            current = []
            chunk_start = start
        current.append(seg)
    if current:
        chunks.append(current)
    return chunks


def analyze_transcript(
    llm: Any,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    output_format: str = "markdown",
    chunk_minutes: int = 4,
    max_tokens: int = 3000,
) -> dict[str, Any]:
    chunks = chunk_segments(segments, max(60, int(chunk_minutes) * 60))
    chunk_notes = []
    for index, chunk in enumerate(chunks, start=1):
        result = llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是视频文稿分析助手。基于带时间戳文稿提取本段的核心观点、"
                        "论据、反驳对象、关键事实。忠于原文，不要补充外部信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"视频标题：{metadata.get('title', '')}\n"
                        f"UP主：{metadata.get('owner', '')}\n"
                        f"片段 {index}/{len(chunks)}：\n{transcript_to_text(chunk)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            purpose="bilibili_chunk_summary",
        )
        chunk_notes.append(
            {
                "index": index,
                "start": format_ts(chunk[0].get("start", 0)),
                "end": format_ts(chunk[-1].get("end", 0)),
                "summary": result.text,
            }
        )

    instructions = (
        "你是B站视频观点分析专家，基于下面带时间戳的视频分段笔记完成结构化分析。"
        "必须忠于原文，区分UP主主观观点和客观事实，不脑补额外信息。"
        "每个观点尽量附带时间戳。"
    )
    structured = llm.complete_structured(
        instructions=instructions,
        input=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "metadata": metadata,
                        "chunk_notes": chunk_notes,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        json_schema=ANALYSIS_SCHEMA,
        json_mode=True,
        schema_name="bilibili_video_viewpoint_analysis",
        temperature=0.2,
        max_tokens=max_tokens,
        purpose="bilibili_global_analysis",
    )
    parsed = structured.parsed if isinstance(structured.parsed, dict) else {"summary": structured.text}
    parsed["chunk_notes"] = chunk_notes
    if output_format == "json":
        return parsed
    parsed["markdown"] = render_markdown(parsed)
    return parsed


def render_markdown(data: dict[str, Any]) -> str:
    video = data.get("video") or {}
    lines = [
        "# B站视频观点分析",
        "",
        f"- 标题：{video.get('title', '')}",
        f"- UP主：{video.get('owner', '')}",
        f"- 核心主题：{video.get('theme', '')}",
        "",
        "## 整体主旨",
        data.get("main_thesis", "") or data.get("summary", ""),
        "",
        "## 分维度核心观点",
    ]
    for item in data.get("viewpoints") or []:
        stamps = "、".join(item.get("timestamps") or [])
        lines.append(f"- {item.get('claim', '')}（{stamps}）")
        if item.get("evidence"):
            lines.append(f"  论据：{item.get('evidence')}")
    lines.extend(["", "## 争议/正反观点"])
    for item in data.get("opposing_or_controversial_points") or []:
        stamps = "、".join(item.get("timestamps") or [])
        lines.append(f"- {item.get('point', '')}：{item.get('position', '')}（{stamps}）")
    lines.extend(["", "## 关键事实/案例/结论"])
    for item in data.get("key_facts") or []:
        stamps = "、".join(item.get("timestamps") or [])
        lines.append(f"- {item.get('fact', '')}（{stamps}）")
    lines.extend(["", "## 简要总结", data.get("summary", "")])
    return "\n".join(lines).strip()
