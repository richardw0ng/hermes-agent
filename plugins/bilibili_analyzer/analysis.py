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
        "audience_comments": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "high_information_feedback": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "questions_or_requests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "disagreements_or_corrections": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["video", "main_thesis", "viewpoints", "summary"],
}

MARKET_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "market_sentiment": {
            "type": "object",
            "properties": {
                "overall": {"type": "string"},
                "confidence": {"type": "string"},
                "bullish_signals": {"type": "array", "items": {"type": "string"}},
                "bearish_signals": {"type": "array", "items": {"type": "string"}},
                "neutral_or_wait_and_see_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "consensus_views": {"type": "array", "items": {"type": "string"}},
        "divergent_views": {"type": "array", "items": {"type": "string"}},
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "stance": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "commentary_signals": {
            "type": "object",
            "properties": {
                "high_information_consensus": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "pushback_or_corrections": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "new_clues": {"type": "array", "items": {"type": "string"}},
                "filtered_noise_summary": {"type": "string"},
            },
        },
        "external_market_sources": {
            "type": "object",
            "properties": {
                "summary": {"type": "object"},
                "high_information_items": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "errors": {"type": "array", "items": {"type": "object"}},
            },
        },
        "watchlist": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "source_coverage": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["executive_summary", "market_sentiment", "source_coverage"],
}


def transcript_to_text(
    segments: list[dict[str, Any]], max_chars: int | None = None
) -> str:
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


def chunk_segments(
    segments: list[dict[str, Any]], chunk_seconds: int = 240
) -> list[list[dict[str, Any]]]:
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
    comments: list[dict[str, Any]] | None = None,
    output_format: str = "markdown",
    chunk_minutes: int = 4,
    max_tokens: int = 3000,
    llm_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    chunks = chunk_segments(segments, max(60, int(chunk_minutes) * 60))
    chunk_notes = []
    for index, chunk in enumerate(chunks, start=1):
        result = llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是视频文稿分析助手。基于带时间戳的文稿提取本段的"
                        "核心观点、论据、反驳对象和关键事实。忠于原文，不要补充外部信息。"
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
            timeout=llm_timeout_seconds,
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
        "如果提供了过滤后的评论，只总结高信息量反馈、问题、分歧和纠正；"
        "忽略纯情绪、重复和低信息量表态。"
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
                        "filtered_comments": comments or [],
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
        timeout=llm_timeout_seconds,
        purpose="bilibili_global_analysis",
    )
    parsed = (
        structured.parsed
        if isinstance(structured.parsed, dict)
        else {"summary": structured.text}
    )
    parsed["chunk_notes"] = chunk_notes
    parsed["filtered_comments"] = comments or []
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

    audience = data.get("audience_comments") or {}
    comments = data.get("filtered_comments") or []
    if audience or comments:
        lines.extend(["", "## 评论区高信息量反馈"])
        if audience.get("summary"):
            lines.append(str(audience.get("summary")))
        for label, key in (
            ("高信息量反馈", "high_information_feedback"),
            ("问题/请求", "questions_or_requests"),
            ("分歧/纠正", "disagreements_or_corrections"),
        ):
            items = audience.get(key) or []
            if items:
                lines.extend(["", f"### {label}"])
                for item in items:
                    lines.append(f"- {item}")
        if comments:
            lines.extend(["", "### 代表性评论"])
            for item in comments[:10]:
                score = item.get("information_score", "")
                like = item.get("like", 0)
                user = item.get("user", "")
                lines.append(
                    f"- {item.get('message', '')} "
                    f"(user={user}, like={like}, score={score})"
                )
    return "\n".join(lines).strip()


def analyze_market_batch(
    llm: Any,
    *,
    group: dict[str, Any],
    records: list[dict[str, Any]],
    market_sources: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
    skipped: list[dict[str, Any]] | None = None,
    llm_timeout_seconds: float = 90.0,
    max_tokens: int = 6000,
) -> dict[str, Any]:
    """Synthesize one market report from video transcripts and comments.

    The batch report intentionally consumes partial records too. A per-video
    analysis timeout should not erase the transcript/comment evidence already
    collected for the batch.
    """
    compact_records = [
        _compact_market_record(record)
        for record in records
        if record.get("transcript") or record.get("analysis") or record.get("comments")
    ]
    if not compact_records:
        report = _heuristic_market_report(group, [], failures or [], skipped or [], market_sources)
        report["markdown"] = render_market_report_markdown(report)
        return report

    instructions = (
        "你是市场观点研究助理。请基于一组B站财经/投资视频的字幕、UP主观点和高信息量评论，"
        "生成一份整体市场情绪分析报告。要求：\n"
        "1. 分别采集各方观点，但最终输出批次级综合判断。\n"
        "2. 明确区分UP主观点、评论区补充、评论区反驳/修正。\n"
        "3. 只采纳有信息增量的评论；纯情绪、刷屏、站队、无理由喊涨喊跌要过滤。\n"
        "4. 不提供投资建议，不给买卖指令；只输出市场情绪、分歧、风险、待验证线索。\n"
        "5. 所有结论都要能追溯到source_coverage里的UP主或评论证据。"
    )
    try:
        structured = llm.complete_structured(
            instructions=instructions,
            input=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "group": group,
                            "records": compact_records,
                            "external_market_sources": _compact_external_market_sources(market_sources or {}),
                            "failures": failures or [],
                            "skipped": skipped or [],
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            json_schema=MARKET_REPORT_SCHEMA,
            json_mode=True,
            schema_name="bilibili_batch_market_sentiment_report",
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=llm_timeout_seconds,
            purpose="bilibili_batch_market_report",
        )
        report = structured.parsed if isinstance(structured.parsed, dict) else {}
        if not report:
            report = _heuristic_market_report(group, compact_records, failures or [], skipped or [], market_sources)
            report["llm_warning"] = "Batch report LLM returned non-JSON output."
    except Exception as exc:
        report = _heuristic_market_report(group, compact_records, failures or [], skipped or [], market_sources)
        report["llm_warning"] = f"Batch report LLM failed: {exc}"

    report["group"] = group
    report["video_count"] = len(records)
    report["partial_video_count"] = sum(1 for record in records if not record.get("success"))
    if market_sources:
        report["external_market_sources"] = _compact_external_market_sources(market_sources)
    report["comment_count"] = sum(
        len(((record.get("comments") or {}).get("comments") or []))
        for record in records
    )
    report["markdown"] = render_market_report_markdown(report)
    return report


def _compact_market_record(record: dict[str, Any]) -> dict[str, Any]:
    transcript = record.get("transcript") or {}
    metadata = transcript.get("metadata") or {}
    comments_payload = record.get("comments") or {}
    comments = comments_payload.get("comments") or []
    analysis = record.get("analysis") or {}
    video = record.get("video") or {}
    up = record.get("up") or {}
    return {
        "success": bool(record.get("success")),
        "error": record.get("error") or "",
        "up": up.get("name") or video.get("owner_name") or metadata.get("owner") or "",
        "title": video.get("title") or metadata.get("title") or "",
        "bvid": video.get("bvid") or metadata.get("bvid") or "",
        "duration": video.get("duration") or metadata.get("duration") or 0,
        "analysis": {
            "main_thesis": analysis.get("main_thesis") or "",
            "summary": analysis.get("summary") or "",
            "viewpoints": (analysis.get("viewpoints") or [])[:6],
            "risk_or_controversy": (
                analysis.get("opposing_or_controversial_points") or []
            )[:4],
            "key_facts": (analysis.get("key_facts") or [])[:5],
        },
        "transcript_excerpt": transcript_to_text(
            transcript.get("segments") or [],
            max_chars=3600,
        ),
        "comment_metadata": comments_payload.get("metadata") or {},
        "high_information_comments": _compact_market_comments(comments),
    }


def _compact_market_comments(comments: list[dict[str, Any]], limit: int = 16) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for comment in comments[:limit]:
        entry = {
            "user": comment.get("user") or "",
            "message": str(comment.get("message") or "")[:360],
            "like": int(comment.get("like") or 0),
            "reply_count": int(comment.get("reply_count") or 0),
            "information_score": comment.get("information_score", 0),
            "filter_reasons": comment.get("filter_reasons") or [],
        }
        replies = []
        for reply in comment.get("replies") or []:
            replies.append(
                {
                    "user": reply.get("user") or "",
                    "message": str(reply.get("message") or "")[:240],
                    "like": int(reply.get("like") or 0),
                }
            )
            if len(replies) >= 3:
                break
        if replies:
            entry["replies"] = replies
        compacted.append(entry)
    return compacted


def _compact_external_market_sources(payload: dict[str, Any], limit: int = 24) -> dict[str, Any]:
    items = []
    for item in payload.get("items") or []:
        items.append(
            {
                "source": item.get("source") or "",
                "symbol": item.get("symbol") or "",
                "title": str(item.get("title") or "")[:160],
                "text": str(item.get("text") or "")[:360],
                "sentiment": item.get("sentiment") or "neutral",
                "information_score": item.get("information_score", 0),
                "url": item.get("url") or "",
            }
        )
        if len(items) >= limit:
            break
    return {
        "summary": payload.get("summary") or {},
        "keyword": payload.get("keyword") or "",
        "symbols": payload.get("symbols") or [],
        "high_information_items": items,
        "errors": payload.get("errors") or [],
    }


def _heuristic_market_report(
    group: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    market_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bullish_terms = ("反弹", "上涨", "机会", "看多", "修复", "突破", "景气", "增持", "低估")
    bearish_terms = ("风险", "下跌", "回调", "亏", "减仓", "高估", "压力", "分歧", "谨慎")
    bullish: list[str] = []
    bearish: list[str] = []
    neutral: list[str] = []
    comment_clues: list[str] = []
    coverage: list[str] = []
    for record in records:
        title = record.get("title") or ""
        up = record.get("up") or ""
        text = " ".join(
            str(part or "")
            for part in (
                title,
                (record.get("analysis") or {}).get("main_thesis"),
                (record.get("analysis") or {}).get("summary"),
                record.get("transcript_excerpt"),
            )
        )
        source = f"{up}: {title}".strip(": ")
        if source:
            coverage.append(source)
        line = source or (record.get("bvid") or "unknown video")
        if any(term in text for term in bullish_terms):
            bullish.append(line)
        elif any(term in text for term in bearish_terms):
            bearish.append(line)
        else:
            neutral.append(line)
        for comment in record.get("high_information_comments") or []:
            msg = str(comment.get("message") or "").strip()
            if msg:
                comment_clues.append(f"{line} / 评论: {msg[:160]}")
    overall = "neutral"
    if len(bullish) > len(bearish) + len(neutral) / 2:
        overall = "bullish"
    elif len(bearish) > len(bullish) + len(neutral) / 2:
        overall = "bearish"
    report = {
        "group": group,
        "executive_summary": (
            "已基于可用字幕、单条分析结果和高信息量评论生成降级版市场情绪报告。"
            "由于批次级LLM未完成，以下结论偏向证据整理而非深度综合。"
        ),
        "market_sentiment": {
            "overall": overall,
            "confidence": "low" if failures else "medium",
            "bullish_signals": bullish[:8],
            "bearish_signals": bearish[:8],
            "neutral_or_wait_and_see_signals": neutral[:8],
        },
        "consensus_views": bullish[:5] if bullish else neutral[:5],
        "divergent_views": bearish[:5],
        "themes": [],
        "commentary_signals": {
            "high_information_consensus": comment_clues[:8],
            "pushback_or_corrections": [],
            "new_clues": comment_clues[8:16],
            "filtered_noise_summary": "短表态、重复、无理由情绪输出未进入报告。",
        },
        "watchlist": [],
        "risk_flags": [str(item.get("error") or item) for item in failures[:5]],
        "source_coverage": coverage,
        "skipped": skipped,
    }
    if market_sources:
        report["external_market_sources"] = _compact_external_market_sources(market_sources)
    return report


def render_market_report_markdown(data: dict[str, Any]) -> str:
    sentiment = data.get("market_sentiment") or {}
    comments = data.get("commentary_signals") or {}
    lines = [
        "# Bilibili Market Sentiment Report",
        "",
        f"- Group: {(data.get('group') or {}).get('name', '')}",
        f"- Videos: {data.get('video_count', 0)}",
        f"- Partial videos: {data.get('partial_video_count', 0)}",
        f"- High-information comments: {data.get('comment_count', 0)}",
        f"- Overall sentiment: {sentiment.get('overall', '')}",
        f"- Confidence: {sentiment.get('confidence', '')}",
        "",
        "## Executive Summary",
        "",
        str(data.get("executive_summary") or ""),
    ]
    if data.get("llm_warning"):
        lines.extend(["", "## Generation Warning", "", str(data.get("llm_warning"))])
    for title, key in (
        ("Bullish Signals", "bullish_signals"),
        ("Bearish Signals", "bearish_signals"),
        ("Neutral / Wait-and-see Signals", "neutral_or_wait_and_see_signals"),
    ):
        items = sentiment.get(key) or []
        if items:
            lines.extend(["", f"## {title}", ""])
            lines.extend(f"- {item}" for item in items)
    for title, key in (
        ("Consensus Views", "consensus_views"),
        ("Divergent Views", "divergent_views"),
        ("Risk Flags", "risk_flags"),
        ("Watchlist", "watchlist"),
    ):
        items = data.get(key) or []
        if items:
            lines.extend(["", f"## {title}", ""])
            lines.extend(f"- {item}" for item in items)
    themes = data.get("themes") or []
    if themes:
        lines.extend(["", "## Themes", ""])
        for item in themes:
            lines.append(f"### {item.get('theme', '')} ({item.get('stance', '')})")
            for evidence in item.get("evidence") or []:
                lines.append(f"- {evidence}")
            sources = item.get("sources") or []
            if sources:
                lines.append(f"Sources: {', '.join(str(src) for src in sources)}")
    lines.extend(["", "## Comment Signals", ""])
    for title, key in (
        ("High-information consensus", "high_information_consensus"),
        ("Pushback / corrections", "pushback_or_corrections"),
        ("New clues", "new_clues"),
    ):
        items = comments.get(key) or []
        if items:
            lines.extend([f"### {title}", ""])
            lines.extend(f"- {item}" for item in items)
            lines.append("")
    if comments.get("filtered_noise_summary"):
        lines.extend(["### Filtered Noise", "", str(comments.get("filtered_noise_summary"))])
    external = data.get("external_market_sources") or {}
    external_items = external.get("high_information_items") or []
    if external or external_items:
        lines.extend(["", "## External Market Sources", ""])
        summary = external.get("summary") or {}
        if summary:
            lines.extend(
                [
                    f"- Overall: {summary.get('overall', '')}",
                    f"- Items: {summary.get('item_count', 0)}",
                    f"- Bullish/Bearish/Neutral: {summary.get('bullish_count', 0)}/{summary.get('bearish_count', 0)}/{summary.get('neutral_count', 0)}",
                ]
            )
        for item in external_items:
            source = item.get("source") or "source"
            sentiment_label = item.get("sentiment") or "neutral"
            title = item.get("title") or item.get("text") or ""
            url = item.get("url") or ""
            suffix = f" ({url})" if url else ""
            lines.append(f"- [{source}/{sentiment_label}] {title}{suffix}")
        errors = external.get("errors") or []
        if errors:
            lines.extend(["", "### External Source Warnings", ""])
            lines.extend(f"- {item.get('source', 'source')}: {item.get('error', '')}" for item in errors[:5])
    coverage = data.get("source_coverage") or []
    if coverage:
        lines.extend(["", "## Source Coverage", ""])
        lines.extend(f"- {item}" for item in coverage)
    lines.extend(
        [
            "",
            "> This report is for information synthesis only and is not investment advice.",
        ]
    )
    return "\n".join(lines).strip()
