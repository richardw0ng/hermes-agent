"""Hermes tools for external market sentiment sources."""

from __future__ import annotations

import json
from typing import Any

from .sources import fetch_market_sentiment_sources, research_market_sources


RESEARCH_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "source_coverage": {"type": "array", "items": {"type": "string"}},
        "consensus": {"type": "array", "items": {"type": "string"}},
        "divergences": {"type": "array", "items": {"type": "string"}},
        "high_information_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
        },
        "noise_or_low_confidence": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["executive_summary", "source_coverage"],
}


MARKET_FETCH_SENTIMENT_SOURCES_SCHEMA = {
    "name": "market_fetch_sentiment_sources",
    "description": (
        "Fetch finance sentiment from Eastmoney Guba, Xueqiu, and CLS, "
        "then filter low-information posts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Optional keyword for external market-source search/filtering.",
                "default": "",
            },
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Stock symbols or A-share codes, e.g. 600519, SH600519, NVDA.",
                "default": [],
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Subset of eastmoney_guba, xueqiu, cls. Defaults to all three. "
                    "Xueqiu supports XUEQIU_FETCH_MODE=auto/http/browser and optional "
                    "XUEQIU_STORAGE_STATE for Playwright-backed WAF fallback."
                ),
                "default": [],
            },
            "max_items_per_source": {
                "type": "integer",
                "description": "Maximum high-information items returned per source. Defaults to 20.",
                "default": 20,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "HTTP timeout per source request. Defaults to 15.",
                "default": 15,
            },
        },
        "required": [],
    },
}

MARKET_RESEARCH_SOURCES_SCHEMA = {
    "name": "market_research_sources",
    "description": (
        "Search Bilibili, Eastmoney Guba, Xueqiu, and CLS for a user query, "
        "then synthesize a cross-source finance research summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query, topic, stock, or market question to research.",
            },
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional stock symbols/codes to focus community sources.",
                "default": [],
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Subset of bilibili, eastmoney_guba, xueqiu, cls. Defaults to all. "
                    "Xueqiu supports XUEQIU_FETCH_MODE=auto/http/browser and optional "
                    "XUEQIU_STORAGE_STATE for Playwright-backed WAF fallback."
                ),
                "default": [],
            },
            "max_items_per_source": {
                "type": "integer",
                "description": "Maximum high-information items to keep per source. Defaults to 10.",
                "default": 10,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "HTTP timeout per source request. Defaults to 15.",
                "default": 15,
            },
            "llm_timeout_seconds": {
                "type": "number",
                "description": "Timeout for the synthesis LLM call. Defaults to 45.",
                "default": 45,
            },
        },
        "required": ["query"],
    },
}


def check_requirements() -> bool:
    return True


def handle_fetch_sentiment_sources(args: dict[str, Any], **_kwargs) -> str:
    try:
        payload = fetch_market_sentiment_sources(
            keyword=str(args.get("keyword") or ""),
            symbols=[str(item) for item in (args.get("symbols") or [])],
            sources=[str(item) for item in (args.get("sources") or [])],
            max_items_per_source=int(args.get("max_items_per_source") or 20),
            timeout_seconds=float(args.get("timeout_seconds") if args.get("timeout_seconds") is not None else 15),
        )
        return json.dumps({"success": True, **payload}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


def make_research_sources_handler(llm: Any):
    def _handler(args: dict[str, Any], **_kwargs) -> str:
        try:
            payload = research_market_sources(
                query=str(args.get("query") or ""),
                symbols=[str(item) for item in (args.get("symbols") or [])],
                sources=[str(item) for item in (args.get("sources") or [])],
                max_items_per_source=int(args.get("max_items_per_source") or 10),
                timeout_seconds=float(args.get("timeout_seconds") if args.get("timeout_seconds") is not None else 15),
            )
            analysis = synthesize_research_summary(
                llm,
                query=str(args.get("query") or ""),
                payload=payload,
                timeout_seconds=float(args.get("llm_timeout_seconds") if args.get("llm_timeout_seconds") is not None else 45),
            )
            return json.dumps({"success": True, **payload, "analysis": analysis}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    return _handler


def synthesize_research_summary(
    llm: Any,
    *,
    query: str,
    payload: dict[str, Any],
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    compact = _compact_research_items(payload.get("items") or [])
    if llm is not None and hasattr(llm, "complete_structured") and compact:
        try:
            structured = llm.complete_structured(
                instructions=(
                    "你是市场信息研究助手。请基于跨 source 搜索结果生成结构化总结。"
                    "严格区分不同来源的事实、观点、情绪和噪声；不要补充未出现在材料中的事实。"
                    "输出应覆盖共识、分歧、高信息量证据、风险和低置信度内容。"
                    "这不是投资建议，不给买卖指令。"
                ),
                input=[
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "query": query,
                                "summary": payload.get("summary") or {},
                                "items": compact,
                                "errors": payload.get("errors") or [],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
                json_schema=RESEARCH_SUMMARY_SCHEMA,
                json_mode=True,
                schema_name="market_cross_source_research_summary",
                temperature=0.2,
                max_tokens=4000,
                timeout=timeout_seconds,
                purpose="market_cross_source_research",
            )
            if isinstance(structured.parsed, dict) and structured.parsed:
                result = structured.parsed
                result["markdown"] = render_research_markdown(result, payload=payload)
                return result
        except Exception as exc:
            fallback = _heuristic_research_summary(query, payload)
            fallback["llm_warning"] = str(exc)
            fallback["markdown"] = render_research_markdown(fallback, payload=payload)
            return fallback

    fallback = _heuristic_research_summary(query, payload)
    fallback["markdown"] = render_research_markdown(fallback, payload=payload)
    return fallback


def _compact_research_items(items: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    compact = []
    for item in items[:limit]:
        compact.append(
            {
                "source": item.get("source") or "",
                "title": str(item.get("title") or "")[:180],
                "text": str(item.get("text") or "")[:500],
                "author": item.get("author") or "",
                "sentiment": item.get("sentiment") or "neutral",
                "information_score": item.get("information_score", 0),
                "url": item.get("url") or "",
            }
        )
    return compact


def _heuristic_research_summary(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items") or []
    by_source: dict[str, int] = {}
    evidence = []
    bullish = []
    bearish = []
    neutral = []
    for item in items:
        source = str(item.get("source") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        line = f"{source}: {item.get('title') or item.get('text') or ''}"
        if item.get("sentiment") == "bullish":
            bullish.append(line)
        elif item.get("sentiment") == "bearish":
            bearish.append(line)
        else:
            neutral.append(line)
        evidence.append(
            {
                "source": source,
                "claim": str(item.get("title") or "")[:160],
                "evidence": str(item.get("text") or item.get("title") or "")[:240],
                "url": item.get("url") or "",
            }
        )
    return {
        "executive_summary": f"已围绕“{query}”检索 {len(items)} 条跨来源材料；以下为基于标题、摘要和社区文本的降级总结。",
        "source_coverage": [f"{source}: {count}" for source, count in sorted(by_source.items())],
        "consensus": (bullish or neutral)[:5],
        "divergences": bearish[:5],
        "high_information_evidence": evidence[:8],
        "noise_or_low_confidence": [f"{item.get('source')}: {item.get('error')}" for item in payload.get("errors") or []],
        "risk_flags": ["部分来源可能需要登录态、签名或浏览器态采集；未返回的来源不应被解读为没有市场讨论。"],
    }


def render_research_markdown(data: dict[str, Any], *, payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-source Market Research",
        "",
        "## Executive Summary",
        "",
        str(data.get("executive_summary") or ""),
    ]
    if data.get("llm_warning"):
        lines.extend(["", "## Generation Warning", "", str(data.get("llm_warning"))])
    for title, key in (
        ("Source Coverage", "source_coverage"),
        ("Consensus", "consensus"),
        ("Divergences", "divergences"),
        ("Risk Flags", "risk_flags"),
        ("Noise / Low Confidence", "noise_or_low_confidence"),
    ):
        values = data.get(key) or []
        if values:
            lines.extend(["", f"## {title}", ""])
            lines.extend(f"- {value}" for value in values)
    evidence = data.get("high_information_evidence") or []
    if evidence:
        lines.extend(["", "## High-information Evidence", ""])
        for item in evidence:
            suffix = f" ({item.get('url')})" if item.get("url") else ""
            lines.append(f"- [{item.get('source', '')}] {item.get('claim', '')}: {item.get('evidence', '')}{suffix}")
    errors = payload.get("errors") or []
    if errors:
        lines.extend(["", "## Source Errors", ""])
        lines.extend(f"- {item.get('source')}: {item.get('error')}" for item in errors)
    lines.extend(["", "> This summary is for information synthesis only and is not investment advice."])
    return "\n".join(lines).strip()
