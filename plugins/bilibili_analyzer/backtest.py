"""Backtest creator stock-pick claims against later market prices."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .analysis import transcript_to_text
from .pdf import write_markdown_pdf

STOCK_PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stock_name": {"type": "string"},
                    "symbol": {"type": "string"},
                    "market": {"type": "string"},
                    "stance": {"type": "string"},
                    "horizon_days": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "thesis": {"type": "string"},
                    "evidence": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
                "required": ["stock_name", "stance", "thesis", "evidence"],
            },
        },
        "no_pick_reason": {"type": "string"},
    },
    "required": ["picks"],
}

COMMENT_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "score": {"type": "number"},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "supporting_comments": {"type": "array", "items": {"type": "string"}},
        "critical_comments": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "score", "summary"],
}


def extract_stock_picks_from_transcript(
    llm: Any,
    *,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    llm_timeout_seconds: float = 45.0,
    max_tokens: int = 2500,
) -> dict[str, Any]:
    """Extract explicit stock-pick claims from a creator video transcript."""
    instructions = (
        "Extract only explicit, testable stock selection or directional calls from a finance creator video. "
        "A valid pick must mention a company, ticker/code, sector ETF, or clearly named tradable stock, "
        "and must include a directional stance. Ignore generic macro commentary without a tradable target. "
        "Use stance values: bullish, bearish, neutral, avoid, unknown. "
        "If the creator gives a likely timeframe, convert it to horizon_days; otherwise use 30. "
        "Set confidence between 0 and 1 based on how explicit the call is. "
        "Do not infer outside facts."
    )
    text = transcript_to_text(segments, max_chars=20000)
    result = llm.complete_structured(
        instructions=instructions,
        input=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "metadata": metadata,
                        "transcript": text,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        json_schema=STOCK_PICK_SCHEMA,
        json_mode=True,
        schema_name="bilibili_creator_stock_pick_extraction",
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=llm_timeout_seconds,
        purpose="bilibili_stock_pick_extraction",
    )
    parsed = result.parsed if isinstance(result.parsed, dict) else {"picks": []}
    parsed["raw_text"] = result.text
    parsed["audit"] = result.audit
    return parsed


def normalize_market_symbol(symbol: str, market: str = "") -> str:
    """Normalize common US/HK/CN symbols for Yahoo Finance chart API."""
    raw = (symbol or "").strip().upper()
    market_norm = (market or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("SH.", "").replace("SZ.", "")
    raw = raw.replace(".SH", ".SS")
    if raw.endswith((".SS", ".SZ", ".HK")):
        return raw
    if re.fullmatch(r"\d{6}", raw):
        if raw.startswith(("6", "9")) or market_norm in {"sh", "shanghai", "sse", "a_sh"}:
            return f"{raw}.SS"
        return f"{raw}.SZ"
    if re.fullmatch(r"\d{4,5}", raw) and (market_norm in {"hk", "hongkong", "h-share"} or raw.startswith("0")):
        return f"{raw.zfill(4)}.HK"
    return raw


def fetch_yahoo_price_series(
    symbol: str,
    *,
    start_date: datetime,
    end_date: datetime,
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    """Fetch daily closes from Yahoo Finance's chart endpoint."""
    yahoo_symbol = normalize_market_symbol(symbol)
    if not yahoo_symbol:
        return []
    period1 = int(start_date.timestamp())
    period2 = int((end_date + timedelta(days=1)).timestamp())
    query = urllib.parse.urlencode(
        {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = (((payload.get("chart") or {}).get("result") or []) or [{}])[0]
    timestamps = result.get("timestamp") or []
    quote = ((((result.get("indicators") or {}).get("quote") or []) or [{}])[0])
    closes = quote.get("close") or []
    series = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        try:
            value = float(close)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        series.append(
            {
                "date": datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d"),
                "close": value,
            }
        )
    return series


def score_pick_against_prices(
    pick: dict[str, Any],
    *,
    published_at: datetime,
    now: datetime | None = None,
    price_fetcher=fetch_yahoo_price_series,
) -> dict[str, Any]:
    """Score one pick against later price movement."""
    now = now or datetime.now()
    symbol = normalize_market_symbol(str(pick.get("symbol") or ""), str(pick.get("market") or ""))
    horizon_days = int(pick.get("horizon_days") or 30)
    horizon_days = min(365, max(1, horizon_days))
    end_date = min(now, published_at + timedelta(days=horizon_days))
    if end_date <= published_at:
        end_date = min(now, published_at + timedelta(days=1))
    series = price_fetcher(symbol, start_date=published_at, end_date=end_date) if symbol else []
    if len(series) < 2:
        return {
            "success": False,
            "score": 0,
            "reason": "No enough price data for the normalized symbol.",
            "symbol": symbol,
            "price_series_count": len(series),
        }
    start = series[0]
    end = series[-1]
    start_close = float(start["close"])
    end_close = float(end["close"])
    pct_change = (end_close - start_close) / start_close * 100 if start_close else 0.0
    stance = str(pick.get("stance") or "unknown").lower()
    confidence = max(0.0, min(1.0, float(pick.get("confidence") or 0.5)))
    if stance in {"bullish", "buy", "long"}:
        base = 50.0 + pct_change * 2.2
    elif stance in {"bearish", "avoid", "sell", "short"}:
        base = 50.0 - pct_change * 2.2
    elif stance in {"neutral", "wait", "hold"}:
        base = 80.0 - abs(pct_change) * 4.0
    else:
        base = 40.0
    score = max(0.0, min(100.0, base))
    weighted_score = score * (0.6 + confidence * 0.4)
    return {
        "success": True,
        "score": round(weighted_score, 1),
        "raw_direction_score": round(score, 1),
        "symbol": symbol,
        "stance": stance,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "start_date": start["date"],
        "end_date": end["date"],
        "start_close": round(start_close, 4),
        "end_close": round(end_close, 4),
        "pct_change": round(pct_change, 2),
    }


def analyze_post_verification_comments(
    llm: Any,
    *,
    pick: dict[str, Any],
    video: dict[str, Any],
    comments: list[dict[str, Any]],
    published_at: datetime,
    min_days_after_publish: int = 7,
    llm_timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Classify latest comments as post-hoc audience validation."""
    min_ts = int((published_at + timedelta(days=max(0, int(min_days_after_publish or 0)))).timestamp())
    eligible = [
        comment
        for comment in comments
        if int(comment.get("ctime") or 0) >= min_ts
    ]
    if not eligible:
        eligible = list(comments[:30])
    compact = [
        {
            "message": str(item.get("message") or "")[:300],
            "like": int(item.get("like") or 0),
            "ctime": int(item.get("ctime") or 0),
        }
        for item in eligible[:30]
        if str(item.get("message") or "").strip()
    ]
    if not compact:
        return {
            "success": False,
            "score": None,
            "verdict": "insufficient_comments",
            "summary": "No usable post-verification comments.",
            "comment_count": 0,
        }
    try:
        structured = llm.complete_structured(
            instructions=(
                "Judge whether post-video audience comments validate or refute a finance creator's stock pick. "
                "Focus on comments written after the idea had time to be tested. "
                "Return verdict as one of: validated, mostly_validated, mixed, mostly_refuted, refuted, irrelevant. "
                "score should be 0-100, where 100 means comments strongly agree the call was right, "
                "50 means mixed/unclear, and 0 means comments strongly say the call was wrong. "
                "Ignore pure emotion, spam, memes, and comments without investment information."
            ),
            input=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "video": video,
                            "published_at": published_at.strftime("%Y-%m-%d"),
                            "pick": pick,
                            "comments": compact,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            json_schema=COMMENT_VERDICT_SCHEMA,
            json_mode=True,
            schema_name="bilibili_post_verification_comment_verdict",
            temperature=0.1,
            max_tokens=1800,
            timeout=llm_timeout_seconds,
            purpose="bilibili_comment_verdict",
        )
        parsed = structured.parsed if isinstance(structured.parsed, dict) else {}
    except Exception as exc:
        parsed = _heuristic_comment_verdict(compact)
        parsed["llm_warning"] = str(exc)
    score = parsed.get("score")
    try:
        score_value = max(0.0, min(100.0, float(score)))
    except (TypeError, ValueError):
        score_value = 50.0
    parsed["success"] = True
    parsed["score"] = round(score_value, 1)
    parsed["comment_count"] = len(compact)
    return parsed


def combine_price_and_comment_scores(price_score: dict[str, Any], comment_verdict: dict[str, Any]) -> dict[str, Any]:
    if not price_score.get("success"):
        base = 50.0
        weights = {"price": 0.0, "comments": 1.0 if comment_verdict.get("success") else 0.0}
    else:
        base = float(price_score.get("score") or 0)
        weights = {"price": 0.75, "comments": 0.25 if comment_verdict.get("success") else 0.0}
    if comment_verdict.get("success") and weights["comments"] > 0:
        combined = base * weights["price"] + float(comment_verdict.get("score") or 50) * weights["comments"]
    else:
        combined = base
    return {
        "score": round(max(0.0, min(100.0, combined)), 1),
        "weights": weights,
    }


def _heuristic_comment_verdict(comments: list[dict[str, Any]]) -> dict[str, Any]:
    positive = 0
    negative = 0
    for item in comments:
        text = str(item.get("message") or "")
        if re.search(r"说中了|准|牛|对了|验证|神了|厉害|预判", text):
            positive += 1
        if re.search(r"错了|打脸|亏|反了|不准|割肉|坑人|马后炮", text):
            negative += 1
    total = max(1, positive + negative)
    score = 50 + (positive - negative) / total * 50
    verdict = "mixed"
    if score >= 75:
        verdict = "mostly_validated"
    elif score <= 25:
        verdict = "mostly_refuted"
    return {
        "verdict": verdict,
        "score": round(max(0, min(100, score)), 1),
        "confidence": min(1.0, total / 10),
        "summary": f"Heuristic comment verdict from {len(comments)} latest comments.",
        "supporting_comments": [],
        "critical_comments": [],
    }


def render_creator_backtest_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Bilibili Creator Stock-Pick Backtest",
        "",
        f"- Creator: {report.get('creator_name') or report.get('up_mid')}",
        f"- UP mid: {report.get('up_mid', '')}",
        f"- Period: {report.get('start_date', '')} to {report.get('end_date', '')}",
        f"- Videos checked: {report.get('video_count', 0)}",
        f"- Picks extracted: {report.get('pick_count', 0)}",
        f"- Scored picks: {report.get('scored_pick_count', 0)}",
        f"- Average score: {report.get('average_score', 'N/A')}",
        "",
        "## Scoring Rule",
        "",
        "- Bullish calls score higher when the later price rises.",
        "- Bearish/avoid calls score higher when the later price falls.",
        "- Neutral calls score higher when the later move is small.",
        "- Explicit confidence from the transcript weights the final score.",
        "- Latest comments after the idea had time to play out add an audience-verdict score.",
        "- Final score combines price movement and post-verification comment verdict.",
        "- No price data or no clear symbol is marked unscored.",
        "",
        "## Pick Results",
        "",
    ]
    picks = report.get("picks") or []
    if not picks:
        lines.append("- No explicit, testable stock picks were extracted.")
    for item in picks:
        pick = item.get("pick") or {}
        score = item.get("score") or {}
        video = item.get("video") or {}
        title = video.get("title") or ""
        created = item.get("published_date") or ""
        lines.append(
            f"- {pick.get('stock_name') or pick.get('symbol') or 'unknown'} "
            f"| stance={pick.get('stance', '')} | symbol={score.get('symbol') or pick.get('symbol') or ''} "
            f"| final={item.get('final_score', {}).get('score', 'unscored')} "
            f"| price_score={score.get('score', 'unscored')} | comment_score={(item.get('comment_verdict') or {}).get('score', 'N/A')} "
            f"| move={score.get('pct_change', 'N/A')}% "
            f"| video={title} | date={created}"
        )
        if pick.get("thesis"):
            lines.append(f"  - Thesis: {pick.get('thesis')}")
        if pick.get("evidence"):
            lines.append(f"  - Evidence: {pick.get('evidence')}")
        if not score.get("success"):
            lines.append(f"  - Unscored reason: {score.get('reason', '')}")
        verdict = item.get("comment_verdict") or {}
        if verdict.get("summary"):
            lines.append(f"  - Comment verdict: {verdict.get('verdict', '')}, {verdict.get('summary')}")
    return "\n".join(lines).strip() + "\n"


def persist_creator_backtest_report(output_dir: Path, report: dict[str, Any]) -> tuple[str, str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "stock_pick_backtest.md"
    json_path = output_dir / "stock_pick_backtest.json"
    pdf_path = output_dir / "stock_pick_backtest.pdf"
    markdown = render_creator_backtest_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    write_markdown_pdf(markdown, pdf_path, title="Bilibili Creator Stock-Pick Backtest")
    payload = {key: value for key, value in report.items() if key != "markdown"}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(md_path), str(json_path), str(pdf_path)
