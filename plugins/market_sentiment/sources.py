"""External market-sentiment sources for finance reports."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

SOURCE_EASTMONEY = "eastmoney_guba"
SOURCE_XUEQIU = "xueqiu"
SOURCE_CLS = "cls"
SOURCE_BILIBILI = "bilibili"
DEFAULT_SOURCES = (SOURCE_EASTMONEY, SOURCE_XUEQIU, SOURCE_CLS)
DEFAULT_RESEARCH_SOURCES = (SOURCE_BILIBILI, SOURCE_EASTMONEY, SOURCE_XUEQIU, SOURCE_CLS)

_NOISE_PATTERNS = (
    "666",
    "哈哈",
    "顶",
    "路过",
    "打卡",
    "牛逼",
    "沙发",
)
_BULLISH_TERMS = (
    "看多",
    "反弹",
    "上涨",
    "突破",
    "低估",
    "增持",
    "买入",
    "利好",
    "修复",
    "放量",
)
_BEARISH_TERMS = (
    "看空",
    "下跌",
    "回调",
    "减仓",
    "卖出",
    "高估",
    "风险",
    "利空",
    "破位",
    "套牢",
)


class MarketSourceError(RuntimeError):
    """Raised when one market source cannot be fetched or parsed."""


def fetch_market_sentiment_sources(
    *,
    keyword: str = "",
    symbols: list[str] | None = None,
    max_items_per_source: int = 20,
    sources: list[str] | None = None,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Fetch and normalize market-sentiment items from configured sources."""

    symbols = [str(item).strip() for item in (symbols or []) if str(item).strip()]
    max_items_per_source = min(100, max(1, int(max_items_per_source or 20)))
    selected_sources = _normalize_sources(sources)
    payload: dict[str, Any] = {
        "keyword": keyword,
        "symbols": symbols,
        "sources": selected_sources,
        "items": [],
        "errors": [],
        "fetched_at": int(time.time()),
    }

    for source in selected_sources:
        try:
            if source == SOURCE_EASTMONEY:
                items = fetch_eastmoney_guba(
                    symbols=symbols,
                    keyword=keyword,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            elif source == SOURCE_XUEQIU:
                items = fetch_xueqiu_discussions(
                    symbols=symbols,
                    keyword=keyword,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            elif source == SOURCE_CLS:
                items = fetch_cls_telegraphs(
                    keyword=keyword,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            else:
                raise MarketSourceError(f"Unsupported market source: {source}")
            payload["items"].extend(_select_high_information_items(items, max_items_per_source))
        except Exception as exc:
            payload["errors"].append({"source": source, "error": str(exc)})

    payload["items"] = sorted(
        payload["items"],
        key=lambda item: (
            float(item.get("information_score") or 0),
            int(item.get("like") or 0),
            int(item.get("timestamp") or 0),
        ),
        reverse=True,
    )
    payload["summary"] = summarize_market_source_items(payload["items"], payload["errors"])
    return payload


def research_market_sources(
    *,
    query: str,
    symbols: list[str] | None = None,
    sources: list[str] | None = None,
    max_items_per_source: int = 10,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Search Bilibili and finance-community sources for one research query."""

    query = str(query or "").strip()
    if not query:
        raise ValueError("query is required")
    symbols = [str(item).strip() for item in (symbols or []) if str(item).strip()]
    max_items_per_source = min(50, max(1, int(max_items_per_source or 10)))
    selected_sources = _normalize_research_sources(sources)
    payload: dict[str, Any] = {
        "query": query,
        "symbols": symbols,
        "sources": selected_sources,
        "items": [],
        "errors": [],
        "fetched_at": int(time.time()),
    }

    for source in selected_sources:
        try:
            if source == SOURCE_BILIBILI:
                items = search_bilibili_videos(
                    query=query,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            elif source == SOURCE_EASTMONEY:
                items = fetch_eastmoney_guba(
                    symbols=symbols,
                    keyword=query,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            elif source == SOURCE_XUEQIU:
                items = fetch_xueqiu_discussions(
                    symbols=symbols,
                    keyword=query,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            elif source == SOURCE_CLS:
                items = fetch_cls_telegraphs(
                    keyword=query,
                    limit=max_items_per_source,
                    timeout_seconds=timeout_seconds,
                )
            else:
                raise MarketSourceError(f"Unsupported market source: {source}")
            payload["items"].extend(_select_high_information_items(items, max_items_per_source))
        except Exception as exc:
            payload["errors"].append({"source": source, "error": str(exc)})

    payload["items"] = sorted(
        payload["items"],
        key=lambda item: (
            float(item.get("information_score") or 0),
            int(item.get("like") or 0),
            int(item.get("timestamp") or 0),
        ),
        reverse=True,
    )
    payload["summary"] = summarize_market_source_items(payload["items"], payload["errors"])
    return payload


def search_bilibili_videos(
    *,
    query: str,
    limit: int = 10,
    timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Search Bilibili videos and normalize result snippets as source items."""

    params = {
        "search_type": "video",
        "keyword": query,
        "page": 1,
        "page_size": min(50, max(limit, 10)),
    }
    payload = _fetch_json(
        "https://api.bilibili.com/x/web-interface/search/type?" + urllib.parse.urlencode(params),
        timeout_seconds=timeout_seconds,
        referer="https://search.bilibili.com/",
    )
    rows = _dig_list(payload, ("result", "data", "items", "list"))
    items = []
    for row in rows:
        title = _clean_html(row.get("title") or "")
        description = _clean_html(row.get("description") or row.get("desc") or "")
        if not title and not description:
            continue
        bvid = str(row.get("bvid") or row.get("arcurl") or "")
        url = row.get("arcurl") or (f"https://www.bilibili.com/video/{bvid}" if bvid.startswith("BV") else "")
        items.append(
            _normalize_item(
                source=SOURCE_BILIBILI,
                title=title or description[:80],
                text=description or title,
                symbol="",
                url=_normalize_bilibili_url(str(url)),
                author=row.get("author") or row.get("uname") or "",
                like=row.get("like") or row.get("favorites") or 0,
                reply_count=row.get("review") or row.get("danmaku") or 0,
                timestamp=_parse_time(row.get("pubdate") or row.get("senddate")),
            )
        )
    return items[:limit]


def fetch_eastmoney_guba(
    *,
    symbols: list[str],
    keyword: str = "",
    limit: int = 20,
    timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch Eastmoney Guba discussion posts for A-share symbols.

    The endpoint is a public web endpoint used by guba.eastmoney.com. It is
    intentionally wrapped as best-effort because Eastmoney occasionally changes
    response fields.
    """

    items: list[dict[str, Any]] = []
    for symbol in symbols[:10]:
        code = _eastmoney_guba_code(symbol)
        if not code:
            continue
        query = urllib.parse.urlencode(
            {
                "code": code,
                "sorttype": 1,
                "ps": min(100, max(limit, 20)),
                "p": 1,
                "type": 0,
            }
        )
        payload = _fetch_json(
            f"https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist?{query}",
            timeout_seconds=timeout_seconds,
            referer=f"https://guba.eastmoney.com/list,{code}.html",
        )
        rows = _dig_list(payload, ("re", "data", "list", "article_list", "items"))
        for row in rows:
            text = _clean_html(
                row.get("post_title")
                or row.get("title")
                or row.get("post_content")
                or row.get("content")
                or ""
            )
            if not text:
                continue
            items.append(
                _normalize_item(
                    source=SOURCE_EASTMONEY,
                    title=text,
                    text=_clean_html(row.get("post_content") or row.get("summary") or text),
                    symbol=symbol,
                    url=_eastmoney_post_url(row, code),
                    author=row.get("user_nickname") or row.get("post_user") or "",
                    like=row.get("post_like_count") or row.get("like_count") or 0,
                    reply_count=row.get("post_comment_count") or row.get("reply_count") or 0,
                    timestamp=_parse_time(row.get("post_publish_time") or row.get("publish_time")),
                )
            )
        if not items:
            items.extend(
                _fetch_eastmoney_guba_html(
                    symbol=symbol,
                    code=code,
                    limit=limit,
                    timeout_seconds=timeout_seconds,
                )
            )
    if not items and keyword:
        items.extend(_fetch_eastmoney_search(keyword, limit, timeout_seconds))
    return items[:limit]


def fetch_xueqiu_discussions(
    *,
    symbols: list[str],
    keyword: str = "",
    limit: int = 20,
    timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch Xueqiu status search results.

    Xueqiu can require cookies for stable access. Set XUEQIU_COOKIE when needed.
    If its web WAF returns HTML instead of JSON, XUEQIU_FETCH_MODE=auto will
    try an optional Playwright-backed browser session.
    """

    terms = symbols[:10] or ([keyword] if keyword else [])
    items: list[dict[str, Any]] = []
    for term in terms:
        params = {
            "q": term,
            "count": min(50, max(limit, 10)),
            "page": 1,
        }
        url = "https://xueqiu.com/query/v1/search/status.json?" + urllib.parse.urlencode(params)
        payload = _fetch_xueqiu_json(
            url,
            timeout_seconds=timeout_seconds,
        )
        rows = _dig_list(payload, ("list", "statuses", "data", "items"))
        for row in rows:
            text = _clean_html(
                row.get("text") or row.get("description") or row.get("title") or ""
            )
            if not text:
                continue
            items.append(
                _normalize_item(
                    source=SOURCE_XUEQIU,
                    title=_clean_html(row.get("title") or text[:80]),
                    text=text,
                    symbol=term if _looks_like_symbol(term) else "",
                    url=_xueqiu_url(row),
                    author=((row.get("user") or {}).get("screen_name") if isinstance(row.get("user"), dict) else ""),
                    like=row.get("like_count") or row.get("likes_count") or 0,
                    reply_count=row.get("reply_count") or row.get("comments_count") or 0,
                    timestamp=_parse_time(row.get("created_at")),
                )
            )
    return items[:limit]


def _fetch_xueqiu_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    mode = os.getenv("XUEQIU_FETCH_MODE", "auto").strip().lower() or "auto"
    if mode not in {"auto", "http", "browser"}:
        mode = "auto"

    direct_error: MarketSourceError | None = None
    if mode != "browser":
        try:
            return _fetch_json(
                url,
                timeout_seconds=timeout_seconds,
                referer="https://xueqiu.com/",
                cookie=os.getenv("XUEQIU_COOKIE", ""),
            )
        except MarketSourceError as exc:
            direct_error = exc
            if mode == "http":
                raise

    try:
        return _fetch_xueqiu_json_browser(url, timeout_seconds=timeout_seconds)
    except MarketSourceError as browser_error:
        if direct_error is not None:
            raise MarketSourceError(
                f"Xueqiu HTTP fetch failed ({direct_error}); browser fallback failed ({browser_error})"
            ) from browser_error
        raise


def _fetch_xueqiu_json_browser(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Fetch Xueqiu JSON through a real browser context when WAF blocks HTTP.

    Optional environment:
    - XUEQIU_STORAGE_STATE: Playwright storage_state JSON exported after login.
    - XUEQIU_COOKIE: cookie header, used as a lighter fallback.
    - XUEQIU_BROWSER_BIN: Chrome/Edge/Chromium executable path.
    - XUEQIU_BROWSER_HEADLESS: true/false, defaults to true.
    """

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise MarketSourceError(
            "Xueqiu browser fallback requires Playwright. Install the browser toolchain "
            "or set XUEQIU_FETCH_MODE=http to disable browser fallback."
        ) from exc

    storage_state = os.getenv("XUEQIU_STORAGE_STATE", "").strip()
    browser_bin = os.getenv("XUEQIU_BROWSER_BIN", "").strip()
    headless = os.getenv("XUEQIU_BROWSER_HEADLESS", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    launch_kwargs: dict[str, Any] = {"headless": headless}
    if browser_bin:
        launch_kwargs["executable_path"] = browser_bin
    context_kwargs: dict[str, Any] = {
        "user_agent": USER_AGENT,
        "locale": "zh-CN",
        "timezone_id": "Asia/Shanghai",
        "extra_http_headers": {
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    }
    if storage_state:
        state_path = Path(storage_state).expanduser()
        if not state_path.exists():
            raise MarketSourceError(f"XUEQIU_STORAGE_STATE does not exist: {state_path}")
        context_kwargs["storage_state"] = str(state_path)

    timeout_ms = max(1, int(timeout_seconds * 1000))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)
            try:
                context = browser.new_context(**context_kwargs)
                cookie_header = os.getenv("XUEQIU_COOKIE", "").strip()
                if cookie_header and not storage_state:
                    parsed_cookies = _parse_cookie_header_for_domain(cookie_header, ".xueqiu.com")
                    if parsed_cookies:
                        context.add_cookies(parsed_cookies)

                page = context.new_page()
                page.goto("https://xueqiu.com/", wait_until="domcontentloaded", timeout=timeout_ms)
                response = context.request.get(
                    url,
                    headers={
                        "Accept": "application/json,text/plain,*/*",
                        "Referer": "https://xueqiu.com/",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                    timeout=timeout_ms,
                )
                text = response.text().strip()
                if not response.ok:
                    raise MarketSourceError(f"Xueqiu browser request failed: HTTP {response.status}")
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    preview = _clean_html(text)[:120]
                    raise MarketSourceError(f"Xueqiu browser request returned non-JSON: {preview}") from exc
                return parsed if isinstance(parsed, dict) else {"data": parsed}
            finally:
                browser.close()
    except (PlaywrightError, PlaywrightTimeoutError) as exc:
        raise MarketSourceError(f"Xueqiu browser fallback failed: {exc}") from exc


def _parse_cookie_header_for_domain(cookie_header: str, domain: str) -> list[dict[str, Any]]:
    cookies = []
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": domain,
                "path": "/",
                "httpOnly": False,
                "secure": True,
                "sameSite": "Lax",
            }
        )
    return cookies


def fetch_cls_telegraphs(
    *,
    keyword: str = "",
    limit: int = 20,
    timeout_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch Cailianpress telegraph items as catalyst/sentiment context."""

    rn = min(50, max(limit, 20))
    payload = _fetch_json(
        f"https://www.cls.cn/api/cache?{urllib.parse.urlencode({'rn': rn, 'lastTime': 0, 'name': 'telegraph'})}",
        timeout_seconds=timeout_seconds,
        referer="https://www.cls.cn/telegraph",
    )
    rows = _dig_list(payload, ("data", "roll_data", "telegraphs", "items"))
    if not rows:
        query = urllib.parse.urlencode(
            {
                "refresh_type": 1,
                "rn": rn,
                "last_time": int(time.time()),
                "category": "",
            }
        )
        payload = _fetch_json(
            f"https://www.cls.cn/v1/roll/get_roll_list?{query}",
            timeout_seconds=timeout_seconds,
            referer="https://www.cls.cn/telegraph",
        )
        if str(payload.get("errno") or "") not in ("", "0"):
            raise MarketSourceError(f"CLS roll_list failed: {payload.get('msg') or payload.get('error') or payload.get('errno')}")
    rows = _dig_list(payload, ("data", "roll_data", "telegraphs", "items"))
    items: list[dict[str, Any]] = []
    for row in rows:
        text = _clean_html(
            row.get("content") or row.get("brief") or row.get("title") or ""
        )
        if not text:
            continue
        if keyword and keyword not in text:
            continue
        items.append(
            _normalize_item(
                source=SOURCE_CLS,
                title=_clean_html(row.get("title") or text[:80]),
                text=text,
                symbol="",
                url=_cls_url(row),
                author="财联社",
                like=0,
                reply_count=0,
                timestamp=_parse_time(row.get("ctime") or row.get("time")),
            )
        )
    return items[:limit]


def summarize_market_source_items(
    items: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bullish = sum(1 for item in items if item.get("sentiment") == "bullish")
    bearish = sum(1 for item in items if item.get("sentiment") == "bearish")
    neutral = max(0, len(items) - bullish - bearish)
    overall = "neutral"
    if bullish > bearish + neutral / 2:
        overall = "bullish"
    elif bearish > bullish + neutral / 2:
        overall = "bearish"
    by_source: dict[str, int] = {}
    for item in items:
        by_source[str(item.get("source") or "unknown")] = by_source.get(str(item.get("source") or "unknown"), 0) + 1
    return {
        "overall": overall,
        "item_count": len(items),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "by_source": by_source,
        "error_count": len(errors or []),
    }


def _fetch_eastmoney_search(keyword: str, limit: int, timeout_seconds: float) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"keyword": keyword, "pageindex": 1, "pagesize": limit})
    payload = _fetch_json(
        f"https://gbapi.eastmoney.com/api/Search/ArticleSearch?{query}",
        timeout_seconds=timeout_seconds,
        referer="https://guba.eastmoney.com/",
    )
    rows = _dig_list(payload, ("data", "list", "items"))
    items = []
    for row in rows:
        text = _clean_html(row.get("post_title") or row.get("title") or row.get("content") or "")
        if text:
            items.append(
                _normalize_item(
                    source=SOURCE_EASTMONEY,
                    title=text,
                    text=_clean_html(row.get("post_content") or text),
                    symbol="",
                    url=_eastmoney_post_url(row, ""),
                    author=row.get("user_nickname") or "",
                    like=row.get("post_like_count") or 0,
                    reply_count=row.get("post_comment_count") or 0,
                    timestamp=_parse_time(row.get("post_publish_time")),
                )
            )
    return items


def _fetch_eastmoney_guba_html(
    *,
    symbol: str,
    code: str,
    limit: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    html = _fetch_text(
        f"https://guba.eastmoney.com/list,{code}.html",
        timeout_seconds=timeout_seconds,
        referer="https://guba.eastmoney.com/",
    )
    items: list[dict[str, Any]] = []
    row_pattern = re.compile(
        r'<tr class="listitem">(?P<body>.*?)</tr>',
        re.S,
    )
    for match in row_pattern.finditer(html):
        body = match.group("body")
        title_match = re.search(
            r'<div class="title"><a(?P<attrs>[^>]*)href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            body,
            re.S,
        )
        if not title_match:
            continue
        read = _first_group(body, r'<div class="read">([^<]*)</div>')
        reply = _first_group(body, r'<div class="reply">([^<]*)</div>')
        author = _clean_html(_first_group(body, r'<div class="author">.*?<a[^>]*>(.*?)</a>'))
        title = _clean_html(title_match.group("title"))
        href = title_match.group("href")
        if not title:
            continue
        url = urllib.parse.urljoin("https://guba.eastmoney.com/", href)
        items.append(
            _normalize_item(
                source=SOURCE_EASTMONEY,
                title=title,
                text=title,
                symbol=symbol,
                url=url,
                author=author,
                like=read,
                reply_count=reply,
                timestamp=0,
            )
        )
        if len(items) >= limit:
            break
    return items


def _fetch_json(
    url: str,
    *,
    timeout_seconds: float,
    referer: str = "",
    cookie: str = "",
) -> dict[str, Any]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
    }
    if referer:
        headers["Referer"] = referer
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise MarketSourceError(str(exc)) from exc
    text = raw.strip()
    if text.startswith("callback(") and text.endswith(")"):
        text = text[len("callback(") : -1]
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MarketSourceError(f"Invalid JSON from {urllib.parse.urlparse(url).netloc}") from exc
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _fetch_text(
    url: str,
    *,
    timeout_seconds: float,
    referer: str = "",
    cookie: str = "",
) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise MarketSourceError(str(exc)) from exc


def _dig_list(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _dig_list(value, keys)
            if nested:
                return nested
    for value in payload.values():
        nested = _dig_list(value, keys)
        if nested:
            return nested
    return []


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if not sources:
        return list(DEFAULT_SOURCES)
    selected = []
    for source in sources:
        normalized = str(source).strip().lower()
        if normalized in DEFAULT_SOURCES and normalized not in selected:
            selected.append(normalized)
    return selected or list(DEFAULT_SOURCES)


def _normalize_research_sources(sources: list[str] | None) -> list[str]:
    if not sources:
        return list(DEFAULT_RESEARCH_SOURCES)
    selected = []
    for source in sources:
        normalized = str(source).strip().lower()
        if normalized in DEFAULT_RESEARCH_SOURCES and normalized not in selected:
            selected.append(normalized)
    return selected or list(DEFAULT_RESEARCH_SOURCES)


def _select_high_information_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for item in items:
        text = str(item.get("text") or item.get("title") or "").strip()
        if not text or _is_noise(text):
            continue
        key = re.sub(r"\s+", "", text)[:80]
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return sorted(selected, key=lambda item: float(item.get("information_score") or 0), reverse=True)[:limit]


def _normalize_item(
    *,
    source: str,
    title: str,
    text: str,
    symbol: str,
    url: str,
    author: str,
    like: Any,
    reply_count: Any,
    timestamp: Any,
) -> dict[str, Any]:
    text = _clean_html(text or title)
    title = _clean_html(title or text[:80])
    return {
        "source": source,
        "title": title[:160],
        "text": text[:1000],
        "symbol": symbol,
        "url": url,
        "author": author,
        "like": _safe_int(like),
        "reply_count": _safe_int(reply_count),
        "timestamp": _safe_int(timestamp),
        "sentiment": _classify_sentiment(f"{title} {text}"),
        "information_score": _information_score(text, like=like, reply_count=reply_count),
    }


def _information_score(text: str, *, like: Any = 0, reply_count: Any = 0) -> float:
    score = min(40, len(text) / 8)
    score += min(20, _safe_int(like) / 5)
    score += min(20, _safe_int(reply_count) / 2)
    if any(term in text for term in _BULLISH_TERMS + _BEARISH_TERMS):
        score += 15
    if re.search(r"\d+(\.\d+)?%|\d{4,}|PE|ROE|利润|营收|估值|现金流|订单|库存", text, re.I):
        score += 20
    return round(min(100.0, score), 2)


def _classify_sentiment(text: str) -> str:
    bullish = sum(1 for term in _BULLISH_TERMS if term in text)
    bearish = sum(1 for term in _BEARISH_TERMS if term in text)
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def _is_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 8:
        return True
    if compact in _NOISE_PATTERNS:
        return True
    return any(compact == pattern * max(1, len(compact) // len(pattern)) for pattern in _NOISE_PATTERNS)


def _clean_html(value: Any) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _first_group(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.S)
    return match.group(1) if match else ""


def _eastmoney_guba_code(symbol: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z.]", "", symbol).upper()
    match = re.search(r"(\d{6})", cleaned)
    return match.group(1) if match else ""


def _looks_like_symbol(value: str) -> bool:
    return bool(re.search(r"(^|[A-Z])\d{5,6}|^[A-Z]{1,6}$", value.upper()))


def _eastmoney_post_url(row: dict[str, Any], code: str) -> str:
    post_id = row.get("post_id") or row.get("article_id") or row.get("id")
    if post_id:
        return f"https://guba.eastmoney.com/news,{code},{post_id}.html" if code else f"https://guba.eastmoney.com/news,{post_id}.html"
    return f"https://guba.eastmoney.com/list,{code}.html" if code else "https://guba.eastmoney.com/"


def _xueqiu_url(row: dict[str, Any]) -> str:
    status_id = row.get("id") or row.get("status_id")
    user = row.get("user") if isinstance(row.get("user"), dict) else {}
    uid = user.get("id") or row.get("user_id")
    if status_id and uid:
        return f"https://xueqiu.com/{uid}/{status_id}"
    if status_id:
        return f"https://xueqiu.com/statuses/{status_id}"
    return "https://xueqiu.com/"


def _cls_url(row: dict[str, Any]) -> str:
    item_id = row.get("id") or row.get("telegraph_id")
    return f"https://www.cls.cn/detail/{item_id}" if item_id else "https://www.cls.cn/telegraph"


def _normalize_bilibili_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def _parse_time(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        number = int(value)
        return number // 1000 if number > 10_000_000_000 else number
    text = str(value).strip()
    if text.isdigit():
        return _parse_time(int(text))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%a %b %d %H:%M:%S %z %Y"):
        try:
            import datetime as _dt

            return int(_dt.datetime.strptime(text, fmt).timestamp())
        except ValueError:
            continue
    return 0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0
