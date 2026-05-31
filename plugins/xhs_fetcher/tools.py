"""Hermes tool entrypoints for Xiaohongshu content fetching.

The first implementation wraps the open-source ``xhs-kit`` CLI instead of
copying volatile Xiaohongshu signing code into Hermes.  This keeps the Hermes
plugin small and lets operators update the fetcher independently when XHS
changes its web API.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


XHS_CHECK_STATUS_SCHEMA = {
    "name": "xhs_check_status",
    "description": "Check local xhs-kit installation and cookie configuration.",
    "parameters": {
        "type": "object",
        "properties": {
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds. Defaults to 10.",
                "default": 10,
            }
        },
    },
}

XHS_SEARCH_NOTES_SCHEMA = {
    "name": "xhs_search_notes",
    "description": "Search Xiaohongshu notes through xhs-kit.",
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Search keyword.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum notes to return when xhs-kit supports limiting.",
                "default": 10,
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds. Defaults to 60.",
                "default": 60,
            },
        },
        "required": ["keyword"],
    },
}

XHS_FETCH_NOTE_SCHEMA = {
    "name": "xhs_fetch_note",
    "description": "Fetch a Xiaohongshu note detail through xhs-kit.",
    "parameters": {
        "type": "object",
        "properties": {
            "url_or_feed_id": {
                "type": "string",
                "description": "Xiaohongshu note URL, xhslink URL, or feed/note id.",
            },
            "xsec_token": {
                "type": "string",
                "description": "Optional xsec_token from search results or URL query.",
            },
            "include_comments": {
                "type": "boolean",
                "description": "Ask xhs-kit to include comments when supported.",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds. Defaults to 60.",
                "default": 60,
            },
        },
        "required": ["url_or_feed_id"],
    },
}

XHS_FETCH_COMMENTS_SCHEMA = {
    "name": "xhs_fetch_comments",
    "description": "Fetch Xiaohongshu note comments through xhs-kit.",
    "parameters": {
        "type": "object",
        "properties": {
            "url_or_feed_id": {
                "type": "string",
                "description": "Xiaohongshu note URL, xhslink URL, or feed/note id.",
            },
            "xsec_token": {
                "type": "string",
                "description": "Optional xsec_token from search results or URL query.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum comments to return after parsing.",
                "default": 50,
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds. Defaults to 60.",
                "default": 60,
            },
        },
        "required": ["url_or_feed_id"],
    },
}


def check_requirements() -> bool:
    # Keep the tools visible.  Handlers return actionable setup errors when
    # xhs-kit is missing, which is more useful than hiding the toolset.
    return True


def handle_check_status(args: dict[str, Any], **_kwargs) -> str:
    timeout = _int_arg(args, "timeout", 10)
    payload = _status_payload()
    if payload["xhs_kit_found"]:
        result = _run_xhs(["status"], timeout=timeout, allow_missing=True)
        payload["login_status_ok"] = result["success"]
        payload["login_status_output"] = _trim(result.get("stdout") or result.get("stderr") or "", 2000)
    return _json({"success": True, **payload})


def handle_search_notes(args: dict[str, Any], **_kwargs) -> str:
    keyword = str(args.get("keyword") or "").strip()
    if not keyword:
        return _json({"success": False, "error": "Missing keyword."})
    limit = _int_arg(args, "limit", 10)
    timeout = _int_arg(args, "timeout", 60)

    sdk_result = _run_sdk("search", timeout=timeout, keyword=keyword, limit=limit)
    if sdk_result is not None:
        return _json(sdk_result)

    # xhs-kit currently documents `search -k <keyword>`.  We pass common limit
    # aliases optimistically; if the installed version rejects them, retry the
    # minimal documented form.
    result = _run_xhs(["search", "-k", keyword, "--limit", str(limit)], timeout=timeout)
    if _looks_like_bad_option(result):
        result = _run_xhs(["search", "-k", keyword], timeout=timeout)
    return _json(_normalize_cli_result(result, operation="search", limit=limit))


def handle_fetch_note(args: dict[str, Any], **_kwargs) -> str:
    url_or_feed_id = str(args.get("url_or_feed_id") or "").strip()
    if not url_or_feed_id:
        return _json({"success": False, "error": "Missing url_or_feed_id."})
    timeout = _int_arg(args, "timeout", 60)
    include_comments = bool(args.get("include_comments", False))
    feed_id, xsec_token = _extract_note_ref(url_or_feed_id, str(args.get("xsec_token") or ""))
    if not feed_id:
        return _json(
            {
                "success": False,
                "error": "Could not extract a Xiaohongshu feed/note id.",
                "hint": "Pass a note URL like https://www.xiaohongshu.com/explore/<id> or the feed_id from search results.",
            }
        )
    if not xsec_token:
        return _json(
            {
                "success": False,
                "error": "Missing xsec_token. xhs-kit requires --xsec-token for note detail.",
                "hint": "Use xhs_search_notes first and pass the returned xsec_token, or provide a note URL that includes xsec_token in the query string.",
                "feed_id": feed_id,
            }
        )

    sdk_result = _run_sdk(
        "detail",
        timeout=timeout,
        feed_id=feed_id,
        xsec_token=xsec_token,
        include_comments=include_comments,
    )
    if sdk_result is not None:
        return _json(sdk_result)

    command = ["detail", "--feed-id", feed_id]
    command.extend(["--xsec-token", xsec_token])
    if include_comments:
        command.append("--load-comments")
    result = _run_xhs(command, timeout=timeout)
    return _json(
        {
            **_normalize_cli_result(result, operation="detail"),
            "feed_id": feed_id,
            "xsec_token_present": bool(xsec_token),
        }
    )


def handle_fetch_comments(args: dict[str, Any], **_kwargs) -> str:
    note_result = json.loads(
        handle_fetch_note(
            {
                "url_or_feed_id": args.get("url_or_feed_id"),
                "xsec_token": args.get("xsec_token"),
                "include_comments": True,
                "timeout": args.get("timeout", 60),
            }
        )
    )
    if not note_result.get("success"):
        return _json(note_result)
    limit = _int_arg(args, "limit", 50)
    comments = _find_comments(note_result.get("data"))
    if comments is None:
        return _json(
            {
                "success": True,
                "operation": "comments",
                "comments": [],
                "comment_count": 0,
                "warning": "xhs-kit returned data, but no comments field was recognized.",
                "raw": note_result.get("data") or note_result.get("stdout") or "",
            }
        )
    return _json(
        {
            "success": True,
            "operation": "comments",
            "comments": comments[:limit],
            "comment_count": len(comments),
            "limit": limit,
            "feed_id": note_result.get("feed_id"),
        }
    )


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _int_arg(args: dict[str, Any], name: str, default: int) -> int:
    try:
        value = int(args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _xhs_bin_parts() -> list[str]:
    raw = os.getenv("XHS_KIT_BIN", "xhs-kit").strip() or "xhs-kit"
    return shlex.split(raw)


def _resolved_xhs_parts() -> tuple[list[str], str]:
    parts = _xhs_bin_parts()
    executable = parts[0]
    found = shutil.which(executable)
    if found:
        return [found] + parts[1:], found
    if len(parts) == 1:
        sibling = Path(sys.executable).parent / executable
        if sibling.exists():
            return [str(sibling)], str(sibling)
    return parts, ""


def _cookie_paths() -> list[str]:
    values = []
    for name in ("XHS_COOKIES_PATH", "COOKIES_PATH"):
        value = os.getenv(name, "").strip()
        if value:
            values.append(value)
    default = Path.home() / ".xhs-kit" / "cookies.json"
    values.append(str(default))
    return values


def _browser_bin() -> str:
    configured = os.getenv("XHS_BROWSER_BIN", "").strip()
    if configured:
        return configured
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return ""


def _status_payload() -> dict[str, Any]:
    parts, found_path = _resolved_xhs_parts()
    browser_bin = _browser_bin()
    cookie_paths = _cookie_paths()
    return {
        "xhs_kit_bin": " ".join(parts),
        "xhs_kit_found": bool(found_path),
        "xhs_kit_path": found_path or "",
        "cookie_env_present": bool(os.getenv("XHS_COOKIES_PATH") or os.getenv("COOKIES_PATH") or os.getenv("XHS_COOKIE")),
        "cookie_paths": [
            {"path": path, "exists": Path(path).expanduser().exists()} for path in cookie_paths
        ],
        "browser_bin": browser_bin,
        "browser_bin_found": bool(browser_bin),
        "install_hint": "Install with: python -m pip install xhs-kit",
        "browser_hint": "If Playwright Chromium is not installed, set XHS_BROWSER_BIN to a local Chrome/Chromium executable.",
        "config_hint": "Set XHS_COOKIES_PATH or follow xhs-kit's login flow before fetching authenticated content.",
    }


def _run_sdk(operation: str, *, timeout: int, **kwargs: Any) -> dict[str, Any] | None:
    try:
        from xhs_kit.po.client import XhsClient
    except Exception:
        return None

    browser_bin = _browser_bin()

    async def _main() -> dict[str, Any]:
        async with XhsClient(headless=True, bin_path=browser_bin or None) as client:
            if operation == "search":
                result = await client.search(kwargs["keyword"])
                feeds = [_model_to_dict(feed) for feed in result.feeds]
                limit = int(kwargs.get("limit") or len(feeds))
                return {
                    "success": True,
                    "operation": "search",
                    "data": feeds[:limit],
                    "result_count": result.count,
                    "limit": limit,
                    "browser_bin": browser_bin,
                }
            if operation == "detail":
                data = await client.get_feed_detail(
                    kwargs["feed_id"],
                    kwargs["xsec_token"],
                    bool(kwargs.get("include_comments", False)),
                )
                return {
                    "success": True,
                    "operation": "detail",
                    "data": _model_to_dict(data),
                    "feed_id": kwargs["feed_id"],
                    "xsec_token_present": True,
                    "browser_bin": browser_bin,
                }
        return {"success": False, "operation": operation, "error": "Unsupported SDK operation."}

    try:
        return asyncio.run(asyncio.wait_for(_main(), timeout=timeout))
    except Exception as exc:
        return {
            "success": False,
            "operation": operation,
            "error": str(exc),
            "setup": _status_payload(),
        }


def _model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _model_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_model_to_dict(item) for item in value]
    return value


def _run_xhs(args: list[str], *, timeout: int, allow_missing: bool = False) -> dict[str, Any]:
    parts, found_path = _resolved_xhs_parts()
    executable = parts[0]
    if not found_path:
        return {
            "success": False,
            "exit_code": 127,
            "error": f"xhs-kit executable not found: {executable}",
            **_status_payload(),
        }

    env = os.environ.copy()
    if os.getenv("XHS_COOKIES_PATH") and not env.get("COOKIES_PATH"):
        env["COOKIES_PATH"] = os.getenv("XHS_COOKIES_PATH", "")

    command = parts + args
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "exit_code": -1,
            "error": f"xhs-kit timed out after {timeout}s",
            "stdout": _trim(exc.stdout or "", 4000),
            "stderr": _trim(exc.stderr or "", 4000),
            "command": _redacted_command(command),
        }
    except OSError as exc:
        if allow_missing:
            return {"success": False, "exit_code": 127, "error": str(exc)}
        return {
            "success": False,
            "exit_code": 127,
            "error": str(exc),
            "command": _redacted_command(command),
        }

    return {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": _redacted_command(command),
    }


def _normalize_cli_result(result: dict[str, Any], *, operation: str, limit: int | None = None) -> dict[str, Any]:
    if not result.get("success"):
        return {
            "success": False,
            "operation": operation,
            "exit_code": result.get("exit_code"),
            "error": result.get("error") or _trim(result.get("stderr") or result.get("stdout") or "", 4000),
            "stderr": _trim(result.get("stderr") or "", 4000),
            "stdout": _trim(result.get("stdout") or "", 4000),
            "command": result.get("command", []),
            "setup": _status_payload(),
        }

    parsed = _parse_jsonish(result.get("stdout") or "")
    payload = {
        "success": True,
        "operation": operation,
        "exit_code": result.get("exit_code"),
        "command": result.get("command", []),
        "data": parsed,
        "stdout": "" if parsed is not None else _trim(result.get("stdout") or "", 12000),
        "stderr": _trim(result.get("stderr") or "", 4000),
    }
    if limit is not None and isinstance(parsed, list):
        payload["data"] = parsed[:limit]
        payload["result_count"] = len(parsed)
        payload["limit"] = limit
    return payload


def _parse_jsonish(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1])
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        candidates.append(text[start_arr : end_arr + 1])
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _extract_note_ref(value: str, xsec_token: str) -> tuple[str, str]:
    value = value.strip()
    token = xsec_token.strip()
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        query = parse_qs(parsed.query)
        token = token or _first_query(query, "xsec_token", "xsecToken")
        for key in ("feed_id", "note_id", "item_id", "id"):
            candidate = _first_query(query, key)
            if candidate:
                return candidate, token
        match = re.search(r"/(?:explore|discovery/item|user/profile/[^/]+/note)/([0-9a-fA-F]{16,32})", parsed.path)
        if match:
            return match.group(1), token
        match = re.search(r"([0-9a-fA-F]{24})", parsed.path)
        if match:
            return match.group(1), token
        return "", token
    return value, token


def _first_query(query: dict[str, list[str]], *names: str) -> str:
    for name in names:
        values = query.get(name) or []
        if values:
            return values[0]
    return ""


def _find_comments(data: Any) -> list[Any] | None:
    if isinstance(data, dict):
        for key in ("comments", "comment_list", "commentList"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        for value in data.values():
            found = _find_comments(value)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_comments(item)
            if found is not None:
                return found
    return None


def _looks_like_bad_option(result: dict[str, Any]) -> bool:
    if result.get("success"):
        return False
    text = f"{result.get('stderr') or ''}\n{result.get('stdout') or ''}".lower()
    return "no such option" in text or "unrecognized" in text or "unknown option" in text


def _trim(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _redacted_command(command: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for part in command:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        lowered = part.lower()
        redacted.append(part)
        if lowered in {"--xsec-token", "--cookie", "--cookies", "--token"}:
            skip_next = True
    return redacted
