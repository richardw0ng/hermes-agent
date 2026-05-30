"""Bilibili metadata, subtitle, audio fallback, and cache helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from http.cookiejar import Cookie, CookieJar

from hermes_constants import get_hermes_home

BV_RE = re.compile(r"(BV[0-9A-Za-z]{10,})")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


class BilibiliError(RuntimeError):
    pass


def cache_dir() -> Path:
    path = get_hermes_home() / "cache" / "bilibili_analyzer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_bvid(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise BilibiliError("Missing Bilibili URL or BV id.")
    match = BV_RE.search(text)
    if match:
        return match.group(1)
    if "b23.tv" in text:
        expanded = expand_url(text)
        match = BV_RE.search(expanded)
        if match:
            return match.group(1)
    raise BilibiliError(f"Could not find a BV id in: {value}")


def expand_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    try:
        with opener.open(request, timeout=15) as response:
            return response.geturl()
    except urllib.error.URLError as exc:
        raise BilibiliError(f"Could not expand short URL: {exc}") from exc


def _cookie_header() -> str:
    return "; ".join(
        f"{name}={value}" for name, value in _bilibili_cookie_values().items()
    )


def _bilibili_cookie_values() -> dict[str, str]:
    sessdata = os.getenv("BILIBILI_SESSDATA", "").strip()
    buvid3 = os.getenv("BILIBILI_BUVID3", "").strip()
    hardcoded = _load_hardcoded_cookies()
    file_cookies = _load_cookie_file()
    if not sessdata:
        sessdata = str(
            hardcoded.get("SESSDATA")
            or hardcoded.get("sessdata")
            or file_cookies.get("SESSDATA")
            or file_cookies.get("sessdata")
            or ""
        ).strip()
    if not buvid3:
        buvid3 = str(
            hardcoded.get("BUVID3")
            or hardcoded.get("buvid3")
            or file_cookies.get("BUVID3")
            or file_cookies.get("buvid3")
            or ""
        ).strip()
    result = {}
    if sessdata:
        result["SESSDATA"] = sessdata
    if buvid3:
        result["buvid3"] = buvid3
    return result


def _load_cookie_file() -> dict[str, str]:
    """Load optional Bilibili cookies from a private JSON file.

    Env vars stay the preferred source. The file hook exists for users who run
    the same private checkout across machines and want to sync credentials as a
    repo-local secret file instead of editing every machine's environment.
    """
    path_value = os.getenv("BILIBILI_COOKIE_FILE", "").strip()
    candidates = [Path(path_value)] if path_value else []
    candidates.append(Path.cwd() / ".hermes" / "bilibili_cookies.json")

    for path in candidates:
        try:
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): str(v) for k, v in payload.items()}
        except Exception:
            continue
    return {}


def _load_hardcoded_cookies() -> dict[str, str]:
    """Load optional repo-synced Python constants for private deployments."""
    try:
        from . import private_cookies  # type: ignore
    except Exception:
        return {}

    result = {}
    sessdata = str(getattr(private_cookies, "SESSDATA", "") or "").strip()
    buvid3 = str(getattr(private_cookies, "BUVID3", "") or "").strip()
    if sessdata:
        result["SESSDATA"] = sessdata
    if buvid3:
        result["BUVID3"] = buvid3
    return result


def _get_json(url: str) -> dict[str, Any]:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.bilibili.com/",
    }
    cookie = _cookie_header()
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise BilibiliError(f"Bilibili HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BilibiliError(f"Bilibili request failed: {exc}") from exc


def fetch_video_info(bvid: str) -> dict[str, Any]:
    url = "https://api.bilibili.com/x/web-interface/view?" + urllib.parse.urlencode(
        {"bvid": bvid}
    )
    payload = _get_json(url)
    if payload.get("code") != 0:
        raise BilibiliError(payload.get("message") or f"Could not fetch {bvid}.")
    data = payload.get("data") or {}
    pages = data.get("pages") or []
    if not pages:
        raise BilibiliError(f"No playable pages found for {bvid}.")
    return {
        "bvid": bvid,
        "aid": data.get("aid"),
        "title": data.get("title") or "",
        "owner": (data.get("owner") or {}).get("name") or "",
        "desc": data.get("desc") or "",
        "duration": data.get("duration") or 0,
        "pages": [
            {
                "cid": page.get("cid"),
                "page": page.get("page"),
                "part": page.get("part") or "",
                "duration": page.get("duration") or 0,
            }
            for page in pages
        ],
    }


def _subtitle_candidates(bvid: str, cid: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"bvid": bvid, "cid": cid})
    payload = _get_json(f"https://api.bilibili.com/x/player/v2?{query}")
    if payload.get("code") != 0:
        raise BilibiliError(payload.get("message") or "Could not fetch subtitle list.")
    subtitle = ((payload.get("data") or {}).get("subtitle") or {})
    return subtitle.get("subtitles") or []


def _normalize_subtitle_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def fetch_subtitle_segments(bvid: str, cid: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    for candidate in _subtitle_candidates(bvid, cid):
        url = candidate.get("subtitle_url") or candidate.get("url") or ""
        if not url:
            continue
        payload = _get_json(_normalize_subtitle_url(url))
        body = payload.get("body") or []
        segments = []
        for item in body:
            text = str(item.get("content") or "").strip()
            if not text:
                continue
            start = float(item.get("from") or 0)
            end = float(item.get("to") or start)
            segments.append({"start": start, "end": end, "text": text})
        if segments:
            source_meta = dict(candidate)
            source_meta["fetcher"] = "bilibili_api"
            return segments, source_meta
    return [], None


def fetch_subtitle_segments_ytdlp(
    bvid: str, cookies_file: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Fetch subtitles through yt-dlp's Bilibili extractor when available.

    yt-dlp is a mature open-source downloader with a maintained Bilibili
    extractor. We keep this optional so the plugin still works in minimal
    installs, but use it as a second opinion when Bilibili's raw subtitle API
    is missing or suspicious.
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        return [], None

    options: dict[str, Any] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    cookiejar = _bilibili_cookiejar()
    if cookiejar:
        options["cookiejar"] = cookiejar
    if cookies_file:
        options["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"https://www.bilibili.com/video/{bvid}/", download=False)
    except Exception:
        return [], None

    subtitle_sets = [
        ("subtitles", info.get("subtitles") or {}),
        ("automatic_captions", info.get("automatic_captions") or {}),
    ]
    for set_name, subtitles in subtitle_sets:
        if not isinstance(subtitles, dict):
            continue
        for lang, entries in subtitles.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                segments = _segments_from_ytdlp_subtitle_entry(entry)
                if segments:
                    return segments, {
                        "fetcher": "yt_dlp",
                        "set": set_name,
                        "lang": lang,
                        "ext": entry.get("ext") if isinstance(entry, dict) else "",
                    }
    return [], None


def _segments_from_ytdlp_subtitle_entry(entry: Any) -> list[dict[str, Any]]:
    if not isinstance(entry, dict):
        return []
    data = entry.get("data")
    if not data:
        url = entry.get("url")
        if not url:
            return []
        try:
            data = _get_text(_normalize_subtitle_url(str(url)))
        except Exception:
            return []
    text = str(data)
    ext = str(entry.get("ext") or "").lower()
    if ext == "json" or text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _segments_from_bilibili_subtitle_json(payload)
    if ext in {"vtt", "srt"} or "-->" in text:
        return _segments_from_timed_text(text)
    return []


def _segments_from_bilibili_subtitle_json(payload: dict[str, Any]) -> list[dict[str, Any]]:
    body = payload.get("body") if isinstance(payload, dict) else None
    if not isinstance(body, list):
        return []
    segments = []
    for item in body:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        start = float(item.get("from") or 0)
        end = float(item.get("to") or start)
        segments.append({"start": start, "end": end, "text": content})
    return segments


def _segments_from_timed_text(text: str) -> list[dict[str, Any]]:
    segments = []
    blocks = re.split(r"\r?\n\r?\n+", text.replace("\ufeff", "").strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_idx < 0:
            continue
        start_text, end_text = lines[timing_idx].split("-->", 1)
        start = _parse_subtitle_timestamp(start_text.strip())
        end = _parse_subtitle_timestamp(end_text.split()[0].strip())
        content = " ".join(lines[timing_idx + 1 :]).strip()
        if content:
            segments.append({"start": start, "end": end, "text": content})
    return segments


def _parse_subtitle_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])
    except (TypeError, ValueError):
        return 0.0


def _get_text(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.bilibili.com/",
    }
    cookie = _cookie_header()
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def transcript_cache_path(bvid: str, cid: int) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", f"{bvid}_{cid}.json")
    return cache_dir() / safe


def load_cached_transcript(bvid: str, cid: int, max_age_days: int = 30) -> dict[str, Any] | None:
    path = transcript_cache_path(bvid, cid)
    if not path.exists():
        return None
    return _load_cache_file(path, max_age_days=max_age_days)


def load_cached_transcript_for_page(
    bvid: str, page: int = 1, max_age_days: int = 30
) -> dict[str, Any] | None:
    for path in sorted(cache_dir().glob(f"{bvid}_*.json")):
        payload = _load_cache_file(path, max_age_days=max_age_days)
        if not payload:
            continue
        metadata = payload.get("metadata") or {}
        if int(metadata.get("page") or 1) == int(page):
            return payload
    return None


def _load_cache_file(path: Path, max_age_days: int = 30) -> dict[str, Any] | None:
    if max_age_days > 0:
        age = time.time() - path.stat().st_mtime
        if age > max_age_days * 86400:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def save_cached_transcript(payload: dict[str, Any]) -> None:
    meta = payload.get("metadata") or {}
    bvid = meta.get("bvid")
    cid = meta.get("cid")
    if not bvid or not cid:
        return
    transcript_cache_path(str(bvid), int(cid)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_audio(bvid: str, cid: int | None = None) -> Path:
    outtmpl = str(cache_dir() / f"{bvid}_{cid or 'audio'}.%(ext)s")
    url = f"https://www.bilibili.com/video/{bvid}"
    if cid:
        url = f"{url}?p=1"

    try:
        import yt_dlp  # type: ignore

        options = {
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "noplaylist": True,
        }
        cookiejar = _bilibili_cookiejar()
        if cookiejar:
            options["cookiejar"] = cookiejar
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except ImportError:
        command = [
            "yt-dlp",
            "-f",
            "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "-o",
            outtmpl,
            url,
        ]
        cookie_header = _cookie_header()
        if cookie_header:
            command[1:1] = ["--add-header", f"Cookie:{cookie_header}"]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
        if completed.returncode != 0:
            raise BilibiliError(
                "yt-dlp audio download failed: "
                + (completed.stderr or completed.stdout)[-1000:]
            )

    matches = sorted(cache_dir().glob(f"{bvid}_{cid or 'audio'}.*"))
    audio = next((p for p in matches if p.suffix.lower() in {".m4a", ".mp3", ".wav", ".webm"}), None)
    if not audio:
        raise BilibiliError("Audio download completed but no audio file was found.")
    return audio


def _bilibili_cookiejar() -> CookieJar | None:
    values = _bilibili_cookie_values()
    if not values:
        return None
    jar = CookieJar()
    for name, value in values.items():
        jar.set_cookie(
            Cookie(
                version=0,
                name=name,
                value=value,
                port=None,
                port_specified=False,
                domain=".bilibili.com",
                domain_specified=True,
                domain_initial_dot=True,
                path="/",
                path_specified=True,
                secure=False,
                expires=None,
                discard=True,
                comment=None,
                comment_url=None,
                rest={},
                rfc2109=False,
            )
        )
    return jar


def format_ts(seconds: float | int) -> str:
    total = max(0, int(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
