"""Hermes tool entrypoints for the Bilibili analyzer plugin."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .analysis import analyze_market_batch, analyze_transcript, transcript_to_text
from .asr import has_asr_provider, transcribe_audio
from .backtest import (
    analyze_post_verification_comments,
    combine_price_and_comment_scores,
    extract_stock_picks_from_transcript,
    persist_creator_backtest_report,
    score_pick_against_prices,
)
from .pdf import write_markdown_pdf
from .bilibili import (
    BilibiliError,
    cache_dir,
    download_audio,
    extract_bvid,
    fetch_follow_group_users,
    fetch_video_comments,
    fetch_subtitle_segments,
    fetch_subtitle_segments_ytdlp,
    fetch_user_archive_videos,
    fetch_video_info,
    fetch_user_latest_videos,
    load_cached_comments,
    load_cached_transcript,
    load_cached_transcript_for_page,
    save_cached_comments,
    save_cached_transcript,
)

CACHE_SCHEMA_VERSION = 4
logger = logging.getLogger(__name__)


class _ProgressReporter:
    def __init__(
        self,
        progress_callback: Any = None,
        *,
        label: str = "bilibili_analyzer",
        batch_dir: Path | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.label = label
        self.batch_dir = batch_dir
        self.events: list[dict[str, Any]] = []
        self.current: dict[str, Any] = {}

    @property
    def path(self) -> Path | None:
        if not self.batch_dir:
            return None
        return self.batch_dir / "progress.json"

    def set_batch_dir(self, batch_dir: Path) -> None:
        self.batch_dir = batch_dir

    def emit(self, stage: str, message: str, **fields: Any) -> None:
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "message": message,
            **fields,
        }
        self.current = event
        self.events.append(event)
        self.events = self.events[-200:]
        logger.info("bilibili_analyzer progress [%s]: %s", stage, message)
        self._emit_callback(event)
        self._write_file()

    def _emit_callback(self, event: dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback("tool.progress", self.label, event["message"], event)
        except Exception as exc:
            logger.debug("bilibili progress callback failed: %s", exc)

    def _write_file(self) -> None:
        path = self.path
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "label": self.label,
                "current": self.current,
                "events": self.events,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("failed to persist bilibili progress: %s", exc)

BILI_FETCH_TRANSCRIPT_SCHEMA = {
    "name": "bilibili_fetch_transcript",
    "description": (
        "Fetch a Bilibili video's transcript. Uses Bilibili subtitles first, "
        "then downloads audio and runs configured ASR when subtitles are absent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url_or_bvid": {
                "type": "string",
                "description": "Bilibili URL, b23.tv short link, or BV id.",
            },
            "page": {
                "type": "integer",
                "description": "Video page number for multi-part videos. Defaults to 1.",
                "default": 1,
            },
            "prefer_subtitle": {
                "type": "boolean",
                "description": "Try Bilibili subtitles before ASR. Defaults to true.",
                "default": True,
            },
            "asr_provider": {
                "type": "string",
                "description": "ASR provider: auto, command, or faster_whisper.",
                "default": "auto",
            },
            "use_cache": {
                "type": "boolean",
                "description": "Reuse cached transcript for the same BV/cid. Defaults to true.",
                "default": True,
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Ignore cached transcripts and fetch again.",
                "default": False,
            },
            "force_asr": {
                "type": "boolean",
                "description": "Skip Bilibili subtitles and transcribe downloaded audio.",
                "default": False,
            },
            "allow_suspicious_subtitle": {
                "type": "boolean",
                "description": (
                    "Return suspicious/incomplete Bilibili subtitles instead of failing. "
                    "Defaults to false because mismatched AI subtitles can poison analysis."
                ),
                "default": False,
            },
        },
        "required": ["url_or_bvid"],
    },
}

BILI_ANALYZE_VIDEO_SCHEMA = {
    "name": "bilibili_analyze_video",
    "description": (
        "Analyze a Bilibili video/audio transcript and summarize the video's "
        "main thesis, viewpoints, evidence, controversies, facts, and timestamps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url_or_bvid": {
                "type": "string",
                "description": "Bilibili URL, b23.tv short link, or BV id.",
            },
            "page": {"type": "integer", "description": "Video page number.", "default": 1},
            "output_format": {
                "type": "string",
                "enum": ["markdown", "json", "pdf"],
                "description": "Return markdown plus JSON fields, JSON only, or archive a PDF alongside markdown.",
                "default": "markdown",
            },
            "chunk_minutes": {
                "type": "integer",
                "description": "Transcript chunk size for first-pass summaries.",
                "default": 4,
            },
            "llm_timeout_seconds": {
                "type": "number",
                "description": "Per-attempt timeout for each analysis LLM call. Defaults to 20.",
                "default": 20,
            },
            "asr_provider": {
                "type": "string",
                "description": "ASR provider: auto, command, or faster_whisper.",
                "default": "auto",
            },
            "use_cache": {
                "type": "boolean",
                "description": "Reuse cached transcript for the same BV/cid. Defaults to true.",
                "default": True,
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Ignore cached transcripts and fetch again.",
                "default": False,
            },
            "force_asr": {
                "type": "boolean",
                "description": "Skip Bilibili subtitles and transcribe downloaded audio.",
                "default": False,
            },
            "allow_suspicious_subtitle": {
                "type": "boolean",
                "description": "Analyze suspicious/incomplete subtitles anyway. Defaults to false.",
                "default": False,
            },
            "persist_file": {
                "type": "boolean",
                "description": (
                    "Archive transcript and analysis to the plugin's standard output "
                    "directory. Defaults to true."
                ),
                "default": True,
            },
            "include_comments": {
                "type": "boolean",
                "description": "Fetch and include high-information comments in analysis.",
                "default": False,
            },
            "max_comments": {
                "type": "integer",
                "description": "Maximum raw comments to fetch when include_comments is true.",
                "default": 200,
            },
            "comment_limit": {
                "type": "integer",
                "description": "Maximum filtered comments to pass into analysis.",
                "default": 30,
            },
            "include_comment_replies": {
                "type": "boolean",
                "description": "Fetch a small number of nested replies for comment analysis.",
                "default": False,
            },
            "comment_sort": {
                "type": "string",
                "description": "Comment sort: hot or time. Defaults to hot.",
                "default": "hot",
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional output markdown path. If omitted, the plugin writes to "
                    "outputs/bilibili_analyzer automatically."
                ),
            },
        },
        "required": ["url_or_bvid"],
    },
}

BILI_FETCH_COMMENTS_SCHEMA = {
    "name": "bilibili_fetch_comments",
    "description": (
        "Fetch and filter high-information comments from a Bilibili video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url_or_bvid": {
                "type": "string",
                "description": "Bilibili URL, b23.tv short link, or BV id.",
            },
            "max_comments": {
                "type": "integer",
                "description": "Maximum raw comments to fetch. Defaults to 200.",
                "default": 200,
            },
            "comment_limit": {
                "type": "integer",
                "description": "Maximum filtered comments to return. Defaults to 30.",
                "default": 30,
            },
            "sort": {
                "type": "string",
                "description": "Comment sort: hot or time. Defaults to hot.",
                "default": "hot",
            },
            "include_replies": {
                "type": "boolean",
                "description": "Fetch a small number of nested replies for informative top comments.",
                "default": False,
            },
            "use_cache": {
                "type": "boolean",
                "description": "Reuse cached comments. Defaults to true.",
                "default": True,
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Ignore cached comments and fetch again.",
                "default": False,
            },
        },
        "required": ["url_or_bvid"],
    },
}

BILI_ANALYZE_FOLLOWING_GROUP_LATEST_SCHEMA = {
    "name": "bilibili_analyze_following_group_latest",
    "description": (
        "Analyze and archive the latest videos from Bilibili creators in one "
        "of the authenticated user's follow groups, such as 鎶曡祫."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "group_name": {
                "type": "string",
                "description": "Follow group name. Defaults to 鎶曡祫.",
                "default": "鎶曡祫",
            },
            "tagid": {
                "type": "integer",
                "description": "Optional follow group tag id. Overrides group_name when set.",
            },
            "per_up_limit": {
                "type": "integer",
                "description": "Latest videos to fetch per followed creator. Defaults to 10.",
                "default": 10,
            },
            "max_up": {
                "type": "integer",
                "description": "Maximum creators to process. 0 means all creators in the group.",
                "default": 0,
            },
            "max_videos_total": {
                "type": "integer",
                "description": "Maximum total videos to analyze. 0 means no total cap.",
                "default": 0,
            },
            "days_back": {
                "type": "integer",
                "description": (
                    "Only analyze videos published within this many days. "
                    "Defaults to 7 for weekly reports; 0 disables date filtering."
                ),
                "default": 7,
            },
            "output_dir": {
                "type": "string",
                "description": (
                    "Optional batch archive directory. If omitted, writes under "
                    "outputs/bilibili_analyzer/following/<timestamp>_<group>."
                ),
            },
            "chunk_minutes": {
                "type": "integer",
                "description": "Transcript chunk size for first-pass summaries.",
                "default": 4,
            },
            "llm_timeout_seconds": {
                "type": "number",
                "description": "Per-attempt timeout for each analysis LLM call. Defaults to 20.",
                "default": 20,
            },
            "asr_provider": {
                "type": "string",
                "description": "ASR provider: auto, command, or faster_whisper.",
                "default": "auto",
            },
            "force_asr": {
                "type": "boolean",
                "description": "Skip Bilibili subtitles and transcribe downloaded audio.",
                "default": False,
            },
            "allow_suspicious_subtitle": {
                "type": "boolean",
                "description": "Analyze suspicious/incomplete subtitles anyway. Defaults to false.",
                "default": False,
            },
            "include_comments": {
                "type": "boolean",
                "description": "Fetch and include high-information comments in each video analysis.",
                "default": False,
            },
            "max_comments": {
                "type": "integer",
                "description": "Maximum raw comments to fetch when include_comments is true.",
                "default": 200,
            },
            "comment_limit": {
                "type": "integer",
                "description": "Maximum filtered comments to pass into each analysis.",
                "default": 30,
            },
            "include_comment_replies": {
                "type": "boolean",
                "description": "Fetch a small number of nested replies for comment analysis.",
                "default": True,
            },
            "comment_sort": {
                "type": "string",
                "description": "Comment sort: hot or time. Defaults to hot.",
                "default": "hot",
            },
            "generate_market_report": {
                "type": "boolean",
                "description": "Generate one batch-level market sentiment report. Defaults to true.",
                "default": True,
            },
            "market_report_timeout_seconds": {
                "type": "number",
                "description": "Timeout for the batch market report LLM call. Defaults to 90.",
                "default": 90,
            },
            "market_report_max_tokens": {
                "type": "integer",
                "description": "Max output tokens for the batch market report. Defaults to 6000.",
                "default": 6000,
            },
            "use_cache": {
                "type": "boolean",
                "description": "Reuse cached transcripts/comments. Defaults to true.",
                "default": True,
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Ignore cached transcripts/comments and fetch again.",
                "default": False,
            },
            "continue_on_error": {
                "type": "boolean",
                "description": "Keep processing later videos if one video fails. Defaults to true.",
                "default": True,
            },
            "retry_attempts": {
                "type": "integer",
                "description": "Retry attempts for Bilibili API/video operations. Defaults to 3.",
                "default": 3,
            },
            "request_delay_seconds": {
                "type": "number",
                "description": "Delay between videos to reduce Bilibili rate limiting. Defaults to 4.",
                "default": 4,
            },
            "rate_limit_pause_seconds": {
                "type": "number",
                "description": "Extra pause after 412/rate-limit errors. Defaults to 60.",
                "default": 60,
            },
        },
        "required": [],
    },
}

BILI_CREATOR_STOCK_PICK_BACKTEST_SCHEMA = {
    "name": "bilibili_backtest_creator_stock_picks",
    "description": (
        "Backtest a finance creator's historical stock-pick claims by extracting "
        "explicit picks from Bilibili videos and comparing them with later prices."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "up_mid": {
                "type": "integer",
                "description": "Bilibili creator mid. Preferred when known.",
            },
            "up_name": {
                "type": "string",
                "description": "Creator name to find inside a follow group when up_mid is omitted.",
            },
            "group_name": {
                "type": "string",
                "description": "Follow group used to resolve up_name. Defaults to 投资.",
                "default": "投资",
            },
            "start_date": {
                "type": "string",
                "description": "Historical video start date, YYYY-MM-DD.",
            },
            "end_date": {
                "type": "string",
                "description": "Historical video end date, YYYY-MM-DD.",
            },
            "max_videos": {
                "type": "integer",
                "description": "Maximum historical videos to inspect. Defaults to 20.",
                "default": 20,
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum Bilibili archive pages to scan. Defaults to 5.",
                "default": 5,
            },
            "default_horizon_days": {
                "type": "integer",
                "description": "Fallback price evaluation horizon when the video gives none. Defaults to 30.",
                "default": 30,
            },
            "include_post_verification_comments": {
                "type": "boolean",
                "description": "Use latest comments as a post-hoc audience verdict dimension. Defaults to true.",
                "default": True,
            },
            "comment_verdict_min_days_after_publish": {
                "type": "integer",
                "description": "Only treat comments after this many days as post-verification when possible. Defaults to 7.",
                "default": 7,
            },
            "max_verdict_comments": {
                "type": "integer",
                "description": "Maximum latest comments to fetch for audience verdict. Defaults to 100.",
                "default": 100,
            },
            "output_dir": {
                "type": "string",
                "description": "Optional archive directory. If omitted, writes under outputs/bilibili_analyzer/backtests.",
            },
            "llm_timeout_seconds": {
                "type": "number",
                "description": "Timeout for stock-pick extraction LLM calls. Defaults to 45.",
                "default": 45,
            },
            "asr_provider": {
                "type": "string",
                "description": "ASR provider: auto, command, or faster_whisper.",
                "default": "auto",
            },
            "force_asr": {
                "type": "boolean",
                "description": "Skip Bilibili subtitles and transcribe downloaded audio.",
                "default": False,
            },
            "allow_suspicious_subtitle": {
                "type": "boolean",
                "description": "Analyze suspicious/incomplete subtitles anyway. Defaults to false.",
                "default": False,
            },
            "use_cache": {
                "type": "boolean",
                "description": "Reuse cached transcripts. Defaults to true.",
                "default": True,
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Ignore cached transcripts and fetch again.",
                "default": False,
            },
        },
        "required": ["start_date", "end_date"],
    },
}


def check_requirements() -> bool:
    return True


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _bool_arg(args: dict[str, Any], name: str, default: bool) -> bool:
    value = args.get(name, default)
    if value is None:
        return default
    return bool(value)


def _prefer_playurl_audio() -> bool:
    command = os.getenv("BILIBILI_ASR_COMMAND", "")
    return (
        os.getenv("BILIBILI_PREFER_PLAYURL_AUDIO", "").strip().lower()
        in {"1", "true", "yes", "on"}
        or bool(
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("BAILIAN_TOKEN_PLAN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        or "bilibili_asr_bailian.py" in command.replace("\\", "/")
    )


def handle_fetch_transcript(args: dict[str, Any], **_kwargs) -> str:
    try:
        payload = fetch_transcript(
            args.get("url_or_bvid") or "",
            page=int(args.get("page") or 1),
            prefer_subtitle=bool(args.get("prefer_subtitle", True)),
            asr_provider=str(args.get("asr_provider") or "auto"),
            use_cache=bool(args.get("use_cache", True)),
            force_refresh=bool(args.get("force_refresh", False)),
            force_asr=bool(args.get("force_asr", False)),
            allow_suspicious_subtitle=bool(args.get("allow_suspicious_subtitle", False)),
        )
        return _json({"success": True, **payload})
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def handle_fetch_comments(args: dict[str, Any], **_kwargs) -> str:
    try:
        payload = fetch_comments(
            args.get("url_or_bvid") or "",
            max_comments=int(args.get("max_comments") or 200),
            comment_limit=int(args.get("comment_limit") or 30),
            sort=str(args.get("sort") or "hot"),
            include_replies=bool(args.get("include_replies", False)),
            use_cache=bool(args.get("use_cache", True)),
            force_refresh=bool(args.get("force_refresh", False)),
        )
        return _json({"success": True, **payload})
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def make_following_group_latest_handler(llm: Any):
    def _handler(args: dict[str, Any], **_kwargs) -> str:
        try:
            payload = analyze_following_group_latest(
                llm,
                group_name=str(args.get("group_name") or "鎶曡祫"),
                tagid=args.get("tagid"),
                per_up_limit=int(args.get("per_up_limit") or 10),
                max_up=int(args.get("max_up") or 0),
                max_videos_total=int(args.get("max_videos_total") or 0),
                days_back=int(args.get("days_back") if args.get("days_back") is not None else 7),
                output_dir=str(args.get("output_dir") or ""),
                chunk_minutes=int(args.get("chunk_minutes") or 4),
                llm_timeout_seconds=float(args.get("llm_timeout_seconds") if args.get("llm_timeout_seconds") is not None else 20),
                asr_provider=str(args.get("asr_provider") or "auto"),
                force_asr=bool(args.get("force_asr", False)),
                allow_suspicious_subtitle=bool(args.get("allow_suspicious_subtitle", False)),
                include_comments=bool(args.get("include_comments", False)),
                max_comments=int(args.get("max_comments") or 200),
                comment_limit=int(args.get("comment_limit") or 30),
                include_comment_replies=bool(args.get("include_comment_replies", True)),
                comment_sort=str(args.get("comment_sort") or "hot"),
                generate_market_report=bool(args.get("generate_market_report", True)),
                market_report_timeout_seconds=float(args.get("market_report_timeout_seconds") if args.get("market_report_timeout_seconds") is not None else 90),
                market_report_max_tokens=int(args.get("market_report_max_tokens") or 6000),
                use_cache=bool(args.get("use_cache", True)),
                force_refresh=bool(args.get("force_refresh", False)),
                continue_on_error=bool(args.get("continue_on_error", True)),
                retry_attempts=int(args.get("retry_attempts") or 3),
                request_delay_seconds=float(args.get("request_delay_seconds") if args.get("request_delay_seconds") is not None else 4),
                rate_limit_pause_seconds=float(args.get("rate_limit_pause_seconds") if args.get("rate_limit_pause_seconds") is not None else 60),
                progress_callback=_kwargs.get("progress_callback"),
            )
            return _json({"success": True, **payload})
        except Exception as exc:
            return _json({"success": False, "error": str(exc)})

    return _handler


def make_creator_stock_pick_backtest_handler(llm: Any):
    def _handler(args: dict[str, Any], **_kwargs) -> str:
        try:
            payload = backtest_creator_stock_picks(
                llm,
                up_mid=args.get("up_mid"),
                up_name=str(args.get("up_name") or ""),
                group_name=str(args.get("group_name") or "投资"),
                start_date=str(args.get("start_date") or ""),
                end_date=str(args.get("end_date") or ""),
                max_videos=int(args.get("max_videos") or 20),
                max_pages=int(args.get("max_pages") or 5),
                default_horizon_days=int(args.get("default_horizon_days") or 30),
                include_post_verification_comments=bool(args.get("include_post_verification_comments", True)),
                comment_verdict_min_days_after_publish=int(args.get("comment_verdict_min_days_after_publish") or 7),
                max_verdict_comments=int(args.get("max_verdict_comments") or 100),
                output_dir=str(args.get("output_dir") or ""),
                llm_timeout_seconds=float(args.get("llm_timeout_seconds") if args.get("llm_timeout_seconds") is not None else 45),
                asr_provider=str(args.get("asr_provider") or "auto"),
                force_asr=bool(args.get("force_asr", False)),
                allow_suspicious_subtitle=bool(args.get("allow_suspicious_subtitle", False)),
                use_cache=bool(args.get("use_cache", True)),
                force_refresh=bool(args.get("force_refresh", False)),
                progress_callback=_kwargs.get("progress_callback"),
            )
            return _json({"success": True, **payload})
        except Exception as exc:
            return _json({"success": False, "error": str(exc)})

    return _handler


def backtest_creator_stock_picks(
    llm: Any,
    *,
    up_mid: int | str | None = None,
    up_name: str = "",
    group_name: str = "投资",
    start_date: str,
    end_date: str,
    max_videos: int = 20,
    max_pages: int = 5,
    default_horizon_days: int = 30,
    include_post_verification_comments: bool = True,
    comment_verdict_min_days_after_publish: int = 7,
    max_verdict_comments: int = 100,
    output_dir: str = "",
    llm_timeout_seconds: float = 45.0,
    asr_provider: str = "auto",
    force_asr: bool = False,
    allow_suspicious_subtitle: bool = False,
    use_cache: bool = True,
    force_refresh: bool = False,
    progress_callback: Any = None,
) -> dict[str, Any]:
    start_dt = _parse_date_arg(start_date, "start_date")
    end_dt = _parse_date_arg(end_date, "end_date")
    if end_dt < start_dt:
        raise ValueError("end_date must be on or after start_date")
    max_videos = min(100, max(1, int(max_videos or 20)))
    max_pages = max(1, int(max_pages or 5))
    default_horizon_days = min(365, max(1, int(default_horizon_days or 30)))
    max_verdict_comments = min(200, max(1, int(max_verdict_comments or 100)))
    reporter = _ProgressReporter(
        progress_callback,
        label="bilibili_backtest_creator_stock_picks",
    )
    resolved_mid = int(up_mid) if up_mid not in (None, "") else 0
    creator_name = up_name.strip()
    if not resolved_mid:
        reporter.emit("creator_resolve_start", f"Resolving creator '{creator_name}' in group '{group_name}'")
        group_payload = fetch_follow_group_users(group_name=group_name, max_users=0)
        match = _find_follow_user(group_payload.get("users") or [], creator_name)
        if not match:
            raise BilibiliError(f"Could not find creator '{creator_name}' in follow group '{group_name}'.")
        resolved_mid = int(match["mid"])
        creator_name = str(match.get("name") or creator_name)
    if not creator_name:
        creator_name = str(resolved_mid)
    batch_dir = _backtest_output_dir(output_dir, creator_name=creator_name)
    reporter.set_batch_dir(batch_dir)
    reporter.emit(
        "video_archive_fetch_start",
        f"Fetching historical videos for {creator_name} from {start_date} to {end_date}",
        up_mid=resolved_mid,
        creator_name=creator_name,
        start_date=start_date,
        end_date=end_date,
    )
    videos = fetch_user_archive_videos(
        resolved_mid,
        start_ts=int(start_dt.timestamp()),
        end_ts=int((end_dt + timedelta(days=1)).timestamp()) - 1,
        max_pages=max_pages,
        page_size=50,
    )[:max_videos]
    reporter.emit(
        "video_archive_fetch_complete",
        f"Fetched {len(videos)} videos in historical window",
        video_count=len(videos),
    )
    pick_items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        bvid = str(video.get("bvid") or "")
        reporter.emit(
            "video_extract_start",
            f"[Video {index}/{len(videos)}] Extracting stock picks from {bvid}",
            video_index=index,
            video_total=len(videos),
            video=video,
        )
        try:
            transcript = fetch_transcript(
                bvid,
                page=1,
                prefer_subtitle=True,
                asr_provider=asr_provider,
                use_cache=use_cache,
                force_refresh=force_refresh,
                force_asr=force_asr,
                allow_suspicious_subtitle=allow_suspicious_subtitle,
            )
            if not transcript.get("metadata", {}).get("usable_for_analysis", True):
                raise BilibiliError("Transcript is marked unusable for backtest extraction.")
            extracted = extract_stock_picks_from_transcript(
                llm,
                metadata=transcript.get("metadata") or {},
                segments=transcript.get("segments") or [],
                llm_timeout_seconds=llm_timeout_seconds,
            )
            published_at = datetime.fromtimestamp(int(video.get("created") or 0)) if video.get("created") else start_dt
            latest_comments: list[dict[str, Any]] = []
            if include_post_verification_comments:
                try:
                    latest_comments_payload = fetch_comments(
                        bvid,
                        max_comments=max_verdict_comments,
                        comment_limit=max_verdict_comments,
                        sort="time",
                        include_replies=False,
                        use_cache=use_cache,
                        force_refresh=force_refresh,
                    )
                    latest_comments = latest_comments_payload.get("comments") or []
                except Exception as exc:
                    logger.info("Could not fetch post-verification comments for %s: %s", bvid, exc)
            for pick in extracted.get("picks") or []:
                if not pick.get("horizon_days"):
                    pick["horizon_days"] = default_horizon_days
                score = score_pick_against_prices(pick, published_at=published_at)
                comment_verdict = {}
                if include_post_verification_comments:
                    comment_verdict = analyze_post_verification_comments(
                        llm,
                        pick=pick,
                        video=video,
                        comments=latest_comments,
                        published_at=published_at,
                        min_days_after_publish=comment_verdict_min_days_after_publish,
                        llm_timeout_seconds=llm_timeout_seconds,
                    )
                final_score = combine_price_and_comment_scores(score, comment_verdict)
                pick_items.append(
                    {
                        "video": video,
                        "published_date": published_at.strftime("%Y-%m-%d"),
                        "pick": pick,
                        "score": score,
                        "comment_verdict": comment_verdict,
                        "final_score": final_score,
                    }
                )
            reporter.emit(
                "video_extract_complete",
                f"[Video {index}/{len(videos)}] Extracted {len(extracted.get('picks') or [])} picks",
                video_index=index,
                video_total=len(videos),
                pick_count=len(extracted.get("picks") or []),
            )
        except Exception as exc:
            failures.append({"video": video, "error": str(exc)})
            reporter.emit(
                "video_extract_failed",
                f"[Video {index}/{len(videos)}] Failed {bvid}: {exc}",
                video_index=index,
                video_total=len(videos),
                error=str(exc),
            )
    scored = [item for item in pick_items if (item.get("score") or {}).get("success")]
    average_score = round(
        sum(float((item.get("final_score") or {}).get("score") or 0) for item in scored) / len(scored),
        1,
    ) if scored else None
    report = {
        "creator_name": creator_name,
        "up_mid": resolved_mid,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "video_count": len(videos),
        "pick_count": len(pick_items),
        "scored_pick_count": len(scored),
        "average_score": average_score,
        "score_dimensions": {
            "price": "Later market price movement after publication.",
            "comments": "Latest/post-verification comments judging whether the call was right, wrong, or disputed.",
        },
        "picks": pick_items,
        "failures": failures,
        "method": "Yahoo Finance chart API for later price movement; LLM extraction for explicit video stock-pick claims.",
    }
    report_path, json_path, pdf_path = persist_creator_backtest_report(batch_dir, report)
    progress_path = str(reporter.path or "")
    reporter.emit(
        "backtest_complete",
        f"Backtest complete: {len(scored)}/{len(pick_items)} picks scored, average={average_score}",
        report_path=report_path,
        json_path=json_path,
        pdf_path=pdf_path,
        progress_path=progress_path,
    )
    return {
        **report,
        "report_path": report_path,
        "json_path": json_path,
        "pdf_path": pdf_path,
        "progress_path": progress_path,
        "batch_dir": str(batch_dir),
    }


def fetch_comments(
    url_or_bvid: str,
    *,
    max_comments: int = 200,
    comment_limit: int = 30,
    sort: str = "hot",
    include_replies: bool = False,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict[str, Any]:
    bvid = extract_bvid(url_or_bvid)
    info = fetch_video_info(bvid)
    aid = int(info.get("aid") or 0)
    if not aid:
        raise BilibiliError(f"Could not resolve aid for {bvid}.")

    if use_cache and not force_refresh:
        cached = load_cached_comments(bvid, aid)
        if cached:
            cached["cache_hit"] = True
            return cached

    raw_comments = fetch_video_comments(
        bvid,
        aid,
        max_comments=max_comments,
        sort=sort,
        include_replies=include_replies,
    )
    filtered_comments = filter_informative_comments(
        raw_comments,
        limit=comment_limit,
    )
    payload = {
        "metadata": {
            "bvid": bvid,
            "aid": aid,
            "title": info.get("title") or "",
            "owner": info.get("owner") or "",
            "sort": sort,
            "include_replies": include_replies,
            "raw_comment_count": len(raw_comments),
            "filtered_comment_count": len(filtered_comments),
            "cache_schema_version": 1,
            "filter": "information_score_v1",
        },
        "comments": filtered_comments,
        "raw_sample": raw_comments[: min(10, len(raw_comments))],
        "cache_hit": False,
    }
    if use_cache:
        save_cached_comments(payload)
    return payload


def analyze_following_group_latest(
    llm: Any,
    *,
    group_name: str = "鎶曡祫",
    tagid: int | None = None,
    per_up_limit: int = 10,
    max_up: int = 0,
    max_videos_total: int = 0,
    days_back: int = 7,
    output_dir: str = "",
    chunk_minutes: int = 4,
    llm_timeout_seconds: float = 20.0,
    asr_provider: str = "auto",
    force_asr: bool = False,
    allow_suspicious_subtitle: bool = False,
    include_comments: bool = False,
    max_comments: int = 200,
    comment_limit: int = 30,
    include_comment_replies: bool = True,
    comment_sort: str = "hot",
    generate_market_report: bool = True,
    market_report_timeout_seconds: float = 90,
    market_report_max_tokens: int = 6000,
    use_cache: bool = True,
    force_refresh: bool = False,
    continue_on_error: bool = True,
    retry_attempts: int = 3,
    request_delay_seconds: float = 4,
    rate_limit_pause_seconds: float = 60,
    progress_callback: Any = None,
) -> dict[str, Any]:
    per_up_limit = min(50, max(1, int(per_up_limit or 10)))
    max_up = max(0, int(max_up or 0))
    max_videos_total = max(0, int(max_videos_total or 0))
    days_back = max(0, int(days_back or 0))
    retry_attempts = max(1, int(retry_attempts or 3))
    request_delay_seconds = max(0.0, float(request_delay_seconds or 0))
    rate_limit_pause_seconds = max(0.0, float(rate_limit_pause_seconds or 0))
    reporter = _ProgressReporter(
        progress_callback,
        label="bilibili_analyze_following_group_latest",
    )
    reporter.emit(
        "group_fetch_start",
        f"Fetching follow group '{group_name}'",
        group_name=group_name,
        tagid=tagid,
    )
    group_payload = _retry_bilibili_operation(
        lambda: fetch_follow_group_users(
            group_name=group_name,
            tagid=int(tagid) if tagid is not None else None,
            max_users=max_up,
        ),
        label=f"fetch_follow_group:{group_name}",
        attempts=retry_attempts,
        rate_limit_pause_seconds=rate_limit_pause_seconds,
    )
    group = group_payload["group"]
    users = group_payload["users"]
    batch_dir = _following_batch_output_dir(output_dir, group_name=str(group.get("name") or group_name))
    reporter.set_batch_dir(batch_dir)
    reporter.emit(
        "group_fetch_complete",
        f"Fetched {len(users)} UPs from group '{group.get('name') or group_name}'",
        group=group,
        up_count=len(users),
        batch_dir=str(batch_dir),
    )
    videos: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    processed = 0
    latest_fetch_limit = min(50, max(per_up_limit, 30 if days_back else per_up_limit))
    planned_total = max_videos_total or max(1, len(users) * per_up_limit)

    for user_index, user in enumerate(users, start=1):
        reporter.emit(
            "up_latest_fetch_start",
            f"[UP {user_index}/{len(users)}] Fetching latest videos for {user.get('name') or user.get('mid')}",
            up_index=user_index,
            up_total=len(users),
            up=user,
        )
        try:
            latest = _retry_bilibili_operation(
                lambda user=user: fetch_user_latest_videos(int(user["mid"]), limit=latest_fetch_limit),
                label=f"fetch_latest_videos:{user.get('name') or user.get('mid')}",
                attempts=retry_attempts,
                rate_limit_pause_seconds=rate_limit_pause_seconds,
            )
        except Exception as exc:
            failure = {
                "stage": "fetch_latest_videos",
                "up": user,
                "error": str(exc),
            }
            failures.append(failure)
            reporter.emit(
                "up_latest_fetch_failed",
                f"[UP {user_index}/{len(users)}] Failed to fetch latest videos: {failure['error']}",
                up_index=user_index,
                up_total=len(users),
                up=user,
                error=failure["error"],
            )
            if not continue_on_error:
                break
            continue
        selected_videos = _filter_recent_videos(latest, days_back=days_back)[:per_up_limit]
        reporter.emit(
            "up_latest_fetch_complete",
            f"[UP {user_index}/{len(users)}] Selected {len(selected_videos)} videos from {len(latest)} latest items",
            up_index=user_index,
            up_total=len(users),
            up=user,
            latest_count=len(latest),
            selected_count=len(selected_videos),
        )
        if not selected_videos:
            skipped.append(
                {
                    "stage": "date_filter",
                    "up": user,
                    "reason": f"No videos within the last {days_back} days." if days_back else "No videos returned.",
                }
            )
            continue
        for video in selected_videos:
            if max_videos_total and len(videos) >= max_videos_total:
                break
            video["owner_mid"] = user["mid"]
            video["owner_name"] = user.get("name") or video.get("author") or ""
            videos.append(video)
            video_index = len(videos)
            output_path = batch_dir / _up_video_archive_filename(user, video)
            args = {
                "url_or_bvid": video["bvid"],
                "page": 1,
                "output_format": "markdown",
                "chunk_minutes": chunk_minutes,
                "llm_timeout_seconds": llm_timeout_seconds,
                "asr_provider": asr_provider,
                "use_cache": use_cache,
                "force_refresh": force_refresh,
                "force_asr": force_asr,
                "allow_suspicious_subtitle": allow_suspicious_subtitle,
                "persist_file": True,
                "include_comments": include_comments,
                "max_comments": max_comments,
                "comment_limit": comment_limit,
                "include_comment_replies": include_comment_replies,
                "comment_sort": comment_sort,
                "output_path": str(output_path),
            }
            if videos or results or failures:
                _sleep_between_bilibili_requests(request_delay_seconds)
            reporter.emit(
                "video_analyze_start",
                f"[Video {video_index}/{planned_total}] Analyzing {video.get('bvid')} - {video.get('title') or ''}",
                video_index=video_index,
                video_total=planned_total,
                up=user,
                video=video,
                output_path=str(output_path),
            )
            try:
                raw = _retry_bilibili_operation(
                    lambda args=args: make_analyze_handler(llm)(
                        args,
                        progress_callback=progress_callback,
                        progress_reporter=reporter,
                        progress_context={
                            "video_index": video_index,
                            "video_total": planned_total,
                            "up": user,
                            "video": video,
                        },
                    ),
                    label=f"analyze_video:{video.get('bvid')}",
                    attempts=retry_attempts,
                    rate_limit_pause_seconds=rate_limit_pause_seconds,
                    retry_result_fn=_tool_result_is_retryable,
                )
            except Exception as exc:
                raw = _json({"success": False, "stage": "analyze_video", "error": str(exc)})
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                item = {"success": False, "error": raw}
            item["up"] = user
            item["video"] = video
            video_records.append(
                {
                    "success": bool(item.get("success")),
                    "up": user,
                    "video": video,
                    "transcript": item.get("transcript"),
                    "comments": item.get("comments"),
                    "analysis": item.get("analysis"),
                    "error": item.get("error") or "",
                    "output_path": item.get("output_path") or str(output_path),
                    "pdf_path": item.get("pdf_path") or _pdf_sibling_path(item.get("output_path") or str(output_path)),
                    "state_path": item.get("state_path") or "",
                }
            )
            if item.get("success"):
                processed += 1
                reporter.emit(
                    "video_analyze_complete",
                    f"[Video {video_index}/{planned_total}] Completed {video.get('bvid')}",
                    video_index=video_index,
                    video_total=planned_total,
                    up=user,
                    video=video,
                    output_path=item.get("output_path") or str(output_path),
                )
                results.append(
                    {
                        "up": user,
                        "video": video,
                        "output_path": item.get("output_path") or str(output_path),
                        "pdf_path": item.get("pdf_path") or _pdf_sibling_path(item.get("output_path") or str(output_path)),
                        "partial_archive": bool(item.get("partial_archive")),
                        "summary": ((item.get("analysis") or {}).get("summary") or ""),
                        "main_thesis": ((item.get("analysis") or {}).get("main_thesis") or ""),
                    }
                )
            else:
                reporter.emit(
                    "video_analyze_failed",
                    f"[Video {video_index}/{planned_total}] Failed {video.get('bvid')}: {item.get('error') or 'Unknown analysis failure.'}",
                    video_index=video_index,
                    video_total=planned_total,
                    up=user,
                    video=video,
                    error=item.get("error") or "Unknown analysis failure.",
                    output_path=item.get("output_path") or "",
                    partial_archive=bool(item.get("partial_archive")),
                )
                failures.append(
                    {
                        "stage": item.get("stage") or "analyze_video",
                        "up": user,
                        "video": video,
                        "error": item.get("error") or "Unknown analysis failure.",
                        "output_path": item.get("output_path") or "",
                        "partial_archive": bool(item.get("partial_archive")),
                    }
                )
                if not continue_on_error:
                    break
        if max_videos_total and len(videos) >= max_videos_total:
            break
        if failures and not continue_on_error:
            break

    market_report: dict[str, Any] | None = None
    market_report_path = ""
    market_report_pdf_path = ""
    if generate_market_report and video_records:
        reporter.emit(
            "market_report_start",
            f"Generating market report from {len(video_records)} video records",
            record_count=len(video_records),
            success_count=len(results),
            failure_count=len(failures),
        )
        market_report = analyze_market_batch(
            llm,
            group=group,
            records=video_records,
            failures=failures,
            skipped=skipped,
            llm_timeout_seconds=market_report_timeout_seconds,
            max_tokens=market_report_max_tokens,
        )
        market_report_path = _persist_market_report(batch_dir, market_report)
        market_report_pdf_path = _pdf_sibling_path(market_report_path)
        reporter.emit(
            "market_report_complete",
            "Market report generated",
            market_report_path=market_report_path,
            market_report_pdf_path=market_report_pdf_path,
        )

    progress_path = str(reporter.path or "")
    index_path = _persist_following_batch_index(
        batch_dir,
        group=group,
        users=users,
        requested_per_up=per_up_limit,
        videos=videos,
        results=results,
        failures=failures,
        skipped=skipped,
        market_report_path=market_report_path,
        market_report_pdf_path=market_report_pdf_path,
        progress_path=progress_path,
    )
    reporter.emit(
        "batch_complete",
        f"Batch complete: {len(results)} succeeded, {len(failures)} failed, {len(skipped)} skipped",
        index_path=str(index_path),
        market_report_path=market_report_path,
        market_report_pdf_path=market_report_pdf_path,
        progress_path=progress_path,
    )
    return {
        "group": group,
        "up_count": len(users),
        "requested_per_up": per_up_limit,
        "video_count": len(videos),
        "analyzed_count": len(results),
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "batch_dir": str(batch_dir),
        "index_path": str(index_path),
        "progress_path": progress_path,
        "market_report_path": market_report_path,
        "market_report_pdf_path": market_report_pdf_path,
        "market_report": market_report,
        "results": results,
        "failures": failures,
        "skipped": skipped,
    }


def _filter_recent_videos(
    videos: list[dict[str, Any]], *, days_back: int = 7
) -> list[dict[str, Any]]:
    if days_back <= 0:
        return list(videos)
    try:
        from hermes_time import now as _hermes_now

        cutoff = int(_hermes_now().timestamp()) - days_back * 86400
    except Exception:
        cutoff = int(time.time()) - days_back * 86400
    return [
        video for video in videos
        if int(video.get("created") or 0) >= cutoff
    ]


def _retry_bilibili_operation(
    fn,
    *,
    label: str,
    attempts: int = 3,
    rate_limit_pause_seconds: float = 60,
    retry_result_fn=None,
):
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts or 1)) + 1):
        try:
            result = fn()
            if retry_result_fn and retry_result_fn(result):
                raise BilibiliError(_retryable_result_error(result))
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            message = str(exc)
            if _looks_like_bilibili_rate_limit(message):
                delay = rate_limit_pause_seconds
            else:
                delay = min(30.0, 2.0 * attempt)
            if delay > 0:
                time.sleep(delay)
    raise last_error or BilibiliError(f"{label} failed")


def _tool_result_is_retryable(raw: str) -> bool:
    try:
        payload = json.loads(raw)
    except Exception:
        return False
    if payload.get("success", True):
        return False
    return _looks_like_bilibili_rate_limit(str(payload.get("error") or ""))


def _retryable_result_error(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return "Retryable tool failure."
    return str(payload.get("error") or "Retryable tool failure.")


def _looks_like_bilibili_rate_limit(message: str) -> bool:
    text = (message or "").lower()
    return (
        "412" in text
        or "precondition failed" in text
        or "璇锋眰杩囦簬棰戠箒" in text
        or "too frequent" in text
        or "connection reset" in text
        or "unexpected_eof" in text
    )


def _sleep_between_bilibili_requests(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def fetch_transcript(
    url_or_bvid: str,
    *,
    page: int = 1,
    prefer_subtitle: bool = True,
    asr_provider: str = "auto",
    use_cache: bool = True,
    force_refresh: bool = False,
    force_asr: bool = False,
    allow_suspicious_subtitle: bool = False,
) -> dict[str, Any]:
    bvid = extract_bvid(url_or_bvid)
    cache_allowed = use_cache and not force_refresh and prefer_subtitle and not force_asr
    if cache_allowed:
        cached = load_cached_transcript_for_page(bvid, page)
        if cached and _cache_is_current(cached):
            _raise_if_unusable_cached(cached, allow_suspicious_subtitle)
            cached["cache_hit"] = True
            return cached

    info = fetch_video_info(bvid)
    pages = info["pages"]
    selected = next((p for p in pages if int(p.get("page") or 0) == page), None)
    if selected is None:
        raise BilibiliError(f"Page {page} does not exist for {bvid}.")
    cid = int(selected["cid"])

    if cache_allowed:
        cached = load_cached_transcript(bvid, cid)
        if cached and _cache_is_current(cached):
            _raise_if_unusable_cached(cached, allow_suspicious_subtitle)
            cached["cache_hit"] = True
            return cached

    source = "subtitle"
    subtitle_meta = None
    segments: list[dict[str, Any]] = []
    warnings: list[str] = []
    if prefer_subtitle and not force_asr:
        segments, subtitle_meta = fetch_subtitle_segments(bvid, cid)
        warnings.extend(
            _subtitle_quality_warnings(
                segments,
                selected.get("duration") or info.get("duration") or 0,
                title=info.get("title") or "",
                part=selected.get("part") or "",
            )
        )
        if not segments or warnings:
            ytdlp_segments, ytdlp_meta = fetch_subtitle_segments_ytdlp(bvid)
            ytdlp_warnings = _subtitle_quality_warnings(
                ytdlp_segments,
                selected.get("duration") or info.get("duration") or 0,
                title=info.get("title") or "",
                part=selected.get("part") or "",
            )
            if ytdlp_segments and (not segments or len(ytdlp_warnings) < len(warnings)):
                segments = ytdlp_segments
                subtitle_meta = ytdlp_meta
                warnings = ytdlp_warnings

    audio_path = None
    if not segments:
        source = "asr"
        audio_path = download_audio(bvid, cid, prefer_playurl=_prefer_playurl_audio())
        segments, asr_provider_used = transcribe_audio(audio_path, asr_provider)
        source = f"asr:{asr_provider_used}"
    elif warnings and has_asr_provider(asr_provider):
        try:
            audio_path = download_audio(bvid, cid, prefer_playurl=_prefer_playurl_audio())
            asr_segments, asr_provider_used = transcribe_audio(audio_path, asr_provider)
            if asr_segments:
                segments = asr_segments
                source = f"asr:{asr_provider_used}"
                warnings.append("Bilibili subtitle looked suspicious, so ASR fallback was used.")
        except Exception as exc:
            warnings.append(f"Subtitle looked suspicious, but ASR fallback failed: {exc}")

    if warnings and source == "subtitle" and segments and not allow_suspicious_subtitle:
        raise BilibiliError(
            "Bilibili subtitle quality check failed; refusing to return it as reliable "
            "transcript. The subtitle may be incomplete or mismatched with the video. "
            "Use force_asr=true after configuring BILIBILI_ASR_COMMAND or faster-whisper, "
            "or set allow_suspicious_subtitle=true only if you explicitly want the raw "
            f"suspicious subtitle. Warnings: {'; '.join(warnings)}"
        )

    metadata = {
        "bvid": bvid,
        "cid": cid,
        "page": page,
        "title": info["title"],
        "owner": info["owner"],
        "description": info["desc"],
        "duration": selected.get("duration") or info.get("duration") or 0,
        "part": selected.get("part") or "",
        "source": source,
        "subtitle": subtitle_meta or {},
        "audio_path": str(audio_path) if audio_path else "",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "warnings": warnings,
        "quality_status": "suspect" if warnings else "ok",
        "usable_for_analysis": not warnings or source.startswith("asr:"),
    }
    payload = {
        "metadata": metadata,
        "segments": segments,
        "warnings": warnings,
        "cache_hit": False,
    }
    if use_cache:
        save_cached_transcript(payload)
    return payload


def make_analyze_handler(llm: Any):
    def _handler(args: dict[str, Any], **_kwargs) -> str:
        stage = "fetch_transcript"
        transcript: dict[str, Any] | None = None
        comments_payload: dict[str, Any] | None = None
        analysis: dict[str, Any] | None = None
        output_path = ""
        pdf_path = ""
        state_path = ""
        reporter = _kwargs.get("progress_reporter")
        if not isinstance(reporter, _ProgressReporter):
            reporter = _ProgressReporter(
                _kwargs.get("progress_callback"),
                label="bilibili_analyze_video",
            )
        progress_context = dict(_kwargs.get("progress_context") or {})
        bvid_or_url = args.get("url_or_bvid") or ""
        try:
            reporter.emit(
                "transcript_fetch_start",
                f"Fetching transcript for {bvid_or_url}",
                url_or_bvid=bvid_or_url,
                **progress_context,
            )
            transcript = fetch_transcript(
                bvid_or_url,
                page=int(args.get("page") or 1),
                prefer_subtitle=True,
                asr_provider=str(args.get("asr_provider") or "auto"),
                use_cache=bool(args.get("use_cache", True)),
                force_refresh=bool(args.get("force_refresh", False)),
                force_asr=bool(args.get("force_asr", False)),
                allow_suspicious_subtitle=bool(args.get("allow_suspicious_subtitle", False)),
            )
            if not transcript.get("metadata", {}).get("usable_for_analysis", True):
                raise BilibiliError(
                    "Transcript is marked unusable for analysis. Re-run with force_asr=true "
                    "or explicitly set allow_suspicious_subtitle=true."
                )
            if _bool_arg(args, "persist_file", True):
                output_path = _write_partial_archive(
                    transcript,
                    comments_payload,
                    analysis,
                    stage="transcript_ready",
                    status="partial",
                    output_path=str(args.get("output_path") or ""),
                )
                state_path = _write_archive_state(
                    transcript,
                    comments_payload,
                    analysis,
                    stage="transcript_ready",
                    status="partial",
                    output_path=output_path,
                )
            reporter.emit(
                "transcript_fetch_complete",
                f"Transcript ready for {transcript.get('metadata', {}).get('bvid') or bvid_or_url}",
                bvid=transcript.get("metadata", {}).get("bvid") or bvid_or_url,
                source=transcript.get("metadata", {}).get("source") or "",
                segment_count=len(transcript.get("segments") or []),
                output_path=output_path,
                state_path=state_path,
                **progress_context,
            )
            comments_for_analysis = None
            if bool(args.get("include_comments", False)):
                stage = "fetch_comments"
                reporter.emit(
                    "comments_fetch_start",
                    f"Fetching comments for {bvid_or_url}",
                    url_or_bvid=bvid_or_url,
                    max_comments=int(args.get("max_comments") or 200),
                    comment_limit=int(args.get("comment_limit") or 30),
                    **progress_context,
                )
                comments_payload = fetch_comments(
                    bvid_or_url,
                    max_comments=int(args.get("max_comments") or 200),
                    comment_limit=int(args.get("comment_limit") or 30),
                    sort=str(args.get("comment_sort") or "hot"),
                    include_replies=bool(args.get("include_comment_replies", False)),
                    use_cache=bool(args.get("use_cache", True)),
                    force_refresh=bool(args.get("force_refresh", False)),
                )
                comments_for_analysis = _comments_for_analysis(
                    comments_payload.get("comments") or [],
                    limit=int(args.get("comment_limit") or 30),
                )
                comments_payload["analysis_comment_count"] = len(comments_for_analysis)
                if _bool_arg(args, "persist_file", True):
                    output_path = _write_partial_archive(
                        transcript,
                        comments_payload,
                        analysis,
                        stage="comments_ready",
                        status="partial",
                        output_path=output_path or str(args.get("output_path") or ""),
                    )
                    state_path = _write_archive_state(
                        transcript,
                        comments_payload,
                        analysis,
                        stage="comments_ready",
                        status="partial",
                        output_path=output_path,
                    )
                reporter.emit(
                    "comments_fetch_complete",
                    f"Comments ready: {comments_payload.get('metadata', {}).get('filtered_comment_count', 0)} filtered, {len(comments_for_analysis)} selected for analysis",
                    filtered_comment_count=comments_payload.get("metadata", {}).get("filtered_comment_count", 0),
                    analysis_comment_count=len(comments_for_analysis),
                    output_path=output_path,
                    state_path=state_path,
                    **progress_context,
                )
            stage = "analyze"
            reporter.emit(
                "llm_analysis_start",
                f"Running video analysis for {transcript.get('metadata', {}).get('bvid') or bvid_or_url}",
                bvid=transcript.get("metadata", {}).get("bvid") or bvid_or_url,
                llm_timeout_seconds=float(args.get("llm_timeout_seconds") if args.get("llm_timeout_seconds") is not None else 20),
                **progress_context,
            )
            analysis = analyze_transcript(
                llm,
                transcript["metadata"],
                transcript["segments"],
                comments=comments_for_analysis,
                output_format=str(args.get("output_format") or "markdown"),
                chunk_minutes=int(args.get("chunk_minutes") or 4),
                llm_timeout_seconds=float(args.get("llm_timeout_seconds") if args.get("llm_timeout_seconds") is not None else 20),
            )
            if _bool_arg(args, "persist_file", True):
                stage = "persist"
                output_path = _persist_analysis(
                    transcript,
                    analysis,
                    str(args.get("output_path") or ""),
                )
                pdf_path = _pdf_sibling_path(output_path)
                state_path = _write_archive_state(
                    transcript,
                    comments_payload,
                    analysis,
                    stage="complete",
                    status="complete",
                    output_path=output_path,
                )
            reporter.emit(
                "llm_analysis_complete",
                f"Video analysis complete for {transcript.get('metadata', {}).get('bvid') or bvid_or_url}",
                bvid=transcript.get("metadata", {}).get("bvid") or bvid_or_url,
                output_path=output_path,
                state_path=state_path,
                **progress_context,
            )
            return _json(
                {
                    "success": True,
                    "transcript": transcript,
                    "comments": comments_payload,
                    "analysis": analysis,
                    "output_path": output_path,
                    "pdf_path": pdf_path,
                    "state_path": state_path,
                    "partial_archive": False,
                }
            )
        except Exception as exc:
            reporter.emit(
                "video_stage_failed",
                f"Video analysis failed at {stage}: {exc}",
                stage_failed=stage,
                error=str(exc),
                output_path=output_path,
                state_path=state_path,
                **progress_context,
            )
            if transcript and _bool_arg(args, "persist_file", True):
                try:
                    output_path = _write_partial_archive(
                        transcript,
                        comments_payload,
                        analysis,
                        stage=stage,
                        status="failed",
                        error=str(exc),
                        output_path=output_path or str(args.get("output_path") or ""),
                    )
                    state_path = _write_archive_state(
                        transcript,
                        comments_payload,
                        analysis,
                        stage=stage,
                        status="failed",
                        error=str(exc),
                        output_path=output_path,
                    )
                except Exception:
                    pass
            return _json(
                {
                    "success": False,
                    "stage": stage,
                    "error": str(exc),
                    "transcript": transcript,
                    "comments": comments_payload,
                    "analysis": analysis,
                    "output_path": output_path,
                    "state_path": state_path,
                    "partial_archive": bool(output_path),
                }
            )

    return _handler


GENERIC_COMMENT_PATTERNS = (
    "哈哈",
    "支持",
    "牛逼",
    "厉害",
    "前排",
    "来了",
    "打卡",
    "三连",
    "点赞",
    "收藏",
    "泪目",
    "笑死",
)


def _comments_for_analysis(
    comments: list[dict[str, Any]], *, limit: int = 30, max_chars: int = 320
) -> list[dict[str, Any]]:
    """Compact filtered comments before sending them into the global LLM pass."""
    compacted: list[dict[str, Any]] = []
    for item in comments[: min(max(1, int(limit or 30)), 20)]:
        message = _trim_comment_text(str(item.get("message") or ""), max_chars)
        if not message:
            continue
        compact = {
            "user": item.get("user") or "",
            "message": message,
            "like": int(item.get("like") or 0),
            "reply_count": int(item.get("reply_count") or 0),
            "information_score": item.get("information_score", 0),
            "filter_reasons": item.get("filter_reasons") or [],
            "investment_signal": _comment_investment_signal(message),
        }
        replies = []
        for reply in item.get("replies") or []:
            reply_message = _trim_comment_text(
                str(reply.get("message") or ""),
                max(80, max_chars // 2),
            )
            if reply_message:
                replies.append(
                    {
                        "user": reply.get("user") or "",
                        "message": reply_message,
                        "like": int(reply.get("like") or 0),
                        "investment_signal": _comment_investment_signal(reply_message),
                    }
                )
            if len(replies) >= 3:
                break
        if replies:
            compact["replies"] = replies
        compacted.append(compact)
    return compacted


def _trim_comment_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def filter_informative_comments(
    comments: list[dict[str, Any]], *, limit: int = 30
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    for comment in comments:
        message = str(comment.get("message") or "").strip()
        normalized = _normalize_comment_text(message)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        score, reasons = _comment_information_score(comment, normalized)
        if score < 3:
            continue
        item = dict(comment)
        item["information_score"] = score
        item["filter_reasons"] = reasons
        item.pop("mid", None)
        if item.get("replies"):
            item["replies"] = filter_informative_comments(
                item["replies"], limit=3
            )
        scored.append(item)
    scored.sort(
        key=lambda item: (
            float(item.get("information_score") or 0),
            int(item.get("like") or 0),
            int(item.get("reply_count") or 0),
        ),
        reverse=True,
    )
    return scored[: max(1, int(limit or 30))]


def _normalize_comment_text(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text.lower()


def _comment_information_score(
    comment: dict[str, Any], normalized: str
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    text = str(comment.get("message") or "")
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    unique_ratio = len(set(normalized)) / max(1, len(normalized))
    score = 0.0
    if cjk_count >= 20:
        score += 2.0
        reasons.append("substantial_text")
    elif cjk_count >= 10:
        score += 1.0
    else:
        score -= 2.0
        reasons.append("too_short")
    if unique_ratio < 0.25:
        score -= 3.0
        reasons.append("repetitive_text")
    if any(pattern in text for pattern in GENERIC_COMMENT_PATTERNS) and cjk_count < 30:
        score -= 2.0
        reasons.append("generic_reaction")
    if re.search(r"[?？]|为什么|怎么|如何|请问|区别|逻辑|原因", text):
        score += 2.0
        reasons.append("question_or_reasoning")
    if re.search(r"因为|但是|所以|我觉得|我认为|比如|例如|建议|风险|经验|观点", text):
        score += 2.0
        reasons.append("argument_or_experience")
    if re.search(r"\d|%|年|月|元|万|倍|PE|ROE|仓位|收益|亏|赚|书|交易|短线|长线", text, re.I):
        score += 1.5
        reasons.append("specific_terms")
    if re.search(
        r"持仓|成本|止损|止盈|加仓|减仓|满仓|空仓|半仓|定投|估值|财报|业绩|现金流|"
        r"订单|库存|政策|利率|汇率|美元|纳指|恒生|沪深|创业板|科创|黄金|债|"
        r"板块|个股|龙头|周期|资金|主线|题材|AI|芯片|半导体|新能源|消费|医药",
        text,
        re.I,
    ):
        score += 2.5
        reasons.append("investment_signal")
    if re.search(r"不同意|不认同|反驳|错了|忽略了|其实|但是|不过|问题是|风险是", text):
        score += 2.0
        reasons.append("pushback_or_correction")
    like = int(comment.get("like") or 0)
    reply_count = int(comment.get("reply_count") or 0)
    if like >= 50:
        score += 2.0
        reasons.append("high_like")
    elif like >= 10:
        score += 1.0
    if reply_count >= 5:
        score += 1.0
        reasons.append("discussion_thread")
    return score, reasons


def _comment_investment_signal(text: str) -> str:
    if re.search(r"风险|亏|下跌|回调|减仓|止损|高估|压力", text):
        return "risk_or_bearish"
    if re.search(r"机会|反弹|看多|加仓|低估|修复|突破|上涨", text):
        return "bullish_or_opportunity"
    if re.search(r"持仓|成本|仓位|定投|估值|财报|政策|利率|板块|个股|资金", text):
        return "specific_market_info"
    if re.search(r"为什么|如何|怎么|请问|逻辑|原因", text):
        return "question_or_due_diligence"
    return "general"


def _persist_analysis(
    transcript: dict[str, Any], analysis: dict[str, Any], output_path: str = ""
) -> str:
    metadata = transcript.get("metadata") or {}
    bvid = str(metadata.get("bvid") or "bilibili")
    cid = str(metadata.get("cid") or "unknown")
    path = _archive_output_path(output_path, bvid=bvid, cid=cid, title=str(metadata.get("title") or ""))
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = analysis.get("markdown") or json.dumps(analysis, ensure_ascii=False, indent=2)
    transcript_lines = [
        "",
        "## Transcript",
        "",
    ]
    for seg in transcript.get("segments") or []:
        transcript_lines.append(
            f"- [{seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}] {seg.get('text', '')}"
        )
    full_markdown = markdown + "\n" + "\n".join(transcript_lines)
    path.write_text(full_markdown, encoding="utf-8")
    write_markdown_pdf(full_markdown, path.with_suffix(".pdf"), title=str(metadata.get("title") or bvid))
    return str(path)


def _write_partial_archive(
    transcript: dict[str, Any],
    comments: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    *,
    stage: str,
    status: str,
    output_path: str = "",
    error: str = "",
) -> str:
    metadata = transcript.get("metadata") or {}
    bvid = str(metadata.get("bvid") or "bilibili")
    cid = str(metadata.get("cid") or "unknown")
    path = _archive_output_path(output_path, bvid=bvid, cid=cid, title=str(metadata.get("title") or ""))
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _partial_archive_lines(
        transcript,
        comments,
        analysis,
        stage=stage,
        status=status,
        error=error,
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


def _write_archive_state(
    transcript: dict[str, Any],
    comments: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    *,
    stage: str,
    status: str,
    output_path: str,
    error: str = "",
) -> str:
    metadata = transcript.get("metadata") or {}
    bvid = str(metadata.get("bvid") or "bilibili")
    cid = str(metadata.get("cid") or "unknown")
    path = _archive_output_path(output_path, bvid=bvid, cid=cid, title=str(metadata.get("title") or ""))
    if not path.is_absolute():
        path = Path.cwd() / path
    state_path = path.with_suffix(".run.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "status": status,
        "stage": stage,
        "error": error,
        "output_path": str(path),
        "metadata": metadata,
        "comment_metadata": (comments or {}).get("metadata") if comments else None,
        "analysis_available": bool(analysis),
        "transcript_segment_count": len(transcript.get("segments") or []),
        "comment_count": len((comments or {}).get("comments") or []),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(state_path)


def _partial_archive_lines(
    transcript: dict[str, Any],
    comments: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    *,
    stage: str,
    status: str,
    error: str = "",
) -> list[str]:
    metadata = transcript.get("metadata") or {}
    comments_items = (comments or {}).get("comments") or []
    lines = [
        "# Bilibili Video Analysis",
        "",
        f"- Status: {status}",
        f"- Stage: {stage}",
        f"- BV: {metadata.get('bvid', '')}",
        f"- CID: {metadata.get('cid', '')}",
        f"- Title: {metadata.get('title', '')}",
        f"- Owner: {metadata.get('owner', '')}",
        f"- Source: {metadata.get('source', '')}",
    ]
    if error:
        lines.extend(["", "## Error", "", error])
    if analysis:
        markdown = analysis.get("markdown")
        if markdown:
            lines.extend(["", markdown.strip()])
        else:
            lines.extend(["", "## Analysis JSON", "", "```json", json.dumps(analysis, ensure_ascii=False, indent=2), "```"])
    else:
        lines.extend(
            [
                "",
                "## Analysis",
                "",
                "Analysis has not completed yet. Transcript and comments below were archived for recovery.",
            ]
        )
    if comments_items:
        lines.extend(["", "## Filtered Comments", ""])
        for item in comments_items[:20]:
            like = int(item.get("like") or 0)
            reply_count = int(item.get("reply_count") or 0)
            lines.append(
                f"- {item.get('message', '')} "
                f"(user={item.get('user', '')}, like={like}, replies={reply_count})"
            )
    lines.extend(["", "## Transcript", "", transcript_to_text(transcript.get("segments") or [])])
    return lines


def _archive_output_path(output_path: str, *, bvid: str, cid: str, title: str = "") -> Path:
    if output_path:
        return _normalize_cross_platform_path(output_path)
    filename = "_".join(
        part for part in (bvid, cid, _safe_filename_part(title)) if part
    )[:180]
    return _default_archive_dir() / f"{filename or bvid}_analysis.md"


def _partial_archive_output_path(output_path: str, *, bvid: str, cid: str, title: str = "") -> Path:
    path = _archive_output_path(output_path, bvid=bvid, cid=cid, title=title)
    if path.suffix:
        return path.with_name(f"{path.stem}_partial{path.suffix}")
    return path.with_name(f"{path.name}_partial.md")


def _default_archive_dir() -> Path:
    configured = os.getenv("BILIBILI_ANALYZER_OUTPUT_DIR", "").strip()
    if configured:
        return _normalize_cross_platform_path(configured)

    preferred = Path.cwd() / "outputs" / "bilibili_analyzer"
    if _ensure_writable_dir(preferred):
        return preferred
    return cache_dir().parent.parent / "outputs" / "bilibili_analyzer"


def _ensure_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _normalize_cross_platform_path(path_value: str) -> Path:
    value = (path_value or "").strip()
    if os.name == "nt":
        match = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    elif re.match(r"^[A-Za-z]:[\\/]", value):
        drive = value[0].lower()
        rest = value[2:].lstrip("\\/").replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(value).expanduser()


def _following_batch_output_dir(output_dir: str, *, group_name: str) -> Path:
    if output_dir:
        path = _normalize_cross_platform_path(output_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    group_part = _safe_filename_part(group_name, max_len=32) or "group"
    path = _default_archive_dir() / "following" / f"{stamp}_{group_part}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _backtest_output_dir(output_dir: str, *, creator_name: str) -> Path:
    if output_dir:
        path = _normalize_cross_platform_path(output_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    creator_part = _safe_filename_part(creator_name, max_len=32) or "creator"
    path = _default_archive_dir() / "backtests" / f"{stamp}_{creator_part}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_date_arg(value: str, name: str) -> datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{name} is required in YYYY-MM-DD format")
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be in YYYY-MM-DD format") from exc


def _find_follow_user(users: list[dict[str, Any]], up_name: str) -> dict[str, Any] | None:
    needle = (up_name or "").strip().lower()
    if not needle:
        return None
    for user in users:
        name = str(user.get("name") or "").strip()
        if name.lower() == needle:
            return user
    for user in users:
        name = str(user.get("name") or "").strip()
        if needle in name.lower():
            return user
    return None


def _up_video_archive_filename(user: dict[str, Any], video: dict[str, Any]) -> str:
    owner = _safe_filename_part(str(user.get("name") or video.get("owner_name") or "up"), 28)
    created = int(video.get("created") or 0)
    date_part = datetime.fromtimestamp(created).strftime("%Y%m%d") if created else "unknown-date"
    bvid = str(video.get("bvid") or "video")
    title = _safe_filename_part(str(video.get("title") or ""), 48)
    return f"{owner}_{date_part}_{bvid}_{title or 'analysis'}.md"


def _persist_market_report(batch_dir: Path, report: dict[str, Any]) -> str:
    path = batch_dir / "market_report.md"
    markdown = report.get("markdown") or json.dumps(report, ensure_ascii=False, indent=2)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    write_markdown_pdf(markdown.rstrip() + "\n", path.with_suffix(".pdf"), title="Bilibili Market Report")
    json_path = batch_dir / "market_report.json"
    payload = {key: value for key, value in report.items() if key != "markdown"}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _pdf_sibling_path(path_value: str) -> str:
    if not path_value:
        return ""
    return str(Path(path_value).with_suffix(".pdf"))


def _persist_following_batch_index(
    batch_dir: Path,
    *,
    group: dict[str, Any],
    users: list[dict[str, Any]],
    requested_per_up: int,
    videos: list[dict[str, Any]],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
    market_report_path: str = "",
    market_report_pdf_path: str = "",
    progress_path: str = "",
) -> Path:
    skipped = skipped or []
    path = batch_dir / "index.md"
    lines = [
        "# Bilibili Follow Group Batch Analysis",
        "",
        f"- Group: {group.get('name', '')}",
        f"- tagid: {group.get('tagid', '')}",
        f"- Followed UP count: {len(users)}",
        f"- Requested videos per UP: {requested_per_up}",
        f"- Videos fetched: {len(videos)}",
        f"- Successful analyses: {len(results)}",
        f"- Failures: {len(failures)}",
        f"- Skipped: {len(skipped)}",
        f"- Market report: {market_report_path or 'Not generated'}",
        f"- Market report PDF: {market_report_pdf_path or 'Not generated'}",
        f"- Progress log: {progress_path or 'Not generated'}",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Successful Archives",
        "",
    ]
    if not results:
        lines.append("- None.")
    for item in results:
        up = item.get("up") or {}
        video = item.get("video") or {}
        output_path = str(item.get("output_path") or "")
        try:
            rel = Path(output_path).resolve().relative_to(batch_dir.resolve())
            link = rel.as_posix()
        except Exception:
            link = output_path
        created = int(video.get("created") or 0)
        created_text = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else ""
        lines.append(
            f"- [{video.get('title', '')}]({link}) "
            f"| UP: {up.get('name', '')} | BV: {video.get('bvid', '')} | {created_text}"
        )
        if item.get("partial_archive"):
            lines.append("  - Status: partial archive")
        if item.get("pdf_path"):
            try:
                rel_pdf = Path(str(item.get("pdf_path"))).resolve().relative_to(batch_dir.resolve())
                lines.append(f"  - PDF: {rel_pdf.as_posix()}")
            except Exception:
                lines.append(f"  - PDF: {item.get('pdf_path')}")
        thesis = str(item.get("main_thesis") or item.get("summary") or "").strip()
        if thesis:
            lines.append(f"  - Summary: {thesis[:240]}")

    lines.extend(["", "## Skipped", ""])
    if not skipped:
        lines.append("- None.")
    for item in skipped:
        up = item.get("up") or {}
        lines.append(
            f"- Stage: {item.get('stage', '')} | UP: {up.get('name', '')} | "
            f"Reason: {item.get('reason', '')}"
        )

    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- None.")
    for item in failures:
        up = item.get("up") or {}
        video = item.get("video") or {}
        output_path = str(item.get("output_path") or "")
        suffix = ""
        if output_path:
            try:
                rel = Path(output_path).resolve().relative_to(batch_dir.resolve())
                suffix = f" | Archive: {rel.as_posix()}"
            except Exception:
                suffix = f" | Archive: {output_path}"
        partial = " | Partial archive" if item.get("partial_archive") else ""
        lines.append(
            f"- Stage: {item.get('stage', '')} | UP: {up.get('name', '')} | "
            f"BV: {video.get('bvid', '')} | Error: {item.get('error', '')}"
            f"{partial}{suffix}"
        )

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def _safe_filename_part(value: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", "_", value or "").strip("_")
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", text)
    return text.strip("._-")[:max_len]


def _cache_is_current(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata") or {}
    return int(metadata.get("cache_schema_version") or 0) >= CACHE_SCHEMA_VERSION


def _raise_if_unusable_cached(
    payload: dict[str, Any], allow_suspicious_subtitle: bool
) -> None:
    metadata = payload.get("metadata") or {}
    warnings = payload.get("warnings") or metadata.get("warnings") or []
    if (
        warnings
        and metadata.get("source") == "subtitle"
        and not allow_suspicious_subtitle
    ):
        raise BilibiliError(
            "Cached Bilibili subtitle is marked suspicious; refusing to return it "
            "as reliable transcript. Re-run with force_asr=true after configuring "
            "ASR, or set allow_suspicious_subtitle=true only if you explicitly "
            f"want the cached raw subtitle. Warnings: {'; '.join(map(str, warnings))}"
        )


def _subtitle_quality_warnings(
    segments: list[dict[str, Any]],
    duration: int | float,
    *,
    title: str = "",
    part: str = "",
) -> list[str]:
    if not segments:
        return []
    warnings = []
    try:
        duration_value = float(duration or 0)
        last_end = max(float(seg.get("end") or 0) for seg in segments)
    except Exception:
        return warnings
    if duration_value >= 120 and last_end / duration_value < 0.65:
        warnings.append(
            "Bilibili subtitle covers less than 65% of the video duration; "
            "it may be incomplete or mismatched. Use force_asr=true to verify."
        )
    warnings.extend(_subtitle_topic_warnings(segments, title=title, part=part))
    return warnings


def _subtitle_topic_warnings(
    segments: list[dict[str, Any]], *, title: str = "", part: str = ""
) -> list[str]:
    grams = _cjk_bigrams(" ".join([title or "", part or ""]))
    if len(grams) < 8:
        return []
    transcript_text = "".join(str(seg.get("text") or "") for seg in segments)
    matched = [gram for gram in grams if gram in transcript_text]
    if len(matched) <= max(1, len(grams) // 10):
        return [
            "Bilibili subtitle appears topic-mismatched with the video title; "
            "title keywords barely appear in the transcript. Use force_asr=true "
            "to verify against audio before analysis."
        ]
    return []


def _cjk_bigrams(text: str) -> list[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text or "")
    seen: set[str] = set()
    grams: list[str] = []
    for i in range(len(chars) - 1):
        gram = "".join(chars[i : i + 2])
        if gram not in seen:
            seen.add(gram)
            grams.append(gram)
    return grams
