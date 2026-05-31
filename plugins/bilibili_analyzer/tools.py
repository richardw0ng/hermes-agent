"""Hermes tool entrypoints for the Bilibili analyzer plugin."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .analysis import analyze_transcript
from .asr import has_asr_provider, transcribe_audio
from .bilibili import (
    BilibiliError,
    cache_dir,
    download_audio,
    extract_bvid,
    fetch_follow_group_users,
    fetch_video_comments,
    fetch_subtitle_segments,
    fetch_subtitle_segments_ytdlp,
    fetch_video_info,
    fetch_user_latest_videos,
    load_cached_comments,
    load_cached_transcript,
    load_cached_transcript_for_page,
    save_cached_comments,
    save_cached_transcript,
)

CACHE_SCHEMA_VERSION = 4

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
                "enum": ["markdown", "json"],
                "description": "Return markdown plus JSON fields, or JSON only.",
                "default": "markdown",
            },
            "chunk_minutes": {
                "type": "integer",
                "description": "Transcript chunk size for first-pass summaries.",
                "default": 4,
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
        "of the authenticated user's follow groups, such as 投资."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "group_name": {
                "type": "string",
                "description": "Follow group name. Defaults to 投资.",
                "default": "投资",
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
                "default": False,
            },
            "comment_sort": {
                "type": "string",
                "description": "Comment sort: hot or time. Defaults to hot.",
                "default": "hot",
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
        },
        "required": [],
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
                group_name=str(args.get("group_name") or "投资"),
                tagid=args.get("tagid"),
                per_up_limit=int(args.get("per_up_limit") or 10),
                max_up=int(args.get("max_up") or 0),
                max_videos_total=int(args.get("max_videos_total") or 0),
                output_dir=str(args.get("output_dir") or ""),
                chunk_minutes=int(args.get("chunk_minutes") or 4),
                asr_provider=str(args.get("asr_provider") or "auto"),
                force_asr=bool(args.get("force_asr", False)),
                allow_suspicious_subtitle=bool(args.get("allow_suspicious_subtitle", False)),
                include_comments=bool(args.get("include_comments", False)),
                max_comments=int(args.get("max_comments") or 200),
                comment_limit=int(args.get("comment_limit") or 30),
                include_comment_replies=bool(args.get("include_comment_replies", False)),
                comment_sort=str(args.get("comment_sort") or "hot"),
                use_cache=bool(args.get("use_cache", True)),
                force_refresh=bool(args.get("force_refresh", False)),
                continue_on_error=bool(args.get("continue_on_error", True)),
            )
            return _json({"success": True, **payload})
        except Exception as exc:
            return _json({"success": False, "error": str(exc)})

    return _handler


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
    group_name: str = "投资",
    tagid: int | None = None,
    per_up_limit: int = 10,
    max_up: int = 0,
    max_videos_total: int = 0,
    output_dir: str = "",
    chunk_minutes: int = 4,
    asr_provider: str = "auto",
    force_asr: bool = False,
    allow_suspicious_subtitle: bool = False,
    include_comments: bool = False,
    max_comments: int = 200,
    comment_limit: int = 30,
    include_comment_replies: bool = False,
    comment_sort: str = "hot",
    use_cache: bool = True,
    force_refresh: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    per_up_limit = min(50, max(1, int(per_up_limit or 10)))
    max_up = max(0, int(max_up or 0))
    max_videos_total = max(0, int(max_videos_total or 0))
    group_payload = fetch_follow_group_users(
        group_name=group_name,
        tagid=int(tagid) if tagid is not None else None,
        max_users=max_up,
    )
    group = group_payload["group"]
    users = group_payload["users"]
    batch_dir = _following_batch_output_dir(output_dir, group_name=str(group.get("name") or group_name))
    videos: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    processed = 0

    for user in users:
        try:
            latest = fetch_user_latest_videos(int(user["mid"]), limit=per_up_limit)
        except Exception as exc:
            failure = {
                "stage": "fetch_latest_videos",
                "up": user,
                "error": str(exc),
            }
            failures.append(failure)
            if not continue_on_error:
                break
            continue
        for video in latest[:per_up_limit]:
            if max_videos_total and processed >= max_videos_total:
                break
            video["owner_mid"] = user["mid"]
            video["owner_name"] = user.get("name") or video.get("author") or ""
            videos.append(video)
            output_path = batch_dir / _up_video_archive_filename(user, video)
            args = {
                "url_or_bvid": video["bvid"],
                "page": 1,
                "output_format": "markdown",
                "chunk_minutes": chunk_minutes,
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
            raw = make_analyze_handler(llm)(args)
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                item = {"success": False, "error": raw}
            item["up"] = user
            item["video"] = video
            if item.get("success"):
                processed += 1
                results.append(
                    {
                        "up": user,
                        "video": video,
                        "output_path": item.get("output_path") or str(output_path),
                        "summary": ((item.get("analysis") or {}).get("summary") or ""),
                        "main_thesis": ((item.get("analysis") or {}).get("main_thesis") or ""),
                    }
                )
            else:
                failures.append(
                    {
                        "stage": item.get("stage") or "analyze_video",
                        "up": user,
                        "video": video,
                        "error": item.get("error") or "Unknown analysis failure.",
                    }
                )
                if not continue_on_error:
                    break
        if max_videos_total and processed >= max_videos_total:
            break
        if failures and not continue_on_error:
            break

    index_path = _persist_following_batch_index(
        batch_dir,
        group=group,
        users=users,
        requested_per_up=per_up_limit,
        videos=videos,
        results=results,
        failures=failures,
    )
    return {
        "group": group,
        "up_count": len(users),
        "requested_per_up": per_up_limit,
        "video_count": len(videos),
        "analyzed_count": len(results),
        "failure_count": len(failures),
        "batch_dir": str(batch_dir),
        "index_path": str(index_path),
        "results": results,
        "failures": failures,
    }


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
        try:
            transcript = fetch_transcript(
                args.get("url_or_bvid") or "",
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
            comments_for_analysis = None
            if bool(args.get("include_comments", False)):
                stage = "fetch_comments"
                comments_payload = fetch_comments(
                    args.get("url_or_bvid") or "",
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
            stage = "analyze"
            analysis = analyze_transcript(
                llm,
                transcript["metadata"],
                transcript["segments"],
                comments=comments_for_analysis,
                output_format=str(args.get("output_format") or "markdown"),
                chunk_minutes=int(args.get("chunk_minutes") or 4),
            )
            if _bool_arg(args, "persist_file", True):
                stage = "persist"
                output_path = _persist_analysis(
                    transcript,
                    analysis,
                    str(args.get("output_path") or ""),
                )
            return _json(
                {
                    "success": True,
                    "transcript": transcript,
                    "comments": comments_payload,
                    "analysis": analysis,
                    "output_path": output_path,
                }
            )
        except Exception as exc:
            partial_output_path = ""
            if transcript and _bool_arg(args, "persist_file", True):
                try:
                    partial_output_path = _persist_partial_analysis(
                        transcript,
                        comments_payload=comments_payload,
                        analysis=analysis,
                        failed_stage=stage,
                        error=str(exc),
                        output_path=str(args.get("output_path") or output_path or ""),
                    )
                except Exception:
                    partial_output_path = ""
            return _json(
                {
                    "success": False,
                    "stage": stage,
                    "error": str(exc),
                    "transcript": transcript,
                    "comments": comments_payload,
                    "analysis": analysis,
                    "output_path": partial_output_path,
                    "partial_archive": bool(partial_output_path),
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
    comments: list[dict[str, Any]], *, limit: int = 30, max_chars: int = 240
) -> list[dict[str, Any]]:
    """Compact filtered comments before sending them into the global LLM pass."""
    compacted: list[dict[str, Any]] = []
    for item in comments[: min(max(1, int(limit or 30)), 12)]:
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
                    }
                )
            if len(replies) >= 2:
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
    path.write_text(markdown + "\n" + "\n".join(transcript_lines), encoding="utf-8")
    return str(path)


def _persist_partial_analysis(
    transcript: dict[str, Any],
    *,
    comments_payload: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    failed_stage: str = "",
    error: str = "",
    output_path: str = "",
) -> str:
    metadata = transcript.get("metadata") or {}
    bvid = str(metadata.get("bvid") or "bilibili")
    cid = str(metadata.get("cid") or "unknown")
    title = str(metadata.get("title") or "")
    path = _partial_archive_output_path(output_path, bvid=bvid, cid=cid, title=title)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# B站视频分析未完成归档",
        "",
        "## 失败信息",
        "",
        f"- 阶段：{failed_stage or 'unknown'}",
        f"- 错误：{error}",
        "",
        "## 视频信息",
        "",
        f"- 标题：{title}",
        f"- UP主：{metadata.get('owner', '')}",
        f"- BV：{bvid}",
        f"- CID：{cid}",
        f"- 来源：{metadata.get('source', '')}",
        f"- 时长：{metadata.get('duration', '')}",
        "",
    ]
    warnings = transcript.get("warnings") or metadata.get("warnings") or []
    if warnings:
        lines.extend(["## 警告", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    if analysis:
        lines.extend(["## 部分分析结果", "", "```json"])
        lines.append(json.dumps(analysis, ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
    if comments_payload:
        comments = comments_payload.get("comments") or []
        lines.extend(["## 评论样本", ""])
        for item in comments[:10]:
            lines.append(f"- {item.get('message', '')}")
        lines.append("")
    lines.extend(["## Transcript", ""])
    for seg in transcript.get("segments") or []:
        lines.append(
            f"- [{seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}] {seg.get('text', '')}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


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


def _up_video_archive_filename(user: dict[str, Any], video: dict[str, Any]) -> str:
    owner = _safe_filename_part(str(user.get("name") or video.get("owner_name") or "up"), 28)
    created = int(video.get("created") or 0)
    date_part = datetime.fromtimestamp(created).strftime("%Y%m%d") if created else "unknown-date"
    bvid = str(video.get("bvid") or "video")
    title = _safe_filename_part(str(video.get("title") or ""), 48)
    return f"{owner}_{date_part}_{bvid}_{title or 'analysis'}.md"


def _persist_following_batch_index(
    batch_dir: Path,
    *,
    group: dict[str, Any],
    users: list[dict[str, Any]],
    requested_per_up: int,
    videos: list[dict[str, Any]],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> Path:
    path = batch_dir / "index.md"
    lines = [
        "# B站关注分组批量分析归档",
        "",
        f"- 分组：{group.get('name', '')}",
        f"- tagid：{group.get('tagid', '')}",
        f"- 关注博主数：{len(users)}",
        f"- 每位博主请求最新视频数：{requested_per_up}",
        f"- 获取视频数：{len(videos)}",
        f"- 成功分析数：{len(results)}",
        f"- 失败数：{len(failures)}",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 成功归档",
        "",
    ]
    if not results:
        lines.append("- 暂无。")
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
            f"｜UP：{up.get('name', '')}｜BV：{video.get('bvid', '')}｜{created_text}"
        )
        thesis = str(item.get("main_thesis") or item.get("summary") or "").strip()
        if thesis:
            lines.append(f"  - 摘要：{thesis[:240]}")

    lines.extend(["", "## 失败记录", ""])
    if not failures:
        lines.append("- 无。")
    for item in failures:
        up = item.get("up") or {}
        video = item.get("video") or {}
        lines.append(
            f"- 阶段：{item.get('stage', '')}｜UP：{up.get('name', '')}｜"
            f"BV：{video.get('bvid', '')}｜错误：{item.get('error', '')}"
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
