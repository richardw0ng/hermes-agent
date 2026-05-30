"""Hermes tool entrypoints for the Bilibili analyzer plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import analyze_transcript
from .asr import has_asr_provider, transcribe_audio
from .bilibili import (
    BilibiliError,
    cache_dir,
    download_audio,
    extract_bvid,
    fetch_subtitle_segments,
    fetch_subtitle_segments_ytdlp,
    fetch_video_info,
    load_cached_transcript,
    load_cached_transcript_for_page,
    save_cached_transcript,
)

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
                "description": "Write transcript and analysis to a persistent markdown file.",
                "default": False,
            },
            "output_path": {
                "type": "string",
                "description": "Optional output markdown path when persist_file is true.",
            },
        },
        "required": ["url_or_bvid"],
    },
}


def check_requirements() -> bool:
    return True


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


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
            cached["cache_hit"] = True
            return cached

    source = "subtitle"
    subtitle_meta = None
    segments: list[dict[str, Any]] = []
    warnings: list[str] = []
    if prefer_subtitle and not force_asr:
        segments, subtitle_meta = fetch_subtitle_segments(bvid, cid)
        warnings.extend(_subtitle_quality_warnings(segments, selected.get("duration") or info.get("duration") or 0))
        if not segments or warnings:
            ytdlp_segments, ytdlp_meta = fetch_subtitle_segments_ytdlp(bvid)
            ytdlp_warnings = _subtitle_quality_warnings(
                ytdlp_segments, selected.get("duration") or info.get("duration") or 0
            )
            if ytdlp_segments and (not segments or len(ytdlp_warnings) < len(warnings)):
                segments = ytdlp_segments
                subtitle_meta = ytdlp_meta
                warnings = ytdlp_warnings

    audio_path = None
    if not segments:
        source = "asr"
        audio_path = download_audio(bvid, cid)
        segments, asr_provider_used = transcribe_audio(audio_path, asr_provider)
        source = f"asr:{asr_provider_used}"
    elif warnings and has_asr_provider(asr_provider):
        try:
            audio_path = download_audio(bvid, cid)
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
        "cache_schema_version": 3,
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
            analysis = analyze_transcript(
                llm,
                transcript["metadata"],
                transcript["segments"],
                output_format=str(args.get("output_format") or "markdown"),
                chunk_minutes=int(args.get("chunk_minutes") or 4),
            )
            output_path = ""
            if bool(args.get("persist_file", False)):
                output_path = _persist_analysis(
                    transcript,
                    analysis,
                    str(args.get("output_path") or ""),
                )
            return _json(
                {
                    "success": True,
                    "transcript": transcript,
                    "analysis": analysis,
                    "output_path": output_path,
                }
            )
        except Exception as exc:
            return _json({"success": False, "error": str(exc)})

    return _handler


def _persist_analysis(
    transcript: dict[str, Any], analysis: dict[str, Any], output_path: str = ""
) -> str:
    metadata = transcript.get("metadata") or {}
    bvid = str(metadata.get("bvid") or "bilibili")
    cid = str(metadata.get("cid") or "unknown")
    path = Path(output_path).expanduser() if output_path else (
        cache_dir().parent / "reports" / "bilibili" / f"{bvid}_{cid}_analysis.md"
    )
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


def _cache_is_current(payload: dict[str, Any]) -> bool:
    metadata = payload.get("metadata") or {}
    return int(metadata.get("cache_schema_version") or 0) >= 3


def _subtitle_quality_warnings(
    segments: list[dict[str, Any]], duration: int | float
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
    return warnings
