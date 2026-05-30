"""ASR adapters for Bilibili analyzer."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class AsrError(RuntimeError):
    pass


def has_asr_provider(provider: str = "auto") -> bool:
    provider = (provider or "auto").strip().lower()
    if provider in {"auto", "command"} and os.getenv("BILIBILI_ASR_COMMAND"):
        return True
    if provider in {"auto", "faster_whisper", "faster-whisper", "local"}:
        try:
            import faster_whisper  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def _normalize_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("segments") or value.get("result") or value.get("data") or value
    if isinstance(value, str):
        return [{"start": 0.0, "end": 0.0, "text": value.strip()}] if value.strip() else []
    if not isinstance(value, list):
        raise AsrError("ASR output must be text, a list, or an object containing segments.")
    segments = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            start = end = 0.0
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
            start = float(item.get("start") or item.get("from") or 0)
            end = float(item.get("end") or item.get("to") or start)
        else:
            continue
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def transcribe_audio(audio_path: Path, provider: str = "auto") -> tuple[list[dict[str, Any]], str]:
    provider = (provider or "auto").strip().lower()
    if provider in {"auto", "command"} and os.getenv("BILIBILI_ASR_COMMAND"):
        return _transcribe_with_command(audio_path), "command"
    if provider in {"auto", "faster_whisper", "faster-whisper", "local"}:
        try:
            return _transcribe_with_faster_whisper(audio_path), "faster_whisper"
        except ImportError:
            if provider != "auto":
                raise AsrError("faster-whisper is not installed.")
        except Exception:
            if provider != "auto":
                raise
    raise AsrError(
        "No ASR provider is configured. Set BILIBILI_ASR_COMMAND to a command "
        "that prints transcript JSON, or install faster-whisper for local ASR."
    )


def _transcribe_with_command(audio_path: Path) -> list[dict[str, Any]]:
    template = os.environ["BILIBILI_ASR_COMMAND"]
    command = template.replace("{audio_path}", str(audio_path))
    completed = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("BILIBILI_ASR_TIMEOUT", "3600")),
        check=False,
    )
    if completed.returncode != 0:
        raise AsrError(
            "ASR command failed: " + (completed.stderr or completed.stdout)[-1000:]
        )
    output = completed.stdout.strip()
    try:
        return _normalize_segments(json.loads(output))
    except json.JSONDecodeError:
        return _normalize_segments(output)


def _transcribe_with_faster_whisper(audio_path: Path) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel  # type: ignore

    model_name = os.getenv("BILIBILI_WHISPER_MODEL", "medium")
    device = os.getenv("BILIBILI_WHISPER_DEVICE", "auto")
    compute_type = os.getenv("BILIBILI_WHISPER_COMPUTE_TYPE", "auto")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        language=os.getenv("BILIBILI_ASR_LANGUAGE", "zh"),
        vad_filter=True,
        word_timestamps=False,
    )
    return [
        {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        for seg in segments
        if seg.text and seg.text.strip()
    ]
