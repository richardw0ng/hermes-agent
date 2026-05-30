"""ASR adapters for Bilibili analyzer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class AsrError(RuntimeError):
    pass


def _has_bailian_key() -> bool:
    return bool(
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_TOKEN_PLAN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def has_asr_provider(provider: str = "auto") -> bool:
    provider = (provider or "auto").strip().lower()
    if provider in {"auto", "bailian", "dashscope", "aliyun"} and _has_bailian_key():
        return True
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
    command = os.getenv("BILIBILI_ASR_COMMAND", "").replace("\\", "/")
    if provider in {"bailian", "dashscope", "aliyun"}:
        return _transcribe_with_bailian(audio_path), "dashscope"
    if provider == "auto" and _has_bailian_key() and "bilibili_asr_bailian.py" in command:
        return _transcribe_with_bailian(audio_path), "dashscope"
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
        encoding="utf-8",
        errors="replace",
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


def _transcribe_with_bailian(audio_path: Path) -> list[dict[str, Any]]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "bilibili_asr_bailian.py"
    command = [sys.executable, str(script), str(audio_path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.getenv("BILIBILI_ASR_TIMEOUT", "3600")),
        check=False,
    )
    if completed.returncode != 0:
        raise AsrError(
            "Bailian/DashScope ASR failed: " + (completed.stderr or completed.stdout)[-1000:]
        )
    try:
        return _normalize_segments(json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise AsrError(
            "Bailian/DashScope ASR returned non-JSON output: "
            + completed.stdout[-1000:]
        ) from exc


def _transcribe_with_faster_whisper(audio_path: Path) -> list[dict[str, Any]]:
    if os.getenv("BILIBILI_WHISPER_IN_PROCESS", "").strip() == "1":
        return _transcribe_with_faster_whisper_in_process(audio_path)

    command = [
        sys.executable,
        "-m",
        "plugins.bilibili_analyzer.whisper_worker",
        str(audio_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.getenv("BILIBILI_ASR_TIMEOUT", "3600")),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        exit_hex = f"0x{completed.returncode & 0xFFFFFFFF:08X}"
        raise AsrError(
            "faster-whisper worker failed "
            f"(exit {completed.returncode}, {exit_hex}). "
            f"{detail or 'No traceback was emitted; this is likely a native runtime crash.'}"
        )
    try:
        return _normalize_segments(json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise AsrError(
            "faster-whisper worker returned non-JSON output: "
            + completed.stdout[-1000:]
        ) from exc


def _transcribe_with_faster_whisper_in_process(audio_path: Path) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel  # type: ignore

    model_name = os.getenv("BILIBILI_WHISPER_MODEL", "medium")
    device = os.getenv("BILIBILI_WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("BILIBILI_WHISPER_COMPUTE_TYPE", "int8")
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
