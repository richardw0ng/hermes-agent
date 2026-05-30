"""ASR command adapter for Hermes' Bilibili analyzer using Bailian/DashScope.

The plugin invokes this script through BILIBILI_ASR_COMMAND. It expects the
audio downloader to create a sidecar file named "<audio_path>.url" containing
the original Bilibili CDN URL, then submits that URL to DashScope's recorded
speech transcription API and prints normalized JSON segments to stdout.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com")
SUBMIT_URL = f"{DASHSCOPE_BASE.rstrip('/')}/api/v1/services/audio/asr/transcription"
TASK_URL = f"{DASHSCOPE_BASE.rstrip('/')}/api/v1/tasks"
UPLOAD_POLICY_URL = f"{DASHSCOPE_BASE.rstrip('/')}/api/v1/uploads"


def _api_key() -> str:
    key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_TOKEN_PLAN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(
            "Missing DashScope API key. Set DASHSCOPE_API_KEY, "
            "BAILIAN_TOKEN_PLAN_API_KEY, or OPENAI_API_KEY."
        )
    return key


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    body = None
    headers = {"Authorization": f"Bearer {_api_key()}"}
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["X-DashScope-Async"] = "enable"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope HTTP {exc.code}: {detail}") from exc


def _read_audio_url(audio_path: Path) -> str:
    sidecar = audio_path.with_suffix(audio_path.suffix + ".url")
    if not sidecar.exists():
        raise RuntimeError(
            f"Missing source URL sidecar: {sidecar}. Set "
            "BILIBILI_PREFER_PLAYURL_AUDIO=1 or use the built-in command path "
            "so the plugin downloads audio via Bilibili playurl."
        )
    url = sidecar.read_text(encoding="utf-8").strip()
    if not url:
        raise RuntimeError(f"Source URL sidecar is empty: {sidecar}")
    return url


def _multipart_body(
    fields: dict[str, str], file_field: str, file_path: Path
) -> tuple[bytes, str]:
    boundary = "----hermes-dashscope-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _upload_audio(audio_path: Path) -> str:
    model = os.getenv("BILIBILI_DASHSCOPE_ASR_MODEL", "paraformer-v2")
    query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
    policy = _request_json(f"{UPLOAD_POLICY_URL}?{query}")
    data = policy.get("data") or policy.get("output") or policy
    upload_dir = str(data["upload_dir"]).rstrip("/")
    object_key = f"{upload_dir}/{audio_path.name}"
    fields = {
        "OSSAccessKeyId": str(data["oss_access_key_id"]),
        "Signature": str(data["signature"]),
        "policy": str(data["policy"]),
        "x-oss-object-acl": str(data.get("x_oss_object_acl", "private")),
        "x-oss-forbid-overwrite": str(data.get("x_oss_forbid_overwrite", "true")),
        "key": object_key,
        "success_action_status": "200",
    }
    body, content_type = _multipart_body(fields, "file", audio_path)
    request = urllib.request.Request(
        str(data["upload_host"]),
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope OSS upload HTTP {exc.code}: {detail}") from exc
    return f"oss://{object_key}"


def _submit(audio_url: str) -> str:
    extra_headers = {}
    if audio_url.startswith("oss://"):
        extra_headers["X-DashScope-OssResourceResolve"] = "enable"
    payload = {
        "model": os.getenv("BILIBILI_DASHSCOPE_ASR_MODEL", "paraformer-v2"),
        "input": {"file_urls": [audio_url]},
        "parameters": {
            "language_hints": [os.getenv("BILIBILI_ASR_LANGUAGE", "zh")],
        },
    }
    data = _request_json(
        SUBMIT_URL,
        method="POST",
        payload=payload,
        extra_headers=extra_headers,
    )
    task_id = ((data.get("output") or {}).get("task_id") or data.get("task_id") or "")
    if not task_id:
        raise RuntimeError(f"DashScope did not return task_id: {json.dumps(data, ensure_ascii=False)}")
    return str(task_id)


def _wait(task_id: str) -> dict[str, Any]:
    timeout = int(os.getenv("BILIBILI_ASR_TIMEOUT", "3600"))
    deadline = time.monotonic() + timeout
    interval = 2.0
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _request_json(f"{TASK_URL}/{task_id}")
        output = last.get("output") or {}
        status = str(output.get("task_status") or output.get("status") or "").upper()
        if status in {"SUCCEEDED", "SUCCESS"}:
            return last
        if status in {"FAILED", "CANCELED", "UNKNOWN"}:
            raise RuntimeError(f"DashScope ASR task failed: {json.dumps(last, ensure_ascii=False)}")
        time.sleep(interval)
        interval = min(interval * 1.5, 10.0)
    raise RuntimeError(f"DashScope ASR task timed out: {json.dumps(last, ensure_ascii=False)}")


def _task_failed_code(task: dict[str, Any]) -> str:
    output = task.get("output") or {}
    code = str(output.get("code") or "")
    if code:
        return code
    results = output.get("results") or []
    if isinstance(results, dict):
        results = [results]
    for item in results:
        if isinstance(item, dict) and item.get("code"):
            return str(item["code"])
    return ""


def _fetch_url(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _collect_transcription_payload(task: dict[str, Any]) -> list[Any]:
    output = task.get("output") or {}
    results = output.get("results") or output.get("result") or []
    if isinstance(results, dict):
        results = [results]
    payloads: list[Any] = []
    for item in results:
        if isinstance(item, dict) and item.get("transcription_url"):
            payloads.append(_fetch_url(str(item["transcription_url"])))
        else:
            payloads.append(item)
    if not payloads:
        payloads.append(output)
    return payloads


def _walk_sentences(value: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if isinstance(value, dict):
        sentence_lists = []
        for key in ("sentences", "sentence", "segments"):
            if isinstance(value.get(key), list):
                sentence_lists.append(value[key])
        for key in ("transcripts", "transcript", "results", "result"):
            child = value.get(key)
            if child is not None:
                segments.extend(_walk_sentences(child))
        for items in sentence_lists:
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or item.get("content") or "").strip()
                if not text:
                    continue
                begin = item.get("begin_time", item.get("start", item.get("from", 0)))
                end = item.get("end_time", item.get("end", item.get("to", begin)))
                start = float(begin or 0)
                finish = float(end or start)
                if start > 1000 or finish > 1000:
                    start /= 1000.0
                    finish /= 1000.0
                segments.append({"start": start, "end": finish, "text": text})
        if not segments:
            text = str(value.get("text") or "").strip()
            if text:
                segments.append({"start": 0.0, "end": 0.0, "text": text})
    elif isinstance(value, list):
        for item in value:
            segments.extend(_walk_sentences(item))
    elif isinstance(value, str) and value.strip():
        segments.append({"start": 0.0, "end": 0.0, "text": value.strip()})
    return segments


def transcribe(audio_path: Path) -> dict[str, Any]:
    if os.getenv("BILIBILI_DASHSCOPE_UPLOAD_AUDIO", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        audio_url = _upload_audio(audio_path)
    else:
        audio_url = _read_audio_url(audio_path)
    task_id = _submit(audio_url)
    try:
        task = _wait(task_id)
    except RuntimeError as exc:
        if audio_url.startswith("oss://") or "FILE_403_FORBIDDEN" not in str(exc):
            raise
        audio_url = _upload_audio(audio_path)
        task_id = _submit(audio_url)
        task = _wait(task_id)
    if _task_failed_code(task):
        raise RuntimeError(f"DashScope ASR task failed: {json.dumps(task, ensure_ascii=False)}")
    payloads = _collect_transcription_payload(task)
    segments: list[dict[str, Any]] = []
    for payload in payloads:
        segments.extend(_walk_sentences(payload))
    if not segments:
        raise RuntimeError(f"DashScope ASR returned no transcript: {json.dumps(task, ensure_ascii=False)}")
    return {
        "provider": "dashscope",
        "task_id": task_id,
        "audio_url": audio_url,
        "segments": segments,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python scripts/bilibili_asr_bailian.py <audio_path>", file=sys.stderr)
        return 2
    try:
        result = transcribe(Path(argv[1]))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
