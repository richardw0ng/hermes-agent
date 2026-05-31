"""Plain-text ASR adapter for Hermes using Bailian/DashScope.

This wraps ``scripts/bilibili_asr_bailian.py`` so it can serve both:

- Bilibili analyzer ASR fallback, which may provide ``<audio_path>.url``.
- Generic Hermes STT command providers, which usually pass only a local file.

The underlying adapter returns JSON segments.  This script prints only the
joined transcript text so ``tools.transcription_tools`` can consume it as a
command STT provider.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bilibili_asr_bailian import transcribe


def _ensure_upload_when_no_sidecar(audio_path: Path) -> None:
    sidecar = audio_path.with_suffix(audio_path.suffix + ".url")
    if not sidecar.exists() and not os.getenv("BILIBILI_DASHSCOPE_UPLOAD_AUDIO"):
        os.environ["BILIBILI_DASHSCOPE_UPLOAD_AUDIO"] = "1"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python scripts/asr_bailian_text.py <audio_path>", file=sys.stderr)
        return 2
    audio_path = Path(argv[1]).expanduser()
    try:
        _ensure_upload_when_no_sidecar(audio_path)
        result = transcribe(audio_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    segments = result.get("segments") or []
    text = "\n".join(
        str(item.get("text") or "").strip()
        for item in segments
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ).strip()
    if not text:
        print("DashScope ASR returned no transcript text.", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
