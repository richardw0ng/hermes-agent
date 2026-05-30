"""Subprocess entrypoint for faster-whisper transcription.

Keeping faster-whisper in a worker process prevents native decoder/model
failures from terminating the Hermes agent process.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from .asr import _transcribe_with_faster_whisper_in_process


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m plugins.bilibili_analyzer.whisper_worker <audio_path>", file=sys.stderr)
        return 2
    try:
        segments = _transcribe_with_faster_whisper_in_process(Path(args[0]))
        print(json.dumps({"segments": segments}, ensure_ascii=False))
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
