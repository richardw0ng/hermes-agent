import json
import builtins
import types

from plugins.bilibili_analyzer import analysis
from plugins.bilibili_analyzer import asr
from plugins.bilibili_analyzer import bilibili
from plugins.bilibili_analyzer import tools


def test_extract_bvid_from_url():
    url = "https://www.bilibili.com/video/BV1xx411c7mD/?spm_id_from=333.337"
    assert bilibili.extract_bvid(url) == "BV1xx411c7mD"


def test_chunk_segments_groups_by_time():
    segments = [
        {"start": 0, "end": 2, "text": "a"},
        {"start": 30, "end": 32, "text": "b"},
        {"start": 65, "end": 68, "text": "c"},
    ]
    chunks = analysis.chunk_segments(segments, chunk_seconds=60)
    assert len(chunks) == 2
    assert [item["text"] for item in chunks[0]] == ["a", "b"]
    assert [item["text"] for item in chunks[1]] == ["c"]


def test_command_asr_normalizes_json(monkeypatch, tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_text("x", encoding="utf-8")
    monkeypatch.setenv("BILIBILI_ASR_COMMAND", "fake-asr {audio_path}")

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"segments": [{"from": 1, "to": 3, "content": "你好"}]},
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    segments, provider = asr.transcribe_audio(audio, "command")
    assert provider == "command"
    assert segments == [{"start": 1.0, "end": 3.0, "text": "你好"}]


def test_cookie_header_reads_repo_local_cookie_file(monkeypatch, tmp_path):
    from plugins.bilibili_analyzer import private_cookies

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(private_cookies, "SESSDATA", "")
    monkeypatch.setattr(private_cookies, "BUVID3", "")
    cookie_dir = tmp_path / ".hermes"
    cookie_dir.mkdir()
    (cookie_dir / "bilibili_cookies.json").write_text(
        json.dumps({"SESSDATA": "sess", "BUVID3": "buvid"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("BILIBILI_SESSDATA", raising=False)
    monkeypatch.delenv("BILIBILI_BUVID3", raising=False)
    monkeypatch.delenv("BILIBILI_COOKIE_FILE", raising=False)

    header = bilibili._cookie_header()
    assert "SESSDATA=sess" in header
    assert "buvid3=buvid" in header


def test_cookie_header_reads_private_cookies_module(monkeypatch):
    from plugins.bilibili_analyzer import private_cookies

    monkeypatch.delenv("BILIBILI_SESSDATA", raising=False)
    monkeypatch.delenv("BILIBILI_BUVID3", raising=False)
    monkeypatch.delenv("BILIBILI_COOKIE_FILE", raising=False)
    monkeypatch.setattr(private_cookies, "SESSDATA", "hardcoded_sess")
    monkeypatch.setattr(private_cookies, "BUVID3", "hardcoded_buvid")

    header = bilibili._cookie_header()
    assert "SESSDATA=hardcoded_sess" in header
    assert "buvid3=hardcoded_buvid" in header


def test_cookiejar_contains_bilibili_cookies(monkeypatch):
    from plugins.bilibili_analyzer import private_cookies

    monkeypatch.delenv("BILIBILI_SESSDATA", raising=False)
    monkeypatch.delenv("BILIBILI_BUVID3", raising=False)
    monkeypatch.setattr(private_cookies, "SESSDATA", "sess")
    monkeypatch.setattr(private_cookies, "BUVID3", "buvid")

    jar = bilibili._bilibili_cookiejar()
    values = {cookie.name: cookie.value for cookie in jar}
    assert values["SESSDATA"] == "sess"
    assert values["buvid3"] == "buvid"


def test_fetch_transcript_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    payload = {
        "metadata": {
            "bvid": "BV1xx411c7mD",
            "cid": 123,
            "source": "subtitle",
            "cache_schema_version": 3,
        },
        "segments": [{"start": 0, "end": 1, "text": "cached"}],
    }
    bilibili.save_cached_transcript(payload)

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("network should not be called on cache hit")

    monkeypatch.setattr(tools, "fetch_video_info", fail_fetch)
    result = tools.fetch_transcript("BV1xx411c7mD", use_cache=True)
    assert result["cache_hit"] is True
    assert result["segments"][0]["text"] == "cached"


def test_suspicious_subtitle_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "title": "short trading",
            "owner": "up",
            "desc": "",
            "pages": [{"cid": 1, "page": 1, "duration": 300, "part": ""}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_subtitle_segments",
        lambda _bvid, _cid: ([{"start": 0, "end": 10, "text": "wrong"}], {}),
    )
    monkeypatch.setattr(tools, "has_asr_provider", lambda _provider="auto": False)

    try:
        tools.fetch_transcript("BV1xx411c7mD", use_cache=False)
    except bilibili.BilibiliError as exc:
        assert "quality check failed" in str(exc)
    else:
        raise AssertionError("suspicious subtitle should fail closed")


def test_suspicious_subtitle_can_be_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "title": "short trading",
            "owner": "up",
            "desc": "",
            "pages": [{"cid": 1, "page": 1, "duration": 300, "part": ""}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_subtitle_segments",
        lambda _bvid, _cid: ([{"start": 0, "end": 10, "text": "raw"}], {}),
    )
    monkeypatch.setattr(tools, "has_asr_provider", lambda _provider="auto": False)

    result = tools.fetch_transcript(
        "BV1xx411c7mD",
        use_cache=False,
        allow_suspicious_subtitle=True,
    )
    assert result["metadata"]["quality_status"] == "suspect"
    assert result["metadata"]["usable_for_analysis"] is False


def test_ytdlp_bilibili_json_subtitle_parser():
    segments = bilibili._segments_from_ytdlp_subtitle_entry(
        {
            "ext": "json",
            "data": json.dumps(
                {
                    "body": [
                        {"from": 1.2, "to": 2.5, "content": "hello"},
                        {"from": 2.5, "to": 4.0, "content": "world"},
                    ]
                }
            ),
        }
    )
    assert segments == [
        {"start": 1.2, "end": 2.5, "text": "hello"},
        {"start": 2.5, "end": 4.0, "text": "world"},
    ]


def test_ytdlp_vtt_subtitle_parser():
    segments = bilibili._segments_from_ytdlp_subtitle_entry(
        {
            "ext": "vtt",
            "data": "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nhello\n\n00:00:02.500 --> 00:00:04.000\nworld\n",
        }
    )
    assert segments == [
        {"start": 1.0, "end": 2.5, "text": "hello"},
        {"start": 2.5, "end": 4.0, "text": "world"},
    ]


def test_suspicious_direct_subtitle_can_use_ytdlp_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "title": "short trading",
            "owner": "up",
            "desc": "",
            "pages": [{"cid": 1, "page": 1, "duration": 300, "part": ""}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_subtitle_segments",
        lambda _bvid, _cid: ([{"start": 0, "end": 10, "text": "short"}], {"fetcher": "direct"}),
    )
    monkeypatch.setattr(
        tools,
        "fetch_subtitle_segments_ytdlp",
        lambda _bvid: (
            [{"start": 0, "end": 290, "text": "full"}],
            {"fetcher": "yt_dlp"},
        ),
    )

    result = tools.fetch_transcript("BV1xx411c7mD", use_cache=False)
    assert result["metadata"]["subtitle"]["fetcher"] == "yt_dlp"
    assert result["metadata"]["quality_status"] == "ok"
    assert result["segments"][0]["text"] == "full"


def test_download_audio_cli_avoids_ffmpeg_postprocessor(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yt_dlp":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        out = bilibili.cache_dir() / "BV1xx411c7mD_1.m4a"
        out.write_text("audio", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(bilibili.subprocess, "run", fake_run)
    audio = bilibili.download_audio("BV1xx411c7mD", 1)
    assert audio.suffix == ".m4a"
    assert "-x" not in calls["command"]
    assert "--add-header" in calls["command"]
    assert calls["kwargs"]["encoding"] == "utf-8"
    assert calls["kwargs"]["errors"] == "replace"


def test_plugin_registers_tools():
    from plugins.bilibili_analyzer import register

    registered = []

    class Ctx:
        llm = object()

        def register_tool(self, **kwargs):
            registered.append(kwargs)

    register(Ctx())
    names = {item["name"] for item in registered}
    assert names == {"bilibili_fetch_transcript", "bilibili_analyze_video"}
    assert all(item["toolset"] == "bilibili" for item in registered)
