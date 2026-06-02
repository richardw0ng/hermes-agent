import json
import builtins
import types
from types import SimpleNamespace
from pathlib import Path

from plugins.bilibili_analyzer import analysis
from plugins.bilibili_analyzer import asr
from plugins.bilibili_analyzer import bilibili
from plugins.bilibili_analyzer import tools
from plugins.market_sentiment import sources as market_sources
from plugins.market_sentiment import tools as market_tools


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


def test_faster_whisper_worker_failure_becomes_asr_error(monkeypatch, tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_text("x", encoding="utf-8")

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="native crash")

    monkeypatch.delenv("BILIBILI_WHISPER_IN_PROCESS", raising=False)
    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    try:
        asr.transcribe_audio(audio, "faster_whisper")
    except asr.AsrError as exc:
        assert "worker failed" in str(exc)
        assert "native crash" in str(exc)
    else:
        raise AssertionError("worker failure should raise AsrError")


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
            "cache_schema_version": tools.CACHE_SCHEMA_VERSION,
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


def test_topic_mismatched_subtitle_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "title": "普通人如何学习短线交易技巧",
            "owner": "up",
            "desc": "",
            "pages": [{"cid": 1, "page": 1, "duration": 300, "part": ""}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_subtitle_segments",
        lambda _bvid, _cid: (
            [
                {"start": 0, "end": 120, "text": "沃尔玛超市烘焙区有很多新品"},
                {"start": 120, "end": 295, "text": "这个可颂和零食区的价格比较适合探店"},
            ],
            {"fetcher": "bilibili_api"},
        ),
    )
    monkeypatch.setattr(tools, "fetch_subtitle_segments_ytdlp", lambda *_args: ([], None))
    monkeypatch.setattr(tools, "has_asr_provider", lambda _provider="auto": False)

    try:
        tools.fetch_transcript("BV1xx411c7mD", use_cache=False)
    except bilibili.BilibiliError as exc:
        assert "topic-mismatched" in str(exc)
    else:
        raise AssertionError("topic-mismatched subtitle should fail closed")


def test_topic_matched_subtitle_passes_title_check():
    warnings = tools._subtitle_quality_warnings(
        [
            {"start": 0, "end": 120, "text": "普通人学习短线交易技巧时要先理解风险"},
            {"start": 120, "end": 295, "text": "短线交易不是神秘方法，也要有自己的计划"},
        ],
        300,
        title="普通人如何学习短线交易技巧",
    )
    assert warnings == []


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


def test_download_audio_retries_incomplete_ytdlp_download(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("BILIBILI_AUDIO_DOWNLOAD_ATTEMPTS", "2")
    monkeypatch.setattr(
        bilibili,
        "download_audio_via_playurl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            bilibili.BilibiliError("Bilibili playurl did not return audio streams.")
        ),
    )
    calls = []

    def fake_download(url, outtmpl):
        calls.append((url, outtmpl))
        if len(calls) == 1:
            part = bilibili.cache_dir() / "BV1xx411c7mD_1.m4a.part"
            part.write_bytes(b"partial")
            raise bilibili.BilibiliError("1048576 bytes read, 12088709 more expected")
        out = bilibili.cache_dir() / "BV1xx411c7mD_1.m4a"
        out.write_bytes(b"audio")

    monkeypatch.setattr(bilibili, "_download_audio_ytdlp", fake_download)

    audio = bilibili.download_audio("BV1xx411c7mD", 1)

    assert len(calls) == 2
    assert audio.name == "BV1xx411c7mD_1.m4a"
    assert not (bilibili.cache_dir() / "BV1xx411c7mD_1.m4a.part").exists()


def test_download_audio_falls_back_to_playurl(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    real_import = builtins.__import__

    class FakeYdl:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def download(self, _urls):
            raise RuntimeError("no formats")

    fake_module = types.SimpleNamespace(YoutubeDL=FakeYdl)

    def fake_import(name, *args, **kwargs):
        if name == "yt_dlp":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    def fake_playurl(bvid, cid):
        out = bilibili.cache_dir() / f"{bvid}_{cid}.m4a"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"audio")
        return out

    monkeypatch.setattr(bilibili, "download_audio_via_playurl", fake_playurl)
    audio = bilibili.download_audio("BV1xx411c7mD", 1)
    assert audio.name == "BV1xx411c7mD_1.m4a"


def test_download_audio_via_playurl_selects_highest_bandwidth(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        bilibili,
        "_get_json",
        lambda _url: {
            "code": 0,
            "data": {
                "dash": {
                    "audio": [
                        {"bandwidth": 100, "baseUrl": "https://example.test/low.m4s"},
                        {"bandwidth": 300, "baseUrl": "https://example.test/high.m4s"},
                    ]
                }
            },
        },
    )
    seen = {}

    def fake_download(url, output, referer):
        seen["url"] = url
        seen["referer"] = referer
        output.write_bytes(b"audio")

    monkeypatch.setattr(bilibili, "_download_url_to_file", fake_download)
    path = bilibili.download_audio_via_playurl("BV1xx411c7mD", 1)
    assert seen["url"] == "https://example.test/high.m4s"
    assert seen["referer"].endswith("/BV1xx411c7mD/")
    assert path.read_bytes() == b"audio"
    assert path.with_suffix(path.suffix + ".url").read_text(encoding="utf-8") == "https://example.test/high.m4s"


def test_fetch_transcript_prefers_playurl_for_bailian_asr(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("BILIBILI_ASR_COMMAND", "python scripts/bilibili_asr_bailian.py {audio_path}")
    monkeypatch.setattr(bilibili, "extract_bvid", lambda _value: "BV1xx411c7mD")
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "bvid": "BV1xx411c7mD",
            "aid": 1,
            "title": "title",
            "owner": "owner",
            "desc": "",
            "duration": 10,
            "pages": [{"cid": 1, "page": 1, "part": "p1", "duration": 10}],
        },
    )
    monkeypatch.setattr(tools, "fetch_subtitle_segments", lambda *_args: ([], None))
    monkeypatch.setattr(tools, "fetch_subtitle_segments_ytdlp", lambda *_args: ([], None))
    seen = {}

    def fake_download(bvid, cid, *, prefer_playurl=False):
        seen["prefer_playurl"] = prefer_playurl
        audio = tmp_path / f"{bvid}_{cid}.m4a"
        audio.write_bytes(b"audio")
        return audio

    monkeypatch.setattr(tools, "download_audio", fake_download)
    monkeypatch.setattr(
        tools,
        "transcribe_audio",
        lambda _audio, _provider: ([{"start": 0, "end": 1, "text": "hello"}], "command"),
    )

    result = tools.fetch_transcript("BV1xx411c7mD", use_cache=False)
    assert seen["prefer_playurl"] is True
    assert result["segments"][0]["text"] == "hello"


def test_fetch_comments_filters_low_information_reactions(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        tools,
        "fetch_video_info",
        lambda _bvid: {
            "bvid": "BV1xx411c7mD",
            "aid": 123,
            "title": "短线交易入门",
            "owner": "owner",
            "desc": "",
            "duration": 10,
            "pages": [{"cid": 1, "page": 1, "part": "p1", "duration": 10}],
        },
    )

    def fake_comments(*_args, **_kwargs):
        return [
            {"rpid": 1, "user": "a", "message": "哈哈哈哈", "like": 100, "reply_count": 0, "ctime": 1},
            {"rpid": 2, "user": "b", "message": "支持支持支持", "like": 50, "reply_count": 0, "ctime": 2},
            {
                "rpid": 3,
                "user": "c",
                "message": "我觉得新手先看风险和仓位管理更重要，因为短线交易亏损往往来自瞎买瞎卖。",
                "like": 8,
                "reply_count": 2,
                "ctime": 3,
            },
            {
                "rpid": 4,
                "user": "d",
                "message": "请问视频里提到的书适合完全没有交易经验的人吗？有没有先后顺序？",
                "like": 2,
                "reply_count": 6,
                "ctime": 4,
            },
        ]

    monkeypatch.setattr(tools, "fetch_video_comments", fake_comments)
    result = tools.fetch_comments("BV1xx411c7mD", use_cache=False)
    messages = [item["message"] for item in result["comments"]]
    assert "哈哈哈哈" not in messages
    assert "支持支持支持" not in messages
    assert any("仓位管理" in message for message in messages)
    assert any("先后顺序" in message for message in messages)
    assert result["metadata"]["raw_comment_count"] == 4
    assert result["metadata"]["filtered_comment_count"] == 2


def test_fetch_video_comments_parses_reply_api(monkeypatch):
    calls = []

    def fake_get_json(url):
        calls.append(url)
        if "pn=1" in url:
            return {
                "code": 0,
                "data": {
                    "page": {"count": 2},
                    "replies": [
                        {
                            "rpid": 10,
                            "root": 0,
                            "parent": 0,
                            "member": {"uname": "alice", "mid": "1"},
                            "content": {"message": "这个观点有数据支撑吗？"},
                            "like": 12,
                            "rcount": 0,
                            "ctime": 100,
                        }
                    ],
                },
            }
        return {"code": 0, "data": {"page": {"count": 2}, "replies": []}}

    monkeypatch.setattr(bilibili, "_get_json", fake_get_json)
    comments = bilibili.fetch_video_comments("BV1xx411c7mD", 123, max_comments=1)
    assert len(comments) == 1
    assert comments[0]["user"] == "alice"
    assert comments[0]["message"] == "这个观点有数据支撑吗？"
    assert "type=1" in calls[0]
    assert "oid=123" in calls[0]


def test_fetch_video_comments_retries_when_ps_out_of_bounds(monkeypatch):
    calls = []

    def fake_get_json(url):
        calls.append(url)
        if "ps=20" in url:
            return {"code": -400, "message": "ps out of bounds"}
        return {
            "code": 0,
            "data": {
                "page": {"count": 1},
                "replies": [
                    {
                        "rpid": 10,
                        "root": 0,
                        "parent": 0,
                        "member": {"uname": "alice", "mid": "1"},
                        "content": {"message": "comment with enough detail"},
                        "like": 12,
                        "rcount": 0,
                        "ctime": 100,
                    }
                ],
            },
        }

    monkeypatch.setattr(bilibili, "_get_json", fake_get_json)
    comments = bilibili.fetch_video_comments("BV1xx411c7mD", 123, max_comments=200)
    assert len(comments) == 1
    assert "ps=20" in calls[0]
    assert any("ps=10" in call for call in calls)


def test_analyze_handler_returns_stage_and_partial_data_on_analysis_error(monkeypatch):
    transcript = {
        "metadata": {"bvid": "BV1xx411c7mD", "usable_for_analysis": True},
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
    comments = {
        "metadata": {"bvid": "BV1xx411c7mD"},
        "comments": [
            {
                "user": "alice",
                "message": "x" * 400,
                "like": 10,
                "reply_count": 0,
                "information_score": 5,
                "filter_reasons": ["substantial_text"],
            }
        ],
    }
    seen = {}

    monkeypatch.setattr(tools, "fetch_transcript", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(tools, "fetch_comments", lambda *_args, **_kwargs: comments)

    def fake_analyze(_llm, _metadata, _segments, **kwargs):
        seen["comments"] = kwargs["comments"]
        raise RuntimeError("model timeout")

    monkeypatch.setattr(tools, "analyze_transcript", fake_analyze)
    handler = tools.make_analyze_handler(object())
    result = json.loads(
        handler(
            {
                "url_or_bvid": "BV1xx411c7mD",
                "include_comments": True,
                "comment_limit": 30,
                "persist_file": False,
            }
        )
    )
    assert result["success"] is False
    assert result["stage"] == "analyze"
    assert result["transcript"] == transcript
    assert result["comments"]["analysis_comment_count"] == 1
    assert len(seen["comments"][0]["message"]) <= 320


def test_analyze_handler_archives_to_standard_output_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    transcript = {
        "metadata": {
            "bvid": "BV1xx411c7mD",
            "cid": 123,
            "title": "Sample Title",
            "usable_for_analysis": True,
        },
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }
    analysis_result = {"markdown": "# Analysis\n\nok"}

    monkeypatch.setattr(tools, "fetch_transcript", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(
        tools,
        "analyze_transcript",
        lambda *_args, **_kwargs: analysis_result,
    )
    handler = tools.make_analyze_handler(object())
    result = json.loads(handler({"url_or_bvid": "BV1xx411c7mD"}))

    output_dir = tmp_path / "outputs" / "bilibili_analyzer"
    output_file = output_dir / result["output_path"].split(str(output_dir))[-1].lstrip("\\/")
    assert result["success"] is True
    assert result["output_path"].startswith(str(output_dir))
    assert result["output_path"].endswith("_analysis.md")
    assert result["pdf_path"].endswith("_analysis.pdf")
    assert "# Analysis" in output_file.read_text(encoding="utf-8")
    assert Path(result["pdf_path"]).read_bytes().startswith(b"%PDF-")
    assert result["state_path"].endswith(".run.json")


def test_analyze_handler_archives_partial_file_on_analysis_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    transcript = {
        "metadata": {
            "bvid": "BV1xx411c7mD",
            "cid": 123,
            "title": "Sample Title",
            "usable_for_analysis": True,
        },
        "segments": [{"start": 0, "end": 1, "text": "hello transcript"}],
    }

    monkeypatch.setattr(tools, "fetch_transcript", lambda *_args, **_kwargs: transcript)

    def fail_analyze(*_args, **_kwargs):
        raise RuntimeError("model timeout")

    monkeypatch.setattr(tools, "analyze_transcript", fail_analyze)
    handler = tools.make_analyze_handler(object())
    result = json.loads(handler({"url_or_bvid": "BV1xx411c7mD"}))

    assert result["success"] is False
    assert result["stage"] == "analyze"
    assert result["output_path"].endswith("_analysis.md")
    output_text = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "Status: failed" in output_text
    assert "model timeout" in output_text
    assert "hello transcript" in output_text
    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["stage"] == "analyze"


def test_analyze_handler_can_disable_auto_archive(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    transcript = {
        "metadata": {"bvid": "BV1xx411c7mD", "cid": 123, "usable_for_analysis": True},
        "segments": [{"start": 0, "end": 1, "text": "hello"}],
    }

    monkeypatch.setattr(tools, "fetch_transcript", lambda *_args, **_kwargs: transcript)
    monkeypatch.setattr(
        tools,
        "analyze_transcript",
        lambda *_args, **_kwargs: {"markdown": "# Analysis\n\nok"},
    )
    handler = tools.make_analyze_handler(object())
    result = json.loads(
        handler({"url_or_bvid": "BV1xx411c7mD", "persist_file": False})
    )

    assert result["success"] is True
    assert result["output_path"] == ""
    assert not (tmp_path / "outputs" / "bilibili_analyzer").exists()


def test_analyze_handler_archives_partial_when_analysis_fails(monkeypatch, tmp_path):
    output_path = tmp_path / "failed.md"
    transcript = {
        "metadata": {
            "bvid": "BV1xx411c7mD",
            "cid": 123,
            "title": "Partial Title",
            "owner": "UP",
            "source": "asr:command",
            "usable_for_analysis": True,
        },
        "segments": [{"start": 0, "end": 1, "text": "hello transcript"}],
        "warnings": [],
    }

    monkeypatch.setattr(tools, "fetch_transcript", lambda *_args, **_kwargs: transcript)

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("Connection error")

    monkeypatch.setattr(tools, "analyze_transcript", fail_analysis)
    handler = tools.make_analyze_handler(object())
    result = json.loads(
        handler(
            {
                "url_or_bvid": "BV1xx411c7mD",
                "output_path": str(output_path),
            }
        )
    )

    assert result["success"] is False
    assert result["stage"] == "analyze"
    assert result["partial_archive"] is True
    assert result["output_path"] == str(output_path)
    text = output_path.read_text(encoding="utf-8")
    assert "Connection error" in text
    assert "hello transcript" in text


def test_following_group_latest_archives_batch(monkeypatch, tmp_path):
    class FakeLLM:
        def complete(self, **_kwargs):
            return SimpleNamespace(text="chunk note")

        def complete_structured(self, **_kwargs):
            return SimpleNamespace(
                parsed={
                    "video": {"title": "测试视频", "owner": "财经UP", "theme": "宏观"},
                    "main_thesis": "测试主旨",
                    "viewpoints": [
                        {
                            "claim": "测试观点",
                            "evidence": "测试论据",
                            "timestamps": ["00:00:01"],
                        }
                    ],
                    "opposing_or_controversial_points": [],
                    "key_facts": [],
                    "summary": "测试总结",
                },
                text="",
            )

    monkeypatch.setattr(
        tools,
        "fetch_follow_group_users",
        lambda **_kwargs: {
            "group": {"tagid": 123, "name": "财经", "count": 1},
            "groups": [],
            "users": [{"mid": 42, "name": "财经UP"}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_user_latest_videos",
        lambda *_args, **_kwargs: [
            {
                "bvid": "BV1xx411c7mD",
                "title": "测试视频",
                "created": 1760000000,
                "author": "财经UP",
            }
        ],
    )
    monkeypatch.setattr(
        tools,
        "fetch_transcript",
        lambda url_or_bvid, **_kwargs: {
            "metadata": {
                "bvid": url_or_bvid,
                "cid": 1,
                "page": 1,
                "title": "测试视频",
                "owner": "财经UP",
                "duration": 5,
                "source": "subtitle",
                "usable_for_analysis": True,
            },
            "segments": [{"start": 0, "end": 5, "text": "这是测试文稿"}],
            "warnings": [],
            "cache_hit": False,
        },
    )

    progress_events = []

    def progress_callback(event, name, message, payload):
        progress_events.append((event, name, message, payload))

    result = tools.analyze_following_group_latest(
        FakeLLM(),
        output_dir=str(tmp_path),
        per_up_limit=10,
        days_back=0,
        up_latest_delay_seconds=0,
        progress_callback=progress_callback,
    )

    assert result["analyzed_count"] == 1
    assert result["failure_count"] == 0
    index_path = tmp_path / "index.md"
    progress_path = tmp_path / "progress.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    assert result["index_path"] == str(index_path)
    assert result["progress_path"] == str(progress_path)
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["market_report_path"] == str(tmp_path / "market_report.md")
    assert result["market_report_pdf_path"] == str(tmp_path / "market_report.pdf")
    assert index_path.exists()
    assert progress_path.exists()
    assert checkpoint_path.exists()
    assert (tmp_path / "market_report.md").exists()
    assert (tmp_path / "market_report.pdf").read_bytes().startswith(b"%PDF-")
    assert "测试视频" in index_path.read_text(encoding="utf-8")
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    stages = {event["stage"] for event in progress_payload["events"]}
    assert {"group_fetch_start", "video_analyze_start", "market_report_complete", "batch_complete"} <= stages
    assert any(item[0] == "tool.progress" for item in progress_events)
    assert len(list(tmp_path.glob("*.md"))) == 3


def test_fetch_user_latest_videos_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "cache_dir", lambda: tmp_path)
    calls = []

    def fake_latest(mid, *, limit):
        calls.append((mid, limit))
        return [{"bvid": "BVcache", "created": 1760000000}]

    monkeypatch.setattr(tools, "fetch_user_latest_videos", fake_latest)

    first = tools._fetch_user_latest_videos_cached(
        42,
        limit=30,
        use_cache=True,
        force_refresh=False,
        cache_hours=12,
        attempts=1,
        rate_limit_pause_seconds=0,
    )
    second = tools._fetch_user_latest_videos_cached(
        42,
        limit=30,
        use_cache=True,
        force_refresh=False,
        cache_hours=12,
        attempts=1,
        rate_limit_pause_seconds=0,
    )

    assert len(calls) == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["videos"][0]["bvid"] == "BVcache"


def test_following_group_latest_caps_attempted_videos_when_analysis_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tools,
        "fetch_follow_group_users",
        lambda **_kwargs: {
            "group": {"tagid": 123, "name": "璐㈢粡", "count": 1},
            "groups": [],
            "users": [{"mid": 42, "name": "璐㈢粡UP"}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_user_latest_videos",
        lambda *_args, **_kwargs: [
            {"bvid": f"BV{i}xx411c7mD", "title": f"video {i}", "created": 1760000000 + i}
            for i in range(4)
        ],
    )
    monkeypatch.setattr(
        tools,
        "fetch_transcript",
        lambda url_or_bvid, **_kwargs: {
            "metadata": {
                "bvid": url_or_bvid,
                "cid": 1,
                "page": 1,
                "title": "fail video",
                "owner": "璐㈢粡UP",
                "duration": 5,
                "source": "subtitle",
                "usable_for_analysis": True,
            },
            "segments": [{"start": 0, "end": 5, "text": "transcript"}],
            "warnings": [],
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        tools,
        "analyze_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )

    result = tools.analyze_following_group_latest(
        object(),
        output_dir=str(tmp_path),
        per_up_limit=4,
        max_videos_total=2,
        days_back=0,
        generate_market_report=False,
        request_delay_seconds=0,
        up_latest_delay_seconds=0,
    )

    assert result["video_count"] == 2
    assert result["analyzed_count"] == 0
    assert result["failure_count"] == 2
    assert [item["video"]["bvid"] for item in result["failures"]] == [
        "BV0xx411c7mD",
        "BV1xx411c7mD",
    ]


def test_creator_stock_pick_backtest_archives_report(monkeypatch, tmp_path):
    class FakeLLM:
        def complete_structured(self, **kwargs):
            if kwargs.get("schema_name") == "bilibili_post_verification_comment_verdict":
                return SimpleNamespace(
                    parsed={
                        "verdict": "mostly_validated",
                        "score": 88,
                        "confidence": 0.8,
                        "summary": "Comments say the call was right.",
                    },
                    text='{"score":88}',
                    audit={},
                )
            return SimpleNamespace(
                parsed={
                    "picks": [
                        {
                            "stock_name": "NVIDIA",
                            "symbol": "NVDA",
                            "market": "US",
                            "stance": "bullish",
                            "horizon_days": 30,
                            "confidence": 0.8,
                            "thesis": "AI demand remains strong",
                            "evidence": "creator explicitly says NVDA is still a leader",
                        }
                    ]
                },
                text='{"picks":[]}',
                audit={},
            )

    monkeypatch.setattr(
        tools,
        "fetch_user_archive_videos",
        lambda *_args, **_kwargs: [
            {
                "bvid": "BV1stockpick",
                "title": "NVDA view",
                "created": 1760000000,
                "author": "财经UP",
            }
        ],
    )
    monkeypatch.setattr(
        tools,
        "fetch_transcript",
        lambda url_or_bvid, **_kwargs: {
            "metadata": {
                "bvid": url_or_bvid,
                "title": "NVDA view",
                "owner": "财经UP",
                "source": "subtitle",
                "usable_for_analysis": True,
            },
            "segments": [{"start": 0, "end": 5, "text": "NVDA is bullish"}],
        },
    )
    monkeypatch.setattr(
        tools,
        "score_pick_against_prices",
        lambda pick, **_kwargs: {
            "success": True,
            "score": 88.0,
            "symbol": "NVDA",
            "pct_change": 12.5,
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_comments",
        lambda *_args, **_kwargs: {
            "comments": [
                {
                    "message": "这个NVDA判断确实说中了",
                    "like": 10,
                    "ctime": 1761000000,
                }
            ]
        },
    )

    result = tools.backtest_creator_stock_picks(
        FakeLLM(),
        up_mid=42,
        start_date="2025-10-01",
        end_date="2025-11-01",
        output_dir=str(tmp_path),
    )

    assert result["video_count"] == 1
    assert result["pick_count"] == 1
    assert result["scored_pick_count"] == 1
    assert result["picks"][0]["comment_verdict"]["score"] == 88.0
    assert result["picks"][0]["final_score"]["score"] == 88.0
    assert result["average_score"] == 88.0
    assert result["report_path"] == str(tmp_path / "stock_pick_backtest.md")
    assert result["pdf_path"] == str(tmp_path / "stock_pick_backtest.pdf")
    assert (tmp_path / "stock_pick_backtest.md").exists()
    assert (tmp_path / "stock_pick_backtest.json").exists()
    assert (tmp_path / "stock_pick_backtest.pdf").read_bytes().startswith(b"%PDF-")


def test_market_sentiment_sources_normalize_and_filter(monkeypatch):
    def fake_fetch_json(url, **_kwargs):
        if "eastmoney" in url:
            return {
                "data": {
                    "list": [
                        {
                            "post_title": "新能源订单增长20%，估值仍低，看多后续修复",
                            "post_content": "公司订单增长20%，利润率改善，估值仍低。",
                            "post_like_count": 12,
                            "post_comment_count": 4,
                            "post_id": "1",
                        },
                        {"post_title": "666"},
                    ]
                }
            }
        if "xueqiu" in url:
            return {
                "list": [
                    {
                        "text": "<p>短期回调但现金流好，ROE维持高位，继续观察。</p>",
                        "like_count": 8,
                        "reply_count": 2,
                        "id": "2",
                        "user": {"id": "u1", "screen_name": "alice"},
                    }
                ]
            }
        return {
            "data": {
                "roll_data": [
                    {
                        "content": "财联社：AI芯片订单继续放量，产业链景气度修复。",
                        "id": "3",
                        "ctime": 1760000000,
                    }
                ]
            }
        }

    monkeypatch.setattr(market_sources, "_fetch_json", fake_fetch_json)

    payload = market_sources.fetch_market_sentiment_sources(
        keyword="AI",
        symbols=["600519"],
        max_items_per_source=5,
    )

    assert payload["errors"] == []
    assert payload["summary"]["item_count"] == 3
    assert any(item["source"] == "eastmoney_guba" for item in payload["items"])
    assert all(item["text"] != "666" for item in payload["items"])
    assert payload["summary"]["bullish_count"] >= 1


def test_eastmoney_guba_html_fallback(monkeypatch):
    monkeypatch.setattr(
        market_sources,
        "_fetch_json",
        lambda *_args, **_kwargs: {"re": [], "count": 0, "rc": 0, "me": "系统繁忙"},
    )
    monkeypatch.setattr(
        market_sources,
        "_fetch_text",
        lambda *_args, **_kwargs: """
        <tr class="listitem">
          <td><div class="read">46</div></td>
          <td><div class="reply">1</div></td>
          <td><div class="title"><a href="/news,600519,1719098865.html">贵州茅台这ROE真是稳得一批，最新33.65%</a></div></td>
          <td><div class="author"><a href="//i.eastmoney.com/u">beawan_han</a></div></td>
        </tr>
        """,
    )

    items = market_sources.fetch_eastmoney_guba(symbols=["600519"], limit=3)

    assert len(items) == 1
    assert items[0]["source"] == "eastmoney_guba"
    assert "ROE" in items[0]["title"]
    assert items[0]["reply_count"] == 1


def test_xueqiu_auto_mode_falls_back_to_browser(monkeypatch):
    monkeypatch.delenv("XUEQIU_FETCH_MODE", raising=False)

    def blocked_http(*_args, **_kwargs):
        raise market_sources.MarketSourceError("Invalid JSON from xueqiu.com")

    monkeypatch.setattr(market_sources, "_fetch_json", blocked_http)
    monkeypatch.setattr(
        market_sources,
        "_fetch_xueqiu_json_browser",
        lambda *_args, **_kwargs: {
            "list": [
                {
                    "text": "<p>cash flow improves and ROE remains strong</p>",
                    "like_count": 9,
                    "reply_count": 3,
                    "id": "123",
                    "user": {"id": "456", "screen_name": "alice"},
                }
            ]
        },
    )

    items = market_sources.fetch_xueqiu_discussions(
        symbols=["600519"],
        limit=5,
    )

    assert len(items) == 1
    assert items[0]["source"] == "xueqiu"
    assert items[0]["url"] == "https://xueqiu.com/456/123"


def test_xueqiu_http_mode_does_not_fall_back_to_browser(monkeypatch):
    monkeypatch.setenv("XUEQIU_FETCH_MODE", "http")
    browser_called = False

    def blocked_http(*_args, **_kwargs):
        raise market_sources.MarketSourceError("Invalid JSON from xueqiu.com")

    def browser_fetch(*_args, **_kwargs):
        nonlocal browser_called
        browser_called = True
        return {}

    monkeypatch.setattr(market_sources, "_fetch_json", blocked_http)
    monkeypatch.setattr(market_sources, "_fetch_xueqiu_json_browser", browser_fetch)

    payload = market_sources.fetch_market_sentiment_sources(
        keyword="maotai",
        symbols=["600519"],
        sources=["xueqiu"],
        max_items_per_source=5,
    )

    assert browser_called is False
    assert payload["items"] == []
    assert payload["errors"][0]["source"] == "xueqiu"
    assert "Invalid JSON" in payload["errors"][0]["error"]


def test_xueqiu_cookie_header_parser():
    cookies = market_sources._parse_cookie_header_for_domain(
        "xq_a_token=abc; xq_id_token=def; malformed",
        ".xueqiu.com",
    )

    assert [cookie["name"] for cookie in cookies] == ["xq_a_token", "xq_id_token"]
    assert all(cookie["domain"] == ".xueqiu.com" for cookie in cookies)


def test_fetch_market_sentiment_sources_tool(monkeypatch):
    monkeypatch.setattr(
        market_tools,
        "fetch_market_sentiment_sources",
        lambda **kwargs: {
            "keyword": kwargs["keyword"],
            "symbols": kwargs["symbols"],
            "items": [{"source": "cls", "text": "AI订单放量", "information_score": 80}],
            "errors": [],
            "summary": {"overall": "bullish", "item_count": 1},
        },
    )

    result = json.loads(
        market_tools.handle_fetch_sentiment_sources(
            {"keyword": "AI", "symbols": ["600519"], "max_items_per_source": 5}
        )
    )

    assert result["success"] is True
    assert result["keyword"] == "AI"
    assert result["summary"]["overall"] == "bullish"


def test_research_market_sources_collects_bilibili_and_finance_sources(monkeypatch):
    monkeypatch.setattr(
        market_sources,
        "search_bilibili_videos",
        lambda **_kwargs: [
            {
                "source": "bilibili",
                "title": "AI软件股观点",
                "text": "UP主认为软件股有修复机会",
                "sentiment": "bullish",
                "information_score": 70,
                "url": "https://www.bilibili.com/video/BV1",
            }
        ],
    )
    monkeypatch.setattr(
        market_sources,
        "fetch_eastmoney_guba",
        lambda **_kwargs: [
            {
                "source": "eastmoney_guba",
                "title": "软件板块放量",
                "text": "讨论资金回流和估值修复",
                "sentiment": "bullish",
                "information_score": 80,
                "url": "https://guba.eastmoney.com/news,1.html",
            }
        ],
    )

    payload = market_sources.research_market_sources(
        query="AI软件股",
        sources=["bilibili", "eastmoney_guba"],
        max_items_per_source=5,
    )

    assert payload["errors"] == []
    assert {item["source"] for item in payload["items"]} == {"bilibili", "eastmoney_guba"}
    assert payload["summary"]["bullish_count"] == 2


def test_market_research_sources_tool_synthesizes_with_llm(monkeypatch):
    class FakeLLM:
        def complete_structured(self, **kwargs):
            assert kwargs["schema_name"] == "market_cross_source_research_summary"
            return SimpleNamespace(
                parsed={
                    "executive_summary": "AI软件股讨论偏积极。",
                    "source_coverage": ["bilibili: 1", "eastmoney_guba: 1"],
                    "consensus": ["软件股有修复预期"],
                    "divergences": [],
                    "high_information_evidence": [
                        {
                            "source": "bilibili",
                            "claim": "UP主看好修复",
                            "evidence": "视频标题和摘要提到软件股机会",
                            "url": "https://www.bilibili.com/video/BV1",
                        }
                    ],
                    "risk_flags": ["样本少"],
                },
                text='{"executive_summary":"AI软件股讨论偏积极。"}',
                audit={},
            )

    monkeypatch.setattr(
        market_tools,
        "research_market_sources",
        lambda **_kwargs: {
            "query": "AI软件股",
            "sources": ["bilibili", "eastmoney_guba"],
            "items": [
                {
                    "source": "bilibili",
                    "title": "AI软件股观点",
                    "text": "UP主认为软件股有修复机会",
                    "sentiment": "bullish",
                    "information_score": 70,
                    "url": "https://www.bilibili.com/video/BV1",
                }
            ],
            "errors": [],
            "summary": {"overall": "bullish", "item_count": 1},
        },
    )

    handler = market_tools.make_research_sources_handler(FakeLLM())
    result = json.loads(handler({"query": "AI软件股"}))

    assert result["success"] is True
    assert result["analysis"]["executive_summary"] == "AI软件股讨论偏积极。"
    assert "High-information Evidence" in result["analysis"]["markdown"]


def test_following_group_latest_includes_market_sources(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        tools,
        "fetch_follow_group_users",
        lambda **_kwargs: {
            "group": {"tagid": 123, "name": "投资", "count": 1},
            "groups": [],
            "users": [{"mid": 42, "name": "财经UP"}],
        },
    )
    monkeypatch.setattr(
        tools,
        "fetch_user_latest_videos",
        lambda *_args, **_kwargs: [
            {"bvid": "BV1xx411c7mD", "title": "AI观点", "created": 1760000000}
        ],
    )
    monkeypatch.setattr(
        tools,
        "make_analyze_handler",
        lambda _llm: lambda *_args, **_kwargs: json.dumps(
            {
                "success": True,
                "transcript": {"segments": [{"start": 0, "end": 1, "text": "AI继续修复"}]},
                "analysis": {"summary": "AI继续修复", "main_thesis": "看多AI"},
                "comments": {"comments": []},
                "output_path": str(tmp_path / "video.md"),
            },
            ensure_ascii=False,
        ),
    )
    monkeypatch.setattr(
        tools,
        "fetch_market_sentiment_sources",
        lambda **_kwargs: {
            "items": [{"source": "cls", "title": "AI订单放量", "text": "AI订单放量", "sentiment": "bullish"}],
            "summary": {"overall": "bullish", "item_count": 1},
            "errors": [],
        },
    )

    def fake_analyze_market_batch(_llm, **kwargs):
        captured["market_sources"] = kwargs.get("market_sources")
        return {
            "executive_summary": "summary",
            "market_sentiment": {"overall": "bullish"},
            "source_coverage": [],
            "markdown": "# report",
        }

    monkeypatch.setattr(tools, "analyze_market_batch", fake_analyze_market_batch)

    result = tools.analyze_following_group_latest(
        object(),
        output_dir=str(tmp_path),
        max_up=1,
        per_up_limit=1,
        max_videos_total=1,
        days_back=0,
        request_delay_seconds=0,
        up_latest_delay_seconds=0,
        include_market_sources=True,
        market_source_symbols=["600519"],
        market_source_keyword="AI",
    )

    assert result["market_sources"]["summary"]["overall"] == "bullish"
    assert captured["market_sources"]["items"][0]["source"] == "cls"
    assert (tmp_path / "market_report.md").exists()


def test_plugin_registers_tools():
    from plugins.bilibili_analyzer import register

    registered = []

    class Ctx:
        llm = object()

        def register_auxiliary_task(self, **_kwargs):
            pass

        def register_tool(self, **kwargs):
            registered.append(kwargs)

    register(Ctx())
    names = {item["name"] for item in registered}
    assert names == {
        "bilibili_fetch_transcript",
        "bilibili_analyze_video",
        "bilibili_fetch_comments",
        "bilibili_analyze_following_group_latest",
        "bilibili_backtest_creator_stock_picks",
    }
    assert all(item["toolset"] == "bilibili" for item in registered)


def test_market_sentiment_plugin_registers_tool():
    from plugins.market_sentiment import register

    registered = []

    class Ctx:
        def register_tool(self, **kwargs):
            registered.append(kwargs)

    register(Ctx())
    assert {item["name"] for item in registered} == {
        "market_fetch_sentiment_sources",
        "market_research_sources",
    }
    assert all(item["toolset"] == "market_sentiment" for item in registered)
