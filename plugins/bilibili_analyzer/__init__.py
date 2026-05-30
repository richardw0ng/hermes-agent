"""Bilibili video transcript, comments, and viewpoint analysis plugin."""

from __future__ import annotations

from .tools import (
    BILI_ANALYZE_VIDEO_SCHEMA,
    BILI_FETCH_COMMENTS_SCHEMA,
    BILI_FETCH_TRANSCRIPT_SCHEMA,
    check_requirements,
    handle_fetch_comments,
    handle_fetch_transcript,
    make_analyze_handler,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="bilibili_fetch_transcript",
        toolset="bilibili",
        schema=BILI_FETCH_TRANSCRIPT_SCHEMA,
        handler=handle_fetch_transcript,
        check_fn=check_requirements,
        emoji="BV",
    )
    ctx.register_tool(
        name="bilibili_analyze_video",
        toolset="bilibili",
        schema=BILI_ANALYZE_VIDEO_SCHEMA,
        handler=make_analyze_handler(ctx.llm),
        check_fn=check_requirements,
        emoji="观点",
    )
    ctx.register_tool(
        name="bilibili_fetch_comments",
        toolset="bilibili",
        schema=BILI_FETCH_COMMENTS_SCHEMA,
        handler=handle_fetch_comments,
        check_fn=check_requirements,
        emoji="评论",
    )
