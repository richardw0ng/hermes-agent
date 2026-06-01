"""Bilibili video transcript, comments, and viewpoint analysis plugin."""

from __future__ import annotations

from .tools import (
    BILI_ANALYZE_VIDEO_SCHEMA,
    BILI_ANALYZE_FOLLOWING_GROUP_LATEST_SCHEMA,
    BILI_CREATOR_STOCK_PICK_BACKTEST_SCHEMA,
    BILI_FETCH_COMMENTS_SCHEMA,
    BILI_FETCH_TRANSCRIPT_SCHEMA,
    check_requirements,
    handle_fetch_comments,
    handle_fetch_transcript,
    make_analyze_handler,
    make_creator_stock_pick_backtest_handler,
    make_following_group_latest_handler,
)


def register(ctx) -> None:
    register_aux = getattr(ctx, "register_auxiliary_task", None)
    if register_aux:
        register_aux(
            key="plugin_bilibili_analyzer_bilibili_chunk_summary",
            display_name="Bilibili chunk summary",
            description="Summarize one Bilibili transcript chunk.",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 30,
                "extra_body": {},
            },
        )
        register_aux(
            key="plugin_bilibili_analyzer_bilibili_global_analysis",
            display_name="Bilibili video analysis",
            description="Extract structured viewpoints from one video.",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 45,
                "extra_body": {},
            },
        )
        register_aux(
            key="plugin_bilibili_analyzer_bilibili_batch_market_report",
            display_name="Bilibili market report",
            description="Synthesize video and comment signals into one report.",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 90,
                "extra_body": {},
            },
        )
        register_aux(
            key="plugin_bilibili_analyzer_bilibili_stock_pick_extraction",
            display_name="Bilibili stock-pick extraction",
            description="Extract explicit creator stock-pick claims.",
            defaults={
                "provider": "auto",
                "model": "",
                "timeout": 45,
                "extra_body": {},
            },
        )
    ctx.register_tool(
        name="bilibili_fetch_transcript",
        toolset="bilibili",
        schema=BILI_FETCH_TRANSCRIPT_SCHEMA,
        handler=handle_fetch_transcript,
        check_fn=check_requirements,
        emoji="BV",
    )
    ctx.register_tool(
        name="bilibili_backtest_creator_stock_picks",
        toolset="bilibili",
        schema=BILI_CREATOR_STOCK_PICK_BACKTEST_SCHEMA,
        handler=make_creator_stock_pick_backtest_handler(ctx.llm),
        check_fn=check_requirements,
        emoji="backtest",
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
    ctx.register_tool(
        name="bilibili_analyze_following_group_latest",
        toolset="bilibili",
        schema=BILI_ANALYZE_FOLLOWING_GROUP_LATEST_SCHEMA,
        handler=make_following_group_latest_handler(ctx.llm),
        check_fn=check_requirements,
        emoji="财经",
    )
