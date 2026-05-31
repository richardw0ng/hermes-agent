"""Xiaohongshu / RedNote content fetching plugin."""

from __future__ import annotations

from .tools import (
    XHS_CHECK_STATUS_SCHEMA,
    XHS_FETCH_COMMENTS_SCHEMA,
    XHS_FETCH_NOTE_SCHEMA,
    XHS_SEARCH_NOTES_SCHEMA,
    check_requirements,
    handle_check_status,
    handle_fetch_comments,
    handle_fetch_note,
    handle_search_notes,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="xhs_check_status",
        toolset="xhs",
        schema=XHS_CHECK_STATUS_SCHEMA,
        handler=handle_check_status,
        check_fn=check_requirements,
        emoji="XHS",
    )
    ctx.register_tool(
        name="xhs_search_notes",
        toolset="xhs",
        schema=XHS_SEARCH_NOTES_SCHEMA,
        handler=handle_search_notes,
        check_fn=check_requirements,
        emoji="搜",
    )
    ctx.register_tool(
        name="xhs_fetch_note",
        toolset="xhs",
        schema=XHS_FETCH_NOTE_SCHEMA,
        handler=handle_fetch_note,
        check_fn=check_requirements,
        emoji="文",
    )
    ctx.register_tool(
        name="xhs_fetch_comments",
        toolset="xhs",
        schema=XHS_FETCH_COMMENTS_SCHEMA,
        handler=handle_fetch_comments,
        check_fn=check_requirements,
        emoji="评",
    )
