"""External market sentiment source plugin."""

from __future__ import annotations

from .tools import (
    MARKET_FETCH_SENTIMENT_SOURCES_SCHEMA,
    MARKET_RESEARCH_SOURCES_SCHEMA,
    check_requirements,
    handle_fetch_sentiment_sources,
    make_research_sources_handler,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="market_fetch_sentiment_sources",
        toolset="market_sentiment",
        schema=MARKET_FETCH_SENTIMENT_SOURCES_SCHEMA,
        handler=handle_fetch_sentiment_sources,
        check_fn=check_requirements,
        emoji="market",
    )
    ctx.register_tool(
        name="market_research_sources",
        toolset="market_sentiment",
        schema=MARKET_RESEARCH_SOURCES_SCHEMA,
        handler=make_research_sources_handler(getattr(ctx, "llm", None)),
        check_fn=check_requirements,
        emoji="research",
    )
