"""FastAPI service exposing the existing audience-mining pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent_layer.audience_agent import (
    AudienceAgent,
    AudiencePortfolio,
    MCPConnectionError,
    PersistentMCPToolClient,
    PipelineError,
    PipelineValidationError,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

LOGGER = logging.getLogger(__name__)
GENERIC_ERROR_MESSAGE = "An unexpected error occurred. Check server logs."


class TrendCapturingMCPClient:
    """Delegate the one MCP call while retaining its validated ticker data.

    ``AudienceAgent`` remains the sole pipeline owner and still performs exactly
    one MCP tool invocation. The wrapper only observes that return value so the
    frontend can render honest article-level view counts without changing the
    agent or its portfolio schema.
    """

    def __init__(self, client: PersistentMCPToolClient) -> None:
        self._client = client
        self.latest_articles: list[dict[str, int | str]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._client.call_tool(name, arguments)
        if name == "fetch_trending_wikipedia_articles":
            articles = AudienceAgent._coerce_articles(result)
            self.latest_articles = [
                {"title": article.title, "views": article.views}
                for article in articles
            ]
        return result


def create_app(
    *,
    client_factory: Callable[[], PersistentMCPToolClient] = PersistentMCPToolClient,
    agent_factory: Callable[[Any], AudienceAgent] = AudienceAgent,
) -> FastAPI:
    """Create the service, with injectable factories for isolated API tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        client = client_factory()
        client.start()
        application.state.mcp_client = client
        application.state.trend_client = TrendCapturingMCPClient(client)
        application.state.latest_portfolio = None
        application.state.agent_factory = agent_factory
        try:
            yield
        finally:
            client.close()

    application = FastAPI(
        title="Autonomous Audience Trend Miner API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").rstrip("/")
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/portfolio/latest")
    async def latest_portfolio() -> dict[str, Any]:
        portfolio: AudiencePortfolio | None = application.state.latest_portfolio
        if portfolio is None:
            raise HTTPException(
                status_code=404,
                detail="No audience portfolio is available. Run the miner first.",
            )
        return portfolio.model_dump(mode="json")

    @application.get("/api/trends/latest")
    async def latest_trends() -> dict[str, list[dict[str, int | str]]]:
        articles = application.state.trend_client.latest_articles
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No trend snapshot is available. Run the miner first.",
            )
        return {"articles": articles}

    @application.websocket("/ws/run")
    async def run_pipeline(websocket: WebSocket) -> None:
        await websocket.accept()
        progress_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()

        def report_progress(stage: str, detail: str) -> None:
            progress_queue.put_nowait(
                {"type": "progress", "stage": stage, "detail": detail}
            )

        agent = application.state.agent_factory(application.state.trend_client)
        run_task = asyncio.create_task(agent.run(report_progress))

        try:
            while not run_task.done() or not progress_queue.empty():
                try:
                    progress = await asyncio.wait_for(
                        progress_queue.get(), timeout=0.1
                    )
                except TimeoutError:
                    continue
                await websocket.send_json(progress)

            portfolio = await run_task
            application.state.latest_portfolio = portfolio
            await websocket.send_json(
                {
                    "type": "result",
                    "portfolio": portfolio.model_dump(mode="json"),
                }
            )
        except WebSocketDisconnect:
            LOGGER.info("Audience-mining WebSocket disconnected before completion")
        except (PipelineError, MCPConnectionError, PipelineValidationError) as exc:
            LOGGER.warning("Audience pipeline failed: %s", exc)
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            LOGGER.exception("Unexpected audience pipeline failure")
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_json(
                    {"type": "error", "message": GENERIC_ERROR_MESSAGE}
                )
        finally:
            if not run_task.done():
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.close(code=1000)

    return application


app = create_app()
