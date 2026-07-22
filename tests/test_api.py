from __future__ import annotations

import asyncio
import unittest

from fastapi.testclient import TestClient

from agent_layer.audience_agent import (
    AudiencePortfolio,
    AudienceSegment,
    BuyingPowerAssessment,
    PipelineValidationError,
    PlacementDecision,
)
from api_layer.main import GENERIC_ERROR_MESSAGE, create_app


def sample_portfolio() -> AudiencePortfolio:
    decision = PlacementDecision(
        article_title="Heat pump",
        cluster_name="Eco Home Upgraders",
        primary_relevance="Residential energy upgrade",
        fit_rationale="The topic represents a practical household efficiency investment.",
        ambiguity_resolution="No material ambiguity exists for this household upgrade.",
    )
    return AudiencePortfolio(
        segments=[
            AudienceSegment(
                source_cluster_name="Eco Home Upgraders",
                audience_name="Efficient Home Optimizers",
                audience_description=(
                    "Rising traffic around Heat pump signals attention on practical "
                    "home-efficiency upgrades. This audience may matter to HVAC and "
                    "home-improvement brands serving energy-conscious households."
                ),
                estimated_size_index=42.5,
                potential_buying_power=BuyingPowerAssessment(
                    level="High",
                    rationale="Large-ticket household upgrades create valuable consideration.",
                    brand_categories=["HVAC", "Home improvement"],
                ),
                source_articles=["Heat pump"],
                placement_decisions=[decision],
            )
        ]
    )


class FakeMCPClient:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    async def call_tool(self, name, arguments):
        if name != "fetch_trending_wikipedia_articles":
            raise AssertionError(f"Unexpected tool: {name}")
        return [{"title": "Heat pump", "views": 12_345}]


class SuccessfulAgent:
    def __init__(self, client) -> None:
        self.client = client

    async def run(self, progress_callback):
        progress_callback("fetch", "Fetching trends via MCP")
        await self.client.call_tool("fetch_trending_wikipedia_articles", {})
        await asyncio.sleep(0)
        progress_callback("cluster", "Clustering into audience themes")
        progress_callback("portfolio", "Generating audience portfolio")
        return sample_portfolio()


class FailingAgent:
    def __init__(self, _client) -> None:
        pass

    async def run(self, progress_callback):
        progress_callback("fetch", "Fetching trends via MCP")
        raise PipelineValidationError("The model returned an invalid portfolio.")


class UnexpectedFailureAgent:
    def __init__(self, _client) -> None:
        pass

    async def run(self, _progress_callback):
        raise RuntimeError("private server detail")


class FastAPITests(unittest.TestCase):
    def test_lifespan_health_websocket_and_latest_results(self) -> None:
        fake_client = FakeMCPClient()
        app = create_app(
            client_factory=lambda: fake_client,  # type: ignore[arg-type]
            agent_factory=SuccessfulAgent,  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
            preflight = client.options(
                "/api/health",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(
                preflight.headers["access-control-allow-origin"],
                "http://localhost:5173",
            )
            missing = client.get("/api/portfolio/latest")
            self.assertEqual(missing.status_code, 404)
            self.assertIn("Run the miner first", missing.json()["detail"])

            with client.websocket_connect("/ws/run") as websocket:
                messages = [websocket.receive_json() for _ in range(4)]

            self.assertEqual(
                [message["type"] for message in messages],
                ["progress", "progress", "progress", "result"],
            )
            self.assertEqual(
                [message["stage"] for message in messages[:-1]],
                ["fetch", "cluster", "portfolio"],
            )
            self.assertEqual(
                messages[-1]["portfolio"]["segments"][0]["audience_name"],
                "Efficient Home Optimizers",
            )

            latest = client.get("/api/portfolio/latest")
            self.assertEqual(latest.status_code, 200)
            self.assertEqual(latest.json(), sample_portfolio().model_dump(mode="json"))

            trends = client.get("/api/trends/latest")
            self.assertEqual(
                trends.json(),
                {"articles": [{"title": "Heat pump", "views": 12_345}]},
            )

        self.assertEqual(fake_client.started, 1)
        self.assertEqual(fake_client.closed, 1)

    def test_pipeline_errors_are_human_readable(self) -> None:
        app = create_app(
            client_factory=FakeMCPClient,  # type: ignore[arg-type]
            agent_factory=FailingAgent,  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/run") as websocket:
                progress = websocket.receive_json()
                error = websocket.receive_json()

        self.assertEqual(progress["type"], "progress")
        self.assertEqual(
            error,
            {
                "type": "error",
                "message": "The model returned an invalid portfolio.",
            },
        )

    def test_unexpected_errors_do_not_leak_details(self) -> None:
        app = create_app(
            client_factory=FakeMCPClient,  # type: ignore[arg-type]
            agent_factory=UnexpectedFailureAgent,  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/run") as websocket:
                error = websocket.receive_json()

        self.assertEqual(
            error, {"type": "error", "message": GENERIC_ERROR_MESSAGE}
        )


if __name__ == "__main__":
    unittest.main()
