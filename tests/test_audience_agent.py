from __future__ import annotations

import asyncio
import unittest

from agent_layer.audience_agent import (
    Article,
    AudienceAgent,
    AudiencePortfolio,
    AudienceSegment,
    BuyingPowerAssessment,
    CandidateCluster,
    ClusterCritique,
    ClusterSet,
    InitialClusterSet,
    PipelineValidationError,
)


class FakeMCPClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


class FakeStructuredModel:
    def __init__(self, owner, schema):
        self.owner = owner
        self.schema = schema

    async def ainvoke(self, messages):
        self.owner.invocations.append((self.schema, messages))
        result = self.owner.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    def with_structured_output(self, schema, method):
        if method != "json_schema":
            raise AssertionError("Pipeline must use native JSON-schema output")
        return FakeStructuredModel(self, schema)


class AudienceAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.articles = [
            Article(title="Solar panel", views=600),
            Article(title="Heat pump", views=300),
            Article(title="Film premiere", views=100),
        ]

    def test_size_index_is_relative_to_total_trending_traffic(self) -> None:
        clusters = ClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Eco Home Upgraders",
                    article_titles=["Solar panel", "Heat pump"],
                    rationale="Shared interest in practical residential energy upgrades.",
                )
            ]
        )

        metrics = AudienceAgent._cluster_metrics(clusters, self.articles)

        self.assertEqual(metrics["Eco Home Upgraders"]["cluster_views"], 900)
        self.assertEqual(metrics["Eco Home Upgraders"]["estimated_size_index"], 90.0)

    def test_unsupported_refined_article_fails_validation(self) -> None:
        clusters = ClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Invented Theme",
                    article_titles=["Made up article"],
                    rationale="This rationale is long enough but not supported by input.",
                )
            ]
        )
        with self.assertRaisesRegex(PipelineValidationError, "unsupported"):
            AudienceAgent._validate_supported_articles(clusters, self.articles)

    def test_coerces_langchain_mcp_text_content_blocks(self) -> None:
        raw_result = [
            {
                "type": "text",
                "text": '{"title": "Solar panel", "views": 600}',
                "id": "one",
            },
            {
                "type": "text",
                "text": '{"title": "Heat pump", "views": 300}',
                "id": "two",
            },
        ]

        articles = AudienceAgent._coerce_articles(raw_result)

        self.assertEqual(articles, self.articles[:2])

    def test_pipeline_calls_mcp_once_and_runs_bounded_llm_sequence(self) -> None:
        cluster = CandidateCluster(
            cluster_name="Eco Home Upgraders",
            article_titles=["Solar panel", "Heat pump"],
            rationale="Shared interest in practical residential energy upgrades.",
        )
        initial_clusters = InitialClusterSet(
            clusters=[
                cluster,
                cluster.model_copy(update={"cluster_name": "Green Transport Explorers"}),
                cluster.model_copy(update={"cluster_name": "Sustainable Tech Researchers"}),
                cluster.model_copy(update={"cluster_name": "Energy Cost Optimizers"}),
                cluster.model_copy(update={"cluster_name": "Climate Conscious Consumers"}),
            ]
        )
        refined_clusters = ClusterSet(clusters=[cluster])
        critique = ClusterCritique(
            needs_refinement=False,
            overall_assessment="The candidate is coherent and commercially actionable.",
            issues=[],
        )
        portfolio = AudiencePortfolio(
            segments=[
                AudienceSegment(
                    source_cluster_name="Eco Home Upgraders",
                    audience_name="Efficient Home Optimizers",
                    audience_description="Rising reading around two energy upgrades signals active home-efficiency curiosity.",
                    estimated_size_index=1.0,
                    potential_buying_power=BuyingPowerAssessment(
                        level="High",
                        rationale="Large-ticket home projects support valuable brand engagement.",
                        brand_categories=["HVAC", "Solar installers"],
                    ),
                    source_articles=["Solar panel", "Heat pump"],
                )
            ]
        )
        fake_llm = FakeLLM(
            [initial_clusters, critique, refined_clusters, portfolio]
        )
        fake_mcp = FakeMCPClient(
            [article.model_dump() for article in self.articles]
        )
        agent = AudienceAgent(fake_mcp, llm=fake_llm)  # type: ignore[arg-type]

        result = asyncio.run(agent.run())

        self.assertEqual(len(fake_mcp.calls), 1)
        self.assertEqual(
            [schema.__name__ for schema, _messages in fake_llm.invocations],
            [
                "InitialClusterSet",
                "ClusterCritique",
                "ClusterSet",
                "AudiencePortfolio",
            ],
        )
        self.assertEqual(result.segments[0].estimated_size_index, 90.0)

    def test_duplicate_article_assignment_fails_validation(self) -> None:
        clusters = ClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Eco Home Upgraders",
                    article_titles=["Solar panel"],
                    rationale="Interest in residential renewable-energy improvement products.",
                ),
                CandidateCluster(
                    cluster_name="Clean Energy Shoppers",
                    article_titles=["Solar panel"],
                    rationale="Interest in household products that can reduce energy impact.",
                ),
            ]
        )
        with self.assertRaisesRegex(PipelineValidationError, "multiple clusters"):
            AudienceAgent._validate_supported_articles(clusters, self.articles)

    def test_portfolio_parsing_failure_retries_once(self) -> None:
        cluster = CandidateCluster(
            cluster_name="Eco Home Upgraders",
            article_titles=["Solar panel", "Heat pump"],
            rationale="Shared interest in practical residential energy upgrades.",
        )
        refined = ClusterSet(clusters=[cluster])
        metrics = AudienceAgent._cluster_metrics(refined, self.articles)
        portfolio = AudiencePortfolio(
            segments=[
                AudienceSegment(
                    source_cluster_name="Eco Home Upgraders",
                    audience_name="Efficient Home Optimizers",
                    audience_description="Rising reading around energy upgrades signals active efficiency curiosity.",
                    estimated_size_index=90.0,
                    potential_buying_power=BuyingPowerAssessment(
                        level="High",
                        rationale="Large-ticket projects can support valuable brand engagement.",
                        brand_categories=["HVAC", "Solar installers"],
                    ),
                    source_articles=["Solar panel", "Heat pump"],
                )
            ]
        )
        fake_llm = FakeLLM([ValueError("invalid structured response"), portfolio])
        agent = AudienceAgent(FakeMCPClient([]), llm=fake_llm)  # type: ignore[arg-type]

        result = asyncio.run(agent._portfolio_pass(refined, metrics))

        self.assertEqual(result, portfolio)
        self.assertEqual(len(fake_llm.invocations), 2)


if __name__ == "__main__":
    unittest.main()
