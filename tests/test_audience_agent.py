from __future__ import annotations

import asyncio
import unittest

from agent_layer.audience_agent import (
    Article,
    ArticlePlacementReview,
    AudienceAgent,
    AudiencePortfolio,
    AudienceSegment,
    BuyingPowerAssessment,
    CandidateCluster,
    ClusterCritique,
    ClusterSet,
    CrossClusterOverlap,
    InitialClusterSet,
    PlacementDecision,
    PipelineValidationError,
    RefinedClusterSet,
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
        pipeline_articles = self.articles + [
            Article(title="Electric bicycle", views=100),
            Article(title="Home battery", views=100),
            Article(title="Recycling", views=100),
        ]
        initial_clusters = InitialClusterSet(
            clusters=[
                cluster,
                CandidateCluster(
                    cluster_name="Entertainment Launch Followers",
                    article_titles=["Film premiere"],
                    rationale="Interest in upcoming entertainment releases and launch moments.",
                ),
                CandidateCluster(
                    cluster_name="Green Transport Explorers",
                    article_titles=["Electric bicycle"],
                    rationale="Interest in lower-emission personal transportation choices.",
                ),
                CandidateCluster(
                    cluster_name="Energy Cost Optimizers",
                    article_titles=["Home battery"],
                    rationale="Interest in technology that changes household energy economics.",
                ),
                CandidateCluster(
                    cluster_name="Climate Conscious Consumers",
                    article_titles=["Recycling"],
                    rationale="Interest in lower-waste habits and circular consumer behavior.",
                ),
            ]
        )
        placement_reviews = [
            ArticlePlacementReview(
                article_title=title,
                assigned_cluster=candidate.cluster_name,
                fit="strong",
                recommended_cluster=candidate.cluster_name,
                reasoning="The article has a direct and commercially coherent fit with this theme.",
            )
            for candidate in initial_clusters.clusters
            for title in candidate.article_titles
        ]
        refined_clusters = RefinedClusterSet(
            clusters=[cluster],
            placement_decisions=[
                PlacementDecision(
                    article_title=title,
                    cluster_name=cluster.cluster_name,
                    primary_relevance="Residential energy upgrades",
                    fit_rationale="The topic directly represents a practical household efficiency investment.",
                    ambiguity_resolution="No material ambiguity exists for this household upgrade topic.",
                )
                for title in cluster.article_titles
            ],
        )
        critique = ClusterCritique(
            needs_refinement=False,
            overall_assessment="The candidate is coherent and commercially actionable.",
            issues=[],
            placement_reviews=placement_reviews,
            cross_cluster_overlaps=[],
        )
        portfolio = AudiencePortfolio(
            segments=[
                AudienceSegment(
                    source_cluster_name="Eco Home Upgraders",
                    audience_name="Efficient Home Optimizers",
                    audience_description=(
                        "Rising Wikipedia traffic around Solar panel and Heat pump signals "
                        "fresh attention on practical home-efficiency upgrades. This audience "
                        "shares an interest in reducing household energy use, making it relevant "
                        "to HVAC, renewable-energy, and home-improvement brands."
                    ),
                    estimated_size_index=1.0,
                    potential_buying_power=BuyingPowerAssessment(
                        level="High",
                        rationale="Large-ticket home projects support valuable brand engagement.",
                        brand_categories=["HVAC", "Solar installers"],
                    ),
                    source_articles=["Solar panel", "Heat pump"],
                    placement_decisions=refined_clusters.placement_decisions,
                )
            ]
        )
        fake_llm = FakeLLM(
            [initial_clusters, critique, refined_clusters, portfolio]
        )
        fake_mcp = FakeMCPClient(
            [article.model_dump() for article in pipeline_articles]
        )
        agent = AudienceAgent(fake_mcp, llm=fake_llm)  # type: ignore[arg-type]

        result = asyncio.run(agent.run())

        self.assertEqual(len(fake_mcp.calls), 1)
        self.assertEqual(
            [schema.__name__ for schema, _messages in fake_llm.invocations],
            [
                "InitialClusterSet",
                "ClusterCritique",
                "RefinedClusterSet",
                "AudiencePortfolio",
            ],
        )
        self.assertEqual(result.segments[0].estimated_size_index, 69.2)

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
        refined = RefinedClusterSet(
            clusters=[cluster],
            placement_decisions=[
                PlacementDecision(
                    article_title=title,
                    cluster_name=cluster.cluster_name,
                    primary_relevance="Residential energy upgrades",
                    fit_rationale="The topic directly represents a practical household efficiency investment.",
                    ambiguity_resolution="No material ambiguity exists for this household upgrade topic.",
                )
                for title in cluster.article_titles
            ],
        )
        metrics = AudienceAgent._cluster_metrics(refined, self.articles)
        portfolio = AudiencePortfolio(
            segments=[
                AudienceSegment(
                    source_cluster_name="Eco Home Upgraders",
                    audience_name="Efficient Home Optimizers",
                    audience_description=(
                        "Rising Wikipedia traffic around Solar panel and Heat pump signals "
                        "fresh attention on practical home-efficiency upgrades. This audience "
                        "shares an interest in reducing household energy use, making it relevant "
                        "to HVAC, renewable-energy, and home-improvement brands."
                    ),
                    estimated_size_index=90.0,
                    potential_buying_power=BuyingPowerAssessment(
                        level="High",
                        rationale="Large-ticket projects can support valuable brand engagement.",
                        brand_categories=["HVAC", "Solar installers"],
                    ),
                    source_articles=["Solar panel", "Heat pump"],
                    placement_decisions=refined.placement_decisions,
                )
            ]
        )
        fake_llm = FakeLLM([ValueError("invalid structured response"), portfolio])
        agent = AudienceAgent(FakeMCPClient([]), llm=fake_llm)  # type: ignore[arg-type]

        result = asyncio.run(agent._portfolio_pass(refined, metrics))

        self.assertEqual(result, portfolio)
        self.assertEqual(len(fake_llm.invocations), 2)

    def test_flagged_cross_cluster_article_requires_resolution(self) -> None:
        articles = [
            Article(title="David Beckham", views=500),
            Article(title="Zendaya", views=400),
            Article(title="Association football", views=300),
        ]
        clusters = [
            CandidateCluster(
                cluster_name="Pop Culture Aficionados",
                article_titles=["David Beckham", "Zendaya"],
                rationale="Interest in widely recognized entertainment and celebrity figures.",
            ),
            CandidateCluster(
                cluster_name="Football Enthusiasts",
                article_titles=["Association football"],
                rationale="Interest in professional football personalities and competitions.",
            ),
        ]
        critique = ClusterCritique(
            needs_refinement=True,
            overall_assessment="A cross-domain sports celebrity needs a dominant placement.",
            issues=[],
            placement_reviews=[
                ArticlePlacementReview(
                    article_title="David Beckham",
                    assigned_cluster="Pop Culture Aficionados",
                    fit="weak",
                    recommended_cluster="Football Enthusiasts",
                    reasoning="His defining professional domain creates a stronger football audience signal.",
                )
            ],
            cross_cluster_overlaps=[
                CrossClusterOverlap(
                    article_title="David Beckham",
                    current_cluster="Pop Culture Aficionados",
                    competing_cluster="Football Enthusiasts",
                    overlap_reason="He has both celebrity visibility and a defining professional football identity.",
                    recommended_resolution="Use football as the dominant theme and avoid counting broad fame twice.",
                )
            ],
        )
        unresolved = RefinedClusterSet(
            clusters=clusters,
            placement_decisions=[
                PlacementDecision(
                    article_title="David Beckham",
                    cluster_name="Pop Culture Aficionados",
                    primary_relevance="Celebrity culture",
                    fit_rationale="He is a globally visible public figure with entertainment relevance.",
                    ambiguity_resolution="No material ambiguity",
                ),
                PlacementDecision(
                    article_title="Zendaya",
                    cluster_name="Pop Culture Aficionados",
                    primary_relevance="Film and celebrity culture",
                    fit_rationale="Her primary public relevance directly supports this entertainment theme.",
                    ambiguity_resolution="No material ambiguity exists for this entertainment figure.",
                ),
                PlacementDecision(
                    article_title="Association football",
                    cluster_name="Football Enthusiasts",
                    primary_relevance="Professional sport",
                    fit_rationale="The topic directly represents the core interest of the football audience.",
                    ambiguity_resolution="No material ambiguity exists for this professional sports topic.",
                ),
            ],
        )

        with self.assertRaisesRegex(PipelineValidationError, "ambiguity"):
            AudienceAgent._validate_refinement(unresolved, critique, articles)

        pruned = AudienceAgent._prune_unresolved_assignments(unresolved, critique)
        self.assertNotIn(
            "David Beckham",
            [title for cluster in pruned.clusters for title in cluster.article_titles],
        )
        AudienceAgent._validate_refinement(pruned, critique, articles)

        resolved = unresolved.model_copy(
            update={
                "clusters": [
                    clusters[0].model_copy(update={"article_titles": ["Zendaya"]}),
                    clusters[1].model_copy(
                        update={
                            "article_titles": ["Association football", "David Beckham"]
                        }
                    ),
                ],
                "placement_decisions": [
                    decision
                    for decision in unresolved.placement_decisions
                    if decision.article_title != "David Beckham"
                ]
                + [
                    PlacementDecision(
                        article_title="David Beckham",
                        cluster_name="Football Enthusiasts",
                        primary_relevance="Professional football",
                        fit_rationale="His defining career and strongest domain relevance align with football audiences.",
                        ambiguity_resolution="Football is the dominant fit; celebrity visibility is secondary and is not counted separately.",
                    )
                ],
            }
        )
        AudienceAgent._validate_refinement(resolved, critique, articles)

    def test_critique_must_flag_exact_cross_cluster_duplicates(self) -> None:
        clusters = InitialClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name=f"Candidate Theme {index}",
                    article_titles=["David Beckham" if index < 2 else f"Topic {index}"],
                    rationale="A sufficiently detailed candidate audience rationale for review.",
                )
                for index in range(5)
            ]
        )
        articles = [
            Article(title="David Beckham", views=500),
            Article(title="Topic 2", views=300),
            Article(title="Topic 3", views=200),
            Article(title="Topic 4", views=100),
        ]
        reviews = [
            ArticlePlacementReview(
                article_title=title,
                assigned_cluster=cluster.cluster_name,
                fit="strong",
                recommended_cluster=cluster.cluster_name,
                reasoning="The article appears to fit this candidate theme at initial review.",
            )
            for cluster in clusters.clusters
            for title in cluster.article_titles
        ]
        critique = ClusterCritique(
            needs_refinement=False,
            overall_assessment="No overlap was reported despite duplicate placement.",
            issues=[],
            placement_reviews=reviews,
            cross_cluster_overlaps=[],
        )

        with self.assertRaisesRegex(PipelineValidationError, "duplicates"):
            AudienceAgent._validate_critique_coverage(
                critique, clusters, articles
            )

        completed = AudienceAgent._complete_critique_coverage(critique, clusters)
        self.assertEqual(len(completed.cross_cluster_overlaps), 1)
        self.assertEqual(
            completed.cross_cluster_overlaps[0].article_title,
            "David Beckham",
        )
        AudienceAgent._validate_critique_coverage(completed, clusters, articles)

    def test_missing_critique_reviews_are_conservatively_dropped(self) -> None:
        clusters = InitialClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name=f"Candidate Theme {index}",
                    article_titles=[f"Topic {index}"],
                    rationale="A sufficiently detailed candidate audience rationale for review.",
                )
                for index in range(5)
            ]
        )
        articles = [
            Article(title=f"Topic {index}", views=500 - index * 50)
            for index in range(5)
        ]
        critique = ClusterCritique(
            needs_refinement=False,
            overall_assessment="The returned critique accidentally omitted one assignment.",
            issues=[],
            placement_reviews=[
                ArticlePlacementReview(
                    article_title=f"Topic {index}",
                    assigned_cluster=f"Candidate Theme {index}",
                    fit="strong",
                    recommended_cluster=f"Candidate Theme {index}",
                    reasoning="The article directly supports the candidate audience theme.",
                )
                for index in range(4)
            ],
            cross_cluster_overlaps=[],
        )

        completed = AudienceAgent._complete_critique_coverage(critique, clusters)

        self.assertTrue(completed.needs_refinement)
        self.assertEqual(len(completed.placement_reviews), 5)
        fallback = completed.placement_reviews[-1]
        self.assertEqual(fallback.article_title, "Topic 4")
        self.assertEqual(fallback.fit, "noise")
        self.assertEqual(fallback.recommended_cluster, "DROP")
        AudienceAgent._validate_critique_coverage(completed, clusters, articles)

        refined = RefinedClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Candidate Theme 4",
                    article_titles=["Topic 4"],
                    rationale="A sufficiently detailed candidate audience rationale for review.",
                )
            ],
            placement_decisions=[
                PlacementDecision(
                    article_title="Topic 4",
                    cluster_name="Candidate Theme 4",
                    primary_relevance="Candidate audience topic",
                    fit_rationale="The refinement attempted to retain the omitted candidate assignment.",
                    ambiguity_resolution=(
                        "The refinement supplied a detailed explanation despite "
                        "the deterministic DROP recommendation."
                    ),
                )
            ],
        )
        pruned = AudienceAgent._prune_unresolved_assignments(refined, completed)
        self.assertEqual(pruned.clusters, [])
        self.assertEqual(pruned.placement_decisions, [])

    def test_duplicate_refined_article_keeps_one_placement(self) -> None:
        refined = RefinedClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="World Cup Historians",
                    article_titles=["2022 FIFA World Cup"],
                    rationale="Interest in the history and outcomes of major football tournaments.",
                ),
                CandidateCluster(
                    cluster_name="Football Event Followers",
                    article_titles=["2022 FIFA World Cup", "UEFA Euro 2024"],
                    rationale="Interest in major international football events and competitions.",
                ),
            ],
            placement_decisions=[
                PlacementDecision(
                    article_title="2022 FIFA World Cup",
                    cluster_name="World Cup Historians",
                    primary_relevance="World Cup history",
                    fit_rationale="This tournament directly supports a historical World Cup audience.",
                    ambiguity_resolution="Historical tournament interest is the dominant cluster fit.",
                ),
                PlacementDecision(
                    article_title="2022 FIFA World Cup",
                    cluster_name="Football Event Followers",
                    primary_relevance="International football event",
                    fit_rationale="This tournament also relates to international football events.",
                    ambiguity_resolution="Event interest provides a plausible but secondary cluster fit.",
                ),
                PlacementDecision(
                    article_title="UEFA Euro 2024",
                    cluster_name="Football Event Followers",
                    primary_relevance="International football event",
                    fit_rationale="This tournament directly supports current football event interest.",
                    ambiguity_resolution="No material ambiguity exists for this event assignment.",
                ),
            ],
        )

        deduplicated = AudienceAgent._deduplicate_refined_assignments(refined)

        assignments = [
            (cluster.cluster_name, title)
            for cluster in deduplicated.clusters
            for title in cluster.article_titles
        ]
        self.assertEqual(
            assignments,
            [
                ("World Cup Historians", "2022 FIFA World Cup"),
                ("Football Event Followers", "UEFA Euro 2024"),
            ],
        )
        self.assertEqual(len(deduplicated.placement_decisions), 2)

    def test_conflicting_critique_labels_are_normalized(self) -> None:
        clusters = InitialClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name=f"Candidate Theme {index}",
                    article_titles=[f"Topic {index}"],
                    rationale="A sufficiently detailed candidate audience rationale for review.",
                )
                for index in range(5)
            ]
        )
        articles = [
            Article(title=f"Topic {index}", views=500 - index * 50)
            for index in range(5)
        ]
        reviews = [
            ArticlePlacementReview(
                article_title="Topic 0",
                assigned_cluster="Candidate Theme 0",
                fit="strong",
                recommended_cluster="Candidate Theme 1",
                reasoning="A stronger competing cluster was identified for this article.",
            ),
            ArticlePlacementReview(
                article_title="Topic 1",
                assigned_cluster="Candidate Theme 1",
                fit="noise",
                recommended_cluster="Candidate Theme 1",
                reasoning="The label and recommendation contradict one another.",
            ),
            ArticlePlacementReview(
                article_title="Topic 2",
                assigned_cluster="Candidate Theme 2",
                fit="misassigned",
                recommended_cluster="Candidate Theme 2",
                reasoning="The label and recommendation contradict one another.",
            ),
            ArticlePlacementReview(
                article_title="Topic 3",
                assigned_cluster="Candidate Theme 3",
                fit="strong",
                recommended_cluster="Invented Theme",
                reasoning="The recommendation does not match a real candidate cluster.",
            ),
            ArticlePlacementReview(
                article_title="Topic 4",
                assigned_cluster="Candidate Theme 4",
                fit="strong",
                recommended_cluster="Candidate Theme 4",
                reasoning="The article directly supports the candidate audience theme.",
            ),
        ]
        critique = ClusterCritique(
            needs_refinement=False,
            overall_assessment="Several structured labels conflict with recommendations.",
            issues=[],
            placement_reviews=reviews,
            cross_cluster_overlaps=[],
        )

        normalized = AudienceAgent._complete_critique_coverage(critique, clusters)
        by_title = {
            review.article_title: review for review in normalized.placement_reviews
        }

        self.assertEqual(by_title["Topic 0"].fit, "misassigned")
        self.assertEqual(by_title["Topic 1"].fit, "weak")
        self.assertEqual(by_title["Topic 2"].fit, "weak")
        self.assertEqual(by_title["Topic 3"].fit, "noise")
        self.assertEqual(by_title["Topic 3"].recommended_cluster, "DROP")
        self.assertTrue(normalized.needs_refinement)
        self.assertEqual(len(normalized.cross_cluster_overlaps), 1)
        AudienceAgent._validate_critique_coverage(
            normalized, clusters, articles
        )

    def test_vague_people_catchall_is_removed_after_refinement(self) -> None:
        refined = RefinedClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Celebrity and Public Figures",
                    article_titles=["Zendaya", "Mitch McConnell"],
                    rationale=(
                        "Interest in recognizable people from entertainment and public life."
                    ),
                ),
                CandidateCluster(
                    cluster_name="Film and TV Buffs",
                    article_titles=["The Odyssey"],
                    rationale="Interest in upcoming film and television entertainment releases.",
                ),
            ],
            placement_decisions=[
                PlacementDecision(
                    article_title="Zendaya",
                    cluster_name="Celebrity and Public Figures",
                    primary_relevance="Notable public figure",
                    fit_rationale="The person is widely recognized by entertainment audiences.",
                    ambiguity_resolution="The model used broad fame as the cluster boundary.",
                ),
                PlacementDecision(
                    article_title="Mitch McConnell",
                    cluster_name="Celebrity and Public Figures",
                    primary_relevance="Notable public figure",
                    fit_rationale="The person is widely recognized by current-events audiences.",
                    ambiguity_resolution="The model used broad fame as the cluster boundary.",
                ),
                PlacementDecision(
                    article_title="The Odyssey",
                    cluster_name="Film and TV Buffs",
                    primary_relevance="Upcoming film",
                    fit_rationale="The title directly supports interest in forthcoming cinema releases.",
                    ambiguity_resolution="No material ambiguity exists for this film assignment.",
                ),
            ],
        )

        pruned = AudienceAgent._prune_vague_people_clusters(refined)

        self.assertEqual(
            [cluster.cluster_name for cluster in pruned.clusters],
            ["Film and TV Buffs"],
        )
        self.assertEqual(
            [decision.article_title for decision in pruned.placement_decisions],
            ["The Odyssey"],
        )

    def test_vague_role_is_removed_from_specific_cluster(self) -> None:
        refined = RefinedClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Cinephiles and TV Buffs",
                    article_titles=["Zendaya", "The Odyssey"],
                    rationale="Interest in actors and upcoming screen entertainment releases.",
                )
            ],
            placement_decisions=[
                PlacementDecision(
                    article_title="Zendaya",
                    cluster_name="Cinephiles and TV Buffs",
                    primary_relevance="Public figure",
                    fit_rationale="The person is broadly known to entertainment audiences.",
                    ambiguity_resolution="The explanation does not identify a specific professional role.",
                ),
                PlacementDecision(
                    article_title="The Odyssey",
                    cluster_name="Cinephiles and TV Buffs",
                    primary_relevance="Upcoming film",
                    fit_rationale="The title directly supports interest in forthcoming cinema releases.",
                    ambiguity_resolution="No material ambiguity exists for this film assignment.",
                ),
            ],
        )

        pruned = AudienceAgent._prune_vague_people_clusters(refined)

        self.assertEqual(pruned.clusters[0].article_titles, ["The Odyssey"])
        self.assertEqual(len(pruned.placement_decisions), 1)

    def test_refinement_decisions_follow_uniquely_renamed_clusters(self) -> None:
        refined = RefinedClusterSet(
            clusters=[
                CandidateCluster(
                    cluster_name="Mixed Martial Arts Fans",
                    article_titles=["Conor McGregor", "Max Holloway"],
                    rationale="Interest in professional mixed martial arts competitors and events.",
                )
            ],
            placement_decisions=[
                PlacementDecision(
                    article_title=title,
                    cluster_name="Athletes and Sports Figures",
                    primary_relevance="Mixed martial artist",
                    fit_rationale="The athlete's primary professional domain is mixed martial arts.",
                    ambiguity_resolution="Mixed martial arts is the specific dominant fit, not broad sports fame.",
                )
                for title in ["Conor McGregor", "Max Holloway"]
            ],
        )

        aligned = AudienceAgent._align_refinement_decisions(refined)

        self.assertEqual(
            [decision.cluster_name for decision in aligned.placement_decisions],
            ["Mixed Martial Arts Fans", "Mixed Martial Arts Fans"],
        )
        self.assertEqual(
            aligned.clusters[0].article_titles,
            ["Conor McGregor", "Max Holloway"],
        )


if __name__ == "__main__":
    unittest.main()
