from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent_layer.audience_agent import AudiencePortfolio
from agent_layer.portfolio_chat import (
    MISSING_KEY_TEXT,
    REFUSAL_TEXT,
    answer_portfolio_question,
)


class FakeChatModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(content=self.response_text)


class FailingChatModel:
    def invoke(self, messages):
        raise RuntimeError("simulated provider failure")


class PortfolioChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            {
                "source_cluster_name": "Football Stars",
                "audience_name": "Football Fanatics",
                "audience_description": (
                    "Attention around leading football players signals strong "
                    "interest in the sport and its personalities."
                ),
                "estimated_size_index": 13.1,
                "potential_buying_power": {
                    "level": "Medium",
                    "rationale": "Fans may engage with football-related products.",
                    "brand_categories": ["Sports Apparel", "Merchandise"],
                },
                "source_articles": ["Jude Bellingham", "Erling Haaland"],
            },
            {
                "source_cluster_name": "Film and Entertainment",
                "audience_name": "Cinephile Trendsetters",
                "audience_description": (
                    "Upcoming film pages signal interest in new cinematic releases "
                    "and entertainment experiences."
                ),
                "estimated_size_index": 8.6,
                "potential_buying_power": {
                    "level": "High",
                    "rationale": "Film fans may pay for content and experiences.",
                    "brand_categories": ["Streaming Services", "Cinema"],
                },
                "source_articles": ["The Odyssey (2026 film)", "Moana (2026 film)"],
            },
        ]

    def _portfolio(self) -> AudiencePortfolio:
        portfolio_segments = []
        for segment in self.segments:
            portfolio_segment = dict(segment)
            portfolio_segment["placement_decisions"] = [
                {
                    "article_title": title,
                    "cluster_name": segment["source_cluster_name"],
                    "primary_relevance": "Direct audience signal",
                    "fit_rationale": (
                        f"{title} directly supports the defined audience interest."
                    ),
                    "ambiguity_resolution": (
                        "No material ambiguity exists for this assignment."
                    ),
                }
                for title in segment["source_articles"]
            ]
            portfolio_segments.append(portfolio_segment)
        return AudiencePortfolio.model_validate({"segments": portfolio_segments})

    def test_audience_name_question_passes_relevance_check(self) -> None:
        llm = FakeChatModel("Football Fanatics has a 13.1% size index.")

        result = answer_portfolio_question(
            "What is the size index for Football Fanatics?",
            self.segments,
            llm=llm,  # type: ignore[arg-type]
        )

        self.assertTrue(result["answered"])
        self.assertEqual(len(llm.invocations), 1)

    def test_source_article_question_passes_relevance_check(self) -> None:
        llm = FakeChatModel("Jude Bellingham supports Football Fanatics.")

        result = answer_portfolio_question(
            "Which audience includes Jude Bellingham?",
            self.segments,
            llm=llm,  # type: ignore[arg-type]
        )

        self.assertTrue(result["answered"])
        self.assertEqual(len(llm.invocations), 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("agent_layer.portfolio_chat.ChatOpenAI")
    def test_out_of_scope_question_skips_llm(
        self, mock_chat_openai
    ) -> None:
        result = answer_portfolio_question(
            "What's the weather today?",
            self.segments,
        )

        self.assertEqual(
            result,
            {"answered": False, "text": REFUSAL_TEXT},
        )
        mock_chat_openai.assert_not_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_missing_api_key_returns_unavailable_without_raising(self) -> None:
        result = answer_portfolio_question(
            "What is the Football Fanatics size index?",
            self.segments,
        )

        self.assertEqual(
            result,
            {"answered": False, "text": MISSING_KEY_TEXT},
        )

    def test_layer_one_and_layer_two_use_exact_refusal_text(self) -> None:
        layer_one_result = answer_portfolio_question(
            "What's the weather today?",
            self.segments,
            llm=FakeChatModel("This must not be called"),  # type: ignore[arg-type]
        )
        refusing_llm = FakeChatModel(REFUSAL_TEXT)
        layer_two_result = answer_portfolio_question(
            "Tell me something unknown about Football Fanatics.",
            self.segments,
            llm=refusing_llm,  # type: ignore[arg-type]
        )

        expected = {"answered": False, "text": REFUSAL_TEXT}
        self.assertEqual(layer_one_result, expected)
        self.assertEqual(layer_two_result, expected)
        self.assertEqual(len(refusing_llm.invocations), 1)

    def test_llm_failure_returns_safe_message(self) -> None:
        result = answer_portfolio_question(
            "What is the Football Fanatics size index?",
            self.segments,
            llm=FailingChatModel(),  # type: ignore[arg-type]
        )

        self.assertEqual(
            result,
            {
                "answered": False,
                "text": "Something went wrong answering that question. Please try again.",
            },
        )

    def test_streamlit_chat_renders_and_styles_refusal_as_info(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "ui_layer" / "app.py"
        app = AppTest.from_file(str(app_path))

        app.run()
        self.assertFalse(app.exception)
        self.assertIn(
            "Run the pipeline first to ask questions about the results.",
            [caption.value for caption in app.caption],
        )

        app.session_state["portfolio"] = self._portfolio()
        app.run()
        self.assertFalse(app.exception)
        self.assertIn(
            "Ask about this dashboard",
            [expander.label for expander in app.expander],
        )
        self.assertEqual(len(app.chat_input), 1)

        app.chat_input[0].set_value("What's the weather today?").run()
        self.assertFalse(app.exception)
        self.assertIn(REFUSAL_TEXT, [message.value for message in app.info])


if __name__ == "__main__":
    unittest.main()
