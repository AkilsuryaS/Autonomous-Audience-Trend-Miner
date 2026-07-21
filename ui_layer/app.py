"""Streamlit dashboard for the Autonomous Audience Trend Miner."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_layer.audience_agent import (  # noqa: E402
    AudienceAgent,
    AudiencePortfolio,
    MCPConnectionError,
    PersistentMCPToolClient,
    PipelineError,
)
from agent_layer.portfolio_chat import answer_portfolio_question  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="Autonomous Audience Trend Miner",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_mcp_client() -> PersistentMCPToolClient:
    """Create one stdio subprocess/session and retain it across Streamlit reruns."""

    client = PersistentMCPToolClient()
    client.start()
    return client


async def run_pipeline(
    mcp_client: PersistentMCPToolClient,
    progress_callback,
) -> AudiencePortfolio:
    # Construct the LLM client inside this short-lived event loop; only the MCP
    # resource crosses Streamlit reruns and it owns its own persistent loop.
    agent = AudienceAgent(mcp_client)
    return await agent.run(progress_callback)


def render_portfolio(portfolio: AudiencePortfolio) -> None:
    st.subheader("Emerging Audience Portfolio")
    st.caption(
        "Size Index is each segment's article traffic as a share of the full "
        "trending list. Signals reflect global English Wikipedia readership."
    )

    columns_per_row = 2
    for start in range(0, len(portfolio.segments), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, segment in zip(
            columns, portfolio.segments[start : start + columns_per_row]
        ):
            buying_power = segment.potential_buying_power
            with column:
                with st.container(border=True):
                    st.markdown(f"### {segment.audience_name}")
                    st.markdown(segment.audience_description)
                    metric_col, power_col = st.columns(2)
                    metric_col.metric(
                        "Size Index", f"{segment.estimated_size_index:.1f}%"
                    )
                    power_col.metric("Buying Power", buying_power.level)
                    st.progress(
                        min(max(segment.estimated_size_index / 100, 0.0), 1.0)
                    )
                    st.divider()
                    st.markdown("**Commercial opportunity**")
                    st.write(buying_power.rationale)
                    st.caption(
                        "Brand fit — " + " · ".join(buying_power.brand_categories)
                    )
                    with st.expander("Traffic signals"):
                        st.write(" · ".join(segment.source_articles))
                    with st.expander("Placement sanity check"):
                        for decision in segment.placement_decisions:
                            st.markdown(
                                f"**{decision.article_title}** — "
                                f"{decision.primary_relevance}"
                            )
                            st.caption(decision.fit_rationale)
                            if not decision.ambiguity_resolution.casefold().startswith(
                                "no material ambiguity"
                            ):
                                st.caption(
                                    "Overlap resolved: "
                                    + decision.ambiguity_resolution
                                )


def render_portfolio_chat(portfolio: AudiencePortfolio) -> None:
    """Render Q&A that is limited to the portfolio already on the dashboard."""

    chat_history = st.session_state.setdefault("chat_history", [])
    with st.expander("Ask about this dashboard"):
        for turn in chat_history:
            if turn["role"] == "assistant" and not turn.get("answered", True):
                st.info(turn["content"])
            else:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

        question = st.chat_input("Ask a question about the audience segments...")
        if question:
            chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            result = answer_portfolio_question(
                question,
                portfolio.model_dump()["segments"],
            )
            chat_history.append(
                {
                    "role": "assistant",
                    "content": result["text"],
                    "answered": result["answered"],
                }
            )
            if result["answered"]:
                with st.chat_message("assistant"):
                    st.markdown(result["text"])
            else:
                st.info(result["text"])


st.title("InMarket Prototype: Autonomous Audience Trend Miner")
st.write(
    "Turn the latest processed English Wikipedia traffic into coherent, "
    "commercially actionable audience segments."
)
st.info(
    "Wikipedia publishes global English-language readership totals, not "
    "country-scoped or US-specific traffic."
)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = None

if st.button("Mine Trends & Generate Audiences", type="primary", use_container_width=True):
    try:
        mcp_client = get_mcp_client()
        with st.status("Mining audience trends...", expanded=True) as status:

            def update_progress(stage: str, detail: str) -> None:
                icons = {"fetch": "🔌", "cluster": "🧠", "portfolio": "📊"}
                status.write(f"{icons.get(stage, '•')} {detail}")

            portfolio = asyncio.run(run_pipeline(mcp_client, update_progress))
            st.session_state.portfolio = portfolio
            status.update(label="Audience portfolio ready", state="complete")
    except (MCPConnectionError, PipelineError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(
            "The audience miner hit an unexpected error. Check the terminal logs "
            f"for details. ({type(exc).__name__}: {exc})"
        )

if st.session_state.portfolio is not None:
    render_portfolio(st.session_state.portfolio)

if st.session_state.portfolio is not None:
    render_portfolio_chat(st.session_state.portfolio)
else:
    st.caption("Run the pipeline first to ask questions about the results.")
