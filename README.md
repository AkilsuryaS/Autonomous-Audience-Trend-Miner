# InMarket Prototype: Autonomous Audience Trend Miner

This prototype turns public Wikipedia Pageviews into an **Emerging Audience Portfolio** for brand marketers. It is modeled on the way a company such as InMarket packages broad behavioral signals into coherent, sellable audience segments, while making an important distinction: Wikipedia traffic is an aggregate interest signal, not individual-level identity or purchase-intent data.

The application fetches the latest processed English Wikipedia trends, aggregates seven days of traffic, removes obvious utility-page noise, and asks `gpt-4o-mini` to discover commercially meaningful themes. A fixed generation → critique → refinement loop makes the reasoning visible and cost-bounded before a final LLM call writes the market-facing portfolio.

## What the app produces

Each audience card includes:

- a market-friendly audience name;
- a two-sentence, stakeholder-friendly brief that names representative traffic
  signals and explains the cluster's shared interest and brand relevance;
- an Estimated Size Index calculated as the cluster's article views divided by all views in the fetched trending list; and
- a High/Medium/Low buying-power assessment with relevant brand categories.

The size percentages do not necessarily sum to 100%. The denominator includes the full fetched trend list, while the agent intentionally drops non-commercial or incoherent topics.

## Architecture

```text
Streamlit UI
    │  calls one deterministic async pipeline
    ▼
LangChain agent layer
    │  one LangChain MCP tool invocation
    ▼
FastMCP Wikipedia service ─────► Wikimedia Pageviews API
    │
    └── returns normalized, aggregated article JSON

Agent reasoning over returned JSON:
generation → critique → refinement → Pydantic validation → portfolio generation
```

The separation is deliberate:

- `mcp_server/wikipedia_server.py` owns HTTP access, date selection, normalization, aggregation, and data-level filtering. It has no LLM or UI imports.
- `agent_layer/audience_agent.py` can access trend data only through the MCP protocol. `langchain-mcp-adapters` loads the server tool into LangChain, and the tool is genuinely invoked over stdio rather than importing the fetch function directly.
- `ui_layer/app.py` owns display and interaction only. It does not know Wikimedia response shapes or clustering prompts.

The Streamlit resource cache retains a `PersistentMCPToolClient`. That resource gives the stdio session a dedicated long-lived event loop, so repeated Streamlit reruns reuse the same spawned MCP process even though the button callback uses `asyncio.run()`.

## Agentic clustering loop

The clustering stage is visibly implemented as three separate LLM calls, not one prompt-to-JSON request:

1. **Generation:** create 5-10 initial commercial clusters from exact article titles.
2. **Critique:** independently audit every article assignment, commercial relevance, semantic cross-cluster overlap, unsupported/misassigned articles, and residual noise. Multi-domain people must receive an explicit competing-cluster review. If structured output omits an assignment or a required duplicate-overlap record, a deterministic fallback marks the unaudited item for removal and supplies the missing overlap instruction without making another LLM call.
3. **Refinement:** apply that critique exactly once and emit one placement decision per retained article. The deterministic guardrails remove retained noise, deduplicate final placements, and conservatively drop flagged articles that lack a concrete ambiguity resolution.

Every pass prints a separately labeled structured payload to the app terminal for debugging and video demonstration. The dashboard also exposes the final article-level decisions in each card's **Placement sanity check** expander. The refined clusters are Pydantic-validated and checked against the original article-title set. Portfolio generation is a fourth, sequential structured-output call. If final parsing or cluster mapping fails, it retries once with stricter formatting instructions.

This fixed pipeline is intentionally deterministic: the LLM does not decide whether to fetch again, critique again, or call an unrelated tool. That keeps the demo debuggable, limits cost, and makes the assignment's clustering/critique loop unambiguous.

## Project structure

```text
.
├── mcp_server/
│   └── wikipedia_server.py
├── agent_layer/
│   └── audience_agent.py
├── ui_layer/
│   └── app.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your OpenAI key:

```dotenv
OPENAI_API_KEY=your-real-key
```

The Wikimedia endpoint is public and unauthenticated, so it needs no API key. `.env` is gitignored and must never be committed.

## Run the app

```bash
streamlit run ui_layer/app.py
```

No second terminal or manual server process is required. The cached LangChain MCP client automatically spawns `mcp_server/wikipedia_server.py` over **stdio** and retains that connection across Streamlit reruns.

Click **Mine Trends & Generate Audiences**. The status panel shows:

1. Fetching trends via MCP
2. Clustering into audience themes (generation, critique, refinement)
3. Generating audience portfolio

The terminal shows the full structured output of each reasoning pass.

## Tests

The unit tests cover the three-day lag anchor, missing-day behavior, title normalization/filtering, multi-day aggregation, exact-article validation, and size-index math.

```bash
python -m unittest discover -v
python -m compileall agent_layer mcp_server ui_layer tests
```

An end-to-end run additionally requires network access to Wikimedia and a valid `OPENAI_API_KEY`.

For a terminal-only demonstration that prints every reasoning state and the final portfolio:

```bash
python -u scripts/run_pipeline.py
```

## Known limitations

- A "week" is approximated as the last seven processed days. The newest requested date is three days before today to buffer Wikimedia's typical 1-2 day processing delay. Any still-missing day is logged and skipped.
- Trending data reflects **English Wikipedia globally**, not US-specific traffic. The Pageviews top endpoint has no country-scoped variant.
- Wikipedia interest is contextual evidence, not proof that a reader belongs to a demographic or intends to buy. The output is suitable for ideation and human review, not autonomous ad targeting.
- The top-article endpoint favors large cultural moments. The LLM removes additional tragedy, crime, politics, obituary, and weak commercial themes after the server's deterministic namespace/noise filter.
- Audience size is a normalized traffic index, not a population estimate or forecast.

## Deployment Strategy

For the local challenge demo, the most reliable image would package the Streamlit app and MCP server code together. Streamlit would spawn the server as a child process over stdio exactly as it does locally. The container would receive `OPENAI_API_KEY` from the hosting platform's secret manager, run as a non-root user, expose only Streamlit's port, and send logs to stdout/stderr.

For production scaling, the two layers can become separate containers or managed processes:

- an MCP data-service container exposing authenticated **Streamable HTTP** instead of stdio; and
- a Streamlit (or API + web frontend) container using the same LangChain adapter boundary.

That transport change is necessary because stdio is a local parent/child transport and does not cross container hosts. On AWS ECS, the services could run in one task for low-latency sidecar-style deployment or as separate services behind private service discovery. Render and Fly.io can run the two processes as private services. Autoscaling, request timeouts, retries, secret injection, egress controls, structured logs, and a short-lived cache for the daily Wikimedia response would be added before production.

Serverless functions are possible after adapting the MCP service to stateless HTTP, but long-lived stdio subprocesses are a poor fit for per-request functions. A scheduled function could pre-aggregate Wikimedia data into object storage, while an API function and the LLM application consume that cached snapshot. This reduces cold-start work and avoids repeatedly fetching the same daily public data.

## Security and reliability notes

- API keys are loaded with `python-dotenv` and `os.getenv`; they are never passed to the public MCP server or hardcoded.
- The MCP server uses request timeouts, a descriptive Wikimedia User-Agent, graceful per-day error handling, and explicit input bounds.
- Article titles are treated as untrusted data in every LLM prompt and cannot introduce instructions.
- Structured output, Pydantic models, exact source-title checks, a deterministic size calculation, and a single retry constrain common LLM failure modes.

## Portfolio Q&A Chat

The dashboard chat answers questions only from the already-generated
`AudiencePortfolio`. It performs no new Wikimedia fetch, clustering pass, or
tool call. The chat receives a plain copy of the segments already stored in
Streamlit session state, so it remains separate from the main mining pipeline.

The feature uses two guardrail layers: a free deterministic keyword-overlap
check before any LLM call, followed by a strict system prompt that permits only
facts in the supplied segment JSON. If the pre-check allows a question that the
portfolio cannot actually support, the model must return the fixed refusal:
`Please ask a question relevant to the dashboard.`

The chat is stateless with respect to the main pipeline and cannot trigger a
new mining run.
