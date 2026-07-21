"""Run one complete audience-mining cycle outside Streamlit.

Useful for validation and for demonstrating the labeled generation, critique,
and refinement outputs in a terminal.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_layer.audience_agent import AudienceAgent, PersistentMCPToolClient


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    client = PersistentMCPToolClient()
    try:
        client.start()
        portfolio = asyncio.run(AudienceAgent(client).run())
        print("\n=== FINAL PORTFOLIO ===", flush=True)
        print(
            json.dumps(portfolio.model_dump(), indent=2, ensure_ascii=False),
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
