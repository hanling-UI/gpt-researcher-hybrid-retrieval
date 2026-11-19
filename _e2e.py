"""Minimal end-to-end run: search -> scrape -> report, with the real .env."""
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from gpt_researcher import GPTResearcher


async def main():
    researcher = GPTResearcher(
        query="What is retrieval augmented generation? Answer briefly.",
        report_type="research_report",
    )
    await researcher.conduct_research()
    report = await researcher.write_report()
    print("\n=== report chars:", len(report), "===")
    print(report[:1000])


asyncio.run(main())
