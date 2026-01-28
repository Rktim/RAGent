import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from crawl4ai import AsyncWebCrawler

async def main(url: str):
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
