
from ddgs import DDGS

def ddg_search(query: str, max_results: int = 5) -> str:
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                f"- {r['title']}\n  {r['href']}\n  {r['body']}"
            )

    if not results:
        return "No relevant web results found."

    return "\n\n".join(results)
