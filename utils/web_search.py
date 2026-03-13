from config.config import SERPER_API_KEY

def _search_serper(query: str, num_results: int = 5) -> list:
    try:
        import requests
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", [])[:num_results]:
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            link    = item.get("link", "")
            results.append({"title": title, "snippet": snippet, "link": link})

        return results[:num_results]
    except Exception as e:
        print(f"[WebSearch] Serper error: {e}")
        return []

def _search_duckduckgo(query: str, num_results: int = 5) -> list:
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                results.append({"title": title, "snippet": body, "link": href})

        return results[:num_results]
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo error: {e}")
        return []

def format_search_results(results: list) -> str:
    try:
        if not results:
            return ""
        parts = []
        for item in results:
            parts.append(
                f"• {item.get('title', '')}\n"
                f"  {item.get('snippet', '')}\n"
                f"  Source: {item.get('link', '')}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[WebSearch] Formatting error: {e}")
        return ""

def web_search(query: str, num_results: int = 5) -> list:
    if SERPER_API_KEY:
        results = _search_serper(query, num_results=num_results)
        if results:
            return results
        print("[WebSearch] Serper returned no results, falling back to DuckDuckGo.")

    return _search_duckduckgo(query, num_results=num_results)
