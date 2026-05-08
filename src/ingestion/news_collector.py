import os
import json
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi.newsapi_client import NewsApiClient

# Load environment variables
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise ValueError("You must set NEWS_API_KEY in your environment variables.")

OUTPUT_FILE = "data/raw/news_articles.jsonl"


def _parse_queries_from_env(default_queries: list[str]) -> list[str]:
    """Read comma-separated queries from NEWS_QUERIES env var."""
    raw = os.getenv("NEWS_QUERIES")
    if not raw:
        return default_queries
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or default_queries

def get_all_sources(news_api):
    """Gets all valid English sources and returns them as a single comma-separated string."""
    response = news_api.get_sources(language='en')
    # Automatically filter out null IDs
    valid_ids = [s['id'] for s in response.get('sources', []) if s.get('id')]
    
    # Intentionally join ALL of them at once to test the limit
    return ",".join(valid_ids)

def fetch_news(query, max_results=100):
    """Fetches news articles related to the given query."""
    news_api = NewsApiClient(api_key=API_KEY)
    all_sources = get_all_sources(news_api)
    
    # Free tier can search up to exactly 1 month (approx 30 days) ago
    from_date = (datetime.utcnow() - timedelta(days=28)).strftime('%Y-%m-%d')
    
    raw_articles = []

    try:
        # Pass all ~80 sources at once
        response = news_api.get_everything(
            q=query,
            sources=all_sources,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page=1,                 
            page_size=max_results
        )

        for article in response.get("articles", []):
            url = article.get("url")
            if url:
                article["platform"] = "news"
                article["collected_at"] = datetime.utcnow().isoformat()
                raw_articles.append(article)
    except Exception as e:
        print(f"Error fetching: {e}")
            
    return raw_articles

def save_articles(articles):
    """Saves articles to the raw JSONL file, avoiding duplicates by URL."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    existing_urls = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "url" in data:
                        existing_urls.add(data["url"])
                except json.JSONDecodeError:
                    continue

    new_articles = [a for a in articles if a.get("url") and a.get("url") not in existing_urls]

    if not new_articles:
        print(f"No new articles to save (filtered {len(articles)} duplicates).")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for article in new_articles:
            f.write(json.dumps(article) + "\n")
            
    print(f"Saved {len(new_articles)} new articles (filtered out {len(articles) - len(new_articles)} duplicates).")

def main(queries: list[str] | None = None, overwrite_output: bool = False):
    default_queries = [
        "Adidas",
        '"Adidas" AND "Nike"',  # Quotes ensure exact matching if needed
    ]

    queries = queries or _parse_queries_from_env(default_queries)
    
    if overwrite_output and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    all_articles = []

    for query in queries:
        print(f"Fetching news for query: {query}")
        try:
            articles = fetch_news(query, max_results=50)
            print(f"Found {len(articles)} high-quality articles.")
            all_articles.extend(articles)
        except Exception as e:
            print(f"Error fetching news for '{query}': {e}")
            
    save_articles(all_articles)
    print("Done collecting news articles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News articles collector")
    parser.add_argument(
        "--queries",
        nargs="*",
        help="One or more search queries. If omitted, uses NEWS_QUERIES env var or defaults.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite data/raw/news_articles.jsonl instead of dedup+append.",
    )
    args = parser.parse_args()
    main(queries=args.queries, overwrite_output=args.overwrite_output)
