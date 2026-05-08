import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("You must set YOUTUBE_API_KEY in your environment variables.")

# Initialize YouTube client
youtube = build("youtube", "v3", developerKey=API_KEY)


OUTPUT_FILE = "data/raw/youtube_comments.jsonl"


def _parse_queries_from_env(default_queries: list[str]) -> list[str]:
    """Read comma-separated queries from YOUTUBE_QUERIES env var."""
    raw = os.getenv("YOUTUBE_QUERIES")
    if not raw:
        return default_queries
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or default_queries


def search_videos(query, max_results=5):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        order="relevance"
    )

    response = request.execute()
    return [item["id"]["videoId"] for item in response["items"]]


def get_comments(video_id, max_comments=500):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )

            response = request.execute()
        except HttpError as e:
            if "commentsDisabled" in str(e):
                print(f"Skipping video {video_id}: comments are disabled.")
                break
            else:
                print(f"An error occurred while fetching comments for {video_id}: {e}")
                break

        for item in response.get("items", []):
            comment_data = item["snippet"]["topLevelComment"]["snippet"]

            comment = {
                "platform": "youtube",
                "video_id": video_id,
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "text": comment_data["textDisplay"],
                "author": comment_data["authorDisplayName"],
                "like_count": comment_data["likeCount"],
                "published_at": comment_data["publishedAt"],
                "collected_at": datetime.utcnow().isoformat()
            }

            comments.append(comment)

            if len(comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def save_comments(comments, mode: str = "a"):
    os.makedirs("data/raw", exist_ok=True)

    with open(OUTPUT_FILE, mode, encoding="utf-8") as f:
        for comment in comments:
            f.write(json.dumps(comment) + "\n")


def main(
    queries: list[str] | None = None,
    max_results: int = 5,
    max_comments: int = 300,
    overwrite_output: bool = False,
):
    default_queries = [
        "Adidas review",
    ]

    queries = queries or _parse_queries_from_env(default_queries)

    first_write_mode = "w" if overwrite_output else "a"
    wrote_any = False

    for query in queries:
        print(f"Searching videos for: {query}")
        video_ids = search_videos(query, max_results=max_results)

        for video_id in video_ids:
            print(f"Collecting comments from video: {video_id}")
            comments = get_comments(video_id, max_comments=max_comments)
            # First write overwrites (optional), subsequent writes append.
            mode = first_write_mode if not wrote_any else "a"
            save_comments(comments, mode=mode)
            wrote_any = True

    print("Done collecting comments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube comments collector")
    parser.add_argument(
        "--queries",
        nargs="*",
        help="One or more search queries. If omitted, uses YOUTUBE_QUERIES env var or defaults.",
    )
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-comments", type=int, default=300)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite data/raw/youtube_comments.jsonl instead of appending.",
    )
    args = parser.parse_args()

    main(
        queries=args.queries,
        max_results=args.max_results,
        max_comments=args.max_comments,
    overwrite_output=args.overwrite_output,
    )
