import os
import json
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("You must set YOUTUBE_API_KEY in your environment variables.")

# Initialize YouTube client
youtube = build("youtube", "v3", developerKey=API_KEY)


OUTPUT_FILE = "data/raw/youtube_comments.jsonl"


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
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response["items"]:
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


def save_comments(comments):
    os.makedirs("data/raw", exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for comment in comments:
            f.write(json.dumps(comment) + "\n")


def main():
    queries = [
        "Adidas review",
#        "Adidas quality",
#        "Adidas shoes",
#        "Adidas complaint"
    ]

    for query in queries:
        print(f"Searching videos for: {query}")
        video_ids = search_videos(query)

        for video_id in video_ids:
            print(f"Collecting comments from video: {video_id}")
            comments = get_comments(video_id, max_comments=300)
            save_comments(comments)

    print("Done collecting comments.")


if __name__ == "__main__":
    main()
