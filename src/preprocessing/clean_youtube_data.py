import json
import os

INPUT_FILE = "data/raw/youtube_comments.jsonl"
OUTPUT_FILE = "data/processed/youtube_comments_clean.jsonl"


def load_comments():
    comments = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            comments.append(json.loads(line))
    return comments


def remove_duplicates(comments):
    seen_ids = set()
    unique_comments = []

    for comment in comments:
        comment_id = comment["comment_id"]

        if comment_id not in seen_ids:
            seen_ids.add(comment_id)
            unique_comments.append(comment)

    return unique_comments


def remove_short_comments(comments, min_words=3):
    filtered = []

    for comment in comments:
        text = comment["text"].strip()
        if len(text.split()) >= min_words:
            filtered.append(comment)

    return filtered


def save_comments(comments):
    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for comment in comments:
            f.write(json.dumps(comment) + "\n")


def main():
    print("Loading raw comments...")
    comments = load_comments()
    print(f"Total raw comments: {len(comments)}")

    comments = remove_duplicates(comments)
    print(f"After removing duplicates: {len(comments)}")

    comments = remove_short_comments(comments)
    print(f"After removing short comments: {len(comments)}")

    save_comments(comments)
    print("Cleaned data saved.")


if __name__ == "__main__":
    main()
