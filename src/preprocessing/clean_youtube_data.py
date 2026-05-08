import json
import os

INPUT_FILES = [
    "data/raw/youtube_comments.jsonl",
    "data/raw/news_articles.jsonl",
]

# This file is not only YouTube; it contains cleaned items from all raw sources.
OUTPUT_FILE = "data/processed/cleaned_raw.jsonl"


def load_data():
    data = []
    for file_path in INPUT_FILES:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            print(f"Warning: {file_path} not found. Skipping.")
    return data


def remove_duplicates(items):
    seen_ids = set()
    unique_items = []

    for item in items:
        # News uses 'source_id', YouTube uses 'comment_id'
        item_id = item.get("comment_id") or item.get("source_id")

        # Fallback to hash of text if no ID is found
        if not item_id:
            item_id = hash(item.get("text", ""))

        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_items.append(item)

    return unique_items


def remove_short_texts(items, min_words=3):
    filtered = []

    for item in items:
        text = item.get("text", "").strip()
        if len(text.split()) >= min_words:
            filtered.append(item)

    return filtered


def save_data(items):
    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def main():
    print("Loading raw data...")
    items = load_data()
    print(f"Total raw items: {len(items)}")

    items = remove_duplicates(items)
    print(f"After removing duplicates: {len(items)}")

    items = remove_short_texts(items)
    print(f"After removing short texts: {len(items)}")

    save_data(items)
    print("Cleaned data saved.")


if __name__ == "__main__":
    main()
