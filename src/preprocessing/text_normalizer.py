import json
import os
import re
import emoji

INPUT_FILE = "data/processed/youtube_comments_clean.jsonl"
OUTPUT_FILE = "data/processed/youtube_comments_normalized.jsonl"


def normalize_text(text):
    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters except basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_comments():
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        count = 0

        for line in infile:
            comment = json.loads(line)
            original_text = comment["text"]

            normalized = normalize_text(original_text)
            comment["normalized_text"] = normalized

            outfile.write(json.dumps(comment) + "\n")
            count += 1

    print(f"Normalized {count} comments.")


if __name__ == "__main__":
    process_comments()
