import json
import os
import re
import emoji

FILE_SCHEMA = [
    {
    "input_file": "data/processed/cleaned_raw.jsonl",
        "input_type": "jsonl",
        "text_field": "text",
        "output_field": "normalized_text"
    },
    {
        "input_file": "data/raw/news_articles_summed.json",
        "input_type": "json_batch",
        "text_field": "summary",
        "output_field": "normalized_text"
    }
]

OUTPUT_FILE = "data/processed/normalized_data.jsonl"

def normalize_text(text):
    if not text:
        return ""
    
    # Replace emojis with empty strings instead of translating to phrases
    text = emoji.replace_emoji(text, replace="")
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_data():
    os.makedirs("data/processed", exist_ok=True)
    
    total_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for schema in FILE_SCHEMA:
            input_file = schema["input_file"]
            input_type = schema["input_type"]
            text_field = schema["text_field"]
            output_field = schema["output_field"]
            
            count = 0

            try:
                with open(input_file, "r", encoding="utf-8") as infile:
                    
                    if input_type == "jsonl":
                        for line in infile:
                            item = json.loads(line)
                            original_text = item.get(text_field, "")
                            
                            # Build new output dictionary
                            new_item = {
                                output_field: normalize_text(original_text),
                                "metadata": {}
                            }
                            
                            # Move other fields to metadata
                            for k, v in item.items():
                                if k != text_field:
                                    new_item["metadata"][k] = v
                                    
                            if "platform" not in new_item["metadata"]:
                                new_item["metadata"]["platform"] = "youtube"
                        
                            outfile.write(json.dumps(new_item) + "\n")
                            count += 1
                            total_count += 1
                    
                    elif input_type == "json_batch":
                        data = json.load(infile)
                        for source_batch in data:
                            if "data" in source_batch:
                                for item in source_batch["data"]:
                                    # extract summary text
                                    original_text = item.get(text_field, "")
                                    
                                    new_item = {
                                        output_field: normalize_text(original_text),
                                        "metadata": {}
                                    }
                                    
                                    # Move original metadata fields
                                    if "metadata" in item:
                                        for k, v in item["metadata"].items():
                                            new_item["metadata"][k] = v
                                            
                                    # If there are any other fields that aren't text_field or metadata, add them
                                    for k, v in item.items():
                                        if k not in [text_field, "metadata"]:
                                            new_item["metadata"][k] = v
                                    
                                    if "platform" not in new_item["metadata"]:
                                        new_item["metadata"]["platform"] = "news"
                                        
                                    outfile.write(json.dumps(new_item) + "\n")
                                    count += 1
                                    total_count += 1
                                    
                print(f"Normalized {count} items from {input_file}.")
            except FileNotFoundError:
                print(f"Input file not found: {input_file}. Skipping...")

    print(f"Total normalized elements saved to {OUTPUT_FILE}: {total_count}")

if __name__ == "__main__":
    process_data()
