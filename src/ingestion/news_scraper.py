import json
import os
import time
from newspaper import Article, ArticleException

INPUT_FILE = "data/processed/combined_data_clean.jsonl"
OUTPUT_FILE = "data/raw/news_articles_scraped.jsonl"

def scrape_full_text():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found. Ensure you have run news_collector.py first.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    scraped_articles = []
    
    print("Starting article scraper...")
    print(f"Reading from: {INPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            item = json.loads(line)
            url = item.get("url")
            
            if not url:
                continue

            print(f"Scraping: {url[:70]}...")
            
            # Initialize the Article object
            article = Article(url)
            
            try:
                # Download the HTML
                article.download()
                # Parse the HTML to extract the main article text
                article.parse()
                
                # Check if we successfully extracted text
                if article.text and len(article.text.split()) > 10:
                    item["full_text"] = article.text
                    print(f"  -> Success! Found {len(article.text.split())} words.")
                else:
                    item["full_text"] = None
                    print("  -> Failed: Page rendered but no article text found (possible JS paywall).")
                
            except ArticleException as e:
                item["full_text"] = None
                print(f"  -> Failed to scrape: {e}")
            except Exception as e:
                item["full_text"] = None
                print(f"  -> Unexpected error: {e}")
                
            scraped_articles.append(item)
            # Sleep briefly to avoid getting IP banned
            time.sleep(1)

    print(f"\nScraping complete. Saving {len(scraped_articles)} articles.")
    
    # Save the updated articles to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        count = 0
        for item in scraped_articles:
            # You might want to decide if you ONLY want to save articles where full_text is not None,
            # or save everything (including failed ones). Here we save everything.
            outfile.write(json.dumps(item) + "\n")
            if item.get("full_text"):
                count += 1
                
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Successfully scraped text for {count} out of {len(scraped_articles)} articles.")

if __name__ == "__main__":
    scrape_full_text()
