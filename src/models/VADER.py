import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

INPUT_FILE = "data/processed/normalized_data.jsonl"
OUTPUT_FILE = "data/processed/classified_data.jsonl"

def main():
    sia = SentimentIntensityAnalyzer()
    
    try:
        df = pd.read_json(INPUT_FILE, lines=True)
    except Exception as e:
        print(f"Could not load {INPUT_FILE}: {e}")
        return {}

    text_field = "normalized_text"
    
    if text_field not in df.columns:
        print(f"Field {text_field} not found in {INPUT_FILE}")
        return {}

    # compute sentiment scores using the unified text field
    df['sentiment_scores'] = df[text_field].apply(lambda x: sia.polarity_scores(str(x)))
    df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])

    # classify sentiment
    def classify_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df['sentiment'] = df['compound'].apply(classify_sentiment)

    # distribution
    sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100
    print(f"--- Distribution for {INPUT_FILE} ---")
    print(sentiment_distribution)

    print("\nTop Negative:")
    print(df.sort_values("compound").head(5)[[text_field,'compound']])
    print("\nTop Positive:")
    print(df.sort_values("compound", ascending=False).head(5)[[text_field,'compound']])
    print("\n" + "="*50 + "\n")

    df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)

    return {
        "positive_ratio": sentiment_distribution.get("positive", 0),
        "negative_ratio": sentiment_distribution.get("negative", 0),
        "neutral_ratio": sentiment_distribution.get("neutral", 0)
    }

if __name__ == '__main__':
    main()