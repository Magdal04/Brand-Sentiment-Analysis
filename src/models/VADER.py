import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


OUTPUT_FILE = "data/processed/youtube_comments_classified.jsonl"
def main():
    # load data
    df = pd.read_json("data/processed/youtube_comments_normalized.jsonl", lines=True)

    sia = SentimentIntensityAnalyzer()


    # compute sentiment scores
    df['sentiment_scores'] = df['normalized_text'].apply(lambda x: sia.polarity_scores(str(x)))
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
    print(sentiment_distribution)


    print(df.sort_values("compound").head(10)[['normalized_text','compound']])
    print(df.sort_values("compound", ascending=False).head(10)[['normalized_text','compound']])

    df.to_json(OUTPUT_FILE, orient="records", lines=True, force_ascii=False)

    return {
        "positive_ratio": sentiment_distribution.get("positive", 0),
        "negative_ratio": sentiment_distribution.get("negative", 0),
        "neutral_ratio": sentiment_distribution.get("neutral", 0)
    }

if __name__ == '__main__':
    main()