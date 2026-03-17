import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

load_dotenv()

# Import early steps safely
from src.ingestion.youtube_collector import main as collect_youtube
from src.preprocessing.clean_youtube_data import main as clean_youtube
from src.preprocessing.text_normalizer import process_comments as normalize
from src.models.VADER import main as run_sentiment_analysis
from src.models.topic_modeling_LDA import get_topics


def save_outputs(df):
    output_path = "data/processed/latest.jsonl"
    df.to_json(output_path, orient="records", lines=True)
    logging.info(f"Saved latest outputs to {output_path}")

def save_full_report(df, sentiment_result, topics, keywords, ml_summary):
    # Retrieve the raw text for examples; handle KeyError if 'text' isn't available
    if "text" in df.columns:
        top_negative = df[df["sentiment"] == "negative"].head(5)["text"].tolist()
        top_positive = df[df["sentiment"] == "positive"].head(5)["text"].tolist()
    elif "normalized_text" in df.columns:
        top_negative = df[df["sentiment"] == "negative"].head(5)["normalized_text"].tolist()
        top_positive = df[df["sentiment"] == "positive"].head(5)["normalized_text"].tolist()
    else:
        top_negative = []
        top_positive = []

    full_report = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "total_comments": len(df)
        },
        "sentiment_distribution": sentiment_result,
        "topics": topics,
        "keywords": keywords,
        "ml_summary": ml_summary,
        "top_positive_examples": top_positive,
        "top_negative_examples": top_negative
    }

    output_path = "data/processed/ai_payload.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)
    logging.info(f"Saved complete AI payload report to {output_path}")


def run_pipeline(skip_collection=False):
    try:
        if not skip_collection:
            logging.info("Step 1: Collecting raw data...")
            collect_youtube()
        else:
            logging.info("Step 1: Skipping YouTube collection.")

        logging.info("Step 2: Cleaning raw data...")
        clean_youtube()

        logging.info("Step 3: Normalizing text...")
        normalize()

        logging.info("Step 4: Running VADER Sentiment Analysis...")
        sentiment_result = run_sentiment_analysis()

        # At this point, the classified JSONL file is guaranteed to exist.
        from src.models.ML_Interpretetion import main as ml_interpretation
        from src.models.keyword_extraction_TF_IDF import main as keyword_extraction

        logging.info("Step 5: Loading classified data for advanced modeling...")
        df = pd.read_json("data/processed/youtube_comments_classified.jsonl", lines=True)

        if df.empty:
            logging.warning("Proceeding with empty dataframe!")
            return

        logging.info("Step 6: Extracting Topics (LDA)...")
        topics = get_topics(df)

        logging.info("Step 7: Running ML Interpretation...")
        ml_summary = ml_interpretation(df)

        logging.info("Step 8: Extracting TF-IDF Keywords...")
        keywords = keyword_extraction(df)
        
        logging.info("Step 9: Consolidating outputs...")
        save_outputs(df)
        save_full_report(df, sentiment_result, topics, keywords, ml_summary)
        
        logging.info("Pipeline completed successfully! \u2728")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brand Sentiment Analysis Pipeline")
    parser.add_argument("--collect", action="store_true", help="Run YouTube collection step")
    args = parser.parse_args()
    
    # We invert the argument for the function semantic
    run_pipeline(skip_collection=not args.collect)
