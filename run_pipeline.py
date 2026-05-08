import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
import tempfile
import re
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
from src.ingestion.news_collector import main as collect_news
from src.preprocessing.clean_youtube_data import main as clean_youtube
from src.preprocessing.text_normalizer import process_data as normalize
from src.models.VADER import main as run_sentiment_analysis
from src.models.topic_modeling_LDA import get_topics


def save_history_snapshot(payload: dict):
    """Persist a timestamped snapshot of the AI payload for trend/history charts."""
    history_dir = "data/history"
    os.makedirs(history_dir, exist_ok=True)

    ts = payload.get("metadata", {}).get("generated_at") or datetime.utcnow().isoformat()
    safe_ts = (
        str(ts)
        .replace(":", "-")
        .replace(".", "-")
        .replace("Z", "")
        .replace("+", "-")
    )

    # Include a run_key to prevent mixing unrelated brands/queries in history.
    meta = payload.get("metadata") or {}
    run_key = (meta.get("brand") or meta.get("run_key") or "").strip()
    if not run_key:
        yt = meta.get("youtube_queries") or ""
        news = meta.get("news_queries") or ""
        combined = f"{yt} {news}".strip().lower()
        combined = re.sub(r"\s+", " ", combined)
        run_key = re.sub(r"[^a-z0-9]+", "-", combined).strip("-")[:48] or "unknown"

    history_path = os.path.join(history_dir, f"ai_payload_{run_key}_{safe_ts}.json")

    # Atomic write to avoid partial JSON reads.
    fd, tmp_path = tempfile.mkstemp(prefix="ai_payload_", suffix=".json", dir=history_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, history_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    logging.info(f"Saved history snapshot to {history_path}")


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
            "total_texts": len(df),
            "brand": os.getenv("BRAND") or "",
            "youtube_queries": os.getenv("YOUTUBE_QUERIES"),
            "news_queries": os.getenv("NEWS_QUERIES"),
        },
        "sentiment_distribution": sentiment_result,
        "topics": topics,
        "keywords": keywords,
        "ml_summary": ml_summary,
        "top_positive_examples": top_positive,
        "top_negative_examples": top_negative
    }

    # Stable-ish grouping key for history and UI filtering.
    meta = full_report.get("metadata") or {}
    meta["run_key"] = meta.get("brand")
    if not meta["run_key"]:
        combined = f"{meta.get('youtube_queries') or ''} {meta.get('news_queries') or ''}".strip().lower()
        combined = re.sub(r"\s+", " ", combined)
        meta["run_key"] = re.sub(r"[^a-z0-9]+", "-", combined).strip("-")[:48] or "unknown"
 
    output_path = "data/processed/ai_payload.json"

    # Atomic write to avoid Streamlit reading a partially-written JSON.
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="ai_payload_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    logging.info(f"Saved complete AI payload report to {output_path}")

    # history snapshot
    save_history_snapshot(full_report)


def run_pipeline(skip_collection=True):
    try:
        if not skip_collection:
            logging.info("Step 1: Collecting raw data...")
            yt_env = os.getenv("YOUTUBE_QUERIES", "").strip()
            news_env = os.getenv("NEWS_QUERIES", "").strip()

            yt_queries = [q.strip() for q in yt_env.split(",") if q.strip()] if yt_env else None
            news_queries = [q.strip() for q in news_env.split(",") if q.strip()] if news_env else None

            logging.info("YouTube queries used: %s", yt_queries or "<defaults>")
            logging.info("News queries used: %s", news_queries or "<defaults>")

            # Pass explicitly so defaults can't accidentally win.
            # Also overwrite raw collection outputs so each run reflects only the current queries.
            collect_youtube(queries=yt_queries, overwrite_output=True)
            collect_news(queries=news_queries, overwrite_output=True)
            
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
        df = pd.read_json("data/processed/classified_data.jsonl", lines=True)

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
        
        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brand Sentiment Analysis Pipeline")
    parser.add_argument("--collect", action="store_true", help="Run YouTube collection step")
    args = parser.parse_args()
    
    # We invert the argument for the function semantic
    run_pipeline(skip_collection=not args.collect)
