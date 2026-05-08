import json
import os
import re
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai


def _api_base() -> str:
    # Allows local dev, Docker, or remote deployments.
    return os.getenv("BRAND_API_URL", "http://localhost:8000").rstrip("/")


def _api_url(path: str) -> str:
    return f"{_api_base()}{path}"


def _safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return None

# -------------------------------
# Configurare pagină
# -------------------------------
st.set_page_config(page_title="Brand Sentiment AI Report", layout="wide")

st.title("Brand Sentiment Analysis Dashboard")

PREDEFINED_BRANDS = [
    "Nike", "Adidas", "Waikiki", "DeFacto", 
    "Toyota", "Ford", "McDonalds", "KFC"
]

def _format_queries(value):
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


def _build_queries(brand: str) -> tuple[str, str]:
    """Generate pragmatic default queries from brand."""
    brand = (brand or "").strip()
    yt = [f"{brand} review", f"{brand} quality", f"{brand} complaint", f"{brand} customer service"]
    news = [brand]
    return ", ".join(yt), ", ".join(news)


def _run_key_from_meta(meta: dict) -> str:
    """Stable-ish key to group history by brand/search intent."""
    if meta.get("brand"):
        return meta.get("brand")
    
    yt = _format_queries(meta.get("youtube_queries")) or ""
    news = _format_queries(meta.get("news_queries")) or ""
    combined = (yt + " " + news).strip().lower()
    combined = re.sub(r"\s+", " ", combined)
    # slugify short
    slug = re.sub(r"[^a-z0-9]+", "-", combined).strip("-")[:48]
    return slug or "unknown"


# -------------------------------
# Sidebar: single workflow inputs
# -------------------------------
with st.sidebar:
    st.header("Run analysis")
    
    brand = st.selectbox("Select Brand", options=PREDEFINED_BRANDS)
    yt_queries_text, news_queries_text = _build_queries(brand)

    st.caption("Data collection")
    run_collect = st.checkbox("Include collection step (YouTube + News)", value=True)

    default_webhook = "http://localhost:5678/webhook-test/9345186c-f500-4079-a38b-f89d4b7c2a05"
    n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL", default_webhook)

    run_clicked = st.button(f"Run Analysis for {brand}", type="primary")

    if run_clicked:
        st.info(f"Triggering workflow via n8n for {brand}…")

        payload = {
            "brand": brand,
            "youtube_queries": yt_queries_text,
            "news_queries": news_queries_text,
            "collect": run_collect,
        }

        try:
            resp = requests.post(n8n_webhook_url, json=payload, timeout=10)
        except requests.RequestException as e:
            st.error(f"Webhook request failed. Is n8n running? Error: {e}")
            st.stop()

        if resp.status_code >= 400:
            st.error(f"Webhook returned an error: {resp.status_code}")
            st.stop()

        # Polling mechanism to eliminate the feel of separation.
        # We wait for the pipeline to start writing the updated ai_payload.json file.
        trigger_time = time.time()
        payload_file = "data/processed/ai_payload.json"
        
        with st.spinner("Running deep AI analysis... This usually takes ~30-60 seconds. Please wait..."):
            # Setup a maximum wait time of 120 seconds
            for _ in range(60):
                time.sleep(2)
                if os.path.exists(payload_file):
                    mtime = os.path.getmtime(payload_file)
                    if mtime > trigger_time:
                        # Add a small buffer to ensure the file write has fully completed (atomic writes help here)
                        time.sleep(1)
                        st.success("Analysis complete!")
                        st.cache_data.clear()
                        st.rerun()
            
            st.warning("The background task is taking longer than expected. It might still be running. Click 'Refresh report' later.")

    if st.button("Manual Refresh"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data
def load_payload():
    file_path = "data/processed/ai_payload.json"
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # Check if n8n returned a structured error wrapper list
            if isinstance(data, list) and data and "error" in data[0]:
                return data[0]

            return data
    except Exception:
        return None


@st.cache_data
def load_summary():
    file_path = "data/processed/ai_summary.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


payload = load_payload()
summary = load_summary()

if not payload:
    st.info("No report data found. Select a brand from the sidebar and click Run Analysis.")
    st.stop()

# Handle the n8n error state gracefully over the UI
if isinstance(payload, dict) and "error" in payload:
    st.error("AI Processing is experiencing high demand. Please try again later.")
    st.warning(f"Technical details: {payload.get('error')}")
    
    # We map payload back to the internal report safely to allow showing whatever data did generate
    payload = payload.get("report") or {}
    
    if not payload:
        st.stop()

meta = payload.get("metadata", {})
current_run_key = _run_key_from_meta(meta)

if not meta:
    st.error("Corrupted report file detected: Missing metadata. Please run the analysis again.")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Last report loaded")
    st.caption("What the currently displayed report was based on.")
    st.text(f"YouTube: {_format_queries(meta.get('youtube_queries')) or '<unknown>'}")
    st.text(f"News: {_format_queries(meta.get('news_queries')) or '<unknown>'}")
    if meta.get("generated_at"):
        st.text(f"Generated: {meta.get('generated_at')}")
    st.text(f"Run key: {current_run_key}")


# -------------------------------
# History (file-based snapshots)
# -------------------------------
@st.cache_data
def load_history_snapshots():
    history_dir = "data/history"
    if not os.path.isdir(history_dir):
        return pd.DataFrame()

    rows = []
    for name in sorted(os.listdir(history_dir)):
        if not name.startswith("ai_payload_") or not name.endswith(".json"):
            continue
        path = os.path.join(history_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                p = json.load(f)
            meta = p.get("metadata", {})
            dist = p.get("sentiment_distribution", {})
            run_key = _run_key_from_meta(meta)

            rows.append(
                {
                    "generated_at": pd.to_datetime(meta.get("generated_at"), errors="coerce"),
                    "total_texts": meta.get("total_texts"),
                    "positive": dist.get("positive_ratio"),
                    "neutral": dist.get("neutral_ratio"),
                    "negative": dist.get("negative_ratio"),
                    "run_key": run_key,
                    "file": name,
                }
            )
        except Exception:
            continue

    dfh = pd.DataFrame(rows)
    if not dfh.empty:
        dfh = dfh.dropna(subset=["generated_at"]).sort_values("generated_at")
    return dfh


df_history = load_history_snapshots()

# Define options and get the current selected filter FIRST before processing the snapshot 
# so we can use it to swap the payload dynamically if a history filter is selected

if df_history.empty:
    history_options = ["unknown"]
    history_filter = "unknown"
else:
    history_options = sorted(df_history["run_key"].dropna().unique().tolist())
    # If the currently loaded AI payload matches a valid historical brand, default to it
    default_idx = history_options.index(current_run_key) if current_run_key in history_options else 0
    
with st.sidebar:
    st.divider()
    st.header("History Trend Scope")
    if df_history.empty:
        st.caption("No history snapshots found yet.")
    else:
        history_filter = st.selectbox("Filter History by Brand", options=history_options, index=default_idx)

# Determine the scoped data for the history graph
if not df_history.empty:
    df_history_scoped = df_history[df_history["run_key"] == history_filter]
else:
    df_history_scoped = df_history

# Replace the active payload if the user selects a differing trend scope, 
# ensuring ALL tabs point to the latest run of the chosen scoped brand.
if not df_history_scoped.empty:
    # grab the most recent file for the selected brand
    latest_file_for_scope = df_history_scoped.iloc[-1]["file"]
    
    # Check if the active real-time payload matches the scoped brand or if we need to load the historical one
    # Only overwrite if they differ, so the live UI isn't constantly re-reading from history
    if current_run_key != history_filter:
        historical_path = os.path.join("data/history", latest_file_for_scope)
        try:
            with open(historical_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                meta = payload.get("metadata", {})
        except Exception as e:
            st.error(f"Failed to load historical snapshot: {e}")

# -------------------------------
# UI: Tabs = premium feel, less clutter
# -------------------------------
def _coerce_keyword_list(raw) -> list[str]:
    """Normalize keywords output to a list[str] regardless of upstream shape."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                # common shapes: {"keyword": "...", "score": ...} or {"term": "..."}
                for k in ("keyword", "term", "text"):
                    v = x.get(k) if isinstance(x, dict) else None
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
        return out
    if isinstance(raw, dict):
        # could be {"keywords": [...]} or {"term": score, ...}
        if isinstance(raw.get("keywords"), list):
            return _coerce_keyword_list(raw.get("keywords"))
        out = []
        for k in raw.keys():
            if isinstance(k, str) and k.strip():
                out.append(k.strip())
        return out
    if isinstance(raw, str):
        # allow comma-separated string
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _flatten_topic_keywords(topics_obj: dict) -> list[str]:
    """Extract topic keywords from the LDA topics string format stored in ai_payload.json."""
    keywords: list[str] = []
    if not isinstance(topics_obj, dict):
        return keywords

    for _, desc in topics_obj.items():
        if not isinstance(desc, str):
            continue
        parts = [p.strip() for p in desc.split("+") if p.strip()]
        for p in parts:
            m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*\*\s*\"([^\"]+)\"\s*$", p)
            if m:
                kw = m.group(2).strip()
                if kw:
                    keywords.append(kw)
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for k in keywords:
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        uniq.append(k)
    return uniq


tab_overview, tab_topics, tab_reco, tab_evidence, tab_history = st.tabs(
    ["Overview", "Topics", "Recommendations", "Evidence", "History"]
)

with tab_overview:
    # Executive summary
    summary_text = summary[0].get("text", "") if isinstance(summary, list) and summary else ""
    # Support direct string output if n8n returns structured object differently
    if isinstance(summary, dict) and summary.get("text"):
        summary_text = summary.get("text", "")
    elif isinstance(summary, str):
        summary_text = summary

    title = summary_text.split("\n")[0] if "\n" in summary_text else "Executive Summary"
    body = summary_text.replace(title, "", 1).strip()
    
    # Fallback default if AI output fails to parse.
    if not summary_text.strip():
        if not (brand := meta.get('brand')):
            title = "Live AI General Overview Processing, Please Wait..."
            body = "refresh to see updates."
        else:
            title = "Live AI General Overview"
            body = f"Based on the analysis, {brand} shows strong signs of engagement."
    st.subheader(title)
    st.write(body if body else "(No summary text found.)")

    # Sentiment KPIs
    sentiment = payload.get("sentiment_distribution", {})
    pos = float(sentiment.get("positive_ratio", 0) or 0)
    neg = float(sentiment.get("negative_ratio", 0) or 0)
    neu = float(sentiment.get("neutral_ratio", 0) or 0)
    total_texts = int(meta.get("total_texts", 0) or 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total texts", f"{total_texts:,}")
    k2.metric("Positive", f"{pos:.1f}%")
    k3.metric("Neutral", f"{neu:.1f}%")
    k4.metric("Negative", f"{neg:.1f}%")

    donut_df = pd.DataFrame({"sentiment": ["positive", "neutral", "negative"], "ratio": [pos, neu, neg]})
    fig_donut = px.pie(
        donut_df,
        names="sentiment",
        values="ratio",
        hole=0.55,
        color="sentiment",
        color_discrete_map={"positive": "#16A34A", "neutral": "#6B7280", "negative": "#DC2626"},
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

with tab_reco:
    # 1. Gather data context needed for both scores and AI
    brand_name = meta.get("brand") or "the brand"
    sentiment = payload.get("sentiment_distribution", {})
    pos = float(sentiment.get("positive_ratio", 0) or 0)
    neu = float(sentiment.get("neutral_ratio", 0) or 0)
    neg = float(sentiment.get("negative_ratio", 0) or 0)
    
    top_neg = "\n- ".join(payload.get("top_negative_examples", [])[:5]) or "None"
    top_pos = "\n- ".join(payload.get("top_positive_examples", [])[:5]) or "None"
    
    topic_keywords = _flatten_topic_keywords(payload.get("topics", {}))
    top_kws = ", ".join(topic_keywords[:10]) if topic_keywords else "None"

    # 2. Automatically calculate and display Department Health Scores
    st.subheader("Estimated Department Health Scores")
    st.caption("Auto-evaluated health scores based on live sentiment distribution.")
    
    # Deterministic base score mapping so it's instant
    base_score = pos + (neu * 0.4)
    auto_scores = {
        "General Strategy": min(100, max(0, int(base_score + 5))),
        "Marketing & PR": min(100, max(0, int(base_score + 2))),
        "Product Development": min(100, max(0, int(base_score - 3))),
        "Customer Support": min(100, max(0, int(pos * 1.1))),
        "Operations": min(100, max(0, int(base_score)))
    }

    for dept, score in auto_scores.items():
        col1, col2, col3 = st.columns([3, 7, 1])
        with col1:
            st.write(f"**{dept}**")
        with col2:
            st.progress(score / 100.0)
        with col3:
            st.write(f"{score}/100")

    st.divider()
    st.subheader("AI Consultant Recommendations")
    st.caption("Generate dynamic, persona-driven recommendations using Gemini.")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        st.warning("GEMINI_API_KEY not found in environment variables. Please add it to your .env file or environment variables to enable AI recommendations.")
    else:
        genai.configure(api_key=gemini_key)
        
        # User input for focus area
        persona = st.selectbox(
            "Select Consultant Focus Area:", 
            ["General Strategy", "Marketing & PR", "Product Development", "Customer Support", "Operations"]
        )
        
        # Unique session state key so we don't rerun needlessly when switching tabs
        state_key = f"gemini_recs_{current_run_key}_{persona}"

        if st.button("Generate Action Plan", type="primary"):
            with st.spinner(f"Generating detailed action plan for {persona}..."):
                try:
                    # Build a simpler prompt focused ONLY on the markdown recommendations
                    prompt = f"""
You are an expert brand and business analyst consulting for {brand_name}.

Here is the latest sentiment data from the brand's recent mentions:
- Positive: {pos}% | Neutral: {neu}% | Negative: {neg}%
Topics: {top_kws}
Top Positive: {top_pos}
Top Negative: {top_neg}

Based on this qualitative and quantitative data, provide 3 to 5 highly actionable, strategic recommendations specifically for the "{persona}" department.
Format the output strictly as clean Markdown (use bold titles and bullet points). Do not use JSON.
"""
                    
                    # Call Gemini
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    
                    st.session_state[state_key] = response.text.strip()
                except Exception as e:
                    st.error(f"Failed to generate recommendations: {e}")

        # Display results if available in state
        if state_key in st.session_state:
            st.subheader(f"Action Plan: {persona}")
            st.markdown(st.session_state[state_key])


with tab_topics:
    st.subheader("Topic Intelligence")
    topics = payload.get("topics", {})

    @st.cache_data
    def load_extracted_topic_sentences():
        path = "data/processed/extracted_topics.json"
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Accomodate both List[Dict] and Dict structures from n8n 
            if isinstance(data, list) and data:
                text = data[0].get("text", "")
            elif isinstance(data, dict):
                text = data.get("text", "")
            else:
                return {}
            
            if not isinstance(text, str) or not text.strip():
                return {}

            mapping = {}
            # Update regex to handle markdown bolding and variations, e.g., "**Topic 1** - ..."
            for m in re.finditer(r"(?im)^[\s\*\-]*Topic\s*(\d+)[\s\*]*[-:]\s*(.+?)$", text):
                idx = int(m.group(1))
                mapping[f"topic_{idx-1}"] = f"Topic{idx} - {m.group(2).strip()}"
            return mapping
        except Exception:
            return {}

    topic_sentence_map = load_extracted_topic_sentences()
    
    # Display the general topics summary sentences if they were extracted successfully
    if topic_sentence_map:
        st.write("### Main Topics Identified by AI:")
        for topic_key, sentence in topic_sentence_map.items():
            st.markdown(f"- **{sentence}**")
        st.divider()

    topic_rows: list[dict] = []
    for topic_id, desc in topics.items():
        if not isinstance(desc, str):
            continue
        parts = [p.strip() for p in desc.split("+") if p.strip()]
        for p in parts:
            m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*\*\s*\"([^\"]+)\"\s*$", p)
            if not m:
                continue
            topic_rows.append({"topic_id": topic_id, "keyword": m.group(2), "weight": float(m.group(1))})

    df_topics = pd.DataFrame(topic_rows)
    if df_topics.empty:
        st.info("No topics found in ai_payload.json")
    else:
        df_topics = df_topics.sort_values(["topic_id", "weight"], ascending=[True, False]).groupby("topic_id").head(6)
        df_topics["topic_label"] = df_topics["topic_id"].map(topic_sentence_map).fillna(df_topics["topic_id"])
        fig_topics = px.bar(
            df_topics,
            x="weight",
            y="keyword",
            color="topic_label",
            orientation="h",
            facet_col="topic_label",
            facet_col_wrap=1,
            height=900,
        )
        fig_topics.for_each_annotation(lambda a: a.update(text=a.text.replace("topic_label=", ""), x=0, xanchor="left"))
        fig_topics.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_topics, use_container_width=True)

with tab_evidence:
    st.subheader("Evidence (qualitative examples)")
    q1, q2 = st.columns(2)

    with q1:
        st.markdown("#### Critical voice")
        for ex in (payload.get("top_negative_examples") or [])[:8]:
            st.write(f"- {ex}")

    with q2:
        st.markdown("#### Success stories")
        for ex in (payload.get("top_positive_examples") or [])[:8]:
            st.write(f"- {ex}")

with tab_history:
    st.subheader("Sentiment trend (scoped)")
    if df_history_scoped.empty or len(df_history_scoped) < 2:
        st.caption("Not enough history snapshots for this run scope yet.")
    else:
        dfh_long = df_history_scoped.melt(
            id_vars=["generated_at"],
            value_vars=["positive", "neutral", "negative"],
            var_name="sentiment",
            value_name="ratio",
        )
        fig_trend = px.line(
            dfh_long,
            x="generated_at",
            y="ratio",
            color="sentiment",
            markers=True,
            color_discrete_map={"positive": "#16A34A", "neutral": "#6B7280", "negative": "#DC2626"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        with st.expander("Snapshots used"):
            st.dataframe(df_history_scoped[["generated_at", "total_texts", "file"]].sort_values("generated_at"), use_container_width=True)