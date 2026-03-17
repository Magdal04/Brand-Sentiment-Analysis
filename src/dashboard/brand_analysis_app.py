import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# -------------------------------
# Configurare pagină
# -------------------------------
st.set_page_config(
    page_title="Brand Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Brand Sentiment Dashboard - Pragmatic Insights")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Filtre și Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV cu comentarii", type=["csv", "json", "jsonl"])

sentiment_filter = st.sidebar.multiselect(
    "Selectează sentiment",
    options=["positive", "neutral", "negative"],
    default=["positive", "neutral", "negative"]
)

show_top_n = st.sidebar.slider("Număr top keywords", min_value=5, max_value=50, value=20)

# -------------------------------
# Load data
# -------------------------------
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".jsonl"):
        df = pd.read_json(uploaded_file, lines=True)
    else:
        df = pd.read_json(uploaded_file)

    # Filtrare sentiment
    df = df[df['sentiment'].isin(sentiment_filter)]

    st.subheader("Distribuție Sentiment")
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)
    ax.set_ylabel("Număr comentarii")
    st.pyplot(fig)

    # Alert for high negative sentiment
    negative_ratio = len(df[df['sentiment']=='negative']) / len(df)
    threshold = 0.3  # 30% comentarii negative

    if negative_ratio > threshold:
        st.warning(f"⚠️ ALERT: High negative sentiment detected! ({negative_ratio:.0%} negative)")
    else:
        st.success(f"✅ Negative sentiment normal ({negative_ratio:.0%})")


    # Trend over time
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'])
        sentiment_time = df.groupby([pd.Grouper(key='published_at', freq='D'), 'sentiment']).size().unstack(fill_value=0)

        st.subheader("Sentiment Trend Over Time")
        fig2, ax2 = plt.subplots(figsize=(10,4))
        sentiment_time.plot(ax=ax2)
        ax2.set_ylabel("Număr comentarii")
        ax2.set_xlabel("Dată")
        st.pyplot(fig2)
    else:
        st.info("Nu există coloană 'timestamp' pentru trend over time.")

    # -------------------------------
    # Wordcloud Negative / Positive
    # -------------------------------
    st.subheader("Wordcloud")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Top cuvinte negative")
        neg_text = " ".join(df[df['sentiment'] == "negative"]["normalized_text"].astype(str))
        if neg_text.strip():
            wc_neg = WordCloud(width=800, height=400, background_color="white").generate(neg_text)
            st.image(wc_neg.to_array())
        else:
            st.write("Nu există comentarii negative în selecție")

    with col2:
        st.write("Top cuvinte pozitive")
        pos_text = " ".join(df[df['sentiment'] == "positive"]["normalized_text"].astype(str))
        if pos_text.strip():
            wc_pos = WordCloud(width=800, height=400, background_color="white").generate(pos_text)
            st.image(wc_pos.to_array())
        else:
            st.write("Nu există comentarii pozitive în selecție")

    # -------------------------------
    # Top Keywords
    # -------------------------------
    st.subheader("Top Keywords")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2), min_df=2)
    X = vectorizer.fit_transform(df['normalized_text'].astype(str))
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    top_keywords = tfidf_df.sum().sort_values(ascending=False).head(show_top_n)
    st.table(top_keywords)

    # -------------------------------
    # Sample Comments
    # -------------------------------
    st.subheader("Most Relevant Comments")
    st.dataframe(df[['normalized_text','sentiment']].head(20))

    # -------------------------------
    # Export CSV / Report
    # -------------------------------
    st.subheader("Export Insights")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered CSV",
        data=csv,
        file_name='sentiment_filtered.csv',
        mime='text/csv',
    )

else:
    st.warning("Încarcă un fișier CSV/JSONL pentru a vizualiza dashboard-ul.")
