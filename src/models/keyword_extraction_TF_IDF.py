from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
import pandas as pd

def main(df):
    texts = df['normalized_text'].dropna().tolist()

    custom_stop_words = list(text.ENGLISH_STOP_WORDS)
    custom_stop_words.extend(['adidas', 'like', 'shoe', 'shoes', 'just', 'face', 'smiling'])

    vectorizer = TfidfVectorizer(
        stop_words= custom_stop_words,
        max_features=1000,
        ngram_range=(2,3),
        min_df=5
    )

    X = vectorizer.fit_transform(texts)

    feature_names = np.array(vectorizer.get_feature_names_out())

    # mean tfidf score per term
    mean_scores = X.mean(axis=0).A1

    top_indices = mean_scores.argsort()[-20:][::-1]

    top_terms = feature_names[top_indices].tolist()

    for term in top_terms:
        print(term)


    # Keyword from Negative comments
    neg_texts = df[df['sentiment'] == 'negative']['normalized_text'].dropna().tolist()

    X_neg = vectorizer.fit_transform(neg_texts)

    feature_names_neg = np.array(vectorizer.get_feature_names_out())
    mean_scores_neg = X_neg.mean(axis=0).A1
    top_indices_neg = mean_scores_neg.argsort()[-20:][::-1]
    top_terms_neg = feature_names_neg[top_indices_neg].tolist()

    print("\nTop negative terms:")
    for term in top_terms_neg:
        print(term)

    return {
        "top_terms": top_terms,
        "top_negative_terms": top_terms_neg
    }

if __name__ == "__main__":
    df = pd.read_json("data/processed/youtube_comments_classified.jsonl", lines=True)
    main(df)