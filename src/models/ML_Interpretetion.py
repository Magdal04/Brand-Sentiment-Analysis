import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def main(df):
    # Keep only positive and negative
    df_binary = df[df['sentiment'].isin(['positive', 'negative'])].copy()

    # Encode target
    df_binary['label'] = df_binary['sentiment'].map({'negative': 0, 'positive': 1})

    X_text = df_binary['normalized_text']
    y = df_binary['label']

    custom_stop_words = list(text.ENGLISH_STOP_WORDS)
    custom_stop_words.extend(['adidas', 'like', 'shoe', 'shoes', 'just', 'face', 'smiling'])

    # Vectorizer (folosește stopwords custom)
    vectorizer = TfidfVectorizer(
        stop_words=custom_stop_words,
        max_features=5000,
        ngram_range=(1,2),
        min_df=5
    )

    X = vectorizer.fit_transform(X_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    # Top negative predictors
    top_negative = feature_names[np.argsort(coefficients)[:20]].tolist()

    # Top positive predictors
    top_positive = feature_names[np.argsort(coefficients)[-20:]].tolist()

    print("Top Negative Words:")
    print(top_negative)

    print("\nTop Positive Words:")
    print(top_positive)

    return {
        "classification_report": report,
        "top_negative_words": top_negative,
        "top_positive_words": top_positive
    }


if __name__ == "__main__":
    df = pd.read_json("data/processed/youtube_comments_classified.jsonl", lines=True)
    main(df)