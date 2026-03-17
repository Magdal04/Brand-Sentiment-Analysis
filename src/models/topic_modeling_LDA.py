import gensim
from gensim import corpora
from nltk.corpus import stopwords

def get_topics(df, num_topics=5):
    """Extrage topicurile din DataFrame-ul oferit."""
    if df.empty or len(df) < 10: return {}

    texts = [str(text).split() for text in df['normalized_text'].dropna()]
    stop_words = set(stopwords.words('english'))
    texts = [[word for word in doc if word not in stop_words] for doc in texts]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    if len(dictionary) == 0: return {}

    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(
        corpus=corpus, id2word=dictionary, 
        num_topics=num_topics, random_state=42, passes=10
    )

    # Returnăm topicurile formatate frumos
    results = {}
    for idx, topic in lda_model.print_topics(-1):
        results[f"topic_{idx}"] = topic
    return results