import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ---------------------------------------------------
# 1️⃣ TF-IDF Vectorization
# ---------------------------------------------------
def vectorize_text(
    df,
    text_column,
    max_features=1000,
    stop_words=True,
    ngram_range=(1, 1)
):
    """
    Safe TF-IDF vectorization.
    Handles empty text, NaN values, and stopword-only columns.
    """

    # Clean column
    corpus = df[text_column].fillna("").astype(str)

    # Remove completely empty rows
    corpus = corpus[corpus.str.strip() != ""]

    if corpus.shape[0] == 0:
        raise ValueError("Selected column contains no valid text data.")

    stop = "english" if stop_words else None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop,
        ngram_range=ngram_range
    )

    try:
        X_tfidf = vectorizer.fit_transform(corpus)

        if len(vectorizer.get_feature_names_out()) == 0:
            raise ValueError("No valid words found after preprocessing.")

    except ValueError:
        raise ValueError(
            "TF-IDF failed: The selected column may contain only stopwords "
            "or non-text values. Try disabling stopwords or choosing another column."
        )

    return X_tfidf, vectorizer
# 2️⃣ Generate Dendrogram (subset only)
# ---------------------------------------------------
def generate_dendrogram(X_tfidf, linkage_method="ward", sample_size=50):
    """
    Generates dendrogram for subset of data.
    Returns matplotlib figure.
    """

    sample_size = min(sample_size, X_tfidf.shape[0])

    subset = X_tfidf[:sample_size].toarray()

    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(
        sch.linkage(subset, method=linkage_method),
        ax=ax
    )

    ax.set_title("Dendrogram")
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Distance")

    return fig


# ---------------------------------------------------
# 3️⃣ Apply Hierarchical Clustering
# ---------------------------------------------------
def apply_clustering(
    X_tfidf,
    n_clusters=5,
    linkage_method="ward",
    metric="euclidean"
):
    """
    Applies Agglomerative Clustering.
    Returns labels and silhouette score.
    """

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric
    )

    labels = model.fit_predict(X_tfidf.toarray())

    score = silhouette_score(X_tfidf, labels)

    return labels, score


# ---------------------------------------------------
# 4️⃣ PCA Projection (2D)
# ---------------------------------------------------
def reduce_dimensionality(X_tfidf):
    """
    Reduces high dimensional TF-IDF to 2D using PCA.
    """

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_tfidf.toarray())

    return reduced


# ---------------------------------------------------
# 5️⃣ Cluster Summary (Top Keywords + Snippet)
# ---------------------------------------------------
def generate_cluster_summary(
    df,
    labels,
    X_tfidf,
    vectorizer,
    text_column,
    n_clusters
):
    """
    Generates cluster business summary.
    """

    feature_names = vectorizer.get_feature_names_out()
    summary_data = []

    for cluster_id in range(n_clusters):

        cluster_indices = np.where(labels == cluster_id)[0]

        cluster_matrix = X_tfidf[cluster_indices]

        mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()

        top_indices = mean_tfidf.argsort()[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]

        snippet = df.iloc[cluster_indices[0]][text_column][:200]

        summary_data.append({
            "Cluster ID": cluster_id,
            "Number of Articles": len(cluster_indices),
            "Top Keywords": ", ".join(top_keywords),
            "Sample Snippet": snippet + "..."
        })

    return pd.DataFrame(summary_data)
from sklearn.metrics import silhouette_score

score = silhouette_score(X_tfidf, labels)
print("Silhouette Score:", score)
