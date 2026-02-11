import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.markdown("""
This system uses **Hierarchical Clustering** to automatically group similar news articles 
based on textual similarity.

👉 Discover hidden themes without defining categories upfront.
""")

# --------------------------------------------------
# Sidebar - Dataset Upload
# --------------------------------------------------
st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, encoding="latin1")
    st.success("Dataset loaded successfully!")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Select Text Column
    # --------------------------------------------------
    text_columns = df.select_dtypes(include=["object"]).columns

    if len(text_columns) == 0:
        st.error("No text columns detected in dataset.")
        st.stop()

    text_column = st.sidebar.selectbox("Select Text Column", text_columns)

    corpus = df[text_column].fillna("").astype(str)
    corpus = corpus[corpus.str.strip() != ""]

    if corpus.shape[0] == 0:
        st.error("Selected column contains no valid text.")
        st.stop()

    # --------------------------------------------------
    # TF-IDF Controls
    # --------------------------------------------------
    st.sidebar.header("📝 Text Vectorization Controls")

    max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)
    use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

    ngram_option = st.sidebar.selectbox(
        "N-gram Range",
        ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
    )

    if ngram_option == "Unigrams":
        ngram_range = (1, 1)
    elif ngram_option == "Bigrams":
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 2)

    stop_words = "english" if use_stopwords else None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    X_tfidf = vectorizer.fit_transform(corpus)

    # --------------------------------------------------
    # Clustering Controls
    # --------------------------------------------------
    st.sidebar.header("🌳 Hierarchical Clustering Controls")

    linkage_method = st.sidebar.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

    # --------------------------------------------------
    # Apply Clustering
    # --------------------------------------------------
    if st.button("🟩 Apply Clustering"):

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )

        labels = model.fit_predict(X_tfidf.toarray())

        df = df.loc[corpus.index].copy()
        df["Cluster"] = labels

        # --------------------------------------------------
        # PCA Visualization (Plotly)
        # --------------------------------------------------
        st.subheader("📊 Cluster Visualization (2D Projection)")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_tfidf.toarray())

        viz_df = pd.DataFrame({
            "PCA1": reduced[:, 0],
            "PCA2": reduced[:, 1],
            "Cluster": labels,
            "Snippet": corpus.values
        })

        fig = px.scatter(
            viz_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            hover_data=["Snippet"]
        )

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------
        # Silhouette Score
        # --------------------------------------------------
        st.subheader("📊 Validation")

        if len(set(labels)) > 1:
            score = silhouette_score(X_tfidf, labels)
            st.write("Silhouette Score:", round(score, 3))

            if score > 0.5:
                st.success("Clusters are well separated.")
            elif score > 0:
                st.warning("Clusters overlap moderately.")
            else:
                st.error("Poor clustering structure.")
        else:
            st.error("Silhouette score cannot be computed with only one cluster.")

        # --------------------------------------------------
        # Cluster Summary
        # --------------------------------------------------
        st.subheader("📋 Cluster Summary")

        feature_names = vectorizer.get_feature_names_out()
        summary_data = []

        for cluster_id in range(n_clusters):

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_matrix = X_tfidf[cluster_indices]

            mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]

            snippet = corpus.iloc[cluster_indices[0]][:200]

            summary_data.append({
                "Cluster ID": cluster_id,
                "Number of Articles": len(cluster_indices),
                "Top Keywords": ", ".join(top_keywords),
                "Sample Snippet": snippet + "..."
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)

        st.info("""
Articles grouped in the same cluster share similar vocabulary and themes. 
These clusters can be used for automatic tagging, recommendations, and content organization.
""")

else:
    st.warning("Please upload a CSV file to begin.")
