import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Hierarchical Clustering App", layout="wide")

st.title("🟣 Hierarchical Clustering for Text Data")

st.markdown("""
Upload any CSV file containing text data.
The app will perform:
- TF-IDF Vectorization
- Hierarchical Clustering
- PCA Visualization
- Silhouette Score Evaluation
""")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
    except:
        df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully ✅")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Detect Text Columns
    # --------------------------------------------------
    text_columns = df.select_dtypes(include=["object"]).columns

    if len(text_columns) == 0:
        st.error("No text column found in dataset.")
        st.stop()

    text_column = st.selectbox("Select Text Column", text_columns)

    corpus = df[text_column].fillna("").astype(str)
    corpus = corpus[corpus.str.strip() != ""]

    if corpus.empty:
        st.error("Selected column contains no valid text.")
        st.stop()

    # --------------------------------------------------
    # TF-IDF Parameters
    # --------------------------------------------------
    st.sidebar.header("TF-IDF Settings")

    max_features = st.sidebar.slider("Max Features", 100, 2000, 1000)
    use_stopwords = st.sidebar.checkbox("Remove English Stopwords", True)

    stop_words = "english" if use_stopwords else None

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )

        X_tfidf = vectorizer.fit_transform(corpus)

    except ValueError:
        st.error("Empty vocabulary error. Try disabling stopwords or increasing max_features.")
        st.stop()

    # --------------------------------------------------
    # Clustering Settings
    # --------------------------------------------------
    st.sidebar.header("Clustering Settings")

    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

    if st.button("Run Clustering"):

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        )

        labels = model.fit_predict(X_tfidf.toarray())

        df = df.loc[corpus.index].copy()
        df["Cluster"] = labels

        st.success("Clustering Completed ✅")

        # --------------------------------------------------
        # PCA Visualization (Streamlit Native)
        # --------------------------------------------------
        st.subheader("📊 Cluster Visualization (PCA 2D)")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_tfidf.toarray())

        viz_df = pd.DataFrame({
            "PCA1": reduced[:, 0],
            "PCA2": reduced[:, 1],
            "Cluster": labels.astype(str)
        })

        st.scatter_chart(
            viz_df,
            x="PCA1",
            y="PCA2",
            color="Cluster"
        )

        # --------------------------------------------------
        # Silhouette Score
        # --------------------------------------------------
        st.subheader("📈 Silhouette Score")

        if len(set(labels)) > 1:
            score = silhouette_score(X_tfidf, labels)
            st.write("Silhouette Score:", round(score, 3))
        else:
            st.warning("Silhouette score requires at least 2 clusters.")

        # --------------------------------------------------
        # Cluster Summary
        # --------------------------------------------------
        st.subheader("📋 Cluster Summary")

        feature_names = vectorizer.get_feature_names_out()
        summary = []

        for cluster_id in range(n_clusters):

            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_matrix = X_tfidf[cluster_indices]

            mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]

            summary.append({
                "Cluster": cluster_id,
                "Documents": len(cluster_indices),
                "Top Keywords": ", ".join(top_words)
            })

        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df)

else:
    st.info("Please upload a CSV file to begin.")
