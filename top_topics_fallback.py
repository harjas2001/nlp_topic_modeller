"""
top_topics_fallback.py
─────────────────────────────────────────────────────────────────────────────
NLP topic modelling pipeline for voicebot fallback / no-match utterances.

Clusters raw fallback utterances using TF-IDF + K-Means, automatically
selects the optimal number of clusters via silhouette scoring, and exports
a ranked topic report to a master CSV. The CSV is appended on each run
(with a timestamp column) so it can be consumed directly by a PowerBI
dashboard as a rolling dataset.

Pipeline
────────
  1. Load & preprocess  — filter to English, lemmatize, remove stopwords
  2. Feature extraction — TF-IDF (unigrams + bigrams, top 1000 features)
  3. Clustering         — K-Means with silhouette-optimised K (up to MAX_CLUSTERS)
  4. Topic labelling    — top TF-IDF terms per cluster centroid
  5. Ranking            — clusters sorted by size (most common fallback topics first)
  6. Export             — appends ranked results to master CSV for PowerBI

Configuration:
  All paths and parameters are loaded from .env (see .env.example).
─────────────────────────────────────────────────────────────────────────────
"""

import os
import datetime
import string
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV       = os.getenv("INPUT_CSV",        "data/fallback.csv")
OUTPUT_CSV      = os.getenv("OUTPUT_CSV",        "output/fallback_master_output.csv")
OUTPUT_TOP1_CSV = os.getenv("OUTPUT_TOP1_CSV",   "output/fallback_top1_topic.csv")
MAX_CLUSTERS    = int(os.getenv("MAX_CLUSTERS",  30))
TOP_N_CLUSTERS  = int(os.getenv("TOP_N_CLUSTERS", 20))
TOP_N_WORDS     = int(os.getenv("TOP_N_WORDS",   15))

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Text Processing ───────────────────────────────────────────────────────────
class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        DetectorFactory.seed = 0

    def preprocess_text(self, text):
        """Lowercase, remove punctuation and digits, lemmatize, strip stopwords."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation and not t.isdigit()]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return " ".join(tokens)

    def is_english(self, text):
        """Return True if langdetect identifies the text as English."""
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    def load_and_preprocess_data(self, file_path):
        """Load CSV, filter to English utterances, and apply text preprocessing."""
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")

        df["utterance"] = df["utterance"].astype(str)
        df = df[df["utterance"].apply(self.is_english)].copy()
        df["processed_text"] = df["utterance"].apply(self.preprocess_text)
        return df


# ── Feature Extraction ────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    def extract_features(self, df):
        """Fit TF-IDF on preprocessed text and return the feature matrix."""
        tfidf_matrix = self.vectorizer.fit_transform(df["processed_text"].tolist())
        return tfidf_matrix, self.vectorizer


# ── Clustering ────────────────────────────────────────────────────────────────
class ClusteringModel:
    def __init__(self):
        self.kmeans = None

    def find_optimal_clusters(self, tfidf_matrix, max_clusters=MAX_CLUSTERS):
        """
        Evaluate K from 2 to max_clusters using silhouette score.
        Returns the K with the highest score.
        """
        silhouette_scores = []
        K = range(2, max_clusters)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"  K={k:>3}  silhouette={score:.4f}")

        optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
        print(f"\n  Optimal K: {optimal_k}")
        return optimal_k

    def plot_silhouette_scores(self, tfidf_matrix, max_clusters=MAX_CLUSTERS):
        """Optional: plot silhouette scores across K values."""
        scores = []
        K = range(2, max_clusters)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            scores.append(silhouette_score(tfidf_matrix, kmeans.labels_))

        plt.figure(figsize=(10, 5))
        plt.plot(K, scores, "bx-")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs Number of Clusters")
        plt.tight_layout()
        plt.show()

    def perform_clustering(self, tfidf_matrix, n_clusters):
        """Fit K-Means with the given number of clusters."""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(tfidf_matrix)
        return self.kmeans


# ── Topic Labelling ───────────────────────────────────────────────────────────
class TopicIdentifier:
    def __init__(self, kmeans, vectorizer):
        self.kmeans = kmeans
        self.vectorizer = vectorizer

    def identify_topics(self, df, top_n_words=TOP_N_WORDS):
        """
        Label each cluster using its top TF-IDF centroid terms.
        Appends a representative example utterance.
        """
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()
        topics = []

        for i in range(self.kmeans.n_clusters):
            topic_terms = [terms[ind] for ind in order_centroids[i, :top_n_words]]
            cluster_utterances = df[df["cluster"] == i]["utterance"]
            representative = cluster_utterances.iloc[0] if not cluster_utterances.empty else ""
            label = (
                f"{topic_terms[0]} and {topic_terms[1]}, "
                f"{', '.join(topic_terms[2:5])} — e.g., '{representative}'"
            )
            topics.append(label)

        return topics


# ── Topic Ranking ─────────────────────────────────────────────────────────────
class TopicRanker:
    def __init__(self, kmeans, processed_texts):
        self.kmeans = kmeans
        self.processed_texts = processed_texts

    def rank_topics(self, top_n=TOP_N_CLUSTERS):
        """Return the top N clusters sorted by size (utterance count)."""
        cluster_sizes = Counter(self.kmeans.labels_)
        return cluster_sizes.most_common(top_n)

    def calc_percentage_total(self, top_clusters):
        """Calculate each cluster's share of the total utterance volume."""
        total_rows = len(self.processed_texts)
        return [(cluster, (size / (total_rows - 1)) * 100) for cluster, size in top_clusters]


# ── Export ────────────────────────────────────────────────────────────────────
class ResultExporter:
    def __init__(self, output_path):
        self.output_path = output_path

    def export_results_to_csv(self, df, topics, top_clusters, percen_of_total):
        """
        Append ranked topic results to the master output CSV.
        Each run is timestamped — PowerBI reads the full file and can
        filter or trend by the Timestamp column.
        """
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        data = []
        timestamp = datetime.datetime.now().strftime("%m-%Y")

        for rank, ((cluster_id, count), (_, percentage)) in enumerate(
            zip(top_clusters, percen_of_total), 1
        ):
            examples = df[df["cluster"] == cluster_id].head(15)
            utterances = examples["utterance"].tolist()
            data.append([rank, topics[cluster_id], count, percentage, timestamp] + utterances)

        columns = (
            ["Rank", "Topic", "Count", "Percentage", "Timestamp"]
            + [f"Utterance_{i + 1}" for i in range(15)]
        )
        df_results = pd.DataFrame(data, columns=columns)

        if not os.path.isfile(self.output_path):
            df_results.to_csv(self.output_path, index=False)
        else:
            df_results.to_csv(self.output_path, mode="a", header=False, index=False)

        print(f"Results appended → {self.output_path}")


class ResultExporterTop1:
    def __init__(self, output_path):
        self.output_path = output_path

    def export_top1_topic(self, df, top_clusters):
        """Export all utterances from the #1 ranked cluster to a separate CSV."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        top_cluster_id = top_clusters[0][0]
        top_utterances = df[df["cluster"] == top_cluster_id]["utterance"]
        pd.DataFrame(top_utterances).to_csv(self.output_path, index=False)
        print(f"Top topic utterances exported → {self.output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(file_path):
    print(f"\nLoading data: {file_path}")
    processor = TextProcessor()
    df = processor.load_and_preprocess_data(file_path)
    print(f"  {len(df):,} English utterances loaded.\n")

    print("Extracting TF-IDF features...")
    extractor = FeatureExtractor()
    tfidf_matrix, vectorizer = extractor.extract_features(df)

    print(f"\nFinding optimal cluster count (max K={MAX_CLUSTERS})...")
    cluster_model = ClusteringModel()
    optimal_k = cluster_model.find_optimal_clusters(tfidf_matrix)
    kmeans = cluster_model.perform_clustering(tfidf_matrix, optimal_k)

    df["cluster"] = kmeans.labels_

    print("\nLabelling topics...")
    identifier = TopicIdentifier(kmeans, vectorizer)
    topics = identifier.identify_topics(df, top_n_words=TOP_N_WORDS)

    ranker = TopicRanker(kmeans, df["processed_text"].tolist())
    top_clusters = ranker.rank_topics(top_n=TOP_N_CLUSTERS)
    percen_of_total = ranker.calc_percentage_total(top_clusters)

    exporter = ResultExporter(OUTPUT_CSV)
    exporter.export_results_to_csv(df, topics, top_clusters, percen_of_total)

    # Uncomment to also export top #1 topic utterances separately
    # top1_exporter = ResultExporterTop1(OUTPUT_TOP1_CSV)
    # top1_exporter.export_top1_topic(df, top_clusters)

    print(f"\n── Top {TOP_N_CLUSTERS} Fallback Topics ──────────────────────────────")
    for rank, ((cluster_id, count), (_, percentage)) in enumerate(
        zip(top_clusters, percen_of_total), 1
    ):
        print(f"\n{rank}. {topics[cluster_id]}")
        print(f"   Count: {count}  |  {percentage:.2f}% of total")
        print("   Examples:")
        for utterance in df[df["cluster"] == cluster_id].head(5)["utterance"]:
            print(f"     - {utterance}")


if __name__ == "__main__":
    main(INPUT_CSV)
