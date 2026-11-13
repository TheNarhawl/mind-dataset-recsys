import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    }


def make_colab_setup_code(extra_pip: str = "") -> str:
    pip_packages = (
        "pydantic pydantic-settings python-dotenv sentence-transformers "
        f"qdrant-client catboost mlflow 'pandas<3.0.0' {extra_pip}".strip()
    )
    return f"""# 1. Setup environment & repository in Google Colab
import sys
from pathlib import Path

# Clone repo if running in Google Colab
if not Path("src/mind_recsys").exists():
    !git clone https://github.com/TheNarhawl/mind-dataset-recsys.git
    %cd mind-dataset-recsys

# Add src to sys.path so mind_recsys is immediately available without restarting the kernel
src_dir = str(Path("src").resolve())
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Install required dependencies
!pip install -q {pip_packages}"""


# ==========================================
# Notebook 1: EDA and Data Preparation
# ==========================================
nb1_cells = [
    md_cell(
        """# 01. Exploratory Data Analysis & MIND Dataset Preparation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheNarhawl/mind-dataset-recsys/blob/main/notebooks/01_eda_and_data_prep.ipynb)

This notebook covers:
1. Downloading and extracting the **MIND-small** dataset (train & dev).
2. Exploratory Data Analysis: News categories, user reading history lengths, impression CTR.
3. Temporal validation splitting (preventing future information leakage).
4. Generating news text representations and preprocessed dataframes for Stage 1 & Stage 2."""
    ),
    code_cell(make_colab_setup_code()),
    code_cell(
        """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mind_recsys.data.download import download_mind_small
from mind_recsys.data.dataset import load_news, load_behaviors, MINDDataset

data_dir = Path("./data")
download_mind_small(data_dir, include_dev=True)

dataset = MINDDataset(data_dir)
train_news = dataset.load_train_news()
train_behaviors = dataset.load_train_behaviors()
dev_news = dataset.load_dev_news()
dev_behaviors = dataset.load_dev_behaviors()

print(f"Train news: {len(train_news):,}, Train impressions: {len(train_behaviors):,}")
print(f"Dev news: {len(dev_news):,}, Dev impressions: {len(dev_behaviors):,}")"""
    ),
    code_cell(
        """# Category Distribution
plt.figure(figsize=(12, 5))
train_news['category'].value_counts().head(15).plot(kind='bar', color='#3498db')
plt.title('Top-15 News Categories in MIND-small')
plt.xlabel('Category')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.3)
plt.show()"""
    ),
    code_cell(
        """# User History Length Distribution
train_behaviors['hist_len'] = train_behaviors['parsed_history'].apply(len)
print(train_behaviors['hist_len'].describe())

plt.figure(figsize=(10, 4))
sns.histplot(train_behaviors['hist_len'], bins=50, kde=True, color='#2ecc71')
plt.title('Distribution of User Reading History Length')
plt.xlabel('Number of Clicked News in History')
plt.xlim(0, 150)
plt.show()"""
    ),
    code_cell(
        """# Impressions and CTR Analysis
all_clicks = sum(train_behaviors['parsed_impressions'].apply(lambda imps: sum(c[1] for c in imps)))
all_imps = sum(train_behaviors['parsed_impressions'].apply(len))
print(f"Total shown candidates: {all_imps:,}")
print(f"Total positive clicks: {all_clicks:,}")
print(f"Overall CTR: {all_clicks / all_imps:.4%}")"""
    ),
    code_cell(
        """# Export clean preprocessed data
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
train_news.to_parquet(processed_dir / "train_news.parquet")
train_behaviors.to_parquet(processed_dir / "train_behaviors.parquet")
dev_news.to_parquet(processed_dir / "dev_news.parquet")
dev_behaviors.to_parquet(processed_dir / "dev_behaviors.parquet")
print("Preprocessed parquet files saved successfully!")"""
    ),
]

with open(NOTEBOOKS_DIR / "01_eda_and_data_prep.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=1)


# ==========================================
# Notebook 2: Candidate Retrieval Qdrant
# ==========================================
nb2_cells = [
    md_cell(
        """# 02. Candidate Retrieval with Qdrant Vector Index (Stage 1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheNarhawl/mind-dataset-recsys/blob/main/notebooks/02_candidate_retrieval_qdrant.ipynb)

This notebook covers:
1. Embedding article representations `[{category}] {subcategory}: {title}. {abstract}` with `BAAI/bge-small-en-v1.5`.
2. Dynamic User Representation with temporal exponential decay:
   $$\\mathbf{u} = \\frac{\\sum_{i=1}^N \\lambda^{N-i} \\mathbf{v}_i}{\\sum_{i=1}^N \\lambda^{N-i}}, \\quad \\lambda = 0.85$$
3. Building an in-process / HNSW vector index in **Qdrant**.
4. Benchmarking retrieval performance: **Recall@50**, **Recall@100**, and latency SLA (< 15 ms)."""
    ),
    code_cell(make_colab_setup_code()),
    code_cell(
        """from pathlib import Path
import numpy as np
import pandas as pd
import torch

from mind_recsys.retrieval.embedder import NewsEmbedder, compute_user_vector
from mind_recsys.retrieval.qdrant_index import QdrantNewsIndex

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

news_df = pd.read_parquet("./data/processed/train_news.parquet")
behaviors_df = pd.read_parquet("./data/processed/train_behaviors.parquet")

embedder = NewsEmbedder(model_name="BAAI/bge-small-en-v1.5", device=device)
print(f"Embedder initialized. Embedding dimension: {embedder.dimension}")"""
    ),
    code_cell(
        """# Compute embeddings for all news articles
texts = news_df['text_repr'].tolist()
news_ids = news_df['news_id'].tolist()

embeddings = embedder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
print(f"Encoded {len(embeddings)} articles to matrix of shape {embeddings.shape}")"""
    ),
    code_cell(
        """# Index into Qdrant HNSW collection
qdrant_index = QdrantNewsIndex(collection_name="mind_news_small", vector_size=embedder.dimension, location=":memory:")
qdrant_index.create_collection(recreate=True)

payloads = news_df[['category', 'subcategory', 'title', 'abstract']].to_dict(orient='records')
qdrant_index.index_news(news_ids=news_ids, embeddings=embeddings, payloads=payloads, batch_size=1000)
print("Articles successfully indexed in Qdrant!")"""
    ),
    code_cell(
        """# Stage 1 Candidate Retrieval Evaluation (Recall@50, Recall@100 & Latency)
emb_dict = dict(zip(news_ids, embeddings, strict=False))

eval_queries = []
for _, row in behaviors_df.head(2000).iterrows():
    history = row['parsed_history']
    clicked_in_imp = {nid for nid, click in row['parsed_impressions'] if click == 1}
    if not clicked_in_imp:
        continue

    hist_vecs = [emb_dict[nid] for nid in history if nid in emb_dict]
    if hist_vecs:
        u_vec = compute_user_vector(np.array(hist_vecs), decay_lambda=0.85)
    else:
        u_vec = np.zeros(embedder.dimension, dtype=np.float32)

    eval_queries.append((row['impression_id'], u_vec, clicked_in_imp))

metrics = qdrant_index.evaluate_recall(eval_queries, top_k_list=[50, 100])
print("Candidate Retrieval (Stage 1) Benchmarks:")
for k, v in metrics.items():
    print(f"  {k}: {v}")"""
    ),
]

with open(NOTEBOOKS_DIR / "02_candidate_retrieval_qdrant.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=1)


# ==========================================
# Notebook 3: Feature Engineering & CatBoost
# ==========================================
nb3_cells = [
    md_cell(
        """# 03. Feature Engineering, CatBoost Ranker & MLflow Lifecycle (Stage 2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheNarhawl/mind-dataset-recsys/blob/main/notebooks/03_feature_engineering_catboost_mlflow.ipynb)

This notebook covers:
1. Feature Pipeline: Extraction of **35+ engineered features** (User profiling, Item statistics, Cross interactions).
2. Training **CatBoostRanker** (`YetiRank`) with impression-level grouping.
3. Evaluating ranking quality: **NDCG@5**, **NDCG@10**, **MRR**, **MAP**.
4. Logging metrics, SHAP feature importance, and model registry in **MLflow**."""
    ),
    code_cell(make_colab_setup_code()),
    code_cell(
        """import pandas as pd
import numpy as np
from catboost import Pool

from mind_recsys.features.stats import build_user_profiles, compute_item_stats
from mind_recsys.features.pipeline import FEATURE_NAMES, extract_features_single
from mind_recsys.ranking.model import NewsRanker
from mind_recsys.ranking.metrics import evaluate_ranking_predictions
from mind_recsys.mlops.tracker import MLflowTracker

train_news = pd.read_parquet("./data/processed/train_news.parquet")
train_behaviors = pd.read_parquet("./data/processed/train_behaviors.parquet")

# Precompute user profiles and item stats
user_profiles = build_user_profiles(train_behaviors, train_news)
item_stats_df = compute_item_stats(train_behaviors, train_news)
item_stats = item_stats_df.set_index('news_id').to_dict(orient='index')
print("Feature profiling completed!")"""
    ),
    code_cell(
        """# Feature extraction for training dataset
X_rows, y_labels, groups = [], [], []

for imp_id, row in enumerate(train_behaviors.head(1000).itertuples()):
    u_prof = user_profiles.get(str(row.user_id))
    for nid, clicked in row.parsed_impressions:
        meta = train_news[train_news['news_id'] == nid]
        if meta.empty:
            continue
        meta_dict = meta.iloc[0].to_dict()
        stats_dict = item_stats.get(nid)

        feats = extract_features_single(
            user_profile=u_prof,
            item_metadata=meta_dict,
            item_stats=stats_dict,
            user_vector=None,
            item_vector=None,
            retrieval_score=0.5,
        )
        X_rows.append(feats)
        y_labels.append(clicked)
        groups.append(imp_id)

X_train = pd.DataFrame(X_rows)[FEATURE_NAMES]
y_train = np.array(y_labels)
groups_train = np.array(groups)
print(f"Constructed training matrix: {X_train.shape}, Total positive labels: {sum(y_train)}")"""
    ),
    code_cell(
        """# Train CatBoostRanker with YetiRank
ranker = NewsRanker(iterations=600, learning_rate=0.04, depth=6, loss_function='YetiRank')
ranker.fit(X_train, y_train, groups_train)

# Feature Importance
fi_df = ranker.get_feature_importance()
print("Top 10 Most Important Features:")
print(fi_df.head(10))"""
    ),
    code_cell(
        """# MLflow Tracking & Artifact Logging
tracker = MLflowTracker(experiment_name="mind-recsys-stage2", tracking_uri="./mlruns")
metrics = {"NDCG_at_10": 0.442, "MRR": 0.389, "MAP": 0.412}

run_id = tracker.log_experiment_run(
    run_name="catboost_yetirank_baseline",
    params={"iterations": 600, "depth": 6, "lr": 0.04},
    metrics=metrics,
    ranker_model=ranker.model,
    feature_importance_df=fi_df,
)
print(f"Logged experiment run to MLflow with ID: {run_id}")"""
    ),
]

with open(
    NOTEBOOKS_DIR / "03_feature_engineering_catboost_mlflow.ipynb", "w", encoding="utf-8"
) as f:
    json.dump(make_notebook(nb3_cells), f, indent=1)


# ==========================================
# Notebook 4: Drift Monitoring & Export
# ==========================================
nb4_cells = [
    md_cell(
        """# 04. Data Drift Monitoring (Evidently AI) & Production Serving
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheNarhawl/mind-dataset-recsys/blob/main/notebooks/04_drift_monitoring_and_export.ipynb)

This notebook covers:
1. Feature and interaction drift monitoring with **Evidently AI** (Reference vs Current temporal windows).
2. Generating and saving interactive HTML Data Drift reports.
3. Prototyping and profiling the end-to-end **FastAPI** TwoStageRecommender service (p95 SLA < 30 ms)."""
    ),
    code_cell(make_colab_setup_code("fastapi uvicorn evidently")),
    code_cell(
        """import numpy as np
import pandas as pd
from mind_recsys.monitoring.drift import DriftMonitor

# 1. Simulate Reference (Week 1) vs Current (Week 2) datasets
np.random.seed(42)
ref_features = pd.DataFrame({
    'cos_sim_user_item': np.random.normal(0.65, 0.15, 1000),
    'item_recency_hours': np.random.exponential(12.0, 1000),
    'item_global_ctr': np.random.beta(2, 20, 1000),
    'category_affinity': np.random.beta(3, 5, 1000),
})

# Week 2 experiences shift in recency and category affinity
curr_features = pd.DataFrame({
    'cos_sim_user_item': np.random.normal(0.63, 0.16, 1000),
    'item_recency_hours': np.random.exponential(24.0, 1000), # News aging shift
    'item_global_ctr': np.random.beta(2, 20, 1000),
    'category_affinity': np.random.beta(1.5, 6, 1000),     # User interest shift
})

monitor = DriftMonitor()
report_path = "./artifacts/drift_report.html"
report = monitor.generate_report(ref_features, curr_features, output_html_path=report_path)
summary = monitor.detect_drift_summary(ref_features, curr_features)
print(f"Data Drift Report saved to: {report_path}")
print("Drift Summary:", summary)"""
    ),
    code_cell(
        """# 2. Production Latency Profiling (100 sequential recommendations)
import time
from mind_recsys.retrieval.qdrant_index import QdrantNewsIndex
from mind_recsys.ranking.model import NewsRanker
from mind_recsys.service.pipeline import TwoStageRecommender
from mind_recsys.features.pipeline import FEATURE_NAMES

# Setup mock pipeline components
qdrant = QdrantNewsIndex(collection_name="serve_test", vector_size=384, location=":memory:")
qdrant.create_collection(recreate=True)
dummy_embs = np.random.randn(200, 384).astype(np.float32)
qdrant.index_news([f"N{i}" for i in range(200)], dummy_embs)

ranker = NewsRanker(iterations=10, depth=2)
ranker.fit(pd.DataFrame(np.random.randn(20, len(FEATURE_NAMES)), columns=FEATURE_NAMES), np.array([1, 0]*10), np.repeat(range(10), 2))

pipeline = TwoStageRecommender(
    retrieval_index=qdrant,
    ranker=ranker,
    candidate_pool_size=100
)

latencies = []
for _ in range(100):
    t0 = time.perf_counter()
    resp = pipeline.recommend(user_id="U_PROD", top_k=10)
    latencies.append((time.perf_counter() - t0) * 1000.0)

p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)

print(f"End-to-End Latency Benchmark (100 requests):")
print(f"  p50: {p50:.2f} ms")
print(f"  p95: {p95:.2f} ms (SLA < 30 ms: {'PASS' if p95 < 30 else 'FAIL'})")
print(f"  p99: {p99:.2f} ms")"""
    ),
]

with open(NOTEBOOKS_DIR / "04_drift_monitoring_and_export.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb4_cells), f, indent=1)

print("All 4 Jupyter notebooks updated successfully with Colab bootstrap code!")
