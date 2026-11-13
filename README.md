# Two-Stage News Recommender System (MIND Dataset)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green.svg)](https://docs.astral.sh/uv/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade, two-stage news recommendation system built on the **Microsoft News Dataset (MIND)**. The project bridges modern vector search and classic tabular Learning-to-Rank, complete with MLOps tracking, data drift monitoring, and low-latency API serving.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Data ["1. Data & Vectorization (MIND-small)"]
        Raw["MIND: News + Impressions"] --> Enc["BGE Embeddings (bge-small-en-v1.5)"]
        Enc --> Qdrant["Qdrant Vector Index (HNSW)"]
    end

    subgraph Stage1 ["2. Stage 1: Candidate Generation (Retrieval)"]
        UserHist["User Click History"] --> UserVec["User Representation (decay-weighted)"]
        UserVec --> QdrantKNN["Qdrant KNN / ANN Search"]
        QdrantKNN --> TopK["Top-100 Candidates (< 10 ms)"]
    end

    subgraph Stage2 ["3. Stage 2: Feature Engineering & Ranking"]
        TopK --> FeatEng["Feature Pipeline (35+ features)"]
        UserStats["User Profiling (entropy, CTR)"] --> FeatEng
        NewsStats["Item Metadata (recency, popularity)"] --> FeatEng
        FeatEng --> CatBoost["CatBoostRanker (YetiRank)"]
        CatBoost --> Ranked["Final Ranked Top-10"]
    end

    subgraph MLOps ["4. MLOps & Production"]
        CatBoost --> MLflow["MLflow: Metrics, Artifacts, Model Registry"]
        TopK --> Evidently["Evidently AI: Data Drift Monitoring"]
        Ranked --> FastAPIServe["FastAPI Endpoint (/recommend)"]
    end
```

---

## 🎯 Key Stages & Capabilities

1. **Candidate Retrieval (Stage 1):**
   - Dense semantic retrieval with `BAAI/bge-small-en-v1.5` (384d).
   - Real-time user interest vector with temporal exponential decay ($\lambda = 0.85$).
   - HNSW indexing in **Qdrant** with Recall@100 evaluation and sub-10ms retrieval.

2. **Feature Engineering & Ranking (Stage 2):**
   - 35+ engineered features: User profile, Item popularity/recency, and Cross-features (lexical overlap, category affinity, cosine similarity).
   - Learning-to-Rank with **CatBoostRanker** (`YetiRank`) grouped by impression.
   - Comprehensive ranking evaluation: **NDCG@5**, **NDCG@10**, **MRR**, **MAP**.

3. **MLOps & Governance:**
   - Experiment tracking, hyperparameter logging, and Model Registry with **MLflow** / **DagsHub**.
   - SHAP summary plots and feature importance analysis.

4. **Data Quality & Drift Detection:**
   - Feature distribution monitoring with **Evidently AI** (Wasserstein distance, Kolmogorov-Smirnov, Jensen-Shannon divergence).
   - Automated HTML drift reports for sliding temporal windows.

5. **Production Serving:**
   - Asynchronous **FastAPI** service (`POST /recommend`).
   - Latency budget: $p95 < 30 \text{ ms}$ on CPU.

---

## 📁 Repository Structure

```
mind-dataset-recsys/
├── .env.example              # Environment variables template
├── pyproject.toml            # Modern project configuration (uv, ruff, pytest)
├── src/
│   └── mind_recsys/
│       ├── __init__.py
│       ├── config.py         # Pydantic Settings & environment variables
│       ├── data/             # MIND dataset loading & pre-processing
│       ├── retrieval/        # Stage 1: Embeddings & Qdrant vector index
│       ├── features/         # Stage 2: 35+ engineered features
│       ├── ranking/          # Stage 2: CatBoostRanker & evaluation metrics
│       ├── monitoring/       # Stage 4: Evidently AI drift monitoring
│       └── service/          # Stage 5: FastAPI microservice
├── notebooks/                # Modular Colab / Jupyter notebooks
│   ├── 01_eda_and_data_prep.ipynb
│   ├── 02_candidate_retrieval_qdrant.ipynb
│   ├── 03_feature_engineering_catboost_mlflow.ipynb
│   └── 04_drift_monitoring_and_export.ipynb
└── tests/                    # Unit & integration tests
```

---

## 🚀 Quickstart

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
# Clone the repository
git clone https://github.com/TheNarhawl/mind-dataset-recsys.git
cd mind-dataset-recsys

# Install dependencies with uv
uv sync --all-groups

# Configure environment variables
cp .env.example .env
# Edit .env and set your HF_TOKEN and optional MLflow/Qdrant credentials
```

### Running Tests & Quality Checks

```bash
# Run test suite
uv run pytest

# Lint and format checks
uv run ruff check .
uv run ruff format --check .
```

