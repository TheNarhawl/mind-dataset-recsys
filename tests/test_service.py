import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from mind_recsys.features.pipeline import FEATURE_NAMES
from mind_recsys.features.stats import UserProfile
from mind_recsys.ranking.model import NewsRanker
from mind_recsys.retrieval.qdrant_index import QdrantNewsIndex
from mind_recsys.service.app import create_app
from mind_recsys.service.pipeline import TwoStageRecommender


def build_mock_recommender() -> TwoStageRecommender:
    # 1. Setup in-memory Qdrant
    qdrant = QdrantNewsIndex(collection_name="test_service", vector_size=4, location=":memory:")
    qdrant.create_collection(recreate=True)

    news_ids = ["N1", "N2", "N3"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    payloads = [
        {
            "title": "Article One",
            "category": "sports",
            "subcategory": "soccer",
            "abstract": "Summary 1",
        },
        {"title": "Article Two", "category": "tech", "subcategory": "ai", "abstract": "Summary 2"},
        {
            "title": "Article Three",
            "category": "finance",
            "subcategory": "stocks",
            "abstract": "Summary 3",
        },
    ]
    qdrant.index_news(news_ids, embeddings, payloads)

    # 2. Setup trained mini ranker
    X_train = pd.DataFrame(
        np.random.RandomState(42).randn(12, len(FEATURE_NAMES)),
        columns=FEATURE_NAMES,
    )
    y_train = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    ranker = NewsRanker(iterations=10, depth=2)
    ranker.fit(X_train, y_train, group_ids)

    # 3. User profile and stats
    profiles = {
        "U100": UserProfile(
            user_id="U100",
            hist_len=2,
            category_entropy=1.0,
            avg_headline_len=15.0,
            top_category="sports",
            category_counts={"sports": 2},
            subcategory_counts={"soccer": 2},
            history_tokens={"article", "one"},
        )
    }

    news_embs = {
        "N1": embeddings[0],
        "N2": embeddings[1],
        "N3": embeddings[2],
    }

    return TwoStageRecommender(
        retrieval_index=qdrant,
        ranker=ranker,
        news_embeddings=news_embs,
        user_profiles=profiles,
        candidate_pool_size=3,
    )


def test_two_stage_recommender():
    recommender = build_mock_recommender()
    response = recommender.recommend(user_id="U100", top_k=2)

    assert response.user_id == "U100"
    assert len(response.recommendations) == 2
    assert response.latency.total_ms > 0.0
    assert response.latency.retrieval_ms >= 0.0
    assert response.latency.feature_engineering_ms >= 0.0
    assert response.latency.ranking_ms >= 0.0

    # Ensure ranking items are sorted by score descending
    scores = [r.score for r in response.recommendations]
    assert scores == sorted(scores, reverse=True)


def test_fastapi_endpoints():
    recommender = build_mock_recommender()
    app = create_app(recommender)
    client = TestClient(app)

    # Test root
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["name"] == "mind-recsys"

    # Test health
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "ok"

    # Test recommend endpoint
    payload = {
        "user_id": "U100",
        "top_k": 2,
    }
    resp_rec = client.post("/api/v1/recommend", json=payload)
    assert resp_rec.status_code == 200
    data = resp_rec.json()
    assert data["user_id"] == "U100"
    assert len(data["recommendations"]) == 2
    assert "latency" in data
