import numpy as np

from mind_recsys.retrieval.embedder import compute_user_vector
from mind_recsys.retrieval.qdrant_index import QdrantNewsIndex


def test_compute_user_vector_empty():
    vec = compute_user_vector(np.empty((0, 384)))
    assert vec.shape == (384,)
    assert np.allclose(vec, 0.0)


def test_compute_user_vector_single():
    v = np.array([[3.0, 4.0]])
    u = compute_user_vector(v, normalize=True)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.allclose(u, [0.6, 0.8])


def test_compute_user_vector_decay():
    # Item 1 is [1, 0], Item 2 (more recent) is [0, 1]
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    # With lambda = 0.5: Item 1 has weight 0.5, Item 2 has weight 1.0
    u = compute_user_vector(v, decay_lambda=0.5, normalize=False)
    # Expected weighted sum: (0.5 * [1, 0] + 1.0 * [0, 1]) / 1.5 = [0.5/1.5, 1.0/1.5] = [1/3, 2/3]
    assert np.isclose(u[0], 1.0 / 3.0)
    assert np.isclose(u[1], 2.0 / 3.0)


def test_qdrant_index_in_memory():
    index = QdrantNewsIndex(collection_name="test_news", vector_size=4, location=":memory:")
    index.create_collection(recreate=True)

    # 3 news items
    news_ids = ["N1", "N2", "N3"]
    # Unit vectors
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    payloads = [
        {"title": "Tech News", "category": "technology"},
        {"title": "Sports News", "category": "sports"},
        {"title": "Finance News", "category": "finance"},
    ]

    index.index_news(news_ids, embeddings, payloads)

    # Search for technology vector
    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = index.search(query_vec, limit=2)

    assert len(results) == 2
    assert results[0]["news_id"] == "N1"
    assert results[0]["title"] == "Tech News"
    assert results[0]["score"] > 0.8


def test_qdrant_evaluate_recall():
    index = QdrantNewsIndex(collection_name="test_recall", vector_size=2, location=":memory:")
    news_ids = ["N1", "N2"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.index_news(news_ids, embeddings)

    # User 1 is interested in N1
    user_queries = [
        ("imp_1", np.array([1.0, 0.0], dtype=np.float32), {"N1"}),
        ("imp_2", np.array([0.0, 1.0], dtype=np.float32), {"N2"}),
    ]

    metrics = index.evaluate_recall(user_queries, top_k_list=[1, 2])
    assert metrics["Recall@1"] == 1.0
    assert metrics["Recall@2"] == 1.0
    assert "avg_latency_ms" in metrics
