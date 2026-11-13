import time
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class QdrantNewsIndex:
    """Manages vector search collection in Qdrant for Stage 1 candidate retrieval."""

    def __init__(
        self,
        collection_name: str = "mind_news",
        vector_size: int = 384,
        client: QdrantClient | None = None,
        location: str | None = ":memory:",
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size

        if client is not None:
            self.client = client
        elif host is not None and port is not None:
            self.client = QdrantClient(host=host, port=port)
        elif location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=location)

    def create_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance.COSINE,
                ),
                hnsw_config=rest.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=1000,
                ),
            )

    def index_news(
        self,
        news_ids: list[str],
        embeddings: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
        batch_size: int = 500,
    ) -> None:
        """Batch upsert embeddings and news metadata into Qdrant collection."""
        self.create_collection(recreate=False)
        total = len(news_ids)

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_ids = list(range(start_idx, end_idx))
            batch_vectors = embeddings[start_idx:end_idx].tolist()

            batch_payloads: list[dict[str, Any]] = []
            for i in range(start_idx, end_idx):
                p = {"news_id": news_ids[i]}
                if payloads and i < len(payloads):
                    p.update(payloads[i])
                batch_payloads.append(p)

            self.client.upsert(
                collection_name=self.collection_name,
                points=rest.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads,
                ),
            )

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search top-K most similar articles for a query (user) vector."""
        vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=limit,
            with_payload=True,
        ).points

        results: list[dict[str, Any]] = []
        for point in points:
            item = dict(point.payload or {})
            item["score"] = float(point.score)
            results.append(item)
        return results

    def batch_search(
        self,
        query_vectors: list[np.ndarray],
        limit: int = 100,
    ) -> list[list[dict[str, Any]]]:
        """Perform batch search for multiple query vectors."""
        results: list[list[dict[str, Any]]] = []
        for qv in query_vectors:
            results.append(self.search(qv, limit=limit))
        return results

    def evaluate_recall(
        self,
        user_queries: list[tuple[str, np.ndarray, set[str]]],
        top_k_list: list[int] | None = None,
    ) -> dict[str, float]:
        """Calculates Recall@K and measures average search latency.

        Args:
            user_queries: List of (impression_id, user_vector, set_of_clicked_news_ids).
            top_k_list: List of K values to evaluate, e.g. [50, 100].

        Returns:
            Dictionary with Recall@K metrics and latency in milliseconds.
        """
        if top_k_list is None:
            top_k_list = [50, 100]

        max_k = max(top_k_list)
        hits_at_k: dict[int, int] = {k: 0 for k in top_k_list}
        total_clicks: int = 0
        total_time_ms: float = 0.0
        n_queries = 0

        for _imp_id, u_vec, true_clicks in user_queries:
            if not true_clicks:
                continue

            t0 = time.perf_counter()
            retrieved = self.search(u_vec, limit=max_k)
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0

            retrieved_ids = [item["news_id"] for item in retrieved]
            for k in top_k_list:
                k_candidates = set(retrieved_ids[:k])
                hits_at_k[k] += len(true_clicks.intersection(k_candidates))

            total_clicks += len(true_clicks)
            n_queries += 1

        metrics: dict[str, float] = {}
        for k in top_k_list:
            recall = (hits_at_k[k] / total_clicks) if total_clicks > 0 else 0.0
            metrics[f"Recall@{k}"] = round(recall, 4)

        avg_latency = (total_time_ms / n_queries) if n_queries > 0 else 0.0
        metrics["avg_latency_ms"] = round(avg_latency, 2)
        return metrics
