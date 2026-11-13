import time
from typing import Any

import numpy as np
import pandas as pd

from mind_recsys.features.pipeline import FEATURE_NAMES, extract_features_single
from mind_recsys.features.stats import UserProfile
from mind_recsys.ranking.model import NewsRanker
from mind_recsys.retrieval.embedder import compute_user_vector
from mind_recsys.retrieval.qdrant_index import QdrantNewsIndex
from mind_recsys.service.schemas import LatencyBreakdown, RecommendationItem, RecommendResponse


class TwoStageRecommender:
    """Production Two-Stage Recommender coordinating Qdrant (Retrieval) and CatBoost (Ranking)."""

    def __init__(
        self,
        retrieval_index: QdrantNewsIndex,
        ranker: NewsRanker,
        news_metadata: dict[str, dict[str, Any]] | None = None,
        news_embeddings: dict[str, np.ndarray] | None = None,
        user_profiles: dict[str, UserProfile] | None = None,
        item_stats: dict[str, dict[str, Any]] | None = None,
        candidate_pool_size: int = 100,
        decay_lambda: float = 0.85,
    ) -> None:
        self.retrieval_index = retrieval_index
        self.ranker = ranker
        self.news_metadata = news_metadata or {}
        self.news_embeddings = news_embeddings or {}
        self.user_profiles = user_profiles or {}
        self.item_stats = item_stats or {}
        self.candidate_pool_size = candidate_pool_size
        self.decay_lambda = decay_lambda

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        history_override: list[str] | None = None,
    ) -> RecommendResponse:
        t_start = time.perf_counter()

        # 1. Resolve User History & Profile
        profile = self.user_profiles.get(user_id)
        if history_override is not None:
            history_nids = history_override
        elif profile is not None:
            history_nids = list(profile.history_tokens)  # or original history
        else:
            history_nids = []

        # 2. Stage 1: Retrieval (Qdrant)
        t_ret_start = time.perf_counter()
        if history_nids and self.news_embeddings:
            history_vecs = [
                self.news_embeddings[nid] for nid in history_nids if nid in self.news_embeddings
            ]
            if history_vecs:
                user_vec = compute_user_vector(
                    np.array(history_vecs),
                    decay_lambda=self.decay_lambda,
                )
            else:
                user_vec = np.zeros(self.retrieval_index.vector_size, dtype=np.float32)
        else:
            user_vec = np.zeros(self.retrieval_index.vector_size, dtype=np.float32)

        candidates = self.retrieval_index.search(user_vec, limit=self.candidate_pool_size)
        t_ret_end = time.perf_counter()
        retrieval_ms = (t_ret_end - t_ret_start) * 1000.0

        if not candidates:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return RecommendResponse(
                user_id=user_id,
                recommendations=[],
                latency=LatencyBreakdown(
                    retrieval_ms=round(retrieval_ms, 2),
                    feature_engineering_ms=0.0,
                    ranking_ms=0.0,
                    total_ms=round(total_ms, 2),
                ),
            )

        # 3. Stage 2: Feature Engineering
        t_feat_start = time.perf_counter()
        features_rows: list[dict[str, float]] = []
        enriched_candidates: list[dict[str, Any]] = []

        for rank_idx, cand in enumerate(candidates, start=1):
            nid = cand.get("news_id", "")
            meta = self.news_metadata.get(nid, {})
            # merge payload if metadata not in dictionary
            if not meta:
                meta = {
                    "news_id": nid,
                    "title": cand.get("title", ""),
                    "category": cand.get("category", "General"),
                    "subcategory": cand.get("subcategory", ""),
                    "abstract": cand.get("abstract", ""),
                }

            stats = self.item_stats.get(nid)
            item_vec = self.news_embeddings.get(nid)
            q_score = float(cand.get("score", 0.0))

            feat_dict = extract_features_single(
                user_profile=profile,
                item_metadata=meta,
                item_stats=stats,
                user_vector=user_vec,
                item_vector=item_vec,
                retrieval_score=q_score,
                retrieval_rank=rank_idx,
            )
            features_rows.append(feat_dict)
            enriched_candidates.append(
                {
                    "news_id": nid,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", "General"),
                    "subcategory": meta.get("subcategory", ""),
                }
            )

        features_df = pd.DataFrame(features_rows)[FEATURE_NAMES]
        t_feat_end = time.perf_counter()
        feat_ms = (t_feat_end - t_feat_start) * 1000.0

        # 4. Stage 2: Ranking (CatBoost)
        t_rank_start = time.perf_counter()
        scored_candidates = self.ranker.rank_candidates(
            features_df=features_df,
            candidates=enriched_candidates,
            top_k=top_k,
        )
        t_rank_end = time.perf_counter()
        ranking_ms = (t_rank_end - t_rank_start) * 1000.0

        # 5. Format Response
        items: list[RecommendationItem] = []
        for rank, item in enumerate(scored_candidates, start=1):
            items.append(
                RecommendationItem(
                    news_id=item["news_id"],
                    score=round(float(item["ranker_score"]), 4),
                    rank=rank,
                    category=item.get("category", "General"),
                    subcategory=item.get("subcategory", ""),
                    title=item.get("title", ""),
                )
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return RecommendResponse(
            user_id=user_id,
            recommendations=items,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                feature_engineering_ms=round(feat_ms, 2),
                ranking_ms=round(ranking_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )
