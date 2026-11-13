from pathlib import Path

import numpy as np
import pandas as pd

from mind_recsys.ranking.metrics import (
    average_precision,
    evaluate_ranking_predictions,
    ndcg_at_k,
    reciprocal_rank,
)
from mind_recsys.ranking.model import NewsRanker


def test_ndcg_at_k_perfect():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.4, 0.2, 0.1])
    assert np.isclose(ndcg_at_k(y_true, y_score, k=5), 1.0)


def test_ndcg_at_k_imperfect():
    y_true = np.array([1, 0, 0, 0])
    # Relevant item ranked last
    y_score = np.array([0.1, 0.4, 0.7, 0.9])
    assert ndcg_at_k(y_true, y_score, k=5) < 1.0


def test_reciprocal_rank():
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.1])  # relevant item at rank 2
    assert np.isclose(reciprocal_rank(y_true, y_score), 0.5)


def test_average_precision():
    y_true = np.array([1, 0, 1])
    # Scores: item 0 (rel=1) is 0.9, item 1 (rel=0) is 0.8, item 2 (rel=1) is 0.5
    # Ranks: rel, non-rel, rel -> precisions: 1/1, 2/3 -> AP = (1 + 2/3) / 2 = 5/6
    y_score = np.array([0.9, 0.8, 0.5])
    assert np.isclose(average_precision(y_true, y_score), (1.0 + 2.0 / 3.0) / 2.0)


def test_evaluate_ranking_predictions():
    y_trues = [np.array([1, 0]), np.array([0, 1])]
    y_scores = [np.array([0.8, 0.2]), np.array([0.1, 0.9])]
    metrics = evaluate_ranking_predictions(y_trues, y_scores)
    assert metrics["NDCG@5"] == 1.0
    assert metrics["NDCG@10"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["MAP"] == 1.0


def test_news_ranker_fit_and_save(tmp_path: Path):
    # Create synthetic ranking dataset with 3 groups
    X = pd.DataFrame(
        {
            "f1": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "f2": [10.0, 50.0, 20.0, 60.0, 15.0, 75.0],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])

    ranker = NewsRanker(iterations=10, depth=2)
    ranker.fit(X, y, groups)

    preds = ranker.predict(X)
    assert len(preds) == 6

    # Test candidate ranking
    candidates = [{"news_id": "A"}, {"news_id": "B"}]
    ranked = ranker.rank_candidates(X.iloc[:2], candidates, top_k=2)
    assert len(ranked) == 2
    assert "ranker_score" in ranked[0]
    assert ranked[0]["ranker_score"] >= ranked[1]["ranker_score"]

    # Test feature importance
    fi = ranker.get_feature_importance()
    assert len(fi) == 2
    assert set(fi["feature"]) == {"f1", "f2"}

    # Test save and load
    model_file = tmp_path / "model.cbm"
    ranker.save_model(model_file)
    assert model_file.exists()

    new_ranker = NewsRanker()
    new_ranker.load_model(model_file)
    preds_after = new_ranker.predict(X)
    assert np.allclose(preds, preds_after)
