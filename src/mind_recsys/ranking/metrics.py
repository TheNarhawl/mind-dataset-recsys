import numpy as np


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    rel = np.asarray(relevance)[:k]
    if rel.size == 0:
        return 0.0
    gains = 2.0**rel - 1.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K for a single query/impression."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.size == 0 or np.sum(y_true) == 0:
        return 0.0

    order = np.argsort(y_score)[::-1]
    sorted_true = y_true[order]

    actual_dcg = dcg_at_k(sorted_true, k)
    ideal_sorted_true = np.sort(y_true)[::-1]
    ideal_dcg = dcg_at_k(ideal_sorted_true, k)

    if ideal_dcg <= 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Reciprocal Rank (RR) for a single query/impression."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.size == 0 or np.sum(y_true) == 0:
        return 0.0

    order = np.argsort(y_score)[::-1]
    sorted_true = y_true[order]

    positive_indices = np.where(sorted_true > 0)[0]
    if positive_indices.size == 0:
        return 0.0
    # 1-based rank
    return float(1.0 / (positive_indices[0] + 1))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision (AP) for a single query/impression."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    total_positive = np.sum(y_true > 0)
    if total_positive == 0:
        return 0.0

    order = np.argsort(y_score)[::-1]
    sorted_true = y_true[order]

    hits = 0.0
    sum_precisions = 0.0
    for i, rel in enumerate(sorted_true):
        if rel > 0:
            hits += 1.0
            sum_precisions += hits / (i + 1.0)

    return float(sum_precisions / total_positive)


def evaluate_ranking_predictions(
    y_trues: list[np.ndarray],
    y_scores: list[np.ndarray],
) -> dict[str, float]:
    """Evaluates NDCG@5, NDCG@10, MRR, and MAP across a collection of impressions."""
    ndcg5_list: list[float] = []
    ndcg10_list: list[float] = []
    mrr_list: list[float] = []
    map_list: list[float] = []

    for yt, ys in zip(y_trues, y_scores, strict=False):
        if np.sum(yt) > 0:
            ndcg5_list.append(ndcg_at_k(yt, ys, k=5))
            ndcg10_list.append(ndcg_at_k(yt, ys, k=10))
            mrr_list.append(reciprocal_rank(yt, ys))
            map_list.append(average_precision(yt, ys))

    return {
        "NDCG@5": round(float(np.mean(ndcg5_list)), 4) if ndcg5_list else 0.0,
        "NDCG@10": round(float(np.mean(ndcg10_list)), 4) if ndcg10_list else 0.0,
        "MRR": round(float(np.mean(mrr_list)), 4) if mrr_list else 0.0,
        "MAP": round(float(np.mean(map_list)), 4) if map_list else 0.0,
    }
