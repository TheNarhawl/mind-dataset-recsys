from typing import Any

import numpy as np

from mind_recsys.features.stats import UserProfile

FEATURE_NAMES = [
    # User Profile Features (1-6)
    "user_hist_len",
    "user_category_entropy",
    "user_avg_headline_len",
    "user_unique_categories",
    "user_unique_subcategories",
    "user_has_history",
    # Item Statistical Features (7-15)
    "item_total_clicks",
    "item_total_impressions",
    "item_global_ctr",
    "item_category_popularity",
    "item_has_entities",
    "item_title_len",
    "item_abstract_len",
    "item_title_word_count",
    "item_abstract_word_count",
    # Interaction & Semantic Cross Features (16-25)
    "cos_sim_user_item",
    "category_match",
    "category_affinity",
    "subcategory_affinity",
    "category_count_in_history",
    "subcategory_count_in_history",
    "lexical_overlap_count",
    "lexical_overlap_ratio",
    "title_contains_user_top_category",
    "item_recency_hours",
    # Retrieval & Comparative Cross Features (26-35)
    "qdrant_retrieval_score",
    "qdrant_retrieval_rank",
    "dot_prod_user_item",
    "diff_title_len_vs_user_avg",
    "ratio_title_len_vs_user_avg",
    "user_interest_concentration",
    "item_is_popular_ctr",
    "item_is_fresh",
    "category_diversity_penalty",
    "combined_semantic_ctr_score",
]

CATEGORICAL_FEATURE_NAMES = ["category", "subcategory"]


def extract_features_single(
    user_profile: UserProfile | None,
    item_metadata: dict[str, Any],
    item_stats: dict[str, Any] | None,
    user_vector: np.ndarray | None,
    item_vector: np.ndarray | None,
    retrieval_score: float = 0.0,
    retrieval_rank: int = 1,
    recency_hours: float = 24.0,
) -> dict[str, float]:
    """Extracts all 35 tabular features for a single (User, Candidate Item) pair."""
    # 1-6 User Features
    if user_profile is not None:
        hist_len = float(user_profile.hist_len)
        cat_entropy = float(user_profile.category_entropy)
        avg_headline_len = float(user_profile.avg_headline_len)
        unique_cats = float(len(user_profile.category_counts))
        unique_subcats = float(len(user_profile.subcategory_counts))
        has_history = 1.0 if hist_len > 0 else 0.0
        top_cat = user_profile.top_category
        cat_counts = user_profile.category_counts
        subcat_counts = user_profile.subcategory_counts
        history_tokens = user_profile.history_tokens
    else:
        hist_len = 0.0
        cat_entropy = 0.0
        avg_headline_len = 0.0
        unique_cats = 0.0
        unique_subcats = 0.0
        has_history = 0.0
        top_cat = ""
        cat_counts = {}
        subcat_counts = {}
        history_tokens = set()

    # 7-15 Item Features
    stats = item_stats or {}
    total_clicks = float(stats.get("item_total_clicks", 0))
    total_imps = float(stats.get("item_total_impressions", 0))
    global_ctr = float(stats.get("item_global_ctr", 0.1))
    cat_pop = float(stats.get("item_category_popularity", 0.05))
    has_entities = float(stats.get("item_has_entities", 0))

    title = str(item_metadata.get("title", ""))
    abstract = str(item_metadata.get("abstract", ""))
    category = str(item_metadata.get("category", ""))
    subcategory = str(item_metadata.get("subcategory", ""))

    title_len = float(len(title))
    abstract_len = float(len(abstract))
    title_words = title.lower().split()
    abstract_words = abstract.lower().split()
    title_word_count = float(len(title_words))
    abstract_word_count = float(len(abstract_words))

    # 16 Semantic Similarity
    if user_vector is not None and item_vector is not None:
        u_norm = np.linalg.norm(user_vector)
        v_norm = np.linalg.norm(item_vector)
        if u_norm > 1e-8 and v_norm > 1e-8:
            cos_sim = float(np.dot(user_vector, item_vector) / (u_norm * v_norm))
            dot_prod = float(np.dot(user_vector, item_vector))
        else:
            cos_sim = 0.0
            dot_prod = 0.0
    else:
        cos_sim = float(retrieval_score)
        dot_prod = float(retrieval_score)

    # 17-24 Cross & Interaction
    cat_match = 1.0 if (top_cat and category and top_cat.lower() == category.lower()) else 0.0
    c_count = float(cat_counts.get(category, 0))
    sc_count = float(subcat_counts.get(subcategory, 0))
    cat_affinity = (c_count / hist_len) if hist_len > 0 else 0.0
    subcat_affinity = (sc_count / hist_len) if hist_len > 0 else 0.0

    title_tokens_set = set(title_words)
    overlap_count = (
        float(len(history_tokens.intersection(title_tokens_set))) if history_tokens else 0.0
    )
    overlap_ratio = (overlap_count / max(len(title_tokens_set), 1)) if title_tokens_set else 0.0
    title_contains_top_cat = 1.0 if (top_cat and top_cat.lower() in title.lower()) else 0.0

    # 25-35 Comparative & Derived
    q_score = float(retrieval_score)
    q_rank = float(retrieval_rank)
    diff_headline_len = abs(title_len - avg_headline_len)
    ratio_headline_len = title_len / (avg_headline_len + 1.0)
    top_cat_count = max(cat_counts.values()) if cat_counts else 0.0
    concentration = (float(top_cat_count) / (hist_len + 1.0)) if hist_len > 0 else 0.0
    is_popular = 1.0 if global_ctr > 0.1 else 0.0
    is_fresh = 1.0 if recency_hours < 24.0 else 0.0
    div_penalty = cat_entropy * (1.0 - cat_affinity)
    comb_score = cos_sim * global_ctr

    return {
        "user_hist_len": hist_len,
        "user_category_entropy": cat_entropy,
        "user_avg_headline_len": avg_headline_len,
        "user_unique_categories": unique_cats,
        "user_unique_subcategories": unique_subcats,
        "user_has_history": has_history,
        "item_total_clicks": total_clicks,
        "item_total_impressions": total_imps,
        "item_global_ctr": global_ctr,
        "item_category_popularity": cat_pop,
        "item_has_entities": has_entities,
        "item_title_len": title_len,
        "item_abstract_len": abstract_len,
        "item_title_word_count": title_word_count,
        "item_abstract_word_count": abstract_word_count,
        "cos_sim_user_item": round(cos_sim, 5),
        "category_match": cat_match,
        "category_affinity": round(cat_affinity, 4),
        "subcategory_affinity": round(subcat_affinity, 4),
        "category_count_in_history": c_count,
        "subcategory_count_in_history": sc_count,
        "lexical_overlap_count": overlap_count,
        "lexical_overlap_ratio": round(overlap_ratio, 4),
        "title_contains_user_top_category": title_contains_top_cat,
        "item_recency_hours": round(recency_hours, 2),
        "qdrant_retrieval_score": round(q_score, 5),
        "qdrant_retrieval_rank": q_rank,
        "dot_prod_user_item": round(dot_prod, 5),
        "diff_title_len_vs_user_avg": round(diff_headline_len, 2),
        "ratio_title_len_vs_user_avg": round(ratio_headline_len, 3),
        "user_interest_concentration": round(concentration, 4),
        "item_is_popular_ctr": is_popular,
        "item_is_fresh": is_fresh,
        "category_diversity_penalty": round(div_penalty, 4),
        "combined_semantic_ctr_score": round(comb_score, 5),
    }
