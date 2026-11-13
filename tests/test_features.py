import numpy as np
import pandas as pd

from mind_recsys.features.pipeline import (
    FEATURE_NAMES,
    extract_features_single,
)
from mind_recsys.features.stats import (
    UserProfile,
    build_user_profiles,
    compute_entropy,
    compute_item_stats,
)


def test_compute_entropy():
    # Single class has 0 entropy
    assert compute_entropy([10]) == 0.0
    # Two equal classes has 1.0 entropy
    assert np.isclose(compute_entropy([5, 5]), 1.0)
    # Empty
    assert compute_entropy([]) == 0.0


def test_build_user_profiles():
    news_df = pd.DataFrame(
        [
            {
                "news_id": "N1",
                "category": "sports",
                "subcategory": "football",
                "title": "Match Day",
            },
            {
                "news_id": "N2",
                "category": "sports",
                "subcategory": "tennis",
                "title": "Grand Slam",
            },
            {
                "news_id": "N3",
                "category": "tech",
                "subcategory": "ai",
                "title": "New AI Model",
            },
        ]
    )

    behaviors_df = pd.DataFrame(
        [
            {
                "user_id": "U10",
                "parsed_history": ["N1", "N2", "N3"],
            }
        ]
    )

    profiles = build_user_profiles(behaviors_df, news_df)
    assert "U10" in profiles
    prof = profiles["U10"]
    assert prof.hist_len == 3
    assert prof.top_category == "sports"
    assert prof.category_counts["sports"] == 2
    assert prof.category_counts["tech"] == 1
    assert "match" in prof.history_tokens
    assert prof.category_entropy > 0.0


def test_compute_item_stats():
    news_df = pd.DataFrame(
        [
            {
                "news_id": "N1",
                "category": "sports",
                "title": "News 1",
                "abstract": "Abstract 1",
                "title_entities": "[]",
                "abstract_entities": "[]",
            },
            {
                "news_id": "N2",
                "category": "news",
                "title": "News 2",
                "abstract": "Abstract 2",
                "title_entities": '[{"Label": "Test"}]',
                "abstract_entities": "[]",
            },
        ]
    )

    behaviors_df = pd.DataFrame(
        [
            {
                "parsed_impressions": [("N1", 1), ("N2", 0)],
            },
            {
                "parsed_impressions": [("N1", 1), ("N2", 1)],
            },
        ]
    )

    stats_df = compute_item_stats(behaviors_df, news_df)
    assert len(stats_df) == 2
    row_n1 = stats_df[stats_df["news_id"] == "N1"].iloc[0]
    assert row_n1["item_total_clicks"] == 2
    assert row_n1["item_total_impressions"] == 2
    # Bayesian CTR: (2 + 10) / (2 + 100) = 12 / 102 ~ 0.11765
    assert np.isclose(row_n1["item_global_ctr"], 12.0 / 102.0, atol=1e-4)

    row_n2 = stats_df[stats_df["news_id"] == "N2"].iloc[0]
    assert row_n2["item_has_entities"] == 1


def test_extract_features_single_completeness():
    profile = UserProfile(
        user_id="U1",
        hist_len=4,
        category_entropy=1.2,
        avg_headline_len=30.0,
        top_category="sports",
        category_counts={"sports": 3, "news": 1},
        subcategory_counts={"football": 3},
        history_tokens={"super", "bowl", "match"},
    )

    item_meta = {
        "news_id": "N5",
        "category": "sports",
        "subcategory": "football",
        "title": "Super Bowl Live Final",
        "abstract": "Exciting game tonight with record viewership.",
    }

    item_stats = {
        "item_total_clicks": 15,
        "item_total_impressions": 100,
        "item_global_ctr": 0.125,
        "item_category_popularity": 0.35,
        "item_has_entities": 1,
    }

    u_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_vec = np.array([0.8, 0.6, 0.0], dtype=np.float32)

    features = extract_features_single(
        user_profile=profile,
        item_metadata=item_meta,
        item_stats=item_stats,
        user_vector=u_vec,
        item_vector=v_vec,
        retrieval_score=0.8,
        retrieval_rank=1,
        recency_hours=2.5,
    )

    # Check all feature names in FEATURE_NAMES exist
    assert len(FEATURE_NAMES) == 35
    for fname in FEATURE_NAMES:
        assert fname in features, f"Missing feature: {fname}"
        val = features[fname]
        assert not np.isnan(val), f"Feature {fname} has NaN value"

    assert features["category_match"] == 1.0
    assert features["category_affinity"] == 0.75  # 3/4
    assert features["cos_sim_user_item"] == 0.8
    assert features["lexical_overlap_count"] == 2.0  # "super", "bowl"
    assert features["item_is_fresh"] == 1.0
