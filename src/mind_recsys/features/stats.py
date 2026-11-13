from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def compute_entropy(counts: list[int] | np.ndarray) -> float:
    """Compute Shannon entropy base 2 for a distribution of categorical counts."""
    arr = np.asarray(counts, dtype=np.float64)
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    probs = arr[arr > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


@dataclass
class UserProfile:
    user_id: str
    hist_len: int
    category_entropy: float
    avg_headline_len: float
    top_category: str
    category_counts: dict[str, int]
    subcategory_counts: dict[str, int]
    history_tokens: set[str]


def build_user_profiles(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> dict[str, UserProfile]:
    """Precompute aggregate profile features for each user based on their reading history."""
    news_map = news_df.set_index("news_id").to_dict(orient="index")
    user_profiles: dict[str, UserProfile] = {}

    for _, row in behaviors_df.iterrows():
        user_id = str(row["user_id"])
        if user_id in user_profiles:
            continue

        history = row.get("parsed_history", [])
        hist_len = len(history)

        categories: list[str] = []
        subcategories: list[str] = []
        headline_lengths: list[int] = []
        all_tokens: set[str] = set()

        for nid in history:
            item = news_map.get(nid)
            if item:
                cat = item.get("category", "")
                subcat = item.get("subcategory", "")
                title = item.get("title", "")
                if cat:
                    categories.append(cat)
                if subcat:
                    subcategories.append(subcat)
                if title:
                    headline_lengths.append(len(title))
                    all_tokens.update(title.lower().split())

        cat_counts = Counter(categories)
        subcat_counts = Counter(subcategories)
        entropy = compute_entropy(list(cat_counts.values()))
        avg_head_len = float(np.mean(headline_lengths)) if headline_lengths else 0.0
        top_cat = cat_counts.most_common(1)[0][0] if cat_counts else ""

        user_profiles[user_id] = UserProfile(
            user_id=user_id,
            hist_len=hist_len,
            category_entropy=round(entropy, 4),
            avg_headline_len=round(avg_head_len, 2),
            top_category=top_cat,
            category_counts=dict(cat_counts),
            subcategory_counts=dict(subcat_counts),
            history_tokens=all_tokens,
        )

    return user_profiles


def compute_item_stats(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes global smoothed CTR, total clicks, impressions, and category popularity."""
    item_clicks: Counter[str] = Counter()
    item_impressions: Counter[str] = Counter()

    for _, row in behaviors_df.iterrows():
        impressions = row.get("parsed_impressions", [])
        for nid, clicked in impressions:
            item_impressions[nid] += 1
            if clicked == 1:
                item_clicks[nid] += 1

    stats_list: list[dict[str, Any]] = []

    # Category totals
    cat_counts = news_df["category"].value_counts().to_dict()
    total_news = max(len(news_df), 1)

    for _, row in news_df.iterrows():
        nid = str(row["news_id"])
        clicks = item_clicks[nid]
        imps = item_impressions[nid]
        # Bayesian smoothed CTR: (clicks + 10) / (impressions + 100)
        smoothed_ctr = (clicks + 10.0) / (imps + 100.0)
        category = str(row["category"])

        title_text = str(row.get("title", ""))
        abstract_text = str(row.get("abstract", ""))

        has_entities = int(
            bool(
                row.get("title_entities")
                and str(row["title_entities"]) not in ("[]", "nan", "")
                or (
                    row.get("abstract_entities")
                    and str(row["abstract_entities"]) not in ("[]", "nan", "")
                )
            )
        )

        stats_list.append(
            {
                "news_id": nid,
                "item_total_clicks": clicks,
                "item_total_impressions": imps,
                "item_global_ctr": round(smoothed_ctr, 5),
                "item_category_popularity": round(cat_counts.get(category, 0) / total_news, 4),
                "item_has_entities": has_entities,
                "item_title_len": len(title_text),
                "item_abstract_len": len(abstract_text),
            }
        )

    return pd.DataFrame(stats_list)
