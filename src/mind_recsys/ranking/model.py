from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


class NewsRanker:
    """Stage 2 Ranking Model wrapping CatBoostRanker with YetiRank / Pairwise loss."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        loss_function: str = "YetiRank",
        eval_metric: str = "NDCG:top=10",
        random_seed: int = 42,
        task_type: str = "CPU",
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.task_type = task_type
        self.feature_names: list[str] = []

        self.model = CatBoostRanker(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            task_type=self.task_type,
            verbose=False,
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        group_ids_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        group_ids_val: pd.Series | np.ndarray | None = None,
        cat_features: list[str | int] | None = None,
    ) -> "NewsRanker":
        """Fit CatBoostRanker with grouped ranking data."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        train_pool = Pool(
            data=X_train,
            label=y_train,
            group_id=group_ids_train,
            cat_features=cat_features,
            feature_names=self.feature_names or None,
        )

        eval_pool = None
        if X_val is not None and y_val is not None and group_ids_val is not None:
            eval_pool = Pool(
                data=X_val,
                label=y_val,
                group_id=group_ids_val,
                cat_features=cat_features,
                feature_names=self.feature_names or None,
            )

        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=50 if eval_pool else None,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict ranking relevance scores."""
        return np.asarray(self.model.predict(X), dtype=np.float32)

    def rank_candidates(
        self,
        features_df: pd.DataFrame,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Scores candidate items and returns them sorted by score in descending order."""
        if features_df.empty or not candidates:
            return []

        scores = self.predict(features_df)
        ranked_list: list[dict[str, Any]] = []

        for item, score in zip(candidates, scores, strict=False):
            candidate_copy = dict(item)
            candidate_copy["ranker_score"] = float(score)
            ranked_list.append(candidate_copy)

        ranked_list.sort(key=lambda x: x["ranker_score"], reverse=True)
        return ranked_list[:top_k]

    def get_feature_importance(
        self,
        data: Pool | None = None,
        fstr_type: str = "PredictionValuesChange",
    ) -> pd.DataFrame:
        """Get dataframe of feature importances sorted descending."""
        if not self.feature_names:
            return pd.DataFrame()

        importances = self.model.get_feature_importance(data=data, type=fstr_type)
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)

    def save_model(self, file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load_model(self, file_path: Path | str) -> "NewsRanker":
        self.model.load_model(str(file_path))
        return self
