import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from catboost import CatBoostRanker


class MLflowTracker:
    """Manages MLflow experiment tracking, artifact logging, and Model Registry."""

    def __init__(
        self,
        experiment_name: str = "mind-two-stage-recsys",
        tracking_uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        # Allow file store if legacy path is passed
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        if tracking_uri is None:
            self.tracking_uri = "sqlite:///mlflow.db"
        else:
            self.tracking_uri = tracking_uri

        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_experiment_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        ranker_model: CatBoostRanker | None = None,
        feature_importance_df: pd.DataFrame | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Starts an MLflow run, logs hyperparams, metrics, artifacts, and returns run_id."""
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)

            clean_metrics = {k.replace("@", "_at_"): float(v) for k, v in metrics.items()}
            mlflow.log_params(params)
            mlflow.log_metrics(clean_metrics)

            # Log feature importance plot
            if feature_importance_df is not None and not feature_importance_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                top_features = feature_importance_df.head(20).sort_values("importance")
                ax.barh(top_features["feature"], top_features["importance"], color="#2b5c8f")
                ax.set_title("Top-20 Feature Importances (CatBoost YetiRank)")
                ax.set_xlabel("Importance")
                plt.tight_layout()

                plot_path = Path("temp_fi_plot.png")
                fig.savefig(plot_path)
                plt.close(fig)

                mlflow.log_artifact(str(plot_path), artifact_path="feature_importance")
                if plot_path.exists():
                    plot_path.unlink()

            # Log CatBoost model
            if ranker_model is not None:
                mlflow.catboost.log_model(
                    cb_model=ranker_model,
                    artifact_path="model",
                )

            return run.info.run_id
