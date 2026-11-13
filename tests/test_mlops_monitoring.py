from pathlib import Path

import numpy as np
import pandas as pd

from mind_recsys.mlops.tracker import MLflowTracker
from mind_recsys.monitoring.drift import DriftMonitor


def test_mlflow_tracker_logging(tmp_path: Path):
    db_file = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    tracker = MLflowTracker(
        experiment_name="test_experiment",
        tracking_uri=tracking_uri,
    )

    fi_df = pd.DataFrame(
        {
            "feature": ["cos_sim", "recency", "ctr"],
            "importance": [45.0, 30.0, 25.0],
        }
    )

    run_id = tracker.log_experiment_run(
        run_name="unit_test_run",
        params={"learning_rate": 0.05, "depth": 6},
        metrics={"NDCG@10": 0.42, "MRR": 0.38},
        feature_importance_df=fi_df,
    )

    assert run_id is not None
    assert len(run_id) > 0


def test_drift_monitor_html_generation(tmp_path: Path):
    np.random.seed(42)
    # Reference: 100 samples from normal distribution
    ref_df = pd.DataFrame(
        {
            "num_feat": np.random.normal(loc=0.0, scale=1.0, size=100),
            "cat_feat": np.random.choice(["sports", "tech", "finance"], size=100),
        }
    )
    # Current: shifted distribution
    curr_df = pd.DataFrame(
        {
            "num_feat": np.random.normal(loc=2.0, scale=1.0, size=100),
            "cat_feat": np.random.choice(
                ["sports", "tech", "finance"], p=[0.8, 0.1, 0.1], size=100
            ),
        }
    )

    monitor = DriftMonitor()
    report_path = tmp_path / "drift_report.html"

    report_dict = monitor.generate_report(
        reference_data=ref_df,
        current_data=curr_df,
        output_html_path=report_path,
    )

    assert report_path.exists()
    assert report_path.stat().st_size > 1000
    assert isinstance(report_dict, dict)

    summary = monitor.detect_drift_summary(ref_df, curr_df)
    assert summary["has_metrics"] is True
    assert summary["metrics_count"] > 0
    assert summary["status"] == "completed"
