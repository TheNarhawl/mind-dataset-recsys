from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


class DriftMonitor:
    """Monitors feature and data distribution drift between time periods using Evidently AI."""

    def __init__(self, stattest_threshold: float = 0.05) -> None:
        self.stattest_threshold = stattest_threshold

    def generate_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Any | None = None,
        output_html_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Generates an Evidently Data Drift Report and optionally exports it to an HTML file."""
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        if output_html_path is not None:
            path = Path(output_html_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(path))

        return snapshot.dict()

    def detect_drift_summary(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Quick summary of dataset drift."""
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=reference_data,
            current_data=current_data,
        )
        return {
            "metrics_count": len(snapshot.metric_results),
            "has_metrics": len(snapshot.metric_results) > 0,
            "status": "completed",
        }
