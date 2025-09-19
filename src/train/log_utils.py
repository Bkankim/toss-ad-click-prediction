from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


METRIC_DIR = Path("logs/metrics")


@dataclass
class MetricRecord:
    run_id: str
    stage: str
    dataset: str
    metrics: Dict[str, float]
    notes: str = ""


def _ensure_metric_dir() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def append_metric(record: MetricRecord) -> Path:
    _ensure_metric_dir()
    filename = METRIC_DIR / f"{record.run_id}.csv"
    headers = ["run_id", "stage", "dataset", "ap", "wll", "competition_score", "notes"]
    row = [
        record.run_id,
        record.stage,
        record.dataset,
        f"{record.metrics.get('ap', float('nan')):.6f}",
        f"{record.metrics.get('wll', float('nan')):.6f}",
        f"{record.metrics.get('competition_score', float('nan')):.6f}",
        record.notes,
    ]
    write_header = not filename.exists()
    with filename.open("a", newline="") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)
    return filename


def default_run_id(prefix: str = "lgbm_baseline") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{prefix}"
