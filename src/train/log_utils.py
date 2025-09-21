from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict


METRIC_DIR = Path("logs/metrics")
ARTIFACT_ROOT = Path("notebooks/artifacts_eda")


@dataclass
# 실험 실행 메트릭을 파일에 기록하기 위한 구조체.
class MetricRecord:
    run_id: str
    stage: str
    dataset: str
    metrics: Dict[str, float]
    notes: str = ""


# 메트릭 디렉터리가 존재하지 않으면 생성한다.
def _ensure_metric_dir() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


# 단일 실험 결과를 CSV 파일로 누적 저장한다.
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


# 실행 시각을 조합한 기본 run_id를 반환한다.
def default_run_id(prefix: str = "lgbm_baseline") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{prefix}"


# 날짜와 run_id를 기준으로 아티팩트 저장 폴더를 준비한다.
def ensure_artifact_directory(run_id: str | None = None, date: str | None = None) -> Path:
    date_str = date or datetime.utcnow().strftime("%Y%m%d")
    base = ARTIFACT_ROOT / date_str
    if run_id:
        base = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base


# 루트 디스크 사용량을 출력해 저장 전 용량 상태를 점검한다.
def report_root_disk_usage() -> None:
    """Run ``du -sh /`` with stderr silenced to report root disk usage."""

    subprocess.run(
        ["bash", "-lc", "du -sh / 2>/dev/null"],
        check=False,
    )
