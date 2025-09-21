from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
# 다운샘플링 작업에 필요한 입출력 경로와 세부 설정을 담는다.
class DownsampleConfig:
    raw_path: str = "data/train.parquet"
    output_path: str = "data/processed/train_downsample_1_2.parquet"
    negative_multiplier: float = 2.0
    seed: int = 42
    shuffle: bool = True
    threads: int = 4


# 다운샘플링된 데이터에서 클래스 비율과 통계를 담는다.
@dataclass
class DownsampleStats:
    dataset_path: str
    positives: int
    negatives: int

    @property
    def neg_to_pos_ratio(self) -> float:
        return float(self.negatives) / self.positives if self.positives else float("inf")

    def within_tolerance(self, expected_ratio: float, tolerance: float = 0.05) -> bool:
        if self.positives == 0:
            return False
        diff = abs(self.neg_to_pos_ratio - expected_ratio)
        return diff <= expected_ratio * tolerance

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "positives": self.positives,
            "negatives": self.negatives,
            "neg_to_pos_ratio": self.neg_to_pos_ratio,
        }


# 대상 파일의 상위 폴더를 보장해 저장 오류를 방지한다.
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# 다운샘플링 결과의 클래스 분포를 계산한다.
def compute_downsample_stats(path: str, target_col: str = "clicked") -> DownsampleStats:
    con = duckdb.connect()
    query = f"""
        SELECT
            SUM(CASE WHEN {target_col} = 1 THEN 1 ELSE 0 END) AS positives,
            SUM(CASE WHEN {target_col} = 0 THEN 1 ELSE 0 END) AS negatives
        FROM read_parquet(?)
    """
    positives, negatives = con.execute(query, [os.fspath(path)]).fetchone()
    return DownsampleStats(dataset_path=os.fspath(path), positives=positives, negatives=negatives)


# 클래스 비율 통계를 JSON 파일로 남긴다.
def log_downsample_stats(stats: DownsampleStats, directory: Optional[str] = None) -> Path:
    base_dir = Path(directory) if directory else Path("logs/downsample")
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = base_dir / f"downsample_stats_{timestamp}.json"
    filename.write_text(json.dumps(stats.to_dict(), indent=2))
    return filename


# 양성/음성 비율을 맞춘 학습용 Parquet을 생성한다.
def create_downsampled_dataset(config: DownsampleConfig, *, overwrite: bool = False) -> Path:
    """Create a down-sampled training set with positive class fully kept and
    negatives sampled at ``negative_multiplier`` times the positives.

    Parameters
    ----------
    config: DownsampleConfig
        Configuration with file paths and sampling parameters.
    overwrite: bool, default False
        Whether to regenerate the dataset if it already exists.

    Returns
    -------
    Path
        Filesystem path to the generated parquet file.
    """

    output_path = Path(config.output_path)
    if output_path.exists() and not overwrite:
        return output_path

    _ensure_parent(output_path)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={config.threads}")
    seed_fraction = (config.seed % 10000) / 10000.0
    con.execute("SELECT setseed(?)", [seed_fraction])

    raw_path = os.fspath(config.raw_path)

    pos_count = con.execute(
        "SELECT COUNT(*) FROM read_parquet('%s') WHERE clicked = 1" % raw_path
    ).fetchone()[0]
    neg_count = con.execute(
        "SELECT COUNT(*) FROM read_parquet('%s') WHERE clicked = 0" % raw_path
    ).fetchone()[0]

    target_neg_rows = int(pos_count * config.negative_multiplier)
    if target_neg_rows >= neg_count:
        sample_query = "SELECT * FROM read_parquet('%s') WHERE clicked = 0" % raw_path
    else:
        sample_query = (
            "SELECT * FROM read_parquet('%s') WHERE clicked = 0 "
            "ORDER BY random() LIMIT %d" % (raw_path, target_neg_rows)
        )

    query = f"""
        WITH pos AS (
            SELECT * FROM read_parquet('{raw_path}') WHERE clicked = 1
        ),
        neg AS (
            {sample_query}
        )
        SELECT * FROM pos
        UNION ALL
        SELECT * FROM neg
    """

    table = con.execute(query).fetch_arrow_table()

    if config.shuffle:
        # shuffle rows deterministically using numpy permutation
        import numpy as np

        rng = np.random.default_rng(config.seed)
        order = rng.permutation(table.num_rows)
        table = table.take(pa.array(order))

    pq.write_table(
        table,
        os.fspath(output_path),
        compression="ZSTD",
        compression_level=3,
        use_dictionary=True,
    )

    return output_path
