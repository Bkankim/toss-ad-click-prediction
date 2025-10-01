"""자동 EDA 리포트를 생성하는 모듈."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import polars as pl


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="토스 CTR 데이터셋 EDA 리포트 생성기")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300_000,
        help="train.parquet에서 로드할 최대 행 수",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/EDA.md"),
        help="생성된 Markdown 리포트를 저장할 경로",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="샘플 추출 시 사용할 난수 시드",
    )
    return parser.parse_args()


# 데이터 파일 경로를 확인한다.
def resolve_data_paths() -> Dict[str, Path]:
    data_root = Path(os.getenv("DATA_ROOT", "/Competition/CTR/toss-ad-click-prediction/data"))
    train_path = Path(os.getenv("TRAIN_PATH", data_root / "train.parquet"))
    test_path = Path(os.getenv("TEST_PATH", data_root / "test.parquet"))
    return {
        "DATA_ROOT": data_root,
        "TRAIN_PATH": train_path,
        "TEST_PATH": test_path,
    }


# 데이터 샘플을 로드한다.
def load_sample(train_path: Path, sample_size: int, seed: int) -> pl.DataFrame:
    if not train_path.exists():
        msg = f"train 데이터가 존재하지 않습니다: {train_path}"
        raise FileNotFoundError(msg)

    if sample_size > 0:
        frame = pl.read_parquet(train_path.as_posix(), n_rows=sample_size)
        if frame.height > 0:
            frame = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        return frame

    return pl.read_parquet(train_path.as_posix())


# 전체 데이터 기준 요약 통계를 계산한다.
def build_dataset_overview(train_path: Path) -> pl.DataFrame:
    scan = pl.scan_parquet(train_path.as_posix())
    aggregated = scan.select(
        pl.len().alias("rows"),
        pl.col("clicked").sum().alias("clicks"),
        pl.col("clicked").mean().alias("click_rate"),
    ).collect()

    metrics = aggregated.to_dicts()[0]
    return pl.DataFrame(
        {
            "metric": ["rows", "clicks", "click_rate"],
            "value": [
                f"{int(metrics['rows']):,}",
                f"{int(metrics['clicks']):,}",
                f"{metrics['click_rate']:.6f}",
            ],
        }
    )


# 전체 데이터 기준 그룹 통계를 계산한다.
def summarize_column(train_path: Path, column: str) -> pl.DataFrame:
    scan = pl.scan_parquet(train_path.as_posix())
    grouped = (
        scan.group_by(column)
        .agg(
            pl.len().alias("rows"),
            pl.col("clicked").sum().alias("clicks"),
            pl.col("clicked").mean().alias("click_rate"),
        )
        .sort(column)
        .collect()
    )
    return grouped.with_columns(pl.col("click_rate").round(6))


# 시퀀스 길이 통계를 계산한다.
def compute_sequence_metrics(frame: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    seq_ready = frame.with_columns(
        pl.when(pl.col("seq").is_not_null())
        .then(pl.col("seq").str.count_matches(",") + 1)
        .otherwise(0)
        .alias("seq_length")
    )

    summary = seq_ready.select(
        pl.col("seq_length").mean().round(2).alias("mean"),
        pl.col("seq_length").median().alias("median"),
        pl.col("seq_length").min().alias("min"),
        pl.col("seq_length").max().alias("max"),
        pl.col("seq_length").quantile(0.25).alias("q1"),
        pl.col("seq_length").quantile(0.75).alias("q3"),
    )

    bucket_expr = (
        pl.when(pl.col("seq_length") < 10)
        .then(pl.lit("0-9"))
        .when(pl.col("seq_length") < 25)
        .then(pl.lit("10-24"))
        .when(pl.col("seq_length") < 50)
        .then(pl.lit("25-49"))
        .when(pl.col("seq_length") < 100)
        .then(pl.lit("50-99"))
        .when(pl.col("seq_length") < 250)
        .then(pl.lit("100-249"))
        .when(pl.col("seq_length") < 500)
        .then(pl.lit("250-499"))
        .when(pl.col("seq_length") < 1_000)
        .then(pl.lit("500-999"))
        .when(pl.col("seq_length") < 2_500)
        .then(pl.lit("1000-2499"))
        .when(pl.col("seq_length") < 5_000)
        .then(pl.lit("2500-4999"))
        .when(pl.col("seq_length") < 10_000)
        .then(pl.lit("5000-9999"))
        .otherwise(pl.lit("10000+"))
        .alias("seq_bucket")
    )

    bucketed = (
        seq_ready.with_columns(bucket_expr)
        .group_by("seq_bucket")
        .agg(
            pl.len().alias("rows"),
            pl.col("clicked").sum().alias("clicks"),
            pl.col("clicked").mean().alias("click_rate"),
        )
        .sort("rows", descending=True)
        .with_columns(pl.col("click_rate").round(6))
    )

    return {
        "summary": summary,
        "buckets": bucketed,
    }


# 사이클릭 피처 요약을 계산한다.
def build_cyclic_overview(train_path: Path) -> Dict[str, pl.DataFrame]:
    scan = pl.scan_parquet(train_path.as_posix())

    day_table = (
        scan.group_by("day_of_week")
        .agg(
            pl.len().alias("rows"),
            pl.col("clicked").mean().alias("click_rate"),
            (((pl.col("day_of_week").cast(pl.Float64) - 1.0) * math.tau / 7.0).sin())
            .mean()
            .alias("mean_sin"),
            (((pl.col("day_of_week").cast(pl.Float64) - 1.0) * math.tau / 7.0).cos())
            .mean()
            .alias("mean_cos"),
        )
        .sort("day_of_week")
        .collect()
        .with_columns(
            pl.col("click_rate").round(6),
            pl.col("mean_sin").round(6),
            pl.col("mean_cos").round(6),
        )
    )

    hour_table = (
        scan.group_by("hour")
        .agg(
            pl.len().alias("rows"),
            pl.col("clicked").mean().alias("click_rate"),
            ((pl.col("hour").cast(pl.Float64) * math.tau / 24.0).sin()).mean().alias(
                "mean_sin"
            ),
            ((pl.col("hour").cast(pl.Float64) * math.tau / 24.0).cos()).mean().alias(
                "mean_cos"
            ),
        )
        .sort("hour")
        .collect()
        .with_columns(
            pl.col("click_rate").round(6),
            pl.col("mean_sin").round(6),
            pl.col("mean_cos").round(6),
        )
    )

    return {
        "day": day_table,
        "hour": hour_table,
    }


# DataFrame을 Markdown 테이블로 변환한다.
def frame_to_markdown(frame: pl.DataFrame, float_precision: int = 6) -> str:
    if frame.is_empty():
        return "_데이터가 없음_"

    headers = frame.columns
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows: List[str] = [header_line, separator_line]

    for row in frame.iter_rows(named=True):
        formatted: List[str] = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                formatted.append(f"{value:.{float_precision}f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


# 리포트를 구성한다.
def build_report(frame: pl.DataFrame, sample_size: int, paths: Dict[str, Path]) -> str:
    overview = build_dataset_overview(paths["TRAIN_PATH"])
    day_stats = summarize_column(paths["TRAIN_PATH"], "day_of_week")
    hour_stats = summarize_column(paths["TRAIN_PATH"], "hour")
    seq_stats = compute_sequence_metrics(frame)
    cyclic_tables = build_cyclic_overview(paths["TRAIN_PATH"])

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    sections: List[str] = [
        "# 자동 생성 EDA 리포트",
        f"- 생성 일시: {generated_at}",
        f"- 샘플 크기: {sample_size:,} (shuffle sample)",
        f"- 데이터 경로: `{paths['TRAIN_PATH']}`",
        "",
        "## 데이터 개요",
        frame_to_markdown(overview, float_precision=6),
        "",
        "## 요일(day_of_week)별 클릭 통계",
        frame_to_markdown(day_stats, float_precision=6),
        "",
        "## 시간(hour)별 클릭 통계",
        frame_to_markdown(hour_stats, float_precision=6),
        "",
        "## 시퀀스 길이 요약",
        frame_to_markdown(seq_stats["summary"], float_precision=2),
        "",
        "### 시퀀스 길이 구간별 클릭률",
        frame_to_markdown(seq_stats["buckets"], float_precision=6),
        "",
        "## 사이클릭 피처 분석",
        "사이클릭 변환(day/hour)에 대한 평균 sin/cos 값과 클릭률 추세입니다.",
        "",
        "### 요일 사이클릭 평균",
        frame_to_markdown(cyclic_tables["day"], float_precision=6),
        "",
        "### 시간 사이클릭 평균",
        frame_to_markdown(cyclic_tables["hour"], float_precision=6),
    ]

    return "\n".join(sections)


# 리포트를 파일로 저장한다.
def write_report(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# 메인 실행 흐름을 제어한다.
def main() -> None:
    args = parse_args()
    paths = resolve_data_paths()
    frame = load_sample(paths["TRAIN_PATH"], args.sample_size, args.random_seed)
    report = build_report(frame, args.sample_size, paths)
    write_report(report, args.output)
    print(f"EDA 리포트를 생성했습니다: {args.output}")


if __name__ == "__main__":
    main()
