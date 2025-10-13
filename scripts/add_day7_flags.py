#!/usr/bin/env python3
"""Add day7 flag features to train/test parquet datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

pl.Config.set_tbl_formatting("ASCII_FULL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append day7 flag columns to datasets")
    parser.add_argument("--input-train", type=Path, required=True)
    parser.add_argument("--input-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--hour-threshold", type=int, default=5, help="hour <= threshold 조건을 day7 flag에 사용")
    return parser.parse_args()


def add_flag(df: pl.DataFrame, threshold: int) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("day_of_week").cast(pl.Int32) == 7)
            & (pl.col("hour").cast(pl.Int32) <= threshold)
        ).cast(pl.Int8).alias("day7_early_hour_flag")
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_parquet(args.input_train.as_posix())
    test_df = pl.read_parquet(args.input_test.as_posix())

    if "day_of_week" not in train_df.columns or "hour" not in train_df.columns:
        raise ValueError("입력 train 데이터에 day_of_week/hour 컬럼이 필요합니다.")
    if "day_of_week" not in test_df.columns or "hour" not in test_df.columns:
        raise ValueError("입력 test 데이터에 day_of_week/hour 컬럼이 필요합니다.")

    train_out = add_flag(train_df, args.hour_threshold)
    test_out = add_flag(test_df, args.hour_threshold)

    train_out.write_parquet((output_dir / "train_processed.parquet").as_posix())
    test_out.write_parquet((output_dir / "test_processed.parquet").as_posix())

    meta = json.loads(args.metadata.read_text())
    if "day7_early_hour_flag" not in meta["feature_columns"]:
        meta["feature_columns"].append("day7_early_hour_flag")
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        "Saved augmented datasets to",
        output_dir,
        "with day7_early_hour_flag (threshold=",
        args.hour_threshold,
        ")",
    )


if __name__ == "__main__":
    main()
