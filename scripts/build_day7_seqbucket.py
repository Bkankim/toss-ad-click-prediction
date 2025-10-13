#!/usr/bin/env python3
"""Build day7 seq_len_bucket based features with cross-fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

pl.Config.set_tbl_formatting("ASCII_FULL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build day7 seq bucket feature")
    parser.add_argument("--input-train", type=Path, required=True)
    parser.add_argument("--input-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def compute_seq_bucket_delta(
    train_df: pl.DataFrame,
    folds: int,
) -> Tuple[np.ndarray, pl.DataFrame, float]:
    y = train_df["clicked"].to_numpy()
    row_idx = train_df["row_idx"].to_numpy()
    n = len(y)
    keys = ["seq_len_bucket_cat"]

    day7 = train_df.filter(pl.col("day_of_week") == 7)
    rest = train_df.filter(pl.col("day_of_week") != 7)
    global_day7_mean = float(day7["clicked"].mean()) if day7.height else float(train_df["clicked"].mean())
    global_rest_mean = float(rest["clicked"].mean()) if rest.height else float(train_df["clicked"].mean())
    global_delta = global_day7_mean - global_rest_mean

    day7_stats_global = (
        day7.group_by(keys)
        .agg(pl.col("clicked").mean().alias("ctr_day7"))
    )
    rest_stats_global = (
        rest.group_by(keys)
        .agg(pl.col("clicked").mean().alias("ctr_rest"))
    )
    global_stats = day7_stats_global.join(rest_stats_global, on=keys, how="outer")
    global_stats = global_stats.with_columns([
        pl.col("ctr_day7").fill_null(global_day7_mean),
        pl.col("ctr_rest").fill_null(global_rest_mean),
        (pl.col("ctr_day7") - pl.col("ctr_rest")).alias("day7_seq_bucket_delta"),
    ]).select(keys + ["day7_seq_bucket_delta"])

    oof = np.zeros(n, dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        train_rows = set(row_idx[train_idx])
        valid_rows = row_idx[valid_idx]

        train_fold = train_df.filter(pl.col("row_idx").is_in(train_rows))
        day7_fold = train_fold.filter(pl.col("day_of_week") == 7)
        rest_fold = train_fold.filter(pl.col("day_of_week") != 7)
        stats_fold = day7_fold.group_by(keys).agg(pl.col("clicked").mean().alias("ctr_day7"))
        stats_fold = stats_fold.join(
            rest_fold.group_by(keys).agg(pl.col("clicked").mean().alias("ctr_rest")),
            on=keys,
            how="outer",
        )
        stats_fold = stats_fold.with_columns([
            pl.col("ctr_day7").fill_null(global_day7_mean),
            pl.col("ctr_rest").fill_null(global_rest_mean),
            (pl.col("ctr_day7") - pl.col("ctr_rest")).alias("day7_seq_bucket_delta"),
        ]).select(keys + ["day7_seq_bucket_delta"])

        valid_fold = train_df.filter(pl.col("row_idx").is_in(valid_rows.tolist()))
        valid_join = (
            valid_fold.join(stats_fold, on=keys, how="left")
            .with_columns(pl.coalesce([pl.col("day7_seq_bucket_delta"), pl.lit(global_delta)]).alias("day7_seq_bucket_delta"))
            .select(["row_idx", "day7_seq_bucket_delta"])
        )
        mapping = dict(zip(valid_join["row_idx"].to_numpy(), valid_join["day7_seq_bucket_delta"].to_numpy()))
        oof[valid_idx] = np.array([mapping.get(r, global_delta) for r in valid_rows], dtype=np.float32)
        print(f"Fold {fold} finished for day7_seq_bucket_delta")

    return oof, global_stats, global_delta


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pl.read_parquet(args.input_train.as_posix())
    test_df = pl.read_parquet(args.input_test.as_posix())

    required_cols = {"row_idx", "clicked", "day_of_week", "seq_len_bucket_cat"}
    if not required_cols.issubset(set(train_df.columns)):
        missing = required_cols - set(train_df.columns)
        raise ValueError(f"train 데이터에 필요한 컬럼이 없습니다: {missing}")
    if not {"row_idx", "day_of_week", "seq_len_bucket_cat"}.issubset(set(test_df.columns)):
        missing = {"row_idx", "day_of_week", "seq_len_bucket_cat"} - set(test_df.columns)
        raise ValueError(f"test 데이터에 필요한 컬럼이 없습니다: {missing}")

    train_df = train_df.with_columns([
        pl.col("day_of_week").cast(pl.Int32),
        pl.col("seq_len_bucket_cat").cast(pl.Int32),
    ])
    test_df = test_df.with_columns([
        pl.col("day_of_week").cast(pl.Int32),
        pl.col("seq_len_bucket_cat").cast(pl.Int32),
    ])

    oof_vals, global_stats, global_delta = compute_seq_bucket_delta(train_df, args.folds)

    train_feat = pl.DataFrame({"row_idx": train_df["row_idx"].to_numpy(), "day7_seq_bucket_delta": oof_vals})
    test_feat = (
        test_df.join(global_stats, on=["seq_len_bucket_cat"], how="left")
        .with_columns(pl.coalesce([pl.col("day7_seq_bucket_delta"), pl.lit(global_delta)]).alias("day7_seq_bucket_delta"))
        .select(["row_idx", "day7_seq_bucket_delta"])
    )

    print("Writing augmented datasets...")
    train_aug = (
        pl.scan_parquet(args.input_train.as_posix())
        .join(train_feat.lazy(), on="row_idx", how="left")
        .collect(streaming=True)
    )
    train_aug.write_parquet((output_dir / "train_processed.parquet").as_posix())

    test_aug = (
        pl.scan_parquet(args.input_test.as_posix())
        .join(test_feat.lazy(), on="row_idx", how="left")
        .collect(streaming=True)
    )
    test_aug.write_parquet((output_dir / "test_processed.parquet").as_posix())

    print("Updating metadata...")
    meta = json.loads(args.metadata.read_text())
    if "day7_seq_bucket_delta" not in meta["feature_columns"]:
        meta["feature_columns"].append("day7_seq_bucket_delta")
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Finished building day7_seq_bucket_delta.")


if __name__ == "__main__":
    main()
