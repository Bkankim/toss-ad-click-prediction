#!/usr/bin/env python3
"""Generate day-7 special features with cross-fit aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

pl.Config.set_tbl_formatting("ASCII_FULL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build day-7 specific features")
    parser.add_argument("--input-train", type=Path, required=True, help="Base train parquet path")
    parser.add_argument("--input-test", type=Path, required=True, help="Base test parquet path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for augmented dataset")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON path to update")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds for cross-fit")
    return parser.parse_args()


def _aggregate_ctr(df: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            {**{k: pl.Series(dtype=pl.Int32) for k in keys}, "ctr": pl.Series([], dtype=pl.Float64)}
        ).with_columns(pl.Series("cnt", [], dtype=pl.Int64))
    return df.group_by(keys).agg([pl.col("clicked").mean().alias("ctr"), pl.len().alias("cnt")])


def compute_ctr_delta(
    train_df: pl.DataFrame,
    keys: List[str],
    feature_name: str,
    n_splits: int,
) -> Tuple[np.ndarray, pl.DataFrame, float]:
    y = train_df["clicked"].to_numpy()
    row_idx = train_df["row_idx"].to_numpy()
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)

    global_day7 = train_df.filter(pl.col("day_of_week") == 7)
    global_rest = train_df.filter(pl.col("day_of_week") != 7)
    global_day7_mean = float(global_day7["clicked"].mean()) if global_day7.height else float(train_df["clicked"].mean())
    global_rest_mean = float(global_rest["clicked"].mean()) if global_rest.height else float(train_df["clicked"].mean())
    global_delta = global_day7_mean - global_rest_mean

    day7_stats_global = _aggregate_ctr(global_day7, keys).rename({"ctr": "day7_ctr"})
    rest_stats_global = _aggregate_ctr(global_rest, keys).rename({"ctr": "rest_ctr"})
    global_stats = day7_stats_global.join(rest_stats_global, on=keys, how="outer")
    global_stats = global_stats.with_columns([
        pl.col("day7_ctr").fill_null(global_day7_mean),
        pl.col("rest_ctr").fill_null(global_rest_mean),
        (pl.col("day7_ctr") - pl.col("rest_ctr")).alias(feature_name),
    ]).select(keys + [feature_name])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        train_rows = set(row_idx[train_idx])
        valid_rows = row_idx[valid_idx]

        train_fold = train_df.filter(pl.col("row_idx").is_in(train_rows))
        day7_fold = train_fold.filter(pl.col("day_of_week") == 7)
        rest_fold = train_fold.filter(pl.col("day_of_week") != 7)

        stats_fold = _aggregate_ctr(day7_fold, keys).rename({"ctr": "day7_ctr"})
        stats_fold = stats_fold.join(_aggregate_ctr(rest_fold, keys).rename({"ctr": "rest_ctr"}), on=keys, how="outer")
        stats_fold = stats_fold.with_columns([
            pl.col("day7_ctr").fill_null(global_day7_mean),
            pl.col("rest_ctr").fill_null(global_rest_mean),
            (pl.col("day7_ctr") - pl.col("rest_ctr")).alias(feature_name),
        ]).select(keys + [feature_name])

        valid_fold = train_df.filter(pl.col("row_idx").is_in(valid_rows.tolist()))
        valid_join = (
            valid_fold.join(stats_fold, on=keys, how="left")
            .with_columns(pl.coalesce([pl.col(feature_name), pl.lit(global_delta)]).alias(feature_name))
            .select(["row_idx", feature_name])
        )
        mapping = dict(zip(valid_join["row_idx"].to_numpy(), valid_join[feature_name].to_numpy()))
        oof[valid_idx] = np.array([mapping.get(r, global_delta) for r in valid_rows], dtype=np.float32)
        print(f"Fold {fold} finished for {feature_name}")

    return oof, global_stats, global_delta


def compute_freq_ratio(train_df: pl.DataFrame) -> pl.DataFrame:
    total_counts = train_df.group_by("inventory_id_cat").agg(
        [
            pl.len().alias("total_cnt"),
            pl.col("day_of_week").eq(7).sum().alias("day7_cnt"),
        ]
    )
    if total_counts.is_empty():
        return pl.DataFrame({"inventory_id_cat": [], "day7_freq_ratio_inv": []})
    return total_counts.with_columns(
        ((pl.col("day7_cnt") + 1) / (pl.col("total_cnt") + 2)).alias("day7_freq_ratio_inv")
    ).select(["inventory_id_cat", "day7_freq_ratio_inv"])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cols = [
        "row_idx",
        "clicked",
        "day_of_week",
        "hour",
        "inventory_id_cat",
        "gender_cat",
        "seq_len_bucket_cat",
    ]
    train_df = (
        pl.read_parquet(args.input_train.as_posix(), columns=train_cols)
        .with_columns(
            [
                pl.col("day_of_week").cast(pl.Int32),
                pl.col("hour").cast(pl.Int32),
                pl.col("seq_len_bucket_cat").cast(pl.Int32),
            ]
        )
    )
    test_cols = ["row_idx", "day_of_week", "hour", "inventory_id_cat", "gender_cat", "seq_len_bucket_cat"]
    test_df = (
        pl.read_parquet(args.input_test.as_posix(), columns=test_cols)
        .with_columns(
            [
                pl.col("day_of_week").cast(pl.Int32),
                pl.col("hour").cast(pl.Int32),
                pl.col("seq_len_bucket_cat").cast(pl.Int32),
            ]
        )
    )
    print(f"Train rows: {train_df.height}, Test rows: {test_df.height}")

    train_features: dict[str, np.ndarray] = {"row_idx": train_df["row_idx"].to_numpy()}
    test_features: dict[str, np.ndarray] = {"row_idx": test_df["row_idx"].to_numpy()}

    ctr_delta_configs = [
        (["inventory_id_cat", "hour"], "day7_ctr_delta_inv_hour"),
        (["seq_len_bucket_cat"], "day7_ctr_delta_seq_bucket"),
    ]

    for keys, feat_name in ctr_delta_configs:
        print(f"Computing CTR delta feature {feat_name} with keys {keys}")
        oof_vals, global_stats, global_delta = compute_ctr_delta(train_df, keys, feat_name, args.folds)
        train_features[feat_name] = oof_vals

        test_join = (
            test_df.join(global_stats, on=keys, how="left")
            .with_columns(pl.coalesce([pl.col(feat_name), pl.lit(global_delta)]).alias(feat_name))
            .select(["row_idx", feat_name])
        )
        test_features[feat_name] = test_join[feat_name].to_numpy()

    print("Computing day7 frequency ratio feature")
    freq_stats = compute_freq_ratio(train_df)
    global_ratio = float((train_df["day_of_week"] == 7).sum() / train_df.height)

    train_ratio = (
        train_df.join(freq_stats, on="inventory_id_cat", how="left")
        .with_columns(pl.coalesce([pl.col("day7_freq_ratio_inv"), pl.lit(global_ratio)]).alias("day7_freq_ratio_inv"))
        .select(["row_idx", "day7_freq_ratio_inv"])
    )
    test_ratio = (
        test_df.join(freq_stats, on="inventory_id_cat", how="left")
        .with_columns(pl.coalesce([pl.col("day7_freq_ratio_inv"), pl.lit(global_ratio)]).alias("day7_freq_ratio_inv"))
        .select(["row_idx", "day7_freq_ratio_inv"])
    )
    train_features["day7_freq_ratio_inv"] = train_ratio["day7_freq_ratio_inv"].to_numpy()
    test_features["day7_freq_ratio_inv"] = test_ratio["day7_freq_ratio_inv"].to_numpy()

    train_feat_df = pl.DataFrame(train_features)
    test_feat_df = pl.DataFrame(test_features)

    print("Writing augmented datasets...")
    train_aug = (
        pl.scan_parquet(args.input_train.as_posix())
        .join(train_feat_df.lazy(), on="row_idx", how="left")
        .collect(streaming=True)
    )
    train_aug.write_parquet((output_dir / "train_processed.parquet").as_posix())

    test_aug = (
        pl.scan_parquet(args.input_test.as_posix())
        .join(test_feat_df.lazy(), on="row_idx", how="left")
        .collect(streaming=True)
    )
    test_aug.write_parquet((output_dir / "test_processed.parquet").as_posix())

    print("Updating metadata...")
    meta = json.loads(args.metadata.read_text())
    new_features = [feat for _, feat in ctr_delta_configs] + ["day7_freq_ratio_inv"]
    for feat in new_features:
        if feat not in meta["feature_columns"]:
            meta["feature_columns"].append(feat)
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Finished building day7 features.")


if __name__ == "__main__":
    main()
