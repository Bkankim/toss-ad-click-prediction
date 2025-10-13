#!/usr/bin/env python3
"""Build day-7 history interaction features with cross-fit aggregation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

pl.Config.set_tbl_formatting("ASCII_FULL")


# 히스토리 피처 교차 구성을 표현한다.
@dataclass(frozen=True)
class FeaturePair:
    left: str
    right: str
    name: str


HISTORY_PAIRS: Sequence[FeaturePair] = (
    FeaturePair("history_a_6", "history_b_11", "day7_hist_a6_b11_delta"),
    FeaturePair("history_a_6", "history_b_22", "day7_hist_a6_b22_delta"),
    FeaturePair("history_a_6", "history_b_23", "day7_hist_a6_b23_delta"),
)


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build day-7 history cross-fit features")
    parser.add_argument("--input-train", type=Path, required=True, help="Base train parquet path")
    parser.add_argument("--input-test", type=Path, required=True, help="Base test parquet path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for augmented dataset")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON path to update")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds for cross-fit")
    parser.add_argument("--bins", type=int, default=24, help="Number of quantile bins for history features")
    parser.add_argument(
        "--extra-keys",
        nargs="*",
        default=(),
        help="추가 그룹화 키(예: gender_cat)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=100.0,
        help="CTR 라플라스 스무딩 강도",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=50,
        help="fold 통계에 신뢰를 부여할 최소 샘플 수",
    )
    return parser.parse_args()


# 연속값을 분위수 기반 구간으로 나눈다.
def compute_bins(values: pl.Series, bins: int) -> List[float]:
    arr = values.drop_nulls().to_numpy()
    if arr.size == 0 or np.all(arr == arr[0]):
        center = float(arr[0]) if arr.size else 0.0
        return [-np.inf, center, np.inf]

    quantiles = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    edges = np.quantile(arr, quantiles, method="nearest")
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.array([arr.min(), arr.mean(), arr.max()], dtype=np.float64)
        edges = np.unique(edges)
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


# 값들을 구간 인덱스로 변환한다.
def assign_bins(series: pl.Series, edges: Sequence[float]) -> np.ndarray:
    boundaries = np.asarray(edges[1:-1], dtype=np.float64)
    values = series.to_numpy()
    if boundaries.size == 0:
        return np.zeros_like(values, dtype=np.int32)
    bins = np.digitize(values, boundaries, right=False)
    return bins.astype(np.int32)


# 클릭 합계와 개수를 집계한다.
def aggregate_stats(df: pl.DataFrame, keys: Iterable[str], prefix: str) -> pl.DataFrame:
    if df.is_empty():
        empty = {key: pl.Series(dtype=pl.Int32) for key in keys}
        empty[f"{prefix}_sum"] = pl.Series([], dtype=pl.Float64)
        empty[f"{prefix}_cnt"] = pl.Series([], dtype=pl.Int64)
        return pl.DataFrame(empty)
    return df.group_by(list(keys)).agg(
        [
            pl.col("clicked").sum().alias(f"{prefix}_sum"),
            pl.len().alias(f"{prefix}_cnt"),
        ]
    )


# 스무딩된 CTR 델타를 계산한다.
def compute_ctr_delta(
    train_df: pl.DataFrame,
    keys: List[str],
    feature_name: str,
    n_splits: int,
    smoothing_alpha: float,
    min_count: int,
    global_day7_mean: float,
    global_rest_mean: float,
) -> Tuple[np.ndarray, pl.DataFrame, float]:
    y = train_df["clicked"].to_numpy()
    row_idx = train_df["row_idx"].to_numpy()
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)

    smoothing = max(smoothing_alpha, 1e-6)
    min_cnt = max(min_count, 0)

    day7_stats_global = aggregate_stats(train_df.filter(pl.col("day_of_week") == 7), keys, "day7")
    rest_stats_global = aggregate_stats(train_df.filter(pl.col("day_of_week") != 7), keys, "rest")
    global_stats = day7_stats_global.join(rest_stats_global, on=keys, how="outer").with_columns(
        [
            pl.col("day7_sum").fill_null(0.0),
            pl.col("day7_cnt").fill_null(0),
            pl.col("rest_sum").fill_null(0.0),
            pl.col("rest_cnt").fill_null(0),
        ]
    )
    global_stats = global_stats.with_columns(
        [
            pl.when(pl.col("day7_cnt") >= min_cnt)
            .then((pl.col("day7_sum") + smoothing * global_day7_mean) / (pl.col("day7_cnt") + smoothing))
            .otherwise(pl.lit(global_day7_mean))
            .alias("day7_ctr"),
            pl.when(pl.col("rest_cnt") >= min_cnt)
            .then((pl.col("rest_sum") + smoothing * global_rest_mean) / (pl.col("rest_cnt") + smoothing))
            .otherwise(pl.lit(global_rest_mean))
            .alias("rest_ctr"),
        ]
    ).with_columns(
        (pl.col("day7_ctr") - pl.col("rest_ctr")).alias(feature_name)
    ).select(keys + [feature_name])

    global_delta = global_day7_mean - global_rest_mean

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        train_rows = set(row_idx[train_idx])
        valid_rows = row_idx[valid_idx]

        train_fold = train_df.filter(pl.col("row_idx").is_in(train_rows))
        stats_fold = aggregate_stats(train_fold.filter(pl.col("day_of_week") == 7), keys, "day7")
        stats_fold = stats_fold.join(
            aggregate_stats(train_fold.filter(pl.col("day_of_week") != 7), keys, "rest"),
            on=keys,
            how="outer",
        ).with_columns(
            [
                pl.col("day7_sum").fill_null(0.0),
                pl.col("day7_cnt").fill_null(0),
                pl.col("rest_sum").fill_null(0.0),
                pl.col("rest_cnt").fill_null(0),
            ]
        ).with_columns(
            [
                pl.when(pl.col("day7_cnt") >= min_cnt)
                .then((pl.col("day7_sum") + smoothing * global_day7_mean) / (pl.col("day7_cnt") + smoothing))
                .otherwise(pl.lit(global_day7_mean))
                .alias("day7_ctr"),
                pl.when(pl.col("rest_cnt") >= min_cnt)
                .then((pl.col("rest_sum") + smoothing * global_rest_mean) / (pl.col("rest_cnt") + smoothing))
                .otherwise(pl.lit(global_rest_mean))
                .alias("rest_ctr"),
            ]
        ).with_columns(
            (pl.col("day7_ctr") - pl.col("rest_ctr")).alias(feature_name)
        ).select(keys + [feature_name])

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


# 스크립트 메인 로직을 수행한다.
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_keys = list(dict.fromkeys(args.extra_keys))
    base_columns = [
        "row_idx",
        "clicked",
        "day_of_week",
        "seq_len_bucket_cat",
        *extra_keys,
    ]
    history_columns = sorted({pair.left for pair in HISTORY_PAIRS} | {pair.right for pair in HISTORY_PAIRS})
    train_cols = base_columns + history_columns
    test_cols = [col for col in train_cols if col != "clicked"]

    train_df = pl.read_parquet(args.input_train.as_posix(), columns=train_cols)
    test_df = pl.read_parquet(args.input_test.as_posix(), columns=test_cols)

    missing_extra = [col for col in extra_keys if col not in train_df.columns]
    if missing_extra:
        raise ValueError(f"추가 키 컬럼을 찾을 수 없습니다: {missing_extra}")

    train_df = train_df.with_columns(
        [
            pl.col("day_of_week").cast(pl.Int32),
            pl.col("seq_len_bucket_cat").cast(pl.Int32),
            *[pl.col(col).cast(pl.Int32) for col in extra_keys],
        ]
    )
    test_df = test_df.with_columns(
        [
            pl.col("day_of_week").cast(pl.Int32),
            pl.col("seq_len_bucket_cat").cast(pl.Int32),
            *[pl.col(col).cast(pl.Int32) for col in extra_keys],
        ]
    )

    fill_values: Dict[str, float] = {}
    for column in history_columns:
        median_value = float(train_df.select(pl.col(column).median()).item())
        fill_values[column] = median_value
        train_df = train_df.with_columns(pl.col(column).fill_null(median_value).alias(column))
        test_df = test_df.with_columns(pl.col(column).fill_null(median_value).alias(column))

    global_day7 = train_df.filter(pl.col("day_of_week") == 7)
    global_rest = train_df.filter(pl.col("day_of_week") != 7)
    global_day7_mean = float(global_day7["clicked"].mean()) if global_day7.height else float(train_df["clicked"].mean())
    global_rest_mean = float(global_rest["clicked"].mean()) if global_rest.height else float(train_df["clicked"].mean())

    bin_edges: Dict[str, List[float]] = {}
    bins = max(2, args.bins)
    for column in history_columns:
        edges = compute_bins(train_df[column], bins)
        bin_edges[column] = edges
        train_bins = assign_bins(train_df[column], edges)
        test_bins = assign_bins(test_df[column], edges)
        train_df = train_df.with_columns(pl.Series(f"{column}_bin", train_bins))
        test_df = test_df.with_columns(pl.Series(f"{column}_bin", test_bins))

    train_features: Dict[str, np.ndarray] = {"row_idx": train_df["row_idx"].to_numpy()}
    test_features: Dict[str, np.ndarray] = {"row_idx": test_df["row_idx"].to_numpy()}

    for pair in HISTORY_PAIRS:
        left_bin = f"{pair.left}_bin"
        right_bin = f"{pair.right}_bin"

        total_bins_left = len(bin_edges[pair.left]) - 1
        total_bins_right = len(bin_edges[pair.right]) - 1
        composite_col = f"{pair.left}_{pair.right}_bin_idx"

        train_df = train_df.with_columns(
            (
                (pl.col(left_bin) * total_bins_right) + pl.col(right_bin)
            ).cast(pl.Int32).alias(composite_col)
        )
        test_df = test_df.with_columns(
            (
                (pl.col(left_bin) * total_bins_right) + pl.col(right_bin)
            ).cast(pl.Int32).alias(composite_col)
        )

        keys = ["seq_len_bucket_cat", *extra_keys, composite_col]
        print(f"Computing history delta feature {pair.name} with keys {keys}")
        oof_vals, global_stats, global_delta = compute_ctr_delta(
            train_df,
            keys,
            pair.name,
            args.folds,
            args.smoothing_alpha,
            args.min_count,
            global_day7_mean,
            global_rest_mean,
        )
        train_features[pair.name] = oof_vals

        test_join = (
            test_df.join(global_stats, on=keys, how="left")
            .with_columns(pl.coalesce([pl.col(pair.name), pl.lit(global_delta)]).alias(pair.name))
            .select(["row_idx", pair.name])
        )
        test_features[pair.name] = test_join[pair.name].to_numpy()

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
    for feat in (pair.name for pair in HISTORY_PAIRS):
        if feat not in meta["feature_columns"]:
            meta["feature_columns"].append(feat)
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    feature_info = {
        "fill_values": fill_values,
        "bin_edges": {col: edges for col, edges in bin_edges.items()},
        "pairs": [pair.name for pair in HISTORY_PAIRS],
        "extra_keys": extra_keys,
        "smoothing_alpha": args.smoothing_alpha,
        "min_count": args.min_count,
    }
    (output_dir / "history_feature_metadata.json").write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
    print("Finished building history interaction features.")


if __name__ == "__main__":
    main()
