"""전처리 파이프라인 구현 스크립트."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import polars as pl


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTR 전처리 파이프라인")
    parser.add_argument("--train", type=Path, required=True, help="원본 train Parquet 경로")
    parser.add_argument("--test", type=Path, required=True, help="원본 test Parquet 경로")
    parser.add_argument("--output", type=Path, required=True, help="전처리 결과 저장 디렉터리")
    parser.add_argument(
        "--merge-seq-embedding",
        type=Path,
        default=None,
        help="시퀀스 임베딩이 포함된 디렉터리 (train_seq_embeddings.parquet / test_seq_embeddings.parquet 기대)",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="테스트용 샘플링 행 수 (None이면 전체)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", type=Path, default=None, help="전처리 메타데이터 저장 경로")
    return parser.parse_args()


@dataclass
class MissingFlagGroup:
    name: str
    columns: List[str]


@dataclass
class PreprocessConfig:
    categorical_cols: List[str]
    freq_encode_cols: List[str]
    target_encode_pairs: List[Tuple[str, str]]
    cyclical_cols: List[str]
    time_period_bins: List[Tuple[int, int, str]]
    sequence_columns: List[str]
    sequence_cross: List[Tuple[str, str]]
    numeric_prefixes: List[str]
    missing_flag_groups: List[MissingFlagGroup]
    high_collinearity_drop: List[str]


DEFAULT_CONFIG = PreprocessConfig(
    categorical_cols=["gender", "age_group", "inventory_id", "day_of_week", "hour"],
    freq_encode_cols=["gender", "inventory_id"],
    target_encode_pairs=[
        ("gender", "age_group"),
        ("day_of_week", "hour"),
        ("inventory_id", "age_group"),
        ("inventory_id", "gender"),
    ],
    cyclical_cols=["day_of_week", "hour"],
    time_period_bins=[
        (0, 4, "late_night"),
        (5, 9, "early_morning"),
        (10, 16, "daytime"),
        (17, 20, "evening"),
        (21, 23, "late_evening"),
    ],
    sequence_columns=["seq"],
    sequence_cross=[
        ("seq_len", "hour"),
        ("seq_len", "day_of_week"),
        ("seq_last", "day_of_week"),
        ("seq_last", "hour"),
    ],
    numeric_prefixes=[
        "l_feat_",
        "feat_a_",
        "feat_b_",
        "feat_c_",
        "feat_d_",
        "feat_e_",
        "history_a_",
        "history_b_",
    ],
    missing_flag_groups=[
        MissingFlagGroup("feat_a_all_missing", [f"feat_a_{i}" for i in range(1, 19)]),
        MissingFlagGroup("feat_e_3_missing", ["feat_e_3"]),
        MissingFlagGroup(
            "other_all_missing",
            ["l_feat_2", "l_feat_8", "l_feat_18", "l_feat_19", "l_feat_20", "l_feat_21", "l_feat_22", "l_feat_23", "l_feat_24"],
        ),
    ],
    high_collinearity_drop=[
        "history_b_3",
        "history_b_4",
        "history_b_5",
        "history_b_16",
        "history_b_18",
        "history_b_21",
        "history_b_24",
        "l_feat_16",
        "l_feat_17",
        "l_feat_20",
        "l_feat_23",
    ],
)


def load_parquet(path: Path, sample_size: Optional[int], seed: int) -> pl.DataFrame:
    if sample_size:
        frame = pl.read_parquet(path.as_posix(), n_rows=sample_size)
    else:
        frame = pl.read_parquet(path.as_posix())

    if "_row_idx" not in frame.columns:
        frame = frame.with_columns(pl.arange(0, frame.height).alias("_row_idx"))

    if sample_size:
        frame = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    return frame


def fill_categorical_missing(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    out = frame
    schema = out.schema
    for col in columns:
        if col not in out.columns:
            continue
        dtype = schema[col]
        if dtype.is_numeric():
            out = out.with_columns(pl.col(col).fill_null(-1))
        else:
            out = out.with_columns(pl.col(col).cast(pl.Utf8).fill_null("Unknown"))
    return out


def add_cyclical_features(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    out = frame
    if "day_of_week" in columns and "day_of_week" in out.columns:
        angle = 2 * math.pi * (pl.col("day_of_week").cast(pl.Float32) / 7.0)
        out = out.with_columns([
            angle.sin().alias("dow_sin"),
            angle.cos().alias("dow_cos"),
            pl.col("day_of_week").cast(pl.Int32).is_in([1, 7]).cast(pl.Int8).alias("is_weekend"),
        ])
    if "hour" in columns and "hour" in out.columns:
        angle = 2 * math.pi * (pl.col("hour").cast(pl.Float32) / 24.0)
        out = out.with_columns([
            angle.sin().alias("hour_sin"),
            angle.cos().alias("hour_cos"),
        ])
    return out


def add_time_period(frame: pl.DataFrame, bins: List[Tuple[int, int, str]]) -> pl.DataFrame:
    if "hour" not in frame.columns:
        return frame
    expr = pl.lit("other")
    for start, end, label in bins:
        expr = (
            pl.when(pl.col("hour").cast(pl.Int16).is_between(start, end))
            .then(pl.lit(label))
            .otherwise(expr)
        )
    frame = frame.with_columns(expr.alias("time_period"))
    frame = frame.with_columns(
        pl.concat_str(
            [pl.col("day_of_week").cast(pl.Utf8), pl.lit("|"), pl.col("hour").cast(pl.Utf8)]
        ).alias("dow_hour")
    )
    return frame


def add_seq_features(frame: pl.DataFrame) -> pl.DataFrame:
    if "seq" not in frame.columns:
        return frame
    seq_str = pl.col("seq").cast(pl.Utf8)
    frame = frame.with_columns([
        (seq_str.str.count_matches(",") + 1).alias("seq_len"),
        seq_str.str.extract(r"^(\d+)", 1).cast(pl.Int32).alias("seq_first"),
        seq_str.str.extract(r"(\d+)$", 1).cast(pl.Int32).alias("seq_last"),
        pl.lit(1).alias("seq_available"),
    ])
    frame = frame.with_columns(
        pl.when(pl.col("seq_len") < 10)
        .then(pl.lit("0-9"))
        .when(pl.col("seq_len") < 25)
        .then(pl.lit("10-24"))
        .when(pl.col("seq_len") < 50)
        .then(pl.lit("25-49"))
        .when(pl.col("seq_len") < 100)
        .then(pl.lit("50-99"))
        .when(pl.col("seq_len") < 250)
        .then(pl.lit("100-249"))
        .when(pl.col("seq_len") < 500)
        .then(pl.lit("250-499"))
        .when(pl.col("seq_len") < 1000)
        .then(pl.lit("500-999"))
        .when(pl.col("seq_len") < 2500)
        .then(pl.lit("1000-2499"))
        .when(pl.col("seq_len") < 5000)
        .then(pl.lit("2500-4999"))
        .otherwise(pl.lit("5000+"))
        .alias("seq_len_bucket")
    )
    return frame


def add_seq_cross(frame: pl.DataFrame, crosses: Iterable[Tuple[str, str]]) -> pl.DataFrame:
    out = frame
    for left, right in crosses:
        if left in out.columns and right in out.columns:
            out = out.with_columns(
                (pl.col(left).cast(pl.Float32) * pl.col(right).cast(pl.Float32)).alias(f"{left}_X_{right}")
            )
    return out


def prepare_pair_keys(frame: pl.DataFrame, pairs: Iterable[Tuple[str, str]]) -> Tuple[pl.DataFrame, List[str]]:
    out = frame
    key_names: List[str] = []
    for left, right in pairs:
        if left in out.columns and right in out.columns:
            key = f"{left}__X__{right}"
            out = out.with_columns(
                pl.concat_str([pl.col(left).cast(pl.Utf8), pl.lit("|"), pl.col(right).cast(pl.Utf8)]).alias(key)
            )
            key_names.append(key)
    return out, key_names


def apply_frequency_encoding(train: pl.DataFrame, test: Optional[pl.DataFrame], columns: Iterable[str]) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    out_train = train
    out_test = test
    total = float(out_train.height)
    for col in columns:
        if col not in out_train.columns:
            continue
        freq = (
            out_train.group_by(col)
            .agg(pl.len().alias(f"{col}_freq"))
            .with_columns((pl.col(f"{col}_freq") / total).alias(f"{col}_freq"))
            .select(col, f"{col}_freq")
        )
        out_train = out_train.join(freq, on=col, how="left")
        if out_test is not None and col in out_test.columns:
            out_test = out_test.join(freq, on=col, how="left").with_columns(pl.col(f"{col}_freq").fill_null(0.0))
    return out_train, out_test


def apply_target_encoding(
    train: pl.DataFrame,
    test: Optional[pl.DataFrame],
    keys: Iterable[str],
    target: str = "clicked",
    weight: float = 50.0,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    out_train = train
    out_test = test
    valid_keys = [key for key in keys if key in out_train.columns]
    if not valid_keys or target not in out_train.columns:
        return out_train, out_test

    prior = float(out_train[target].mean())
    fold_col = "__te_fold__"
    fold_ids = np.random.default_rng(seed).integers(0, n_splits, size=out_train.height)
    out_train = out_train.with_columns(pl.Series(fold_col, fold_ids))

    for key in valid_keys:
        overall = out_train.group_by(key).agg([
            pl.col(target).sum().alias("_sum_total"),
            pl.len().alias("_cnt_total"),
        ])
        fold_stats = out_train.group_by([key, fold_col]).agg([
            pl.col(target).sum().alias("_sum_fold"),
            pl.len().alias("_cnt_fold"),
        ])

        fold_te = (
            fold_stats.join(overall, on=key, how="left")
            .with_columns(
                (
                    (pl.col("_sum_total") - pl.col("_sum_fold") + weight * prior)
                    / (pl.col("_cnt_total") - pl.col("_cnt_fold") + weight)
                ).alias(f"{key}_te")
            )
            .select(key, fold_col, f"{key}_te")
        )

        out_train = out_train.join(fold_te, on=[key, fold_col], how="left").with_columns(
            pl.col(f"{key}_te").fill_null(prior)
        )

        mapping = overall.with_columns(
            ((pl.col("_sum_total") + weight * prior) / (pl.col("_cnt_total") + weight)).alias(f"{key}_te")
        ).select(key, f"{key}_te")

        if out_test is not None and key in out_test.columns:
            out_test = out_test.join(mapping, on=key, how="left").with_columns(
                pl.col(f"{key}_te").fill_null(prior)
            )

    out_train = out_train.drop(fold_col)
    return out_train, out_test


def add_missing_flags(frame: pl.DataFrame, groups: Iterable[MissingFlagGroup]) -> pl.DataFrame:
    out = frame
    for group in groups:
        cols = [c for c in group.columns if c in out.columns]
        if not cols:
            continue
        if len(cols) == 1:
            flag_expr = pl.col(cols[0]).is_null().cast(pl.Int8)
        else:
            flag_expr = pl.all_horizontal([pl.col(c).is_null() for c in cols]).cast(pl.Int8)
        out = out.with_columns(flag_expr.alias(group.name))
    return out


def drop_high_collinearity(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    existing = [c for c in columns if c in frame.columns]
    if existing:
        return frame.drop(existing)
    return frame


def encode_categories(train: pl.DataFrame, test: Optional[pl.DataFrame], columns: Iterable[str]) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    out_train = train
    out_test = test
    with pl.StringCache():
        for col in columns:
            if col in out_train.columns:
                out_train = out_train.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{col}_cat")
                )
            if out_test is not None and col in out_test.columns:
                out_test = out_test.with_columns(
                    pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(f"{col}_cat")
                )
    keep_cols = set(columns)
    drop_cols = [c for c in columns if c in out_train.columns]
    if drop_cols:
        out_train = out_train.drop(drop_cols, strict=False)
    if out_test is not None:
        drop_cols_test = [c for c in columns if c in out_test.columns]
        if drop_cols_test:
            out_test = out_test.drop(drop_cols_test, strict=False)
    return out_train, out_test


def merge_embeddings(train: pl.DataFrame, test: pl.DataFrame, embedding_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    train_file = embedding_dir / "train_seq_embeddings.parquet"
    test_file = embedding_dir / "test_seq_embeddings.parquet"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "임베딩 파일을 찾을 수 없습니다. train_seq_embeddings.parquet/test_seq_embeddings.parquet 파일을 준비하세요."
        )
    train_emb = pl.read_parquet(train_file.as_posix())
    test_emb = pl.read_parquet(test_file.as_posix())
    if "_row_idx" not in train_emb.columns:
        train_emb = train_emb.with_columns(pl.arange(0, train_emb.height).alias("_row_idx"))
    if "_row_idx" not in test_emb.columns:
        test_emb = test_emb.with_columns(pl.arange(0, test_emb.height).alias("_row_idx"))
    train = train.join(train_emb, on="_row_idx", how="left")
    test = test.join(test_emb, on="_row_idx", how="left")
    return train, test


def preprocess(  # noqa: PLR0915
    train: pl.DataFrame,
    test: pl.DataFrame,
    config: PreprocessConfig,
    seed: int,
    embedding_dir: Optional[Path] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    if "_row_idx" not in train.columns:
        train = train.with_columns(pl.arange(0, train.height).alias("_row_idx"))
    if "_row_idx" not in test.columns:
        test = test.with_columns(pl.arange(0, test.height).alias("_row_idx"))

    train = fill_categorical_missing(train, config.categorical_cols)
    test = fill_categorical_missing(test, config.categorical_cols)

    train = add_cyclical_features(train, config.cyclical_cols)
    test = add_cyclical_features(test, config.cyclical_cols)

    train = add_time_period(train, config.time_period_bins)
    test = add_time_period(test, config.time_period_bins)

    train = add_seq_features(train)
    test = add_seq_features(test)

    train = add_seq_cross(train, config.sequence_cross)
    test = add_seq_cross(test, config.sequence_cross)

    train, pair_keys = prepare_pair_keys(train, config.target_encode_pairs)
    test, _ = prepare_pair_keys(test, config.target_encode_pairs)

    train, test = apply_frequency_encoding(train, test, config.freq_encode_cols)
    train, test = apply_target_encoding(train, test, pair_keys, n_splits=5, seed=seed)
    if pair_keys:
        train = train.drop(pair_keys, strict=False)
        test = test.drop(pair_keys, strict=False)

    train = add_missing_flags(train, config.missing_flag_groups)
    test = add_missing_flags(test, config.missing_flag_groups)

    train = train.drop("seq", strict=False)
    test = test.drop("seq", strict=False)

    categorical_to_encode = [
        "gender",
        "age_group",
        "inventory_id",
        "time_period",
        "dow_hour",
        "seq_len_bucket",
    ]
    train, test = encode_categories(train, test, categorical_to_encode)

    train = drop_high_collinearity(train, config.high_collinearity_drop)
    test = drop_high_collinearity(test, config.high_collinearity_drop)

    if embedding_dir is not None:
        train, test = merge_embeddings(train, test, embedding_dir)

    train = train.drop("_row_idx")
    test = test.drop("_row_idx")

    metadata = {
        "train_rows": train.height,
        "test_rows": test.height,
        "feature_columns": [c for c in train.columns if c != "clicked"],
        "config": asdict(config),
    }
    return train, test, metadata


def main() -> None:
    args = parse_args()
    train = load_parquet(args.train, args.sample_size, args.seed)
    test = load_parquet(args.test, args.sample_size, args.seed)
    train_processed, test_processed, metadata = preprocess(
        train,
        test,
        DEFAULT_CONFIG,
        seed=args.seed,
        embedding_dir=args.merge_seq_embedding,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    train_out = args.output / "train_processed.parquet"
    test_out = args.output / "test_processed.parquet"
    train_processed.write_parquet(train_out.as_posix(), compression="zstd")
    test_processed.write_parquet(test_out.as_posix(), compression="zstd")

    metadata_path = args.metadata_path or (args.output / "preprocess_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved train to {train_out}")
    print(f"Saved test to {test_out}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
