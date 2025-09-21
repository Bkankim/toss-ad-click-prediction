from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl

FLOAT_DTYPES = {pl.Float32, pl.Float64}


@dataclass
# 피처 선택 및 결측 처리에 사용할 기본 설정을 보관한다.
class FeatureConfig:
    seq_col: str = "seq"
    target_col: str = "clicked"
    exclude_columns: Tuple[str, ...] = ("clicked", "seq", "ID")
    fill_value: float = 0.0
    null_indicator_columns: Tuple[str, ...] = ("feat_e_3",)
    row_null_threshold: Optional[int] = 95
    row_null_indicator_name: str = "row_nulls_ge_95"

    @property
    # 제외 컬럼 튜플을 빠른 조회를 위해 집합으로 노출한다.
    def exclude_set(self) -> set[str]:
        return set(self.exclude_columns)


SEQ_FEATURE_COLUMNS = [
    "seq_mean",
    "seq_std",
    "seq_min",
    "seq_max",
    "seq_len",
    "seq_sum",
    "seq_median",
]


# 시퀀스 문자열에서 통계량 피처를 계산한다.
def extract_sequence_features(df: pl.DataFrame, seq_col: str) -> pl.DataFrame:
    seq_series = df[seq_col].to_list()
    stats = []
    for raw in seq_series:
        if raw is None or raw == "":
            stats.append({name: 0.0 for name in SEQ_FEATURE_COLUMNS})
            continue
        try:
            tokens = [float(token) for token in str(raw).split(",") if token != ""]
        except ValueError:
            tokens = []
        if not tokens:
            stats.append({name: 0.0 for name in SEQ_FEATURE_COLUMNS})
            continue
        arr = np.asarray(tokens, dtype=float)
        stats.append(
            {
                "seq_mean": float(np.mean(arr)),
                "seq_std": float(np.std(arr)),
                "seq_min": float(np.min(arr)),
                "seq_max": float(np.max(arr)),
                "seq_len": float(arr.size),
                "seq_sum": float(np.sum(arr)),
                "seq_median": float(np.median(arr)),
            }
        )
    return pl.DataFrame(stats)


# 지정된 컬럼을 안전하게 float64로 캐스팅한다.
def _cast_numeric_columns(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    cast_exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df.with_columns(cast_exprs) if cast_exprs else df


# 모델 학습/추론용 입력 테이블과 타깃 배열을 준비한다.
def prepare_features(
    df: pl.DataFrame,
    config: FeatureConfig,
    *,
    has_target: bool,
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """Prepare features for LightGBM training/inference.

    Returns pandas DataFrame (LightGBM input) and optionally numpy target array."""

    augmented_df = df
    if config.row_null_threshold is not None:
        base_cols = [c for c in df.columns if c != config.target_col]
        null_count_expr = pl.sum_horizontal(
            [pl.col(col).is_null().cast(pl.UInt16) for col in base_cols]
        ).alias("_row_null_count")
        augmented_df = augmented_df.with_columns(null_count_expr)
        augmented_df = augmented_df.with_columns(
            (
                pl.col("_row_null_count") >= config.row_null_threshold
            )
            .cast(pl.Float32)
            .alias(config.row_null_indicator_name)
        )
        augmented_df = augmented_df.drop("_row_null_count")

    for col in config.null_indicator_columns:
        if col in augmented_df.columns:
            name = f"{col}_is_null"
            augmented_df = augmented_df.with_columns(
                pl.col(col).is_null().cast(pl.Float32).alias(name)
            )

    feature_cols = [c for c in augmented_df.columns if c not in config.exclude_set]
    numeric_df = _cast_numeric_columns(augmented_df, feature_cols)
    seq_df = extract_sequence_features(df, config.seq_col)
    tabular = (
        numeric_df.select(feature_cols)
        .fill_null(config.fill_value)
        .to_pandas()
    )
    combined = pd.concat([tabular, seq_df.to_pandas()], axis=1)

    if has_target:
        target = (
            df[config.target_col]
            .cast(pl.Float32, strict=False)
            .fill_null(0.0)
            .to_numpy()
        )
        return combined, target
    return combined, None


# 설정에 따라 사용할 피처 컬럼 목록을 반환한다.
def select_feature_columns(df: pl.DataFrame, config: FeatureConfig) -> List[str]:
    return [c for c in df.columns if c not in config.exclude_set]
