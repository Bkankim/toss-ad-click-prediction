from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

FLOAT_DTYPES = {pl.Float32, pl.Float64}


@dataclass
class FeatureConfig:
    seq_col: str = "seq"
    target_col: str = "clicked"
    exclude_columns: Tuple[str, ...] = ("clicked", "seq", "ID")
    fill_value: float = 0.0

    @property
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


def _cast_numeric_columns(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    cast_exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df.with_columns(cast_exprs) if cast_exprs else df


def prepare_features(
    df: pl.DataFrame,
    config: FeatureConfig,
    *,
    has_target: bool,
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """Prepare features for LightGBM training/inference.

    Returns pandas DataFrame (LightGBM input) and optionally numpy target array."""

    feature_cols = [c for c in df.columns if c not in config.exclude_set]
    numeric_df = _cast_numeric_columns(df, feature_cols)
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


def select_feature_columns(df: pl.DataFrame, config: FeatureConfig) -> List[str]:
    return [c for c in df.columns if c not in config.exclude_set]
