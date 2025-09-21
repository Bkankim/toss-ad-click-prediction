from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl


@dataclass
# Null 분석에 필요한 공통 설정을 묶는다.
class NullAnalysisConfig:
    data_path: str
    feature_col: str = "feat_e_3"
    target_col: str = "clicked"
    high_null_threshold: int = 95


# Parquet을 지연 로딩해 대규모 데이터도 메모리 효율적으로 다룬다.
def _scan_dataset(cfg: NullAnalysisConfig) -> pl.LazyFrame:
    return pl.scan_parquet(cfg.data_path)


# feat_e_3 결측 비율을 전체·타깃별로 집계한다.
def feat_e3_overview(cfg: NullAnalysisConfig) -> pl.DataFrame:
    lf = _scan_dataset(cfg)
    feature = pl.col(cfg.feature_col)
    target = pl.col(cfg.target_col)
    overall = lf.select(
        [
            pl.len().alias("total_rows"),
            feature.is_null().sum().alias("null_count"),
            feature.is_null().mean().alias("null_ratio"),
        ]
    )
    by_target = (
        lf.group_by(cfg.target_col)
        .agg(
            [
                pl.len().alias("rows"),
                feature.is_null().sum().alias("null_count"),
                feature.is_null().mean().alias("null_ratio"),
            ]
        )
        .sort(cfg.target_col)
    )
    return pl.concat([overall, by_target], how="diagonal_relaxed").collect(engine="streaming")


# 지정된 컬럼 단위로 feat_e_3 결측 통계를 계산한다.
def feat_e3_by_column(cfg: NullAnalysisConfig, column: str, top_n: Optional[int] = 10) -> pl.DataFrame:
    lf = _scan_dataset(cfg)
    feature = pl.col(cfg.feature_col)
    result = (
        lf.group_by(column)
        .agg(
            [
                pl.len().alias("rows"),
                feature.is_null().sum().alias("null_count"),
                feature.is_null().mean().alias("null_ratio"),
            ]
        )
        .filter(pl.col("rows") > 0)
        .sort(pl.col("null_ratio"), descending=True)
    )
    if top_n is not None:
        result = result.head(top_n)
    return result.collect(engine="streaming")


# 각 행의 Null 개수를 계산해 추가 컬럼으로 붙인다.
def _with_null_count(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    columns = [pl.col(name).is_null().cast(pl.UInt16) for name in schema]
    null_count_expr = pl.sum_horizontal(columns).alias("null_count")
    return lf.with_columns(null_count_expr)


# 행 기준 Null 개수 통계를 요약한다.
def high_null_overview(cfg: NullAnalysisConfig) -> pl.DataFrame:
    lf = _with_null_count(_scan_dataset(cfg))
    threshold = cfg.high_null_threshold
    target = pl.col(cfg.target_col)
    flag = (pl.col("null_count") >= threshold).alias("high_null")
    summary = lf.with_columns(flag).select(
        [
            pl.len().alias("total_rows"),
            pl.col("high_null").sum().alias("high_null_rows"),
            (pl.col("high_null").mean()).alias("high_null_ratio"),
        ]
    )
    by_target = (
        lf.filter(pl.col("null_count") >= threshold)
        .group_by(cfg.target_col)
        .agg(
            [
                pl.len().alias("rows"),
                target.cast(pl.Float64).mean().alias("target_mean"),
            ]
        )
        .sort(cfg.target_col)
    )
    return pl.concat([summary, by_target], how="diagonal_relaxed").collect(engine="streaming")


# 높은 Null 행을 특정 컬럼 기준으로 집계한다.
def high_null_by_column(
    cfg: NullAnalysisConfig, column: str, top_n: Optional[int] = 10
) -> pl.DataFrame:
    lf = _with_null_count(_scan_dataset(cfg))
    threshold = cfg.high_null_threshold
    filtered = lf.filter(pl.col("null_count") >= threshold)
    result = (
        filtered.group_by(column)
        .agg(
            [
                pl.len().alias("rows"),
                pl.col(cfg.target_col).cast(pl.Float64).mean().alias("target_mean"),
                pl.col("null_count").mean().alias("avg_nulls"),
            ]
        )
        .filter(pl.col("rows") > 0)
        .sort(pl.col("rows"), descending=True)
    )
    if top_n is not None:
        result = result.head(top_n)
    return result.collect(engine="streaming")
