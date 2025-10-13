"""다운샘플링 데이터셋을 생성하기 위한 도구 모듈."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl

DEFAULT_STRAT_COLS: List[str] = ["inventory_id", "day_of_week", "hour", "device"]


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTR 학습용 다운샘플 데이터 생성")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/train.parquet"),
        help="원본 학습 데이터 Parquet 경로",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/samples/train_downsample_phase_a.parquet"),
        help="생성한 다운샘플 데이터를 저장할 경로",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("logs/downsampling/phase_a_metadata.json"),
        help="생성 결과 메타데이터 저장 경로",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=4.0,
        help="양성에 비해 음성을 최대 몇 배까지 유지할지 비율",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링용 난수 시드",
    )
    parser.add_argument(
        "--strat-cols",
        nargs="*",
        default=None,
        help="층화 기준으로 사용할 컬럼 목록(생략 시 기본값)",
    )
    return parser.parse_args()


# 실제 데이터에 존재하는 층화 컬럼을 결정한다.
def resolve_strat_columns(frame: pl.DataFrame, candidates: Iterable[str]) -> List[str]:
    available = [col for col in candidates if col in frame.columns]
    if not available:
        raise ValueError("층화에 사용할 수 있는 컬럼이 존재하지 않습니다.")
    return available


# 음성 데이터를 층화 비율에 맞춰 샘플링한다.
def sample_negative_groups(
    negative_frame: pl.DataFrame,
    strat_cols: List[str],
    fraction: float,
    seed: int,
) -> pl.DataFrame:
    concat_series = [pl.col(col).cast(pl.Utf8) for col in strat_cols]
    negative_frame = negative_frame.with_columns(
        pl.concat_str(concat_series, separator="|").alias("__stratum__")
    )

    sampled_groups: List[pl.DataFrame] = []
    for _, group in negative_frame.group_by("__stratum__"):
        take = max(1, min(group.height, int(round(group.height * fraction))))
        sampled_groups.append(group.sample(n=take, with_replacement=False, seed=seed))

    if not sampled_groups:
        raise ValueError("층화 샘플 결과가 비어 있습니다.")

    sampled = pl.concat(sampled_groups, how="vertical")
    return sampled.drop("__stratum__")


# 주어진 프레임에서 층화된 음성 샘플을 추출한다.
def sample_negatives(
    frame: pl.DataFrame,
    strat_cols: List[str],
    target_ratio: float,
    seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, dict]:
    negative_frame = frame.filter(pl.col("clicked") == 0)
    positive_frame = frame.filter(pl.col("clicked") == 1)

    total_pos = positive_frame.height
    total_neg = negative_frame.height
    if total_pos == 0 or total_neg == 0:
        raise ValueError("양성 또는 음성 샘플 수가 0입니다.")

    target_neg = min(int(total_pos * target_ratio), total_neg)
    fraction = target_neg / total_neg

    sampled_negative = sample_negative_groups(
        negative_frame=negative_frame,
        strat_cols=strat_cols,
        fraction=fraction,
        seed=seed,
    )

    stats = {
        "total_pos": total_pos,
        "total_neg": total_neg,
        "target_neg": target_neg,
        "actual_neg": sampled_negative.height,
        "ratio": sampled_negative.height / total_pos,
    }

    return positive_frame, sampled_negative, stats


# 생성된 데이터와 메타데이터를 저장한다.
def save_outputs(
    positives: pl.DataFrame,
    negatives: pl.DataFrame,
    output_path: Path,
    metadata_path: Path,
    metadata: dict,
    seed: int,
) -> None:
    combined = pl.concat([positives, negatives], how="vertical")
    combined = combined.sample(n=combined.height, with_replacement=False, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path, compression="zstd")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# 전체 실행 흐름을 담당한다.
def main() -> None:
    args = parse_args()
    train_path = args.train_path
    if not train_path.exists():
        raise FileNotFoundError(f"학습 데이터가 존재하지 않습니다: {train_path}")

    frame = pl.read_parquet(train_path)
    strat_cols = resolve_strat_columns(frame, args.strat_cols or DEFAULT_STRAT_COLS)
    positives, sampled_negatives, stats = sample_negatives(
        frame=frame,
        strat_cols=strat_cols,
        target_ratio=args.negative_ratio,
        seed=args.seed,
    )

    metadata = {
        "train_path": str(train_path),
        "output_path": str(args.output_path),
        "metadata_path": str(args.metadata_path),
        "negative_ratio": args.negative_ratio,
        "strat_columns": strat_cols,
        "random_seed": args.seed,
    }
    metadata.update(stats)

    save_outputs(
        positives=positives,
        negatives=sampled_negatives,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        metadata=metadata,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
