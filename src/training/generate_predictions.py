"""Utility to generate LightGBM predictions for the full dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import polars as pl


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM 모델로 전체 데이터 예측")
    parser.add_argument("--data-path", type=Path, required=True, help="전처리된 Parquet 경로")
    parser.add_argument("--metadata-path", type=Path, required=True, help="메타데이터 JSON 경로")
    parser.add_argument("--models-dir", type=Path, required=True, help="LightGBM 모델이 저장된 디렉터리")
    parser.add_argument("--output-path", type=Path, required=True, help="예측을 저장할 Parquet 경로")
    parser.add_argument("--batch-size", type=int, default=500_000, help="배치 단위 예측 크기")
    return parser.parse_args()


# LightGBM 모델을 로드한다.
def load_models(models_dir: Path) -> List[lgb.Booster]:
    boosters: List[lgb.Booster] = []
    for fold in range(1, 6):
        model_path = models_dir / f"fold_{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        boosters.append(lgb.Booster(model_file=str(model_path)))
    return boosters


# 배치 단위로 데이터를 로드하고 예측한다.
def generate_predictions(
    data_path: Path,
    feature_cols: List[str],
    boosters: List[lgb.Booster],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = pl.scan_parquet(data_path.as_posix()).select(pl.len()).collect().item()
    preds_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for offset in range(0, total_rows, batch_size):
        batch = (
            pl.scan_parquet(data_path.as_posix())
            .slice(offset, batch_size)
            .select(feature_cols + ["clicked"])
            .collect()
        )
        if batch.is_empty():
            break
        X = batch.select(feature_cols).to_numpy().astype(np.float32, copy=False)
        batch_preds = np.mean(
            [booster.predict(X, num_iteration=booster.best_iteration) for booster in boosters],
            axis=0,
        )
        preds_list.append(batch_preds.astype(np.float32))
        labels_list.append(batch["clicked"].to_numpy().astype(np.float32))

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return preds, labels


# 전체 실행 흐름을 수행한다.
def main() -> None:
    args = parse_args()

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    _ = metadata.get("feature_columns")  # 정보 기록 유지용

    boosters = load_models(args.models_dir)
    model_feature_cols = boosters[0].feature_name()
    schema = pl.read_parquet(args.data_path.as_posix(), n_rows=1).schema
    feature_cols = [col for col in model_feature_cols if col in schema]
    if len(feature_cols) != len(model_feature_cols):
        missing = set(model_feature_cols) - set(feature_cols)
        if missing:
            raise ValueError(f"데이터에 존재하지 않는 피처가 있습니다: {sorted(missing)}")

    preds, labels = generate_predictions(
        data_path=args.data_path,
        feature_cols=feature_cols,
        boosters=boosters,
        batch_size=args.batch_size,
    )

    output = pl.DataFrame({"clicked": labels, "raw_pred": preds})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(args.output_path.as_posix(), compression="zstd")


if __name__ == "__main__":
    main()
