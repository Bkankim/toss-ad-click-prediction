"""LightGBM 베이스라인 OOF 학습 스크립트."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", 42))
DEFAULT_FOLDS = 5
DEFAULT_SAMPLE_SIZE = 500_000


# 데이터셋을 로드한다.
def load_dataset(
    train_path: Path,
    test_path: Path,
    sample_size: Optional[int],
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if sample_size and sample_size > 0:
        train_frame = pl.read_parquet(train_path.as_posix(), n_rows=sample_size)
    else:
        train_frame = pl.read_parquet(train_path.as_posix())

    test_frame = pl.read_parquet(test_path.as_posix())

    drop_cols = set(drop_columns or [])
    drop_cols.update({"seq"})

    valid_drop_cols = [col for col in drop_cols if col in train_frame.columns]
    if valid_drop_cols:
        train_frame = train_frame.drop(valid_drop_cols)
    if valid_drop_cols:
        test_frame = test_frame.drop([col for col in valid_drop_cols if col in test_frame.columns])

    return train_frame, test_frame


# 숫자형 피처만 선별하여 NumPy 배열로 변환한다.
def polars_to_numpy(frame: pl.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    exclude = exclude or []
    numeric_cols = [
        col
        for col, dtype in zip(frame.columns, frame.dtypes)
        if col not in exclude and dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    if not numeric_cols:
        raise ValueError("No numeric columns available after filtering")

    numeric_frame = frame.select([pl.col(col).cast(pl.Float32) for col in numeric_cols])
    return numeric_frame.to_numpy(), numeric_cols


# pandas DataFrame으로 변환하고 문자열 컬럼을 제거한다.
def prepare_features(
    frame: pl.DataFrame,
    target_column: str = "clicked",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = frame[target_column].to_numpy()
    feature_frame = frame.drop(target_column)
    X, feature_names = polars_to_numpy(feature_frame)
    return X.astype(np.float32), y.astype(np.float32), feature_names


# 테스트 데이터 준비.
def prepare_test_features(
    frame: pl.DataFrame,
    feature_names: List[str],
    id_column: str = "ID",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ids = None
    if id_column in frame.columns:
        ids = frame[id_column].to_numpy()
        frame = frame.drop(id_column)

    numeric_cols = [col for col in feature_names if col in frame.columns]
    missing_cols = [col for col in feature_names if col not in numeric_cols]

    numeric_frame = frame.select([pl.col(col).cast(pl.Float32) for col in numeric_cols])
    X_numeric = numeric_frame.to_numpy() if numeric_cols else np.empty((frame.height, 0), dtype=np.float32)

    if missing_cols:
        zeros = np.zeros((frame.height, len(missing_cols)), dtype=np.float32)
        X = np.concatenate([X_numeric, zeros], axis=1)
    else:
        X = X_numeric

    return X.astype(np.float32), ids


# Weighted LogLoss를 계산한다.
def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred.astype(np.float64), eps, 1 - eps)
    pos_weight = float((y_true == 0).sum()) / max(float((y_true == 1).sum()), 1.0)
    weights = np.where(y_true == 1, pos_weight, 1.0)
    loss = -(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    return loss.sum() / weights.sum()


# 일반 LogLoss를 계산한다.
def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred.astype(np.float64), eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(np.mean(loss))


# 혼합 점수를 계산한다.
def blended_score(ap: float, wll: float) -> float:
    return 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))


# LightGBM 모델을 학습한다.
def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    num_folds: int,
    params: Dict[str, float],
    num_boost_round: int,
    early_stopping_rounds: int,
    output_dir: Optional[Path] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[List[lgb.Booster], Dict[str, float], np.ndarray]:
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=GLOBAL_SEED)

    oof_pred = np.zeros(len(y), dtype=np.float32)
    models: List[lgb.Booster] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        train_weight = sample_weight[train_idx] if sample_weight is not None else None
        valid_weight = sample_weight[valid_idx] if sample_weight is not None else None

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weight,
            feature_name=feature_names,
            free_raw_data=False,
        )
        valid_dataset = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=valid_weight,
            feature_name=feature_names,
            free_raw_data=False,
        )

        booster = lgb.train(
            params,
            train_set=train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        models.append(booster)
        valid_pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
        oof_pred[valid_idx] = valid_pred.astype(np.float32)

        ap = average_precision_score(y_valid, valid_pred)
        ll = logloss(y_valid, valid_pred)
        wll = weighted_logloss(y_valid, valid_pred)
        score = blended_score(ap, wll)
        fold_metrics.append({"fold": fold, "ap": ap, "logloss": ll, "wll": wll, "score": score})

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"fold_{fold}.txt"
            booster.save_model(model_path.as_posix())

    ap_full = average_precision_score(y, oof_pred)
    ll_full = logloss(y, oof_pred)
    wll_full = weighted_logloss(y, oof_pred)
    score_full = blended_score(ap_full, wll_full)

    metrics = {
        "folds": fold_metrics,
        "overall": {"ap": ap_full, "logloss": ll_full, "wll": wll_full, "score": score_full},
    }
    return models, metrics, oof_pred


# 제출 파일을 생성한다.
def create_submission(
    models: List[lgb.Booster],
    X_test: np.ndarray,
    ids: Optional[np.ndarray],
    output_path: Path,
) -> None:
    preds = np.mean([model.predict(X_test, num_iteration=model.best_iteration) for model in models], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ids is not None:
        submission_df = pd.DataFrame({"ID": ids, "clicked": preds})
    else:
        submission_df = pd.DataFrame({"clicked": preds})

    submission_df.to_csv(output_path, index=False)


# 로그를 파일로 저장한다.
def save_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# CLI 실행 흐름.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM 베이스라인 학습")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="훈련에 사용할 최대 행 수 (-1이면 전체)")
    parser.add_argument("--num-boost-round", type=int, default=300, help="LightGBM 부스팅 라운드")
    parser.add_argument("--early-stopping", type=int, default=50, help="Early stopping 라운드")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Stratified K-Fold 수")
    parser.add_argument("--models-dir", type=Path, default=Path("models/lightgbm_baseline"), help="모델 저장 디렉터리")
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/lightgbm_baseline_metrics.json"), help="메트릭 저장 경로")
    parser.add_argument("--submission-path", type=Path, default=Path("submission/lightgbm_baseline.csv"), help="제출 파일 저장 경로")
    parser.add_argument(
        "--device-type",
        type=str,
        default=os.getenv("LIGHTGBM_DEVICE", "cuda"),
        help="LightGBM device_type (cpu/cuda/gpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = {
        "TRAIN_PATH": Path(os.getenv("TRAIN_PATH", "/Competition/CTR/toss-ad-click-prediction/data/train.parquet")),
        "TEST_PATH": Path(os.getenv("TEST_PATH", "/Competition/CTR/toss-ad-click-prediction/data/test.parquet")),
    }

    train_frame, test_frame = load_dataset(paths["TRAIN_PATH"], paths["TEST_PATH"], args.sample_size)
    X, y, feature_names = prepare_features(train_frame)
    num_folds = args.folds

    params = {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": GLOBAL_SEED,
        "device_type": args.device_type,
        "force_row_wise": False,
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)

    models, metrics, _ = train_lightgbm(
        X,
        y,
        feature_names,
        num_folds,
        params,
        args.num_boost_round,
        args.early_stopping,
        args.models_dir,
    )

    print("\nOOF Metrics:")
    overall = metrics["overall"]
    print(f"AP: {overall['ap']:.6f}")
    print(f"LogLoss: {overall['logloss']:.6f}")
    print(f"Weighted LogLoss: {overall['wll']:.6f}")
    print(f"Blended Score: {overall['score']:.6f}")

    X_test, ids = prepare_test_features(test_frame, feature_names)
    create_submission(models, X_test, ids, args.submission_path)
    save_metrics(metrics, args.metrics_path)


if __name__ == "__main__":
    main()
