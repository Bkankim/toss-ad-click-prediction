"""다운샘플링 Phase A LightGBM 파이프라인 실행 스크립트."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .lightgbm_baseline import (
    blended_score,
    load_dataset,
    prepare_features,
    prepare_test_features,
    save_metrics,
    train_lightgbm,
    weighted_logloss,
)


# CLI 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A LightGBM 학습 및 보정")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(os.getenv("PHASE_A_TRAIN_PATH", "data/samples/train_downsample_phase_a.parquet")),
        help="Phase A 학습에 사용할 Parquet 경로",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(os.getenv("TEST_PATH", "data/test.parquet")),
        help="추론용 테스트 Parquet 경로",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/lightgbm_phase_a"),
        help="학습된 모델 저장 디렉터리",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("logs/lightgbm_phase_a_metrics.json"),
        help="메트릭 JSON 저장 경로",
    )
    parser.add_argument(
        "--metrics-log-dir",
        type=Path,
        default=Path("logs/metrics"),
        help="요약 CSV를 저장할 디렉터리",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path("submission"),
        help="제출 파일 저장 디렉터리",
    )
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("logs/downsampling/phase_a_oof.parquet"),
        help="OOF 예측 저장 경로",
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path("logs/downsampling/phase_a_calibration.json"),
        help="캘리브레이션 요약 저장 경로",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=300,
        help="LightGBM 부스팅 라운드 수",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="Early stopping 라운드 수",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default=os.getenv("LIGHTGBM_DEVICE", "cuda"),
        help="LightGBM device_type (cpu/gpu/cuda)",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=1.0,
        help="LightGBM scale_pos_weight 값",
    )
    parser.add_argument(
        "--day7-weight",
        type=float,
        default=1.0,
        help="day_of_week == 7 샘플에 부여할 추가 가중치 (기본 1.0)",
    )
    parser.add_argument(
        "--day7-hour-threshold",
        type=int,
        default=-1,
        help="day_of_week == 7 이면서 hour <= 값인 샘플에만 추가 가중치 적용 (-1이면 비활성)",
    )
    parser.add_argument(
        "--day7-hour-weight",
        type=float,
        default=1.0,
        help="day7-hour-threshold 조건을 만족하는 샘플에 부여할 가중치",
    )
    return parser.parse_args()


# LightGBM 학습 파라미터를 구성한다.
def build_params(device_type: str, scale_pos_weight: float) -> Dict[str, float]:
    return {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
        "device_type": device_type,
        "force_row_wise": False,
        "scale_pos_weight": scale_pos_weight,
    }


# 메트릭을 계산한다.
def _compute_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ap = average_precision_score(y_true, pred)
    wll = weighted_logloss(y_true, pred)
    score = blended_score(ap, wll)
    return {"ap": float(ap), "wll": float(wll), "score": float(score)}


# Platt/Isotonic 보정을 수행하고 결과를 반환한다.
def calibrate_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Dict[str, float]],
    Dict[str, float],
    LogisticRegression,
    IsotonicRegression,
]:
    platt = LogisticRegression(max_iter=1000)
    platt.fit(pred.reshape(-1, 1), y_true)
    platt_oof = platt.predict_proba(pred.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(pred, y_true)
    isotonic_oof = isotonic.transform(pred)

    metrics = {
        "raw": _compute_metrics(y_true, pred),
        "platt": _compute_metrics(y_true, platt_oof),
        "isotonic": _compute_metrics(y_true, isotonic_oof),
    }

    calibration_summary = {
        "platt_coef": platt.coef_.ravel().tolist(),
        "platt_intercept": platt.intercept_.tolist(),
        "isotonic_x_min": float(isotonic.X_min_),
        "isotonic_x_max": float(isotonic.X_max_),
    }

    preds = {
        "raw": pred,
        "platt": platt_oof,
        "isotonic": isotonic_oof,
    }

    return preds, metrics, calibration_summary, platt, isotonic


# Phase A 파이프라인을 실행한다.
def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"학습 데이터가 존재하지 않습니다: {args.train_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"테스트 데이터가 존재하지 않습니다: {args.test_path}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_log_dir.mkdir(parents=True, exist_ok=True)
    args.submission_dir.mkdir(parents=True, exist_ok=True)
    args.oof_path.parent.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = load_dataset(args.train_path, args.test_path, sample_size=None)

    sample_weight: Optional[np.ndarray] = None
    use_day7_weight = args.day7_weight != 1.0
    use_day7_hour_weight = args.day7_hour_threshold >= 0 and args.day7_hour_weight != 1.0

    if use_day7_weight or use_day7_hour_weight:
        if "day_of_week" not in train_frame.columns:
            raise ValueError("day_of_week 컬럼이 없어 가중치를 적용할 수 없습니다.")

        day_np = train_frame["day_of_week"].cast(pl.Int32).to_numpy()
        sample_weight = np.ones(day_np.shape[0], dtype=np.float32)
        day7_mask = day_np == 7
        if not np.any(day7_mask):
            raise ValueError("day=7 샘플이 존재하지 않아 가중치를 적용할 수 없습니다.")

        if use_day7_weight:
            sample_weight[day7_mask] *= np.float32(args.day7_weight)

        if use_day7_hour_weight:
            if "hour" not in train_frame.columns:
                raise ValueError("hour 컬럼이 없어 day7 시간대 가중치를 적용할 수 없습니다.")
            hour_np = train_frame["hour"].cast(pl.Int32).to_numpy()
            early_mask = hour_np <= args.day7_hour_threshold
            combined_mask = day7_mask & early_mask
            if not np.any(combined_mask):
                print(
                    f"경고: hour <= {args.day7_hour_threshold} 조건을 만족하는 day7 샘플이 없습니다."
                )
            else:
                sample_weight[combined_mask] *= np.float32(args.day7_hour_weight)
                print(
                    "Applying day7 hour weight "
                    f"threshold={args.day7_hour_threshold}, weight={args.day7_hour_weight:.3f}, "
                    f"affected={combined_mask.sum()}"
                )

        print(
            f"Final weight stats -> min={sample_weight.min():.3f}, max={sample_weight.max():.3f}, "
            f"mean={sample_weight.mean():.3f}"
        )

    X, y, feature_names = prepare_features(train_frame)
    X_test, ids = prepare_test_features(test_frame, feature_names)

    params = build_params(args.device_type, args.scale_pos_weight)

    models, metrics, oof_pred = train_lightgbm(
        X=X,
        y=y,
        feature_names=feature_names,
        num_folds=5,
        params=params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        output_dir=args.models_dir,
        sample_weight=sample_weight,
    )

    calibrated_preds, calibration_metrics, calibration_summary, platt_model, isotonic_model = calibrate_predictions(y, oof_pred)
    metrics["calibration"] = calibration_metrics
    save_metrics(metrics, args.metrics_json)

    pl.DataFrame(
        {
            "clicked": y,
            "raw_pred": calibrated_preds["raw"],
            "platt_pred": calibrated_preds["platt"],
            "isotonic_pred": calibrated_preds["isotonic"],
        }
    ).write_parquet(args.oof_path, compression="zstd")

    raw_test = np.mean([model.predict(X_test, num_iteration=model.best_iteration) for model in models], axis=0)
    pl.DataFrame({"ID": ids, "clicked": raw_test}).write_csv(args.submission_dir / "lightgbm_phase_a_raw.csv")
    pl.DataFrame({"ID": ids, "clicked": platt_model.predict_proba(raw_test.reshape(-1, 1))[:, 1]}).write_csv(
        args.submission_dir / "lightgbm_phase_a_platt.csv"
    )
    pl.DataFrame({"ID": ids, "clicked": isotonic_model.transform(raw_test)}).write_csv(
        args.submission_dir / "lightgbm_phase_a_isotonic.csv"
    )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_lgbm_phase_a")
    rows = []
    for stage, metric in calibration_metrics.items():
        rows.append(
            {
                "run_id": run_id,
                "stage": f"phase_a_{stage}",
                "dataset": "train_downsample_phase_a",
                "ap": metric["ap"],
                "logloss": metrics["overall"]["logloss"],
                "wll": metric["wll"],
                "competition_score": metric["score"],
                "notes": f"Phase A LightGBM calibration (device_type={args.device_type})",
            }
        )

    pl.DataFrame(rows).write_csv(args.metrics_log_dir / f"{run_id}.csv")
    args.calibration_path.write_text(json.dumps(calibration_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
