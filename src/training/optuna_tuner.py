"""Optuna 기반 LightGBM 하이퍼파라미터 튜너."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from .phase_a import calibrate_predictions
from .lightgbm_baseline import (
    blended_score,
    load_dataset,
    prepare_features,
    prepare_test_features,
    save_metrics,
    train_lightgbm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna로 LightGBM 최적화")
    parser.add_argument("--train-path", type=Path, required=True, help="학습 데이터 Parquet 경로")
    parser.add_argument("--test-path", type=Path, required=True, help="테스트 데이터 Parquet 경로")
    parser.add_argument("--models-dir", type=Path, required=True, help="최종 베스트 모델 저장 디렉터리")
    parser.add_argument("--metrics-json", type=Path, required=True, help="최종 메트릭 저장 경로")
    parser.add_argument("--metrics-log-dir", type=Path, default=Path("logs/metrics"))
    parser.add_argument("--submission-dir", type=Path, required=True, help="최종 제출 파일 디렉터리")
    parser.add_argument("--oof-path", type=Path, required=True, help="최종 OOF 저장 경로")
    parser.add_argument("--calibration-path", type=Path, required=True, help="최종 캘리브레이션 요약 경로")

    parser.add_argument("--study-name", type=str, default="lgbm_optuna")
    parser.add_argument("--storage", type=str, default="sqlite:///logs/optuna/lgbm_random.db")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device-type", type=str, default="cuda")

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-prefix", type=str, default="optuna")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_search_space(trial: optuna.Trial, base_params: Dict[str, float]) -> Dict[str, float]:
    params = base_params.copy()
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 32, 512, step=32),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 2000, step=100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }
    )
    return params


def initialize_storage(storage: str) -> None:
    if storage.startswith("sqlite"):
        path = Path(storage.replace("sqlite:///", ""))
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_log_dir.mkdir(parents=True, exist_ok=True)
    args.submission_dir.mkdir(parents=True, exist_ok=True)
    args.oof_path.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_path.parent.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = load_dataset(args.train_path, args.test_path, sample_size=None)
    X, y, feature_names = prepare_features(train_frame)
    X_test, ids = prepare_test_features(test_frame, feature_names)

    base_params: Dict[str, float] = {
        "objective": "binary",
        "metric": ["auc"],
        "device_type": args.device_type,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": args.seed,
        "force_row_wise": False,
    }

    wandb_enabled = args.wandb_project is not None and wandb is not None

    def objective(trial: optuna.Trial) -> float:
        params = build_search_space(trial, base_params)
        models, metrics, _ = train_lightgbm(
            X=X,
            y=y,
            feature_names=feature_names,
            num_folds=args.folds,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping,
            output_dir=None,
        )
        overall = metrics["overall"]

        if wandb_enabled:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=f"{args.wandb_run_prefix}-trial-{trial.number}",
                reinit=True,
                config=params,
            )
            try:
                wandb.log(
                    {
                        "trial_score": overall["score"],
                        "trial_ap": overall["ap"],
                        "trial_wll": overall["wll"],
                        "trial_logloss": overall["logloss"],
                    }
                )
            finally:
                run.finish()

        trial.set_user_attr("ap", overall["ap"])
        trial.set_user_attr("wll", overall["wll"])
        trial.set_user_attr("logloss", overall["logloss"])
        return overall["score"]

    initialize_storage(args.storage)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_params = base_params.copy()
    best_params.update(study.best_params)

    models, metrics, oof_pred = train_lightgbm(
        X=X,
        y=y,
        feature_names=feature_names,
        num_folds=args.folds,
        params=best_params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        output_dir=args.models_dir,
    )

    preds, calibration_metrics, calibration_summary, platt_model, isotonic_model = calibrate_predictions(y, oof_pred)
    metrics["calibration"] = calibration_metrics
    save_metrics(metrics, args.metrics_json)

    pl.DataFrame(
        {
            "clicked": y,
            "raw_pred": preds["raw"],
            "platt_pred": preds["platt"],
            "isotonic_pred": preds["isotonic"],
        }
    ).write_parquet(args.oof_path.as_posix(), compression="zstd")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_lgbm_optuna")
    rows = []
    for stage, metric in calibration_metrics.items():
        rows.append(
            {
                "run_id": run_id,
                "stage": f"optuna_{stage}",
                "dataset": args.train_path.stem,
                "ap": metric["ap"],
                "logloss": metrics["overall"]["logloss"],
                "wll": metric["wll"],
                "competition_score": metric["score"],
                "notes": f"Optuna tuned LightGBM (device_type={args.device_type})",
            }
        )
    pl.DataFrame(rows).write_csv(args.metrics_log_dir / f"{run_id}.csv")
    args.calibration_path.write_text(json.dumps(calibration_summary, indent=2), encoding="utf-8")

    test_raw = np.mean(
        [model.predict(X_test, num_iteration=model.best_iteration) for model in models],
        axis=0,
    ).astype(np.float32)

    raw_df = pl.DataFrame({"ID": ids, "clicked": test_raw}) if ids is not None else pl.DataFrame({"clicked": test_raw})
    raw_df.write_csv(args.submission_dir / "optuna_raw.csv")

    platt_test = platt_model.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    iso_test = isotonic_model.transform(test_raw)

    if ids is not None:
        pl.DataFrame({"ID": ids, "clicked": platt_test}).write_csv(args.submission_dir / "optuna_platt.csv")
        pl.DataFrame({"ID": ids, "clicked": iso_test}).write_csv(args.submission_dir / "optuna_isotonic.csv")
    else:
        pl.DataFrame({"clicked": platt_test}).write_csv(args.submission_dir / "optuna_platt.csv")
        pl.DataFrame({"clicked": iso_test}).write_csv(args.submission_dir / "optuna_isotonic.csv")

    if wandb_enabled:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{args.wandb_run_prefix}-final",
            reinit=True,
            config=best_params,
        )
        try:
            wandb.log(
                {
                    "final_score": metrics["overall"]["score"],
                    "final_ap": metrics["overall"]["ap"],
                    "final_wll": metrics["overall"]["wll"],
                    "final_logloss": metrics["overall"]["logloss"],
                }
            )
        finally:
            run.finish()

    print("Best trial:", study.best_trial.number)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Best score:", study.best_value)


if __name__ == "__main__":
    main()
