from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.data.features import FeatureConfig, prepare_features
from src.model.metrics import compute_competition_metrics
from src.train.log_utils import MetricRecord, append_metric, default_run_id

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


@dataclass
# LightGBM 학습에 필요한 하이퍼파라미터와 경로 설정을 묶는다.
class LightGBMConfig:
    train_path: str
    target_col: str = "clicked"
    seq_col: str = "seq"
    test_size: float = 0.2
    random_state: int = 42
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    metric_period: int = 50
    learning_rate: float = 0.1
    num_leaves: int = 31
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    enable_wandb: bool = False
    notes: str = ""


@dataclass
# 학습이 끝난 후 결과 요약과 메타데이터를 담는다.
class TrainingResult:
    model: lgb.Booster
    run_id: str
    metrics: dict
    best_iteration: int
    feature_columns: list[str]
    train_shape: tuple[int, int]
    val_shape: tuple[int, int]


PARAM_KEYS = (
    "objective",
    "metric",
    "boosting_type",
    "num_leaves",
    "learning_rate",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "verbose",
    "random_state",
)


# LightGBM용 기본 파라미터 딕셔너리를 구성한다.
def _build_params(cfg: LightGBMConfig) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbose": -1,
        "random_state": cfg.random_state,
    }


# wandb 실행을 초기화하고 세션 핸들을 반환한다.
def _init_wandb(cfg: LightGBMConfig, run_id: str, params: dict) -> Optional["wandb.sdk.wandb_run.Run"]:
    if not cfg.enable_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb package is required but not installed")
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name or run_id,
        config={**params, "test_size": cfg.test_size},
        reinit=True,
    )
    return wandb_run


# LightGBM 모델을 학습하고 결과 요약을 반환한다.
def train_lightgbm(cfg: LightGBMConfig) -> TrainingResult:
    data_path = Path(cfg.train_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pl.read_parquet(data_path)
    feature_cfg = FeatureConfig(seq_col=cfg.seq_col, target_col=cfg.target_col)
    feature_cols = [c for c in df.columns if c not in feature_cfg.exclude_set]

    X, y = prepare_features(df, feature_cfg, has_target=True)
    stratify_labels = y.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
        stratify=stratify_labels,
    )

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = _build_params(cfg)

    run_id = default_run_id()
    wandb_run = _init_wandb(cfg, run_id, params)

    callbacks = [
        lgb.log_evaluation(period=cfg.metric_period),
        lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds),
    ]

    booster = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=cfg.num_boost_round,
        callbacks=callbacks,
    )

    best_iteration = booster.best_iteration or cfg.num_boost_round
    val_preds = booster.predict(X_val, num_iteration=best_iteration)
    comp_metrics = compute_competition_metrics(y_val, val_preds)
    metrics_dict = comp_metrics.to_dict()
    metrics_dict["best_iteration"] = best_iteration

    append_metric(
        MetricRecord(
            run_id=run_id,
            stage="stage0",
            dataset="validation",
            metrics=metrics_dict,
            notes=cfg.notes,
        )
    )

    if wandb_run is not None:
        wandb_run.log({
            "validation/ap": metrics_dict["ap"],
            "validation/wll": metrics_dict["wll"],
            "validation/competition_score": metrics_dict["competition_score"],
            "validation/best_iteration": best_iteration,
        })
        wandb_run.finish()

    return TrainingResult(
        model=booster,
        run_id=run_id,
        metrics=metrics_dict,
        best_iteration=best_iteration,
        feature_columns=feature_cols,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
    )
