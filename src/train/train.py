from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downsample import DownsampleConfig, create_downsampled_dataset
from src.inference import generate_submission
from src.train.lightgbm_runner import LightGBMConfig, train_lightgbm
from src.train.log_utils import report_root_disk_usage

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def parse_args() -> argparse.Namespace:
    # 학습 실행에 필요한 커맨드라인 인자를 정의한다.
    parser = argparse.ArgumentParser(description="Train LightGBM baseline model")
    parser.add_argument("--raw-train", default="data/train.parquet", help="Path to raw train parquet")
    parser.add_argument(
        "--train-path",
        default="data/processed/train_downsample_1_2.parquet",
        help="Path to downsampled training parquet",
    )
    parser.add_argument(
        "--ensure-downsample",
        action="store_true",
        help="Regenerate downsampled dataset before training",
    )
    parser.add_argument("--negative-multiplier", type=float, default=2.0, help="Negative sampling multiplier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--model-dir", default="model", help="Directory to save trained model")
    parser.add_argument("--config-dir", default="model", help="Directory to save config snapshot")
    parser.add_argument("--notes", default="", help="Optional notes for metric logging")
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Generate submission predictions after training",
    )
    parser.add_argument(
        "--inference-input",
        default="data/processed/test.parquet",
        help="Parquet file used for submission inference",
    )
    parser.add_argument(
        "--submission-path",
        default=None,
        help="Optional path for the generated submission CSV",
    )
    return parser.parse_args()


def ensure_wandb_login() -> None:
    # wandb API 키를 사용해 로그인을 보장한다.
    if wandb is None:
        raise ImportError("wandb package is not installed. Install wandb or disable --wandb option.")
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()  # fallback to interactive login


def summarize_config(args: argparse.Namespace, lgbm_cfg: LightGBMConfig) -> dict[str, float | int | str]:
    # 주요 학습/전처리 하이퍼파라미터를 정리해 로그 하단에 출력한다.
    return {
        "NEGATIVE_MULTIPLIER": args.negative_multiplier,
        "TEST_SIZE": lgbm_cfg.test_size,
        "LEARNING_RATE": lgbm_cfg.learning_rate,
        "NUM_LEAVES": lgbm_cfg.num_leaves,
        "NUM_BOOST_ROUND": lgbm_cfg.num_boost_round,
        "EARLY_STOPPING_ROUNDS": lgbm_cfg.early_stopping_rounds,
        "FEATURE_FRACTION": lgbm_cfg.feature_fraction,
        "BAGGING_FRACTION": lgbm_cfg.bagging_fraction,
        "BAGGING_FREQ": lgbm_cfg.bagging_freq,
        "RANDOM_SEED": lgbm_cfg.random_state,
    }


def main() -> None:
    # 다운샘플링부터 LightGBM 학습과 모델 저장까지의 전체 파이프라인을 실행한다.
    args = parse_args()

    downsample_path = Path(args.train_path)
    if args.ensure_downsample or not downsample_path.exists():
        cfg = DownsampleConfig(
            raw_path=args.raw_train,
            output_path=str(downsample_path),
            negative_multiplier=args.negative_multiplier,
            seed=args.seed,
        )
        create_downsampled_dataset(cfg, overwrite=args.ensure_downsample)

    if args.wandb:
        ensure_wandb_login()

    lgbm_cfg = LightGBMConfig(
        train_path=str(downsample_path),
        random_state=args.seed,
        enable_wandb=args.wandb,
        wandb_project="Toss-CTR-Competition" if args.wandb else None,
        wandb_entity="bkan-ai" if args.wandb else None,
        notes=args.notes,
    )

    result = train_lightgbm(lgbm_cfg)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"lgb_model_{timestamp}.txt"
    report_root_disk_usage()
    result.model.save_model(model_path.as_posix())

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"config_{timestamp}.json"

    payload = {
        "feature_columns": result.feature_columns,
        "train_shape": result.train_shape,
        "val_shape": result.val_shape,
        "metrics": result.metrics,
        "best_iteration": result.best_iteration,
        "train_path": str(downsample_path),
        "seed": args.seed,
    }
    report_root_disk_usage()
    config_path.write_text(json.dumps(payload, indent=2))

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Config saved to {config_path}")
    print(
        "📊 Validation Metrics - AP: {ap:.4f}, WLL: {wll:.4f}, Score: {score:.4f}".format(
            ap=result.metrics["ap"],
            wll=result.metrics["wll"],
            score=result.metrics["competition_score"],
        )
    )
    cfg_summary = summarize_config(args, lgbm_cfg)
    print(f"CFG = {cfg_summary}")

    if args.run_inference:
        submission_path = args.submission_path or f"submission/{result.run_id}_preds.csv"
        submission_file = generate_submission(
            model_path=model_path.as_posix(),
            input_path=args.inference_input,
            output_path=submission_path,
            run_id=result.run_id,
        )
        print(f"📤 Submission generated at {submission_file}")


if __name__ == "__main__":
    main()
