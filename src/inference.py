from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import pandas as pd
import polars as pl

from src.data.features import FeatureConfig, prepare_features
from src.train.log_utils import (
    default_run_id,
    ensure_artifact_directory,
    report_root_disk_usage,
)


# 제출 파일 생성을 위한 명령행 인자를 정의한다.
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inference pipeline."""

    parser = argparse.ArgumentParser(description="Generate submission predictions with LightGBM model")
    parser.add_argument("--model-path", required=True, help="Path to the trained LightGBM model")
    parser.add_argument(
        "--input",
        default="data/processed/test.parquet",
        help="Path to inference parquet file",
    )
    parser.add_argument(
        "--output",
        default="submission/preds.csv",
        help="Where to write the submission CSV",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier to align artifacts",
    )
    return parser.parse_args()


# 학습된 LightGBM 모델을 로드한다.
def load_model(model_path: str) -> lgb.Booster:
    """Load a LightGBM booster from disk."""

    return lgb.Booster(model_file=model_path)


# 추론에 사용할 피처 행렬과 ID 시리즈를 준비한다.
def prepare_inference_dataset(path: str, feature_cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature dataframe and ID series for inference."""

    dataset = pl.read_parquet(path)
    if "ID" not in dataset.columns:
        raise ValueError("Inference dataset must contain an 'ID' column")
    features, _ = prepare_features(dataset, feature_cfg, has_target=False)
    id_series = dataset["ID"].to_pandas()
    return features, id_series


# 학습된 모델을 사용해 제출용 CSV를 생성한다.
def generate_submission(
    model_path: str,
    input_path: str,
    output_path: str,
    *,
    run_id: Optional[str] = None,
) -> Path:
    """Create submission file by scoring the input parquet with the trained model."""

    booster = load_model(model_path)
    feature_cfg = FeatureConfig()
    features, ids = prepare_inference_dataset(input_path, feature_cfg)

    predictions = booster.predict(features, num_iteration=booster.best_iteration or booster.num_trees())

    submission_dir = Path(output_path).parent
    submission_dir.mkdir(parents=True, exist_ok=True)

    report_root_disk_usage()

    submission_df = pd.DataFrame({"ID": ids, "clicked": predictions})
    submission_path = Path(output_path)
    submission_df.to_csv(submission_path, index=False)

    artifact_run_id = run_id or default_run_id(prefix="submission")
    artifact_dir = ensure_artifact_directory(run_id=artifact_run_id)
    metadata_path = artifact_dir / "submission.json"
    metadata = {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_path": str(submission_path),
        "run_id": artifact_run_id,
        "num_rows": len(submission_df),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return submission_path


# CLI에서 호출했을 때 추론 파이프라인을 실행한다.
def main() -> None:
    """Entrypoint for CLI execution."""

    args = parse_args()
    run_id = args.run_id or default_run_id(prefix="submission")
    output_path = generate_submission(
        model_path=args.model_path,
        input_path=args.input,
        output_path=args.output,
        run_id=run_id,
    )
    print(f"✅ Submission saved to {output_path}")


if __name__ == "__main__":
    main()
