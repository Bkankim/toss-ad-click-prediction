"""프로젝트 진입점 스크립트."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict


# 명령행 인자를 파싱한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="토스 CTR 파이프라인 유틸리티")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="환경 변수로 정의된 데이터 경로의 존재 여부와 메타 정보를 확인한다.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1000,
        help="샘플 확인 시 출력할 최대 행 수(Polars/Pandas 이용 시).",
    )
    return parser.parse_args()


# 데이터 경로 환경 변수를 로드한다.
def load_data_paths() -> Dict[str, Path]:
    data_root = Path(os.getenv("DATA_ROOT", "/Competition/CTR/data"))
    train_path = Path(os.getenv("TRAIN_PATH", data_root / "train.parquet"))
    test_path = Path(os.getenv("TEST_PATH", data_root / "test.parquet"))
    return {
        "DATA_ROOT": data_root,
        "TRAIN_PATH": train_path,
        "TEST_PATH": test_path,
    }


# Parquet 파일의 기본 메타 정보를 추출한다.
def summarize_parquet(path: Path) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        summary["num_rows"] = parquet_file.metadata.num_rows
        summary["num_columns"] = parquet_file.metadata.num_columns
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - 진단용 경로
        print(f"[WARN] {path} 메타 정보 조회 중 예외 발생: {exc}")
    return summary


# 데이터 경로를 검증하고 샘플 요약을 출력한다.
def check_data(sample_rows: int) -> None:
    paths = load_data_paths()
    for label, path in paths.items():
        if label == "DATA_ROOT":
            exists = path.exists()
            print(f"{label}: {path} (exists={exists})")
            continue

        exists = path.exists()
        print(f"{label}: {path} (exists={exists})")
        if not exists:
            continue

        summary = summarize_parquet(path)
        if summary:
            print(
                f"  → rows={summary.get('num_rows')} cols={summary.get('num_columns')}"
            )

        if sample_rows <= 0:
            continue

        try:
            import polars as pl  # type: ignore

            frame = pl.read_parquet(path, n_rows=sample_rows)
            print(f"  → sample (polars) shape={frame.shape}")
            continue
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - 진단용 경로
            print(f"  → [WARN] Polars 로드 실패: {exc}")

        try:
            import pandas as pd  # type: ignore

            frame_pd = pd.read_parquet(path)
            print(f"  → sample (pandas) shape={frame_pd.head(sample_rows).shape}")
        except ImportError:
            print("  → [INFO] pandas/Polars 미설치로 샘플 출력은 생략합니다.")
        except Exception as exc:  # pragma: no cover - 진단용 경로
            print(f"  → [WARN] pandas 로드 실패: {exc}")


# 스크립트 진입점.
def main() -> None:
    args = parse_args()
    if args.check_data:
        check_data(args.sample_rows)
    else:
        print("토스 CTR 파이프라인 유틸리티 - --check-data 옵션을 사용해 데이터를 검증하세요.")


if __name__ == "__main__":
    main()
