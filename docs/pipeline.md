# 데이터 & 파이프라인 개요
전처리부터 추론까지의 엔드투엔드 흐름을 정리합니다. 최근 세션 로그에 남긴 상세 기록을 기반으로, 재현 가능한 명령과 산출물 정책을 요약했습니다.

## 1. 데이터 요약
| 구분 | 경로 | 생성 방법 | 비고 |
| --- | --- | --- | --- |
| 원본 학습 | `data/train.parquet` | 대회 제공 | 약 1,070만 row, click rate 1.9% |
| 원본 테스트 | `data/test.parquet` | 대회 제공 | 제출용 예측 대상 (day7 집중) |
| 다운샘플 (Phase A) | `data/samples/train_downsample_phase_a.parquet` | `python -m src.data.downsampling` | 양/음성 1:4 층화 샘플 |
| 1:2 다운샘플 | `data/samples/train_downsample_1_2_cv_seqemb*.parquet` | 동일 스크립트, `--negative-ratio 2` | 최신 실험 기본 자산 |
| 파생 피처 | `data/clean_corr96_*` | `scripts/build_day7_*`, `scripts/build_history_cross_features.py` | Git 미추적, 필요 시 재생성 |

> 대용량 산출물은 `.gitignore`로 제외되며, 생성 이력과 파라미터는 세션 로그에 기록해 둡니다.

## 2. 엔드투엔드 플로우
1. **환경 구성**: `./scripts/setup_env.sh` 실행 → `.env` 설정 → 가상환경 활성화.
2. **EDA 스냅샷**: `python -m src.eda.report_generator --sample-size 300000 --output docs/analysis_auto.md` 실행.
3. **파생 피처 생성**: day7·history 전용 스크립트를 순서대로 실행해 `data/clean_*` 갱신.
4. **학습/튜닝**: 베이스라인(`src/training/lightgbm_baseline.py`) → Phase A 파이프라인(`src/training/phase_a.py`) → 필요 시 Optuna(`src/training/optuna_tuner.py`).
5. **추론/제출**: `src/training/generate_predictions.py`로 제출 파일 생성, 외부 스토리지로 백업.
6. **문서 갱신**: `docs/analysis.md`, `docs/experiments.md`에 인사이트·실험 결과를 즉시 반영.

## 3. 데이터 파이프라인
- `scripts/build_day7_features.py`: 7일 이동 윈도우 통계, `day7_ctr_delta_*`, `day7_freq_ratio_inv` 생성.
- `scripts/build_day7_seqbucket.py`: seq 길이 버킷 기반 델타(`day7_seq_bucket_delta`) 추가.
- `scripts/build_history_cross_features.py`: history_a/b 교차 피처(α 조절 가능) 생성.
- `scripts/add_day7_flags.py`: 주말/야간 플래그를 포함해 품질 보정 컬럼 삽입.
- 실행 예시:
  ```bash
  python scripts/build_day7_features.py     --train-path data/train.parquet     --output-dir data/clean_corr96_phase_a
  python scripts/build_history_cross_features.py     --input data/clean_corr96_phase_a/train.parquet     --output data/clean_corr96_phase_a_hist.parquet
  python -m src.data.downsampling     --train-path data/clean_corr96_phase_a_hist.parquet     --output-path data/samples/train_downsample_phase_a.parquet
  ```

## 4. 피쳐 엔지니어링
- `src/features/preprocess.py`에서 공통 인코딩/스케일링(결측 플래그 포함)을 수행합니다.
- `seq` 컬럼은 길이, 다양성, 최근 토큰 기반 지표로 확장됩니다.
- day7 특화 피처와 history 교차 피처는 완전 상관 열 제거 후 LightGBM 입력으로 사용합니다.

## 5. 모델 학습 파이프라인
| 스크립트 | 설명 | 주요 옵션 | 예시 |
| --- | --- | --- | --- |
| `python -m src.training.lightgbm_baseline` | 5-Fold CV, 기본 LightGBM 베이스라인 | `--sample-size`, `--num-boost-round`, `--device-type` | `python -m src.training.lightgbm_baseline --sample-size 500000 --device-type cuda` |
| `python -m src.training.phase_a` | Phase A 전용 파이프라인 + Calibration | `--train-path`, `--num-boost-round`, `--scale-pos-weight` | `python -m src.training.phase_a --train-path data/samples/train_downsample_1_2_cv_seqemb_seqbucket.parquet --num-boost-round 1000` |
| `python -m src.training.optuna_tuner` | Optuna 기반 하이퍼파라미터 탐색 | `--n-trials`, `--storage`, `--study-name` | `python -m src.training.optuna_tuner --n-trials 50 --storage sqlite:///logs/optuna/phase_a.db` |

- 학습 결과는 `models/` 디렉터리에 저장하며 Git에 포함되지 않습니다.
- 요약 메트릭은 `logs/metrics/<timestamp>_*.csv`에 저장되고, 대표 결과는 `docs/experiments.md`에 반영합니다.

## 6. 추론 및 제출 파이프라인
- `python -m src.training.generate_predictions --model-dir models/lightgbm_phase_a --output submission/phase_a_calibrated.csv`
- 캘리브레이션 결과(`platt`, `isotonic`, temperature scaling)는 `logs/downsampling/`·`logs/analysis/`에 기록하고, Raw/Platt/Isotonic 비교 후 최종 선택합니다.
- 제출 파일은 즉시 외부 스토리지(예: GitHub Release, S3)에 백업하고 리더보드 메모는 `docs/experiments.md`에 남깁니다.

## 7. 자동화 & 재현성 체크리스트
- `python main.py --check-data`로 데이터 로딩 스모크 테스트 수행.
- `logs/metrics/README.md` 정책에 따라 핵심 실험만 CSV로 유지하고, 대량 로그는 외부 저장.
- `ARTIFACT_RETENTION_DAYS` 정책에 따라 `models/`, `submission/`, `logs/metrics/`를 주기적으로 정리합니다.
