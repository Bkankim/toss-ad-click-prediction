# toss-ad-click-prediction

토스 NEXT ML Challenge CTR 예측 대회를 위한 LightGBM 기반 파이프라인입니다. 대회 규정 준수와 재현 가능한 실험 관리를 목표로 합니다.

## 빠른 시작
1. 저장소 루트에서 환경 설정 스크립트를 실행합니다.
   ```bash
   cd /Competition/CTR/toss-ad-click-prediction
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```
2. `.env.example`을 복사해 실제 환경 변수 파일을 만듭니다.
   ```bash
   cp .env.example .env
   ```
3. `.env` 파일에서 데이터 경로와 로깅 옵션을 환경에 맞게 수정합니다.
4. 가상환경을 활성화한 뒤 스모크 테스트를 실행해 데이터 적재가 정상 동작하는지 확인합니다.
   ```bash
   source .venv/bin/activate
   python main.py --check-data
   ```
5. 자동 EDA 리포트를 생성해 핵심 통계를 확인합니다.
   ```bash
   python -m src.eda.report_generator --sample-size 300000 --output docs/EDA.md
   ```

## 환경 변수
`.env.example`에 Story 1.1에서 요구하는 최소 변수들이 정의되어 있습니다.

- `DATA_ROOT`: 대회 데이터가 위치한 루트 경로
- `TRAIN_PATH`, `TEST_PATH`: 학습·추론에 사용할 기본 Parquet 파일
- `CACHE_DIR`: 중간 산출물을 저장할 디렉터리 (필요 시 생성)
- `SUBMISSION_DIR`: 제출 파일이 저장될 폴더
- `EXPERIMENT_TRACKING`: `wandb` 고정(필요 시 보조 로그로 `csv` 추가 가능)
- `WANDB_API_KEY`: W&B API 키(현재 값은 예시이므로 실제 키로 교체 필요)
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_RUN_GROUP`, `WANDB_MODE`: 팀/프로젝트 메타데이터와 실행 모드(`online`/`offline`)
- `WANDB_DIR`: W&B 로컬 캐시 경로(정기적으로 정리 필요)
- `GLOBAL_SEED`: 전역 난수 시드
- `USE_GPU`: LightGBM GPU 학습 활용 여부 (0/1)
- `OPTUNA_STORAGE`: Optuna 실험 기록 저장 위치(`sqlite:///optuna.db` 권장)
- `ARTIFACT_RETENTION_DAYS`: 모델·로그 보관 기간(일)

필요한 변수는 `.env`에서 추가하거나 수정한 뒤 `source .venv/bin/activate && export $(cat .env | xargs)`로 적용할 수 있습니다. `WANDB_API_KEY`는 반드시 실제 키로 대체하세요.

## 스토리지 관리 가이드
- 모든 실험/제출 전 `du -sh / 2>/dev/null`로 루트 용량(150GB 한도)을 확인합니다.
- `WANDB_DIR`, `OPTUNA_STORAGE`, `submission/`, `logs/` 등의 아티팩트는 `ARTIFACT_RETENTION_DAYS` 기준으로 주기적으로 정리합니다.
- 필요 이상으로 모델, Optuna trial, W&B 로컬 캐시가 쌓이지 않도록 실험 종료 후 즉시 압축 또는 삭제합니다.

## 데이터 배치 규칙
- 원본 Parquet, CSV 등 대회 데이터는 `/Competition/CTR/toss-ad-click-prediction/data`에 보관합니다.
- Git 추적은 `/Competition/CTR/toss-ad-click-prediction` 내부에서만 수행합니다.
- 대회 규정을 준수하기 위해 외부 데이터는 사용하지 않습니다.

## 개발 체크리스트 스냅샷
- `scripts/setup_env.sh`: Python 3.11 기반 가상환경 생성, uv/pip 의존성 설치, GPU 옵션 안내.
- `.env.example`: 데이터 경로, 캐시 위치, 실험 로깅 옵션 기본값.
- Story 1.1 수용 기준: 메모리 사용 로깅, 실행 방법 문서화, 스모크 테스트 (`python main.py --check-data`) 포함.
- Story 1.2 수용 기준: `src/eda/report_generator.py`로 day/hour, 사이클릭, 시퀀스 통계를 자동 생성하여 `docs/EDA.md` 갱신.

자세한 요구사항과 에픽/스토리 구조는 `docs/prd.md`를 참고하세요.
