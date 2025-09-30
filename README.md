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

## 환경 변수
`.env.example`에 Story 1.1에서 요구하는 최소 변수들이 정의되어 있습니다.

- `DATA_ROOT`: 대회 데이터가 위치한 루트 경로
- `TRAIN_PATH`, `TEST_PATH`: 학습·추론에 사용할 기본 Parquet 파일
- `CACHE_DIR`: 중간 산출물을 저장할 디렉터리 (필요 시 생성)
- `SUBMISSION_DIR`: 제출 파일이 저장될 폴더
- `EXPERIMENT_TRACKING`: `mlflow` 또는 `csv` 중 선택
- `MLFLOW_TRACKING_URI`: MLflow 서버를 사용할 경우 URI
- `GLOBAL_SEED`: 전역 난수 시드
- `USE_GPU`: LightGBM GPU 학습 활용 여부 (0/1)

필요한 변수는 `.env`에서 추가하거나 수정한 뒤 `source .venv/bin/activate && export $(cat .env | xargs)`로 적용할 수 있습니다.

## 데이터 배치 규칙
- 원본 Parquet, CSV 등 대회 데이터는 `/Competition/CTR/toss-ad-click-prediction/data`에 보관합니다.
- Git 추적은 `/Competition/CTR/toss-ad-click-prediction` 내부에서만 수행합니다.
- 대회 규정을 준수하기 위해 외부 데이터는 사용하지 않습니다.

## 개발 체크리스트 스냅샷
- `scripts/setup_env.sh`: Python 3.11 기반 가상환경 생성, uv/pip 의존성 설치, GPU 옵션 안내.
- `.env.example`: 데이터 경로, 캐시 위치, 실험 로깅 옵션 기본값.
- Story 1.1 수용 기준: 메모리 사용 로깅, 실행 방법 문서화, 스모크 테스트 (`python main.py --check-data`) 포함.

자세한 요구사항과 에픽/스토리 구조는 `docs/prd.md`를 참고하세요.
