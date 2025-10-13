# 운영 가이드
환경 구성, 실험 추적, 스토리지 관리 등 운영 요소를 정리합니다. 세션 인수인계 문서와 동일한 정책을 유지합니다.

## 1. 환경 구성
- **의존성 설치**: `./scripts/setup_env.sh` 실행 → Python 3.11 가상환경 생성, uv/pip 의존성 설치.
- **환경 변수**: `.env.example`을 복사해 `.env` 작성 후 `source .venv/bin/activate && export $(cat .env | xargs)`로 로드. 주요 키:
  - `DATA_ROOT`, `TRAIN_PATH`, `TEST_PATH`: 데이터 위치 (`data/` 하위 디렉터리 권장)
  - `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`: 실험 추적용 W&B 설정 (가상 키를 반드시 실제 값으로 교체)
  - `OPTUNA_STORAGE`: `sqlite:///logs/optuna/phase_a.db` 형태 권장
  - `ARTIFACT_RETENTION_DAYS`: 아티팩트 자동 정리 주기(일)
- **스모크 테스트**: `python main.py --check-data`로 데이터 경로와 메모리 로그를 확인.

## 2. 실험 추적
- **W&B**: `.env`에 API 키를 등록하고, 실행 커맨드에 `WANDB_MODE=online|offline`을 명시. 보안상 키는 Git에 포함하지 않습니다.
- **CSV 로그**: `logs/metrics/<timestamp>_*.csv`에 AP/WLL/Score/노트 기록. 핵심 실험만 남기고 나머지는 삭제 또는 외부 보관.
- **Optuna**: `python -m src.training.optuna_tuner --n-trials 50 --storage sqlite:///logs/optuna/phase_a.db` 형태로 실행. `logs/optuna/`는 `.gitignore`로 제외됩니다.

## 3. 스토리지 & 백업 정책
- **산출물 정책**: `models/`, `submission/`, `logs/metrics/`는 Git 추적에서 제외하며 `.gitkeep`과 README로 규칙만 남깁니다.
- **외부 백업**: 제출 파일과 모델은 GitHub Release/S3 등 외부 스토리지에 즉시 백업하고 링크를 문서에 남깁니다.
- **정리 주기**: `ARTIFACT_RETENTION_DAYS`에 따라 오래된 CSV/모델을 삭제하거나 아카이브합니다.

## 4. 협업 & 버전 관리
- **Git 운영**: `main` 브랜치 기반, 기능별 브랜치 → PR → 리뷰 후 머지. 커밋 메시지는 `type(scope): summary` 형태 권장.
- **세션 인수인계**: 장시간 작업 중단 시 `docs/experiments.md`와 세션 로그를 함께 갱신해 다음 담당자가 맥락을 바로 파악할 수 있도록 합니다.
