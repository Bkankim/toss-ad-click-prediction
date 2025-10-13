# toss-ad-click-prediction
토스 NEXT ML Challenge CTR 예측 대회 참가를 위해 구축한 LightGBM 기반 파이프라인입니다. 포트폴리오 용도로 재구성하여 문제 정의, 실험 근거, 재현 절차를 명확히 남겼습니다. 프로젝트에서 관리하는 PRD·히스토리·세션 로그를 참고해 최신 상태를 유지합니다.

## 프로젝트 하이라이트
- **최종 내부 AP 0.6104 / Public LB 0.34126**: Phase A 파이프라인(`20251011_221557_lgbm_phase_a`)이 달성했습니다.
- **베이스라인 대비 +0.40p 개선**: 100만 샘플 베이스라인(0.2069) → 1:2 다운샘플(0.5807) → 최종 파이프라인(0.6104)으로 단계적 향상을 확인했습니다.
- **재현 가능한 워크플로우**: 전처리 스크립트, 학습/추론 CLI, 로그 관리 규칙을 문서화하여 누구나 동일한 결과를 낼 수 있도록 구성했습니다.
- **최종 결과**: Private 리더보드에서 **Top 10%**를 기록하며 대회를 마무리했습니다.
- **평가 지표**: Score = 0.5 × AP + 0.5 × (1 / (1 + WLL)). PRD 목표(Score ≥ 0.700, Public AP ≥ 0.500)를 향해 실험을 지속 중입니다.

## 문서 허브
| 문서 | 설명 |
| --- | --- |
| [`docs/overview.md`](docs/overview.md) | 문제 정의, 핵심 성과, 접근 전략 요약 |
| [`docs/pipeline.md`](docs/pipeline.md) | 데이터 전처리 → 학습 → 추론까지의 명령 모음과 체크리스트 |
| [`docs/analysis.md`](docs/analysis.md) | 자동/수동 EDA 요약, 상관관계 분석, 리스크 평가 |
| [`docs/experiments.md`](docs/experiments.md) | 대표 실험 하이라이트, 제출 이력, 교훈 정리 |
| [`docs/operations.md`](docs/operations.md) | 환경 구성, 스토리지 정책, 실험 추적 가이드 |

## 빠른 시작
1. 저장소 루트로 이동해 환경을 구성합니다.
   ```bash
   cd /Competition/CTR/toss-ad-click-prediction
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```
2. `.env.example`을 복사해 실제 환경 변수 파일을 준비합니다.
   ```bash
   cp .env.example .env
   ```
3. `.env`에서 데이터 경로, W&B, Optuna 설정을 환경에 맞게 수정 후 가상환경을 활성화합니다.
   ```bash
   source .venv/bin/activate
   export $(cat .env | xargs)
   ```
4. 스모크 테스트로 데이터 적재를 확인합니다.
   ```bash
   python main.py --check-data
   ```
5. 자동 EDA 리포트를 생성해 최신 분석을 갱신합니다.
   ```bash
   python -m src.eda.report_generator --sample-size 300000 --output docs/analysis_auto.md
   ```

## 재현 가능한 워크플로우
1. **전처리**
   ```bash
   python scripts/build_day7_features.py --train-path data/train.parquet --output-dir data/clean_corr96_phase_a
   python scripts/build_history_cross_features.py --input data/clean_corr96_phase_a/train.parquet --output data/clean_corr96_phase_a_hist.parquet
   python -m src.data.downsampling --train-path data/clean_corr96_phase_a_hist.parquet --output-path data/samples/train_downsample_phase_a.parquet
   ```
2. **학습**
   ```bash
   python -m src.training.lightgbm_baseline --sample-size 500000 --device-type cuda
   python -m src.training.phase_a --train-path data/samples/train_downsample_1_2_cv_seqemb_seqbucket.parquet --num-boost-round 1000
   ```
3. **Optuna 튜닝 (선택)**
   ```bash
   python -m src.training.optuna_tuner --n-trials 50 --storage sqlite:///logs/optuna/phase_a.db
   ```
4. **추론 및 제출 파일 생성**
   ```bash
   python -m src.training.generate_predictions --model-dir models/lightgbm_phase_a --output submission/phase_a_calibrated.csv
   ```
5. **문서 업데이트**: 실험 결과는 `logs/metrics/*.csv`에 저장되고, 요약은 `docs/experiments.md`와 세션 로그에 반영합니다.

## 스토리지 & 운영 정책
- **용량 가드**: 작업 전 `du -sh / 2>/dev/null`로 루트 디스크 사용량(150GB 한도)을 확인합니다.
- **산출물 관리**: `models/`, `submission/`, `logs/metrics/`는 Git에서 제외하고 `.gitkeep`과 README로 규칙만 남깁니다.
- **외부 백업**: 제출 파일과 모델은 GitHub Release/S3 등 외부 스토리지에 즉시 백업하고 문서에 링크를 남깁니다.
- **실험 추적**: 핵심 실험만 CSV 요약으로 남기고, 대량 로그는 문서 링크만 제공합니다. W&B 사용 시 API 키는 `.env`에서만 관리합니다.

## 저장소 구조 요약
```
├── docs/                # 포트폴리오 문서 (overview, pipeline, analysis, experiments, operations)
├── notebooks/           # 탐색/시각화 노트북 (대표 노트만 보관)
├── scripts/             # 전처리 및 특징 생성 CLI 스크립트
├── src/                 # 데이터, 피처, 학습 파이프라인 모듈
├── logs/metrics/        # 대표 실험 CSV 요약 (산출물 정책 참조)
├── models/, submission/ # 모델/제출 산출물 (외부 스토리지로 백업)
└── data/                # 원본/파생 데이터 (스크립트로 재생성 가능)
```

## 개인 회고
- 테스트 데이터(day7)에 맞춘 holdout과 `day7_ctr_delta_*` 파생만이 안정적인 개선을 가져왔다. 단순 가중치 조정은 오히려 리더보드 점수를 낮췄다.
- Full train 설정과 높은 `scale_pos_weight`는 확률 붕괴를 불러와 다운샘플 + cross-fit 전략이 가장 현실적인 선택임을 깨달았다.
- Calibration(Platt, Isotonic, Temperature) 실험 결과 Raw 점수가 가장 일관되게 좋아 제출 전략을 단순화할 수 있었다.
- 시퀀스 길이/다양성 및 교차 히스토리 피처가 성능 향상을 이끌었고, 이를 문서와 스크립트로 표준화하면서 최종 Private 리더보드 Top 10% 성적을 달성했다.
- 팀 협업 측면: 서로 다른 모델을 나눠 실험하고 앙상블을 구성하려 했지만, 일정과 자원 부족으로 끝내 통합하지 못했다.
- 딥러닝 실험 부재: Transformer·시퀀스 임베딩 기반 모델을 직접 재현해 비교하려 했으나, LightGBM 파이프라인 안정화에 집중하면서 팀원들이 별도로 구현·실험했다.
- 후처리 검증 한계: temperature scaling 등 캘리브레이션 실험은 day7 holdout에서 가능성이 보였지만, 리더보드 제출에 충분히 활용하지 못했다.
