# 실험 · 결과 개요
대표 실험과 리더보드 성과를 한 문서에서 정리합니다. 최근 세션 로그를 간추려, 실험 설계 → 결과 비교 → 제출 기록으로 이어지는 스토리를 제공합니다.

## 1. 실험 기록 포맷
- 각 실험은 “목표 → 설정 → 결과 → 해석 → 후속 액션” 카드 형식으로 기록합니다.
- 실행 커맨드, 사용한 스크립트, 입력 데이터 버전을 명시해 재현성을 담보합니다.
- 원본 로그는 `logs/metrics/*.csv`에 저장되고, 본 문서에는 핵심 수치만 요약합니다.

## 2. 대표 실험 하이라이트
| 실험 ID | 단계 | 검증 데이터 | AP (내부) | LB Public | 주요 설정 & 인사이트 |
| --- | --- | --- | --- | --- | --- |
| `20251006_1408_lgbm_1m` | 베이스라인 샘플 | `train_sample_1m` | 0.2069 | - | 100만 샘플, 기본 LightGBM 설정. 데이터 스케일 확장의 필요성 확인. |
| `20251006_100336_phase_a_spw1` | Seq Window 실험 | `train_downsample_phase_a` | 0.4823 | 0.33953 | 7일 윈도우 파생, scale_pos_weight=1.0. 시간 기반 파생의 가치를 검증. |
| `20251007_053310_lgbm_optuna` | Optuna 튜닝 | `train_downsample_random_clean_corr96` | 0.5116 | - | Optuna로 학습률/leaf 탐색. 과도한 leaf 확장은 과적합으로 이어짐. |
| `20251008_032422_lgbm_phase_1_2_baseline_nb1000` | Phase 1·2 베이스라인 | `train_downsample_1_2` | 0.5807 | 0.34077 | 1:2 다운샘플, num_boost_round=1000. 베이스라인 대비 +37pt 개선. |
| `20251011_221557_lgbm_phase_a` | Phase A 최종 | `train_downsample_phase_a` | **0.6104** | **0.34126** | CUDA 학습 + seq_bucket 델타. 캘리브레이션은 Raw가 가장 안정적. |
| `20251013_final_submission` | 최종 제출 | `train_downsample_phase_a` | 0.6104 | Private Top 10% | Private 리더보드 최종 제출, Raw 점수 유지 |

> 표의 AP/LB 값은 `logs/metrics/*.csv`와 제출 기록을 기반으로 합니다.

## 3. 세부 실험 로그 요약
- **데이터 전략 실험**: `scripts/build_day7_features.py`와 `build_day7_seqbucket.py`로 생성한 day7 파생이 LB 0.34126 달성에 핵심. 반면 전체 학습(full train)이나 과도한 day7 가중치는 확률 붕괴로 폐기.
- **하이퍼파라미터 탐색**: Optuna(`lgbm_optuna`)로 학습률 0.03, num_leaves 256 조합이 가장 안정적. num_boost_round는 600~1000 범위에서 early stopping 50을 유지.
- **Calibration 비교**: Raw > Platt > Isotonic 순으로 LB가 하락. Temperature scaling(T=1.75)은 day7 holdout에서는 개선되나 LB에서는 미확인 → 보조 전략으로 보류.
- **최종 Private 결과**: Raw 예측 그대로 제출해 최종 리더보드 Top 10%를 기록했습니다.

## 4. 리더보드 & 제출 현황
| 날짜 | 제출 파일 | 내부 AP | Public LB | 비고 |
| --- | --- | --- | --- | --- |
| 2025-10-08 | `phase_1_2_baseline_nb1000.csv` | 0.5807 | 0.34077 | 1:2 다운샘플 베이스라인 (Raw) |
| 2025-10-11 | `phase_a_calibrated.csv` | 0.6104 | 0.34126 | seq_bucket 델타 + Raw 확정 버전 |
| 2025-10-12 | `phase_a_offline_cv.csv` | 0.6096 | (미제출) | offline CV 검증 결과 공유 |
| 2025-10-12 | `phase_a_temp175.csv` | 0.6104 | (미제출) | temperature scaling 1.75 테스트 |

> 제출 파일은 Git에 포함하지 않고 외부 스토리지에 백업합니다. 제출 메타데이터는 `docs/experiments.md`와 세션 로그에 동시 기록합니다.

## 5. 시각화 & 리포트 가이드
- 추세 그래프, 피처 중요도, Calibration 비교 차트는 `notebooks/`에서 생성 후 이미지로 저장해 문서에 첨부합니다.
- 핵심 그래프: (1) 실험별 AP/LB 추이, (2) Feature importance 상위 20개, (3) Calibration ROC/PR 곡선, (4) day7 holdout 대비 그래프.

## 6. 교훈 및 회고
- **성공 요인**: (a) 층화 다운샘플과 day7 특화 파생 조합, (b) 시퀀스 길이/다양성 피처 도입, (c) GPU 기반 대규모 학습과 Optuna 탐색.
- **아쉬운 점**: day7 가중치 실험과 full train 재도전은 안정적 파라미터를 찾지 못함. feature pruning 자동화가 미흡.
- **다음 액션**: (1) history 교차 피처 향상, (2) day7 holdout 기반 temperature scaling 재평가, (3) Phase B 데이터 대응 및 앙상블 전략 검토.
