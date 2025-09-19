# Memory Log

## 대회 개요
- 대회명: 토스 NEXT ML Challenge CTR 예측 (예선 기간 2025-09-08~2025-10-13).
- 과제: 토스 앱 외부 광고의 클릭 확률을 예측하는 CTR 모델 개발. 예선은 Private 리더보드 상위 30팀을 본선에 진출시키고, 본선은 코드 검증 및 보고서 평가를 병행.
- 평가 지표: Average Precision 50% + Weighted LogLoss 50% (테스트 30% Public / 70% Private 분할).
- 규칙: 2025-09-08 이전 공개·상업 이용 허용 사전학습 모델만 사용 가능, 외부 데이터·API 기반 모델 금지, Python 필수, 1일 최대 제출 3회.

## 일정 체크
- 팀 병합 마감: 2025-10-06 23:59
- 예선 종료: 2025-10-13 10:00 (제출 마감 동일)
- 본선 산출물 제출: 2025-10-13 12:00 ~ 2025-10-17 10:00
- 본선 평가: 2025-10-17 ~ 2025-10-31
- 최종 발표: 2025-11-03 10:00, 시상식 2025-11-06 저녁

## 데이터 요약
- train.parquet: 10,704,179행, 119컬럼 (타깃 `clicked` 포함). 강한 클래스 불균형 (clicked=1 약 204k).
- test.parquet: 1,527,298행, 118컬럼 (`ID` 포함). 
- 주요 피처: `gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`, 고차원 시퀀스 `seq`, 그룹별 연속형 피처(`l_feat_*`, `feat_[a-e]_*`, `history_[a-b]_` 등).
- `seq` 길이 분포 1~14k+, 고차원 범주형 다수 → 고급 인코딩/임베딩 전략 필요.

## 현재 성과 및 베이스라인
- 내부 최초 제출: `notebooks/lgbm_2.ipynb` 기반 LightGBM, Public 0.330 (2025-09-18 기준 0.33 기록 유지).
- 공유된 GPU 베이스라인: XGBoost + class weight, CV 0.3532±0.0007 / LB 0.3308. NVIDIA Merlin/cuDF 활용으로 5-fold 학습 ~2분(RTX A6000 기준).

## 전략 아이디어 메모
- 모델: DCN-V3 중심 다단계 스태킹, AutoInt·FiBiNET·RAT 등 Transformer 계열 보조 모델, LightGBM/XGBoost/ CatBoost 혼합 앙상블.
- 불균형 대응: Focal Loss 변형, 클래스 균형 손실, BorderlineSMOTE·ADASYN, 비용 민감 학습.
- 특성 공학: Mutual Information 기반 중요도 평가, 시간 특성 Fourier/rolling window, RFM·세션 파생, 계층 target encoding, 해싱/임베딩.
- 정규화 & 안정화: Multi-level dropout, Elastic Net, Mixup/CutMix, SAM, Cosine annealing, adversarial validation.
- 실험 관리: CV-OOF 분리, 실험 로깅, 제출 회수 ≤15회 계획, Public/Private 상관계수 ≥0.85 목표.

## TODO (2025-09-18)
1. CV 스킴 다변화: 랜덤 K-Fold, 시간 기반, inventory/seq 해시 기반 등 3종 이상 구성하고, 각 Fold에 인벤토리·요일 분포를 강제해 OOF-리더보드 상관 분석.
2. row_nulls ≥95 구간 전처리 정책(삭제 vs. 플래그 vs. 별도 모델) 확정 및 파이프라인 초안 작성, `feat_e_3` Null indicator 포함 여부까지 결정.
3. `artifacts_eda/` tail 임계값 기반으로 주요 수치 피처(예: `l_feat_12/14/25`, `feat_e_2`, `feat_e_6`) 클리핑·스케일링 실험을 진행해 LGBM/NN 안정성 영향 평가.
4. seq_len decile 파생(타깃 인코딩, interaction) 피처를 구현하고 기존 LGBM 베이스라인에서 성능 확인.
5. 위 실험과 CV 비교 결과를 주 단위로 MEMORY.md에 업데이트하는 루틴 확립.
6. SHAP·Permutation 최하위 피처(`history_b_*`, `feat_a_*`, `seq_median` 등) 제거/재가공 실험 추가하고 영향 분석.

## MCP 도구 설정 (2025-09-18)
- Node.js 워크스페이스 초기화 (`package.json`) 후 MCP 서버 패키지 devDependencies 추가: `@upstash/context7-mcp`, `@movibe/memory-bank-mcp`, `@modelcontextprotocol/server-sequential-thinking`.
- `mcp.json`에 npx 기반 서버 연결 정의 완료 (context7, memorybank --mode code --path ., sequential-thinking). MCP 지원 클라이언트에서 해당 파일로 설정 불러올 수 있음.
- context7 사용 시 `CONTEXT7_API_KEY` 환경 변수를 별도 설정해야 하며, memorybank는 프로젝트 루트 `memory-bank/` 폴더를 기본 저장소로 사용.
- sequential-thinking 서버는 기본 설정으로 사고 단계 관리 툴 제공, 필요 시 `DISABLE_THOUGHT_LOGGING=true` 환경 변수로 로깅 억제 가능.
- 향후 MCP 관련 변경 사항도 본 섹션에 덧붙여 추적할 것.

## EDA 확장 체크리스트 (2025-09-18)
- Train/Test 분포 격차: `day_of_week`는 Train 1~7 고르게 분포하지만 Test는 7만 존재. Test `seq`의 98.3%가 Train에서 등장하지 않은 시퀀스로 확인됨. 주요 지면(`inventory_id`) 비중은 유사하나 ID 36/46 등에서 ±1~2%포인트 차이 존재.
- 고Null 행 프로파일링: row_nulls=95/96 행이 총 17,208건(Train)으로, 주로 `inventory_id` 2/29/88에 집중돼 있으며 CTR 2.2~5.1%로 전체 평균 대비 편차 존재. 해당 행은 `gender`, `age_group`, 다수 `feat_e_*`/`feat_a_*`가 전부 결측.
- 상호작용 후보: `inventory_id`×`hour` 조합에서 ID 88의 13~21시대 CTR 3.6~4.1%, `seq_len`이 60 미만인 구간과 특정 시간대(13·19시 등)에서 CTR 3%대 확인. 짧은 `seq_len`과 ID 88 결합 시 CTR 4.5%.
- 수치 안정화 가이드: `feat_e_2` tail span이 ~805로 극단적이며, `l_feat_25`·`l_feat_12`도 p95~p999 범위가 400~800 이상. `feat_e_*` 계열은 음수 범위 포함 → 로그/Box-Cox 전환 전 offset 필요.
- Adversarial Validation: `day_of_week`, `seq`, `inventory_id`를 순차 제거해도 AUC 1.0 유지. 근본적으로 Train/Test가 완전히 분리된 이벤트로 간주해야 하며, 시간적 홀드아웃 또는 리샘플링 전략 재설계 필요.

## 체크리스트 상태 (2025-09-18)
- [x] Train/Test 분포 격차 진단 (day_of_week, seq, inventory 비중)
- [x] 고Null 행 클러스터 라벨/CTR 분석
- [x] 상호작용 후보 (inventory×hour, seq_len 조합) 발굴
- [x] 수치형 tail 기준 정리 및 변환 가이드 작성
- [x] Adversarial Validation 컬럼 제거 실험 (근본 분포 격차 확인)

## 다음 우선순위 체크리스트 (2025-09-18)
- [ ] CV 전략 비교 (랜덤 K-Fold, 시간 기반, inventory/seq 해시 기반) 및 인벤토리·요일 분포 보존 여부 검증
- [ ] 고Null 행(≥95) 전처리 + `feat_e_3` Null indicator 정책 확정 및 파이프라인 반영
- [ ] Heavy-tail 피처 클리핑/스케일링 실험 완료 후 결과 정리 (`l_feat_12/14/25`, `feat_e_2`, `feat_e_6` 포함)
- [ ] seq_len decile 파생 피처 모델 검증 및 효과 평가
- [ ] SHAP·Permutation 하위 피처 정리/재가공 실험 수행 및 리포트

## day_of_week=7 분석 (2025-09-18)

> ⚠️ 대회 운영 답변에 따라, 실제 Public/Private 평가는 별도 비공개 테스트셋으로 이루어지며 우리가 가진 `test.parquet`은 형식 검증용임. 따라서 아래 분석은 참고 자료로 보존하고, CV 설계는 다양한 분포를 대비하도록 조정함.
- Train 7일차는 레이블이 있는 1,526,335행으로 다른 요일과 규모가 동일하지만, 시퀀스는 거의 전부 당일 최초 등장(`new_seq_row_ratio ≈ 0.98`).
- Test는 전부 day_of_week=7로 구성되며 1,527,298행 중 25,859행(1.7%)만 train에서 본 seq를 재사용. 이 1.7% 역시 길이 1~10 사이 짧은 시퀀스가 99.97%를 차지하며, 최초 등장일은 대부분 day=1.
- 따라서 Test는 ‘7일차 신규 트래픽’ 시나리오로 볼 수 있고, 기존 CV(랜덤 K-Fold)는 분포를 전혀 재현하지 못함.

### CV 재설계 제안
1. **Time Holdout**: day_of_week=1~6 전체 + day=7의 80%를 학습, 남은 day=7 20%(seq 해시 기반 샘플)로 검증. Test와 동일한 요일·신규 seq 분포를 유지.
2. **OOF Fold 구성**: day=7에 한해 seq 해시로 5개 Fold를 구성하고, 각 Fold별로 나머지 day=7 데이터와 day=1~6 전체를 학습에 포함. Fold 간 seq 중복을 방지해 실제 제출 시 Generalization을 반영.
3. **리더보드 제출 전략**: 최종 모델 학습 시 day=1~7 전체 사용, 단 day=7은 위 Fold 스키마 기반으로 OOF/Model Selection에 활용한 뒤 전체 재학습 → 예측.

> 후속 과제: day=7 신규 seq에 특화된 파생 피처(짧은 seq_len, 초기 token 등)를 집중적으로 공학하고, Holdout 점수와 LB 간 상관관계를 추적해야 함.

## LightGBM Feature Importance 메모 (2025-09-18)

- `feat_e_3`는 결측률이 높음(train ~10.1%, test ~6.0%)에도 Gain/Split 기준 Top5에 위치. 결측 여부가 강력한 신호일 가능성이 있으므로 null indicator 생성 및 결측/비결측 CTR 차이를 분석하고, 값이 존재할 때는 클리핑·log 변환 등으로 안정화 검토.
- Gain 기준 중요도에서 `history_a_1`이 지배적으로 나타나며 `inventory_id`가 그 뒤를 이음. 이는 과거 성과 피처가 적은 분할만으로도 손실을 크게 줄이고 있음을 의미.
- Split(빈도) 기준으로는 `inventory_id`가 최상위, `history_a_1`은 두 번째. ⇒ 모델이 지면 ID에 자주 의존하며 `seq_std`, `seq_mean`, `hour` 같은 새 피처도 다수 분할에서 사용됨.
- history 계열 피처가 target leakage가 아닌지 재검증 필요. 필요 시 decay/정규화 적용을 검토.
- `inventory_id` 의존을 완화하기 위해 시간/사용자/seq 기반 상호작용 피처 확대를 계획.
- seq 통계(`seq_std`, `seq_mean`)가 유의미하므로 seq_len decile 파생 실험을 우선 진행하고, CV 비교 시 gain/split 두 지표 모두 추적할 것.
