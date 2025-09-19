# Experiment & Delivery Pipeline

## Stage 0. 준비 & 관측 체계
- [ ] 최신 LightGBM 베이스라인(`notebooks/lgb_model_20250919_011839.pkl`) 재현 및 로그 저장
- [ ] 다운샘플 비율(1:2) 유지 여부 점검 및 로그화
- [ ] 실험 산출물 저장 구조 확정 (`notebooks/artifacts_eda/YYYYMMDD/...`)
- [ ] 메트릭 로깅 규칙 수립 (OOF AP/WLL, LB 기록, 특성 처리 메타데이터)
- [ ] wandb 프로젝트/자격 설정 및 환경 변수 확인 **(AGENTS.md 참고)**

## Stage 1. 데이터 품질 & Null 처리
- [ ] `feat_e_3` 결측 패턴 검증 및 Null indicator 적용 여부 결정
- [ ] row_nulls ≥95 구간 처리 정책(삭제/플래그/별도 모델) 문서화 및 구현
- [ ] Null 처리 후 재로딩 시 훈련·검증 스케일 변동 모니터링
- **추가 EDA**: 결측 인디케이터 도입 전후 CTR 차이 / row_nulls 구간 피처 분포 비교

## Stage 2. 수치 안정화 & Heavy-tail 대응
- [ ] `l_feat_12/14/25`, `feat_e_2`, `feat_e_6` 등 Tail 컬럼 클리핑·스케일링 실험
- [ ] 변환 방식별(클리핑, 로그+offset, Box-Cox) 성능 비교표 작성
- [ ] `raw_to_clean_tree.py` 자동 변환(Winsorize+transform) 결과와 수동 실험 비교 **(팀원 코드 참고)**
- [ ] 변환 후 SHAP/Permutation 재실행
- **추가 EDA**: 변환 전후 분위수·히스토그램 / Fold별 잔차·예측 분포 점검

## Stage 3. 피처 정리 & 파생 재설계
- [ ] SHAP·Permutation 하위 피처(`history_b_*`, `feat_a_*`, `seq_median`, `l_feat_20/23`) 제거 실험
- [ ] `history_b_*` 생성 로직 재검토 후 decay/ratio 등 대안 파생 검증
- [ ] `raw_to_clean_tree.py` 교차키+Target Encoding 베이스라인 비교 **(팀원 코드 참고)**
- [ ] 시퀀스 파생 확장(길이 버킷, 초기·마지막 토큰 요약 등) 큐 작성 및 실험
- **추가 EDA**: 피처 블록 상관행렬/VIF, 신규 시퀀스 파생 분포·인벤토리별 차이 시각화

## Stage 4. 교차검증 재설계 & 검증 체계
- [ ] 랜덤 K-Fold · 시간 기반 · inventory/seq hash 3종 CV 구현
- [ ] Fold별 인벤토리·요일 분포 보존 여부 검증 리포트 작성
- [ ] day=7 신규 seq 비중 모니터링 지표 구축
- [ ] CV 결과 vs LB 점수 상관계수/편차 기록
- **추가 EDA**: Fold별 `inventory_id`·`day_of_week`·`seq_len` 분포 / Calibration 곡선 비교

## Stage 5. 모델 확장 & 제출 전략
- [ ] 정리된 피처/클리핑 반영 후 LightGBM 재학습 및 OOF 기록
- [ ] `scale_pos_weight` 등 클래스 가중치 튜닝 실험
- [ ] 동일 데이터로 XGBoost·CatBoost 비교 및 앙상블 후보 구성
- [ ] `history_a_1` 정규화·상호작용 실험
- [ ] OOF 기반 확률 보정(Temperature / Isotonic) 적용 및 Weighted LogLoss 확인
- [ ] 제출 스케줄 수립 (day7 특화 모델 vs 전체 모델)
- **추가 EDA**: 모델 간 피처 중요도 비교, 예측 확률 Calibration, 인벤토리별 성능 분해

## Stage 6. 모니터링 & 문서화
- [ ] 각 실험 종료 시 MEMORY.md 업데이트(Score/주요 변화/실패 사유)
- [ ] 성공·실패 실험 요약표 유지
- [ ] 주간 리포트(주요 인사이트, 다음 계획) 작성
- **추가 EDA**: 제출 성과 추이 vs 실험 변경점 분석, 장기 추세 대시보드 초안

