# 실험 로그 관리

- 실행마다 산출되는 OOF/LB 지표는 `logs/metrics/` 아래 CSV 파일로 저장합니다. (실행당 하나)
- 파일명 규칙: `YYYYMMDD_HHMM_model.csv`.
- 필수 컬럼: `run_id, stage, dataset, ap, wll, competition_score, notes`.
- 추가 그래프나 SHAP 산출물은 `notebooks/artifacts_eda/YYYYMMDD/` 디렉터리에 동일한 `run_id`로 보관합니다.
