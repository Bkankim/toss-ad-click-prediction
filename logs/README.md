# Experiment Logging

- Place OOF/LB metrics in CSV under `logs/metrics/` (one file per run).
- Use filename convention: `YYYYMMDD_HHMM_model.csv`.
- Columns: `run_id, stage, dataset, ap, wll, competition_score, notes`.
- Store supplemental plots or SHAP artifacts under `notebooks/artifacts_eda/YYYYMMDD/` using the same `run_id`.
