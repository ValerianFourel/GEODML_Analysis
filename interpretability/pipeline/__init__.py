"""GEODML upstream pipeline ported into GEODML_Analysis.

Stages:
  prompts  - variant-aware rerank prompt (biased vs neutral)
  rerank   - Phase 1: LLM rerank of cached SERPs (cluster GPU)
  features - Phase 2: deterministic T1-T7 + confounder extraction from cached HTML
  merge    - Phase 3: clean + merge into full_experiment_data_{variant}.parquet
  dml      - Phase 4: DoubleML PLR / IRM with LGBM/RF, 5-fold CV
  config   - treatment defs, model list, pool sizes, DML grid
"""
