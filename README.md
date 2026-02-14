# Chronos-2 Project (Deep NLP / Foundation Models)

This repository reproduces **zero-shot time-series forecasting** with **Chronos-2** and adds **two extensions**:
1) **Covariate ablation** (with vs. without covariates) on multivariate/covariate datasets  
2) **Horizon sensitivity** (short vs. long horizons) across datasets  
3) **Domain transfer** to an external dataset (e.g., electricity consumption)

The code is designed to be **reproducible**: fixed seeds, deterministic splits, and all outputs saved as CSV + figures.

## 1) Quickstart

### Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

> Recommended: `chronos-forecasting==2.1.0` (includes bug fixes for covariates during fine-tuning; safe for inference too).

### Sanity check (model + GPU)
```bash
python scripts/sanity_check.py --device cuda
```

## 2) Baseline reproduction (Chronos datasets)

Chronos benchmark datasets are published on Hugging Face as `autogluon/chronos_datasets` and `autogluon/chronos_datasets_extra`.

List available dataset configs:
```bash
python scripts/list_dataset_configs.py --dataset autogluon/chronos_datasets
python scripts/list_dataset_configs.py --dataset autogluon/chronos_datasets_extra
```

Run baseline on a chosen dataset (example config name; use the listing script to pick one):
```bash
python scripts/run_baseline.py \
  --hf_dataset autogluon/chronos_datasets \
  --config australian_electricity_demand \
  --prediction_length 48 \
  --max_series 200 \
  --device cuda
```

Outputs (created automatically):
- `results/<run_id>/metrics.csv` (aggregate + per-series metrics)
- `results/<run_id>/predictions.parquet`
- `results/<run_id>/figures/*.png`

## 3) Extension 1 — Covariate ablation

This compares Chronos-2 performance with full covariates vs. dropping covariates (univariate-only).

```bash
python scripts/run_covariate_ablation.py \
  --hf_dataset autogluon/chronos_datasets_extra \
  --config ETTh1 \
  --prediction_length 24 \
  --max_series 50 \
  --device cuda
```

## 4) Extension 2 — Horizon sweep

```bash
python scripts/run_horizon_sweep.py \
  --hf_dataset autogluon/chronos_datasets \
  --config australian_electricity_demand \
  --horizons 24,48,96,192 \
  --max_series 200 \
  --device cuda
```

## 5) (Optional) Extension 3 — Domain transfer

Example using a public Hugging Face electricity dataset:
```bash
python scripts/run_domain_transfer.py \
  --hf_dataset LeoTungAnh/electricity_hourly \
  --prediction_length 24 \
  --max_series 200 \
  --device cuda
```

## 6) Reproducibility

- All runs save a `config.json` with parameters, package versions, and GPU info.
- Metrics are computed from the **same deterministic split** (last `prediction_length` points used as test).
- Use `--seed` to control randomness.

## 7) Report

A 4-page IEEE LaTeX report is provided in `report/`:
- `report/chronos-report.pdf`
- `report/references.bib`

After running experiments, copy figures/tables into `report/figures` and `report/tables`, then compile.

```bash
cd report
latexmk -pdf main.tex
```

