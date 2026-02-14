.PHONY: sanity baseline ablation horizons domain

sanity:
	python scripts/sanity_check.py --device cuda

baseline:
	python scripts/run_baseline.py --hf_dataset autogluon/chronos_datasets --config australian_electricity_demand --prediction_length 48 --max_series 200 --device cuda

ablation:
	python scripts/run_covariate_ablation.py --hf_dataset autogluon/chronos_datasets_extra --config ETTh1 --prediction_length 24 --max_series 50 --device cuda

horizons:
	python scripts/run_horizon_sweep.py --hf_dataset autogluon/chronos_datasets --config australian_electricity_demand --horizons 24,48,96,192 --max_series 200 --device cuda
