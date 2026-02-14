import argparse
import os
import pandas as pd

from chronos2_project.data import load_hf_timeseries
from chronos2_project.experiment import run_zero_shot, evaluate_predictions
from chronos2_project.utils import now_run_id, ensure_dir, collect_system_info, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--prediction_length", type=int, default=24)
    ap.add_argument("--max_series", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    run_id = now_run_id("ablation_covariates")
    run_dir = os.path.join(args.out_dir, run_id)
    ensure_dir(run_dir)

    ds = load_hf_timeseries(args.hf_dataset, args.config, split=args.split, max_series=args.max_series)
    df_full = ds.df.copy()

    # Variant A: full covariates (as provided)
    trainA, testA, predA, rtA, qcols = run_zero_shot(
        df=df_full,
        id_col=ds.id_col,
        time_col=ds.time_col,
        target_col=ds.target_col,
        prediction_length=args.prediction_length,
        device=args.device,
    )
    perA, aggA = evaluate_predictions(testA, predA, ds.id_col, ds.time_col, ds.target_col, qcols)

    # Variant B: drop covariates => univariate-only
    drop_cols = ds.covariate_cols
    df_uni = df_full.drop(columns=drop_cols) if drop_cols else df_full
    trainB, testB, predB, rtB, _ = run_zero_shot(
        df=df_uni,
        id_col=ds.id_col,
        time_col=ds.time_col,
        target_col=ds.target_col,
        prediction_length=args.prediction_length,
        device=args.device,
    )
    perB, aggB = evaluate_predictions(testB, predB, ds.id_col, ds.time_col, ds.target_col, qcols)

    # Save
    pd.DataFrame([
        {"variant": "with_covariates", **aggA.to_dict(), "runtime_seconds": rtA},
        {"variant": "no_covariates", **aggB.to_dict(), "runtime_seconds": rtB},
    ]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    perA.to_csv(os.path.join(run_dir, "metrics_per_series_with_covariates.csv"), index=False)
    perB.to_csv(os.path.join(run_dir, "metrics_per_series_no_covariates.csv"), index=False)
    predA.to_parquet(os.path.join(run_dir, "predictions_with_covariates.parquet"), index=False)
    predB.to_parquet(os.path.join(run_dir, "predictions_no_covariates.parquet"), index=False)

    cfg = vars(args)
    cfg.update({
        "run_id": run_id,
        "run_dir": run_dir,
        "covariate_cols": ds.covariate_cols,
        "system": collect_system_info(),
        "aggregate": {
            "with_covariates": aggA.to_dict(),
            "no_covariates": aggB.to_dict(),
        }
    })
    save_json(os.path.join(run_dir, "config.json"), cfg)

    print(f"Saved ablation run to: {run_dir}")
    print(pd.DataFrame([
        {"variant": "with_covariates", **aggA.to_dict()},
        {"variant": "no_covariates", **aggB.to_dict()},
    ]))

if __name__ == "__main__":
    main()
