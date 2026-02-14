import argparse
import os
import pandas as pd

from chronos2_project.data import load_hf_timeseries
from chronos2_project.experiment import run_zero_shot, evaluate_predictions
from chronos2_project.utils import now_run_id, ensure_dir, collect_system_info, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", required=True, help="HF dataset name (no config expected).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prediction_length", type=int, default=24)
    ap.add_argument("--max_series", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    run_id = now_run_id("domain_transfer")
    run_dir = os.path.join(args.out_dir, run_id)
    ensure_dir(run_dir)

    ds = load_hf_timeseries(args.hf_dataset, None, split=args.split, max_series=args.max_series)
    df = ds.df

    train, test, pred, runtime, qcols = run_zero_shot(
        df=df,
        id_col=ds.id_col,
        time_col=ds.time_col,
        target_col=ds.target_col,
        prediction_length=args.prediction_length,
        device=args.device,
    )
    per, agg = evaluate_predictions(test, pred, ds.id_col, ds.time_col, ds.target_col, qcols)

    pred.to_parquet(os.path.join(run_dir, "predictions.parquet"), index=False)
    per.to_csv(os.path.join(run_dir, "metrics_per_series.csv"), index=False)
    pd.DataFrame([agg]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    cfg = vars(args)
    cfg.update({
        "run_id": run_id,
        "run_dir": run_dir,
        "dataset_inferred_cols": {
            "id_col": ds.id_col,
            "time_col": ds.time_col,
            "target_col": ds.target_col,
            "covariate_cols": ds.covariate_cols,
        },
        "runtime_seconds": runtime,
        "aggregate_metrics": agg.to_dict(),
        "system": collect_system_info(),
    })
    save_json(os.path.join(run_dir, "config.json"), cfg)

    print(f"Saved domain transfer run to: {run_dir}")
    print(pd.DataFrame([agg]))

if __name__ == "__main__":
    main()
