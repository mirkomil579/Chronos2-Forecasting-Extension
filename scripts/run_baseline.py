import argparse
import os
import pandas as pd

from chronos2_project.data import load_hf_timeseries
from chronos2_project.experiment import run_zero_shot, evaluate_predictions
from chronos2_project.plotting import plot_forecast
from chronos2_project.utils import now_run_id, ensure_dir, collect_system_info, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", required=True)
    ap.add_argument("--config", default=None, help="HF config name (dataset subset).")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prediction_length", type=int, default=48)
    ap.add_argument("--max_series", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    run_id = now_run_id("baseline")
    run_dir = os.path.join(args.out_dir, run_id)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "figures"))

    ds = load_hf_timeseries(args.hf_dataset, args.config, split=args.split, max_series=args.max_series)
    df = ds.df

    train_df, test_df, pred_df, runtime, qcols = run_zero_shot(
        df=df,
        id_col=ds.id_col,
        time_col=ds.time_col,
        target_col=ds.target_col,
        prediction_length=args.prediction_length,
        device=args.device,
    )

    per_series, agg = evaluate_predictions(test_df, pred_df, ds.id_col, ds.time_col, ds.target_col, qcols)

    # Save outputs
    pred_path = os.path.join(run_dir, "predictions.parquet")
    metrics_path = os.path.join(run_dir, "metrics_per_series.csv")
    agg_path = os.path.join(run_dir, "metrics.csv")

    pred_df.to_parquet(pred_path, index=False)
    per_series.to_csv(metrics_path, index=False)
    pd.DataFrame([agg]).to_csv(agg_path, index=False)

    # Plot a few series
    for i, sid in enumerate(per_series["series_id"].head(10)):
        h = train_df[train_df[ds.id_col].astype(str) == str(sid)]
        t = test_df[test_df[ds.id_col].astype(str) == str(sid)]
        p = pred_df[pred_df[ds.id_col].astype(str) == str(sid)]
        if len(h)==0 or len(t)==0 or len(p)==0:
            continue
        plot_forecast(h, t, p, ds.id_col, ds.time_col, ds.target_col, qcols,
                      out_path=os.path.join(run_dir, "figures", f"forecast_{i:02d}.png"),
                      title=f"{args.hf_dataset}:{args.config} id={sid}")

    # Save run config + system info
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

    print(f"Saved baseline run to: {run_dir}")
    print("Aggregate metrics:")
    print(pd.DataFrame([agg]))

if __name__ == "__main__":
    main()
