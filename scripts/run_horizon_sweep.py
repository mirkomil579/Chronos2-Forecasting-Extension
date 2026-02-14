import argparse
import os
import pandas as pd

from chronos2_project.data import load_hf_timeseries
from chronos2_project.experiment import run_zero_shot, evaluate_predictions
from chronos2_project.utils import now_run_id, ensure_dir, collect_system_info, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--horizons", required=True, help="Comma-separated horizons, e.g. 24,48,96")
    ap.add_argument("--max_series", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]

    run_id = now_run_id("horizon_sweep")
    run_dir = os.path.join(args.out_dir, run_id)
    ensure_dir(run_dir)

    ds = load_hf_timeseries(args.hf_dataset, args.config, split=args.split, max_series=args.max_series)
    df = ds.df

    rows = []
    for h in horizons:
        train, test, pred, runtime, qcols = run_zero_shot(
            df=df,
            id_col=ds.id_col,
            time_col=ds.time_col,
            target_col=ds.target_col,
            prediction_length=h,
            device=args.device,
        )
        per, agg = evaluate_predictions(test, pred, ds.id_col, ds.time_col, ds.target_col, qcols)
        rows.append({"prediction_length": h, **agg.to_dict(), "runtime_seconds": runtime, "n_series": len(per)})
        # Save per-horizon metrics
        per.to_csv(os.path.join(run_dir, f"metrics_per_series_h{h}.csv"), index=False)

    out = pd.DataFrame(rows).sort_values("prediction_length")
    out.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

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
        "system": collect_system_info(),
    })
    save_json(os.path.join(run_dir, "config.json"), cfg)

    print(f"Saved horizon sweep to: {run_dir}")
    print(out)

if __name__ == "__main__":
    main()
