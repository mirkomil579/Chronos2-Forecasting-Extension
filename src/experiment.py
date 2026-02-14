from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from chronos import Chronos2Pipeline

from .data import train_test_split_last_horizon
from .metrics import compute_metrics_per_series, aggregate_metrics

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

@dataclass
class RunOutputs:
    run_dir: str
    predictions: pd.DataFrame
    per_series_metrics: pd.DataFrame
    aggregate_metrics: pd.Series
    runtime_seconds: float

def run_zero_shot(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    prediction_length: int,
    device: str,
    model_id: str = "amazon/chronos-2",
    quantiles: Optional[List[float]] = None,
    max_series_plots: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, Dict[float, str]]:
    quantiles = quantiles or DEFAULT_QUANTILES
    qcols = {q: f"q{int(q*100):02d}" for q in quantiles}

    train_df, test_df = train_test_split_last_horizon(df, id_col, time_col, prediction_length)

    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)

    start = time.time()
    # Chronos expects full panel with id/timestamp/target and optional covariates in columns.
    pred = pipeline.predict_df(
        context_df=train_df,
        prediction_length=prediction_length,
        quantiles=quantiles,
        # return_index=True yields id/timestamp keys (depends on version); we handle below
    )
    runtime = time.time() - start

    # Standardize prediction output: ensure id/time present.
    if id_col not in pred.columns:
        if "id" in pred.columns:
            pred.rename(columns={"id": id_col}, inplace=True)
    if time_col not in pred.columns:
        if "timestamp" in pred.columns:
            pred.rename(columns={"timestamp": time_col}, inplace=True)

    # Rename quantile cols if needed
    for q in quantiles:
        # common outputs: 'quantile_0.1' or 'p10' etc; try to match
        for cand in [f"quantile_{q}", f"q{q}", f"p{int(q*100)}", str(q)]:
            if cand in pred.columns:
                pred.rename(columns={cand: qcols[q]}, inplace=True)
                break
    # If the library already uses q10/q50/q90, accept.
    for q in quantiles:
        if qcols[q] not in pred.columns:
            # fallback: maybe 'q10' exists
            alt = f"q{int(q*100)}"
            if alt in pred.columns:
                pred.rename(columns={alt: qcols[q]}, inplace=True)

    return train_df, test_df, pred, runtime, qcols

def evaluate_predictions(
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    qcols: Dict[float, str],
) -> Tuple[pd.DataFrame, pd.Series]:
    per_series = compute_metrics_per_series(
        y_true_df=test_df,
        y_pred_df=pred_df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        quantile_cols=qcols,
        point_quantile=0.5,
    )
    agg = aggregate_metrics(per_series)
    return per_series, agg
