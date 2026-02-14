from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))

def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    # y_pred_q: predicted quantile
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def weighted_quantile_loss(y_true: np.ndarray, quantile_preds: Dict[float, np.ndarray], quantiles: List[float]) -> float:
    losses = [pinball_loss(y_true, quantile_preds[q], q) for q in quantiles]
    return float(np.mean(losses))

@dataclass
class MetricRow:
    series_id: str
    n: int
    mae: float
    rmse: float
    smape: float
    wql: Optional[float] = None

def compute_metrics_per_series(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    quantile_cols: Optional[Dict[float, str]] = None,
    point_quantile: float = 0.5,
) -> pd.DataFrame:
    """y_pred_df must contain the same (id,time) keys, plus quantile columns."""
    quantiles = sorted(list(quantile_cols.keys())) if quantile_cols else []
    rows: List[MetricRow] = []
    for sid, g_true in y_true_df.groupby(id_col, sort=False):
        g_pred = y_pred_df[y_pred_df[id_col] == sid].sort_values(time_col)
        g_true = g_true.sort_values(time_col)
        if len(g_pred) != len(g_true) or len(g_true) == 0:
            continue
        y_true = g_true[target_col].to_numpy(dtype=float)
        if quantile_cols:
            y_point = g_pred[quantile_cols[point_quantile]].to_numpy(dtype=float)
        else:
            # fallback
            y_point = g_pred["prediction"].to_numpy(dtype=float)
        qpreds = {}
        if quantile_cols:
            for q, col in quantile_cols.items():
                qpreds[q] = g_pred[col].to_numpy(dtype=float)
        rows.append(MetricRow(
            series_id=str(sid),
            n=len(y_true),
            mae=mae(y_true, y_point),
            rmse=rmse(y_true, y_point),
            smape=smape(y_true, y_point),
            wql=weighted_quantile_loss(y_true, qpreds, quantiles) if quantile_cols else None,
        ))
    out = pd.DataFrame([r.__dict__ for r in rows])
    return out

def aggregate_metrics(per_series: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["mae","rmse","smape","wql"] if c in per_series.columns]
    return per_series[cols].mean(numeric_only=True)
