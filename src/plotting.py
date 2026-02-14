from __future__ import annotations
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(
    history: pd.DataFrame,
    future_true: pd.DataFrame,
    future_pred: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    quantile_cols: Dict[float, str],
    out_path: str,
    title: Optional[str] = None,
    max_points_history: int = 400,
) -> None:
    plt.figure()
    sid = history[id_col].iloc[0]
    h = history.sort_values(time_col).tail(max_points_history)
    t = future_true.sort_values(time_col)
    p = future_pred.sort_values(time_col)

    plt.plot(h[time_col], h[target_col], label="history")
    plt.plot(t[time_col], t[target_col], label="true")

    # median
    plt.plot(p[time_col], p[quantile_cols[0.5]], label="pred q50")

    # uncertainty band
    if 0.1 in quantile_cols and 0.9 in quantile_cols:
        plt.fill_between(
            p[time_col],
            p[quantile_cols[0.1]],
            p[quantile_cols[0.9]],
            alpha=0.2,
            label="q10-q90",
        )

    plt.xlabel("time")
    plt.ylabel(target_col)
    plt.legend()
    plt.title(title or f"Forecast (id={sid})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
