from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

@dataclass
class LoadedDataset:
    df: pd.DataFrame
    id_col: str
    time_col: str
    target_col: str
    covariate_cols: List[str]
    freq: Optional[str] = None

def list_hf_configs(dataset_name: str) -> List[str]:
    return get_dataset_config_names(dataset_name)

def _infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    # Expected by Chronos pandas API: id, timestamp, target
    candidates = [
        ("id", "timestamp", "target"),
        ("item_id", "timestamp", "target"),
        ("series_id", "timestamp", "target"),
    ]
    for a,b,c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            return a,b,c
    # Some datasets use 'ds'/'y'
    if "ds" in df.columns and "y" in df.columns:
        # create id if missing
        if "id" not in df.columns:
            df["id"] = 0
        df.rename(columns={"ds":"timestamp","y":"target"}, inplace=True)
        return "id","timestamp","target"
    raise ValueError(f"Could not infer (id,timestamp,target) from columns: {list(df.columns)[:50]}")  # pragma: no cover

def load_hf_timeseries(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "test",
    max_series: Optional[int] = None,
) -> LoadedDataset:
    """Loads a Hugging Face time-series dataset and returns a normalized pandas DataFrame.

    The function tries to:
    - load the dataset split as a table
    - infer id/time/target columns
    - convert timestamp to pandas datetime
    - keep covariates as any remaining numeric columns (excluding id/time/target)
    """
    ds = load_dataset(dataset_name, config_name, split=split)
    df = ds.to_pandas()

    id_col, time_col, target_col = _infer_columns(df)

    # Standardize
    df[time_col] = pd.to_datetime(df[time_col])
    # Sort
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Optionally sub-sample series IDs for speed
    if max_series is not None:
        unique_ids = df[id_col].unique()
        keep_ids = unique_ids[:max_series]
        df = df[df[id_col].isin(keep_ids)].reset_index(drop=True)

    # Identify covariates: numeric columns excluding id/time/target
    covariate_cols = []
    for col in df.columns:
        if col in (id_col, time_col, target_col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            covariate_cols.append(col)

    return LoadedDataset(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        covariate_cols=covariate_cols,
        freq=None,
    )

def train_test_split_last_horizon(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    prediction_length: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split: for each series, last `prediction_length` rows are test."""
    parts_train = []
    parts_test = []
    for _, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        if len(g) <= prediction_length:
            continue
        parts_train.append(g.iloc[:-prediction_length])
        parts_test.append(g.iloc[-prediction_length:])
    train = pd.concat(parts_train, axis=0).reset_index(drop=True) if parts_train else df.iloc[0:0].copy()
    test = pd.concat(parts_test, axis=0).reset_index(drop=True) if parts_test else df.iloc[0:0].copy()
    return train, test
