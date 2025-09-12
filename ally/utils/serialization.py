"""
Utilities to normalize datetime-like values for JSON/tool I/O.
Converts pandas/NumPy datetimes to UTC ISO-8601 strings.
"""
from __future__ import annotations
from typing import Any, Mapping, Iterable
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ISO_FMT_Z = "%Y-%m-%dT%H:%M:%SZ"

def _to_utc_py_datetime(dt: Any) -> datetime:
    """Accept pandas.Timestamp, numpy.datetime64, naive/aware datetime"""
    if isinstance(dt, pd.Timestamp):
        ts = dt.tz_convert("UTC") if dt.tzinfo else dt.tz_localize("UTC")
        return ts.to_pydatetime()
    if isinstance(dt, np.datetime64):
        # convert to ns, then to py datetime (UTC)
        ns = dt.astype("datetime64[ns]").astype("int64")
        return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    # Not a datetime-like
    return dt

def to_iso_utc(val: Any) -> Any:
    """Return ISO-8601 Z string for datetime-like; otherwise original."""
    dt = _to_utc_py_datetime(val)
    if isinstance(dt, datetime):
        return dt.strftime(ISO_FMT_Z)
    return val

def convert_timestamps(obj: Any) -> Any:
    """
    Recursively convert datetime-like values to ISO-8601 Z strings.
    Handles dicts, lists/tuples, pandas Series/DataFrames, Index/DatetimeIndex.
    """
    # pandas DataFrame: convert index + datetime columns
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        
        # Use a temporary index name to avoid collisions
        idx_name = "__ally_idx__"
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df = df.reset_index(names=[idx_name])
            if pd.api.types.is_datetime64_any_dtype(df[idx_name]):
                # Fast path for datetime64[ns] columns
                if pd.api.types.is_datetime64_ns_dtype(df[idx_name].dtype):
                    s = df[idx_name].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
                    df[idx_name] = s.dt.strftime(ISO_FMT_Z)
                else:
                    df[idx_name] = df[idx_name].map(to_iso_utc)
            # Only promote the index column to 'timestamp' if there isn't one already
            if "timestamp" not in df.columns:
                df = df.rename(columns={idx_name: "timestamp"})
        
        # Convert datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Fast path for datetime64[ns] columns
                if pd.api.types.is_datetime64_ns_dtype(df[col].dtype):
                    # Check if already timezone-aware
                    if df[col].dt.tz is not None:
                        # Already timezone-aware, convert to UTC and format
                        s = df[col].dt.tz_convert("UTC")
                    else:
                        # Timezone-naive, localize to UTC first
                        s = df[col].dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
                    df[col] = s.dt.strftime(ISO_FMT_Z)
                else:
                    df[col] = df[col].map(to_iso_utc)
        
        return df

    # pandas Series
    if isinstance(obj, pd.Series):
        s = obj.copy()
        if pd.api.types.is_datetime64_any_dtype(s):
            s = s.map(to_iso_utc)
        return s

    # DatetimeIndex / Index element
    if isinstance(obj, pd.DatetimeIndex):
        return [to_iso_utc(x) for x in obj.tolist()]
    if isinstance(obj, pd.PeriodIndex):
        try:
            ts_list = obj.to_timestamp().tolist()  # PeriodIndex → DatetimeIndex
        except Exception:
            ts_list = pd.DatetimeIndex(obj.astype("datetime64[ns]")).tolist()
        return [to_iso_utc(x) for x in ts_list]

    # Mapping
    if isinstance(obj, Mapping):
        return {k: convert_timestamps(v) for k, v in obj.items()}

    # Iterable containers
    if isinstance(obj, (list, tuple)):
        return [convert_timestamps(v) for v in obj]

    # Scalar datetime-like
    iso = to_iso_utc(obj)
    return iso