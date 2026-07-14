"""Normalize provider responses into internal Polars schemas."""
from __future__ import annotations

import polars as pl

from app.indicators.pipeline import filter_halt_days

DAILY_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "quote_ts"]
ADJ_FACTOR_COLS = ["symbol", "trade_date", "ex_factor"]
INSTRUMENT_COLS = ["symbol", "name", "code", "exchange", "asset_type", "source"]


def _safe_from_pandas(df):
    """Convert a pandas-like DataFrame to Polars without requiring pyarrow.

    Handles NumPy 2.x DTypes (Int64DType, Float64DType) and pandas extension
    types (StringDtype, etc.) by converting them to traditional numpy-backed
    types first. This avoids the pyarrow dependency which can cause segfaults
    on macOS (jemalloc conflict).

    Uses a two-phase approach:
      1. Known extension types → explicit numpy-backed conversion.
      2. Any remaining non-numpy columns → generic safe conversion via pandas
         nullable-dtype to standard-dtype mapping, falling back to object.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)

    conversions: dict[str, object] = {}
    for col_name in df.columns:
        dtype = df[col_name].dtype
        dtype_mod = type(dtype).__module__

        # NumPy 2.x DTypes (numpy.dtypes.Int64DType / Float64DType etc.)
        if dtype_mod.startswith("numpy.dtypes"):
            if isinstance(dtype, np.dtypes.Int64DType):
                conversions[col_name] = df[col_name].astype("int64")
            elif isinstance(dtype, np.dtypes.Float64DType):
                conversions[col_name] = df[col_name].astype("float64")
            elif isinstance(dtype, np.dtypes.Int32DType):
                conversions[col_name] = df[col_name].astype("int32")
            elif isinstance(dtype, np.dtypes.Float32DType):
                conversions[col_name] = df[col_name].astype("float32")
            else:
                # Other numpy 2.x DTypes: try conversion via numpy_dtype if available
                try:
                    conversions[col_name] = df[col_name].astype(dtype)
                except Exception:  # noqa: BLE001
                    pass
            continue

        # Pandas extension types (StringDtype, ArrowDtype, Int64, etc.)
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            if isinstance(dtype, pd.StringDtype):
                conversions[col_name] = df[col_name].astype("object")
            elif isinstance(dtype, pd.BooleanDtype):
                conversions[col_name] = df[col_name].astype("object")
            elif isinstance(dtype, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                                    pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype)):
                conversions[col_name] = df[col_name].astype("float64")
            elif isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
                conversions[col_name] = df[col_name].astype("float64")
            else:
                # Fallback for other extension types (ArrowDtype etc.)
                try:
                    conversions[col_name] = df[col_name].astype("object")
                except Exception:  # noqa: BLE001
                    pass
            continue

        # Phase 2: any remaining non-numpy columns (e.g. pandas nullable types
        # that don't subclass ExtensionDtype in some versions, or types from
        # reset_index() that produce non-standard dtypes).
        if not hasattr(dtype, 'kind') or str(getattr(dtype, 'kind', '')) not in 'biufc':
            try:
                # Try to convert to numpy-backed via to_numpy
                converted = pd.Series(df[col_name].to_numpy(), name=col_name)
                # If the series is object, keep it as object; otherwise let
                # astype determine the natural numpy dtype.
                if converted.dtype.kind == 'O':
                    conversions[col_name] = converted
                else:
                    conversions[col_name] = converted.astype(converted.dtype)
            except Exception:  # noqa: BLE001
                try:
                    conversions[col_name] = df[col_name].astype("object")
                except Exception:  # noqa: BLE001
                    pass

    if conversions:
        df = df.copy()
        for col_name, converted in conversions.items():
            df[col_name] = converted

    return pl.from_pandas(df)


def to_polars(data) -> pl.DataFrame:
    if data is None:
        return pl.DataFrame()
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, dict):
        rows: list[dict] = []
        for sym, values in data.items():
            for item in values or []:
                row = dict(item or {})
                row.setdefault("symbol", sym)
                rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()
    if hasattr(data, "reset_index"):
        return _safe_from_pandas(data.reset_index())
    try:
        return pl.DataFrame(data)
    except Exception:  # noqa: BLE001
        return pl.DataFrame()


def normalize_daily(data, default_symbol: str | None = None, source: str = "tickflow") -> pl.DataFrame:  # noqa: ARG001
    df = to_polars(data)
    if df.is_empty():
        return df
    rename_map = {
        "ts_code": "symbol",
        "trade_date": "date",
        "datetime": "date",
        "vol": "volume",
        "amt": "amount",
        "timestamp": "quote_ts",
    }
    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
    if "symbol" not in df.columns and default_symbol:
        df = df.with_columns(pl.lit(default_symbol).alias("symbol"))
    if "date" in df.columns and df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    # quote_ts: 毫秒级行情时间戳, 用于盘后校验/量比折算。保留为 Int64, 缺失则置 null。
    if "quote_ts" in df.columns:
        df = df.with_columns(pl.col("quote_ts").cast(pl.Int64, strict=False))
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    df = filter_halt_days(df)
    keep = [c for c in DAILY_COLS if c in df.columns]
    return df.select(keep) if keep else pl.DataFrame()


def normalize_adj_factors(data, source: str = "tickflow") -> pl.DataFrame:  # noqa: ARG001
    df = to_polars(data)
    if df.is_empty():
        return df
    rename_map = {
        "timestamp": "trade_date",
        "date": "trade_date",
        "adj_factor": "ex_factor",
    }
    df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})
    if "trade_date" in df.columns:
        if df.schema["trade_date"] in {pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64, pl.Float32}:
            df = df.with_columns(
                pl.from_epoch(pl.col("trade_date").cast(pl.Int64), time_unit="ms").dt.date().alias("trade_date")
            )
        else:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date, strict=False))
    if "ex_factor" in df.columns:
        df = df.with_columns(pl.col("ex_factor").cast(pl.Float64, strict=False))
    keep = [c for c in ADJ_FACTOR_COLS if c in df.columns]
    return df.select(keep).drop_nulls() if len(keep) == len(ADJ_FACTOR_COLS) else pl.DataFrame()


def normalize_instruments(rows: list[dict], asset_type: str, source: str = "tickflow") -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    out: list[dict] = []
    for item in rows:
        symbol = item.get("symbol")
        if not symbol:
            continue
        out.append({
            "symbol": str(symbol),
            "name": item.get("name") or str(symbol),
            "code": item.get("code") or str(symbol).split(".")[0],
            "exchange": item.get("exchange"),
            "asset_type": asset_type,
            "source": source,
        })
    if not out:
        return pl.DataFrame()
    return pl.DataFrame(out).select(INSTRUMENT_COLS).unique(subset=["symbol"], keep="last").sort("symbol")
