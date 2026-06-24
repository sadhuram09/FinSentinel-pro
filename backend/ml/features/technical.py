"""Technical indicators computed from OHLCV price history via pandas-ta.

Technical indicators turn raw price/volume series into bounded, comparable
signals that encode momentum, trend and volatility. Tree-based models consume
these directly; without them the model would have to rediscover well-known
market patterns (e.g. mean reversion) from scratch, wasting capacity.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from backend.schemas import TechnicalIndicators

# Standard parameterisations. These are the textbook defaults traders use, so
# the indicator values are interpretable against published thresholds.
RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_LENGTH = 20
BB_STD = 2.0


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Append RSI, MACD and Bollinger Band columns to an OHLCV frame.

    Args:
        ohlcv: DataFrame indexed by date with at least a ``Close`` column.

    Returns:
        The input frame with indicator columns added. Rows where indicators are
        undefined (the warm-up window) contain NaN and should be dropped before
        training/inference.
    """
    df = ohlcv.copy()
    close = df["Close"]

    # RSI: ratio of recent gains to losses. Bounded 0-100, it flags
    # overbought (>70) / oversold (<30) conditions that often precede reversals.
    df["rsi"] = ta.rsi(close, length=RSI_LENGTH)

    # MACD: difference of a fast and slow EMA, plus a signal line and histogram.
    # It captures trend direction and momentum acceleration in one transform.
    # pandas-ta column names carry the parameters; we select by prefix so the
    # code is robust to version-specific suffix changes (0.3.x vs 0.4.x).
    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df["macd"] = _col_by_prefix(macd, "MACD_")
    df["macd_signal"] = _col_by_prefix(macd, "MACDs_")
    df["macd_hist"] = _col_by_prefix(macd, "MACDh_")

    # Bollinger Bands: a moving average +/- N standard deviations. They
    # normalise price against its own recent volatility, so the same model
    # generalises across high- and low-volatility regimes.
    bb = ta.bbands(close, length=BB_LENGTH, std=BB_STD)
    df["bb_lower"] = _col_by_prefix(bb, "BBL_")
    df["bb_middle"] = _col_by_prefix(bb, "BBM_")
    df["bb_upper"] = _col_by_prefix(bb, "BBU_")
    # %B locates price within the band envelope (0 = lower, 1 = upper); it is
    # the single most model-friendly distillation of the bands.
    df["bb_pct"] = _col_by_prefix(bb, "BBP_")

    return df


def _col_by_prefix(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Select the single pandas-ta output column starting with ``prefix``.

    pandas-ta encodes indicator parameters into column names (e.g.
    ``BBL_20_2.0_2.0``), and the exact suffix varies across library versions.
    Matching on the stable prefix keeps feature extraction version-tolerant.
    """
    matches = [c for c in frame.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"No pandas-ta column with prefix '{prefix}' in {list(frame.columns)}")
    return frame[matches[0]]


def latest_technical_snapshot(ohlcv: pd.DataFrame) -> TechnicalIndicators:
    """Return the most recent fully-defined indicator row as a Pydantic model."""
    enriched = compute_technical_features(ohlcv)
    cols = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_pct",
    ]
    row = enriched.dropna(subset=cols).iloc[-1]
    return TechnicalIndicators(
        rsi=float(row["rsi"]),
        macd=float(row["macd"]),
        macd_signal=float(row["macd_signal"]),
        macd_hist=float(row["macd_hist"]),
        bb_upper=float(row["bb_upper"]),
        bb_middle=float(row["bb_middle"]),
        bb_lower=float(row["bb_lower"]),
        bb_pct=float(row["bb_pct"]),
    )
