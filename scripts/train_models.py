"""Offline training for the FinSentinel stacked ensemble.

Trains one XGBoost + LightGBM pair per horizon (5d, 30d) on a *pooled*,
multi-year, multi-ticker dataset and persists each pair to ``models/`` as a
``.joblib`` bundle the inference path loads at startup.

Design choices that matter:

* POOLED, ticker-agnostic models. We concatenate all tickers into one training
  set and train a single model per horizon, rather than one model per ticker.
  Pooling gives each model far more data and forces it to learn signal that
  generalises across names instead of memorising one ticker's idiosyncrasies;
  it also means a brand-new ticker can be scored without retraining. The
  trade-off is that genuinely ticker-specific dynamics are averaged out — for a
  large-cap universe that is an acceptable price for robustness.

* TIME-SERIES SPLIT, never random. Financial rows are ordered in time and the
  label looks into the future, so a random shuffle would put rows from *after*
  the validation period into training and leak the future. We instead pick a
  single global date cutoff at the 80th percentile of the timeline: everything
  on/before it trains, everything after it validates. Labels are built per
  ticker so a shift never crosses a ticker boundary.

* Reported metrics: ROC-AUC (ranking), F1 (thresholded decision quality at 0.5)
  and Brier score (calibration — how honest the probabilities are; lower is
  better). Brier matters here because the judge and the confidence interval
  consume probabilities, not just the argmax.

Run from the project root:

    python scripts/train_models.py
    python scripts/train_models.py --tickers AAPL MSFT NVDA --years 8
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: render plots to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yfinance as yf  # noqa: E402
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score  # noqa: E402

# Make the project root importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  # noqa: E402

from backend.ml.features.technical import compute_technical_features  # noqa: E402
from backend.ml.models.forecaster import (  # noqa: E402
    FEATURE_COLUMNS,
    HORIZONS,
    LGBM_WEIGHT,
    MODEL_VERSION,
    XGB_WEIGHT,
    EnsembleModel,
    build_lgbm,
    build_xgb,
    model_path,
)
from backend.mlops import registry  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
DEFAULT_YEARS = 7  # comfortably above the 5-year minimum
VALIDATION_FRACTION = 0.20  # most-recent 20% of the timeline held out

# Where the drift detector reads the training feature distribution from.
FEATURE_BASELINE_PATH = PROJECT_ROOT / "models" / "feature_baseline.csv"
SHAP_SAMPLE = 300  # rows sampled for the (relatively expensive) SHAP summary plot


def fetch_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Pull ``years`` of daily, split/dividend-adjusted OHLCV for one ticker."""
    start = (datetime.now() - timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price history for '{ticker}'.")
    return df


def build_pooled_dataset(tickers: list[str], years: int) -> pd.DataFrame:
    """Build one pooled frame of technical features + Close + ticker + date.

    Each ticker is featurised independently (so indicator warm-up never bleeds
    across names), then all are concatenated.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        ohlcv = fetch_ohlcv(ticker, years)
        feats = compute_technical_features(ohlcv)
        sub = feats[FEATURE_COLUMNS + ["Close"]].copy()
        sub["ticker"] = ticker
        sub["date"] = feats.index
        frames.append(sub)
        print(f"  {ticker}: {len(sub):>5} rows ({sub['date'].min().date()} -> {sub['date'].max().date()})")
    pooled = pd.concat(frames, ignore_index=True)
    return pooled


def make_labels(pooled: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach a binary 'up over the next `horizon` days' label, computed per ticker."""
    parts: list[pd.DataFrame] = []
    for _, group in pooled.groupby("ticker", sort=False):
        g = group.sort_values("date").copy()
        future = g["Close"].shift(-horizon)
        g["y"] = (future > g["Close"]).astype("float")  # NaN for the unlabelled tail
        parts.append(g)
    labelled = pd.concat(parts, ignore_index=True)
    return labelled.dropna(subset=FEATURE_COLUMNS + ["y"])


def time_series_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a single global date cutoff at the (1 - VALIDATION_FRACTION) point.

    Using the actual calendar (not row index) keeps the holdout strictly the
    most-recent slice of time across every ticker, so no future row can train a
    model that is then validated on the past.
    """
    unique_dates = np.sort(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * (1.0 - VALIDATION_FRACTION))
    cutoff = unique_dates[cutoff_idx]
    train = df[df["date"] <= cutoff]
    val = df[df["date"] > cutoff]
    return train, val, cutoff


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """ROC-AUC, F1 @0.5, and Brier score for a probability vector."""
    preds = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, preds, zero_division=0),
        "brier": brier_score_loss(y_true, proba),
    }


def train_horizon(pooled: pd.DataFrame, horizon: int, models_dir: Path, version: str) -> dict:
    """Train, validate, persist and return the ensemble + metrics for one horizon."""
    labelled = make_labels(pooled, horizon)
    train, val, cutoff = time_series_split(labelled)

    x_train, y_train = train[FEATURE_COLUMNS], train["y"].to_numpy()
    x_val, y_val = val[FEATURE_COLUMNS], val["y"].to_numpy()

    xgb = build_xgb()
    lgbm = build_lgbm()
    xgb.fit(x_train, y_train)
    lgbm.fit(x_train, y_train)

    p_xgb = xgb.predict_proba(x_val)[:, 1]
    p_lgbm = lgbm.predict_proba(x_val)[:, 1]
    p_ens = XGB_WEIGHT * p_xgb + LGBM_WEIGHT * p_lgbm

    metrics = {
        "xgboost": _metrics(y_val, p_xgb),
        "lightgbm": _metrics(y_val, p_lgbm),
        "ensemble": _metrics(y_val, p_ens),
    }

    print(f"\n=== Horizon {horizon}d ===")
    print(f"  train rows: {len(train):>6}  | val rows: {len(val):>6}  | cutoff: {pd.Timestamp(cutoff).date()}")
    print(f"  val up-rate: {y_val.mean():.3f}")
    print(f"  {'model':<10} {'AUC':>7} {'F1':>7} {'Brier':>8}")
    for name, m in metrics.items():
        print(f"  {name:<10} {m['auc']:>7.4f} {m['f1']:>7.4f} {m['brier']:>8.4f}")

    bundle = {
        "xgb": xgb,
        "lgbm": lgbm,
        "feature_columns": FEATURE_COLUMNS,
        "horizon": horizon,
        "model_version": version,
        "tickers": sorted(pooled["ticker"].unique().tolist()),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "validation_metrics": metrics,
    }
    out = model_path(horizon, models_dir)
    joblib.dump(bundle, out)
    print(f"  saved -> {out}")

    return {"xgb": xgb, "lgbm": lgbm, "metrics": metrics, "x_val": x_val, "horizon": horizon}


def feature_importance_plot(xgb, horizon: int, out_dir: Path) -> Path:
    """Save a horizontal bar chart of XGBoost gain importances for one horizon."""
    importances = xgb.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_COLUMNS[i] for i in order], importances[order], color="#4C72B0")
    ax.set_title(f"XGBoost feature importance (horizon {horizon}d)")
    ax.set_xlabel("importance (gain)")
    fig.tight_layout()
    path = out_dir / f"feature_importance_h{horizon}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def shap_summary_plot(xgb, x_val: pd.DataFrame, horizon: int, out_dir: Path) -> Path:
    """Save a SHAP summary (beeswarm) plot of the XGBoost model for one horizon."""
    sample = x_val.sample(min(SHAP_SAMPLE, len(x_val)), random_state=42)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    path = out_dir / f"shap_summary_h{horizon}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_feature_baseline(pooled: pd.DataFrame) -> Path:
    """Persist the training feature distribution as the drift-detection baseline."""
    FEATURE_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pooled[FEATURE_COLUMNS].to_csv(FEATURE_BASELINE_PATH, index=False)
    return FEATURE_BASELINE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist FinSentinel ensembles.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to pool.")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="Years of history (>=5).")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to write .joblib bundles into.",
    )
    args = parser.parse_args()

    if args.years < 5:
        parser.error("Need at least 5 years of history for a meaningful time-series split.")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    version = f"{MODEL_VERSION}-pooled-{datetime.now():%Y%m%d-%H%M%S}"

    print(f"Pooling {len(args.tickers)} tickers over ~{args.years}y: {', '.join(args.tickers)}")
    pooled = build_pooled_dataset(args.tickers, args.years)
    print(f"Pooled dataset: {len(pooled)} feature rows across {pooled['ticker'].nunique()} tickers.")

    # Point MLflow at the local SQLite store and our experiment.
    registry.configure()
    mlflow.set_experiment(experiment_id=registry.ensure_experiment())

    with mlflow.start_run(run_name=version) as run:
        # --- params: what defined this training run -------------------------
        xgb_params = build_xgb().get_params()
        lgbm_params = build_lgbm().get_params()
        mlflow.log_params(
            {
                "model_type": "stacked-ensemble(xgboost+lightgbm)",
                "tickers": ",".join(sorted(args.tickers)),
                "n_tickers": pooled["ticker"].nunique(),
                "years": args.years,
                "date_start": str(pooled["date"].min().date()),
                "date_end": str(pooled["date"].max().date()),
                "n_rows": len(pooled),
                "validation_fraction": VALIDATION_FRACTION,
                "horizons": ",".join(str(h) for h in HORIZONS),
                "feature_columns": ",".join(FEATURE_COLUMNS),
                "xgb_weight": XGB_WEIGHT,
                "lgbm_weight": LGBM_WEIGHT,
                "xgb_n_estimators": xgb_params["n_estimators"],
                "xgb_max_depth": xgb_params["max_depth"],
                "xgb_learning_rate": xgb_params["learning_rate"],
                "lgbm_n_estimators": lgbm_params["n_estimators"],
                "lgbm_max_depth": lgbm_params["max_depth"],
                "lgbm_learning_rate": lgbm_params["learning_rate"],
                "model_version": version,
            }
        )

        # --- train each horizon; log metrics + plots ------------------------
        plot_dir = args.models_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        all_models: dict[int, tuple] = {}
        for horizon in HORIZONS:
            res = train_horizon(pooled, horizon, args.models_dir, version)
            all_models[horizon] = (res["xgb"], res["lgbm"])
            # metrics: auc/f1/brier per model per horizon
            for model_name, m in res["metrics"].items():
                for metric_name, value in m.items():
                    mlflow.log_metric(f"{metric_name}_{model_name}_h{horizon}", value)
            # artifacts: SHAP summary + feature importance plots
            mlflow.log_artifact(str(feature_importance_plot(res["xgb"], horizon, plot_dir)), "plots")
            mlflow.log_artifact(str(shap_summary_plot(res["xgb"], res["x_val"], horizon, plot_dir)), "plots")

        # --- artifact: feature baseline for drift detection -----------------
        baseline = save_feature_baseline(pooled)
        mlflow.log_artifact(str(baseline), "baseline")

        # --- register the ensemble and stage it -----------------------------
        ensemble = EnsembleModel(models=all_models, feature_columns=FEATURE_COLUMNS, model_version=version)
        model_info = mlflow.pyfunc.log_model(
            name="forecaster",
            python_model=ensemble,
            registered_model_name=registry.MODEL_NAME,
        )
        client = mlflow.MlflowClient()
        new_version = model_info.registered_model_version
        # MLflow 3 uses aliases (stages were removed): tag this version "staging".
        client.set_registered_model_alias(registry.MODEL_NAME, registry.STAGING_ALIAS, new_version)

        print(f"\nMLflow run_id   : {run.info.run_id}")
        print(f"Registered model: {registry.MODEL_NAME} v{new_version} -> alias '@{registry.STAGING_ALIAS}'")
        print(f"Tracking URI    : {registry.TRACKING_URI}")
        print("Promote to production with: python scripts/promote_model.py")

    print(f"\nDone. Models written to {args.models_dir}")


if __name__ == "__main__":
    main()
