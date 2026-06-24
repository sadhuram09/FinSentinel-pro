"""SHAP TreeExplainer wrapper.

Financial predictions are only actionable if a human can see *why* the model
leaned a given way. SHAP values decompose a single prediction into additive
per-feature contributions with a sound game-theoretic basis, and
``TreeExplainer`` computes them exactly and cheaply for gradient-boosted trees.
We surface the top-5 contributions so the UI can show the dominant drivers
without overwhelming the analyst.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from backend.schemas import ShapAttribution

TOP_N = 5


def explain_prediction(
    model,
    sample: pd.DataFrame,
    feature_columns: list[str],
    top_n: int = TOP_N,
) -> list[ShapAttribution]:
    """Return the top-N feature attributions for a single-row prediction.

    Args:
        model: A fitted tree model (XGBoost/LightGBM) compatible with
            ``shap.TreeExplainer``.
        sample: One-row DataFrame of feature values to explain.
        feature_columns: Ordered feature names matching the model's inputs.
        top_n: How many of the highest-magnitude contributions to return.

    Returns:
        A list of :class:`ShapAttribution`, ordered by absolute contribution
        descending. Contributions are signed: positive pushes the prediction
        toward the "up" class, negative toward "down".
    """
    explainer = shap.TreeExplainer(model)
    x = sample[feature_columns].to_numpy()
    shap_values = explainer.shap_values(x)

    # For binary classifiers SHAP may return a list (per class) or an array.
    # Normalise to the positive-class contribution vector for the single sample.
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # shape (n_classes, n_samples, n_features) -> positive class, first row
        contribs = values[1][0] if values.shape[0] > 1 else values[0][0]
    elif values.ndim == 2:
        contribs = values[0]
    else:
        contribs = values

    contribs = np.asarray(contribs, dtype=float).ravel()

    attributions = [
        ShapAttribution(
            feature=feature_columns[i],
            value=float(sample.iloc[0][feature_columns[i]]),
            contribution=float(contribs[i]),
        )
        for i in range(len(feature_columns))
    ]
    attributions.sort(key=lambda a: abs(a.contribution), reverse=True)
    return attributions[:top_n]
