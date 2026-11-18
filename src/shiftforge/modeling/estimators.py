"""Train-only preprocessing and deterministic regression estimators."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


FEATURES = ("pickup_zone", "pickup_hour", "pickup_weekday", "pickup_month")


def build_estimators(seed: int) -> dict[str, object]:
    """Construct unfitted models. No encoder may be fitted before train selection.

    The zone feature can contain 265 identifiers, exceeding the 255-category
    limit of histogram gradient boosting. OrdinalEncoder consolidates categories
    using *training frequencies only*. HistGB interprets encoded values as
    nominal categories, not as a numerical ordering.
    """
    ridge_preprocess = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), [0, 1, 2, 3])],
        remainder="drop",
    )
    hist_preprocess = ColumnTransformer(
        [
            ("zone", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                max_categories=128,
            ), [0]),
            ("calendar", "passthrough", [1, 2, 3]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return {
        "median": DummyRegressor(strategy="median"),
        "ridge": Pipeline([
            ("preprocess", ridge_preprocess),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", hist_preprocess),
            ("regressor", HistGradientBoostingRegressor(
                categorical_features=[0],
                max_iter=80,
                max_leaf_nodes=15,
                learning_rate=0.08,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=seed,
            )),
        ]),
    }
