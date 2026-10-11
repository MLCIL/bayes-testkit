"""Allowed option sets for the signed-rank test estimator.

The ``decision_table`` columns follow the t-tests' skeleton, less the location
estimate and its HDI, which the nonparametric test does not estimate.
"""

ALL_SIGNED_RANK_COLUMNS: list[str] = [
    "comparison",
    "p_left",
    "p_rope",
    "p_right",
    "mean_theta_left",
    "mean_theta_rope",
    "mean_theta_right",
    "decision",
    "decision_raw",
]
