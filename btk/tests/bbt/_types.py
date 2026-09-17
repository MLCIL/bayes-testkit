from typing import Literal, get_args

HyperPriorType = Literal[
    "log_normal",
    "cauchy",
    "normal",
]

TieSolverType = Literal["add", "spread", "forget", "davidson"]

ReportedPropertyColumnType = Literal[
    "left_model",
    "right_model",
    "median",
    "mean",
    "hdi_low",
    "hdi_high",
    "delta",
    "above_50",
    "in_rope",
    "weak_interpretation",
    "strong_interpretation",
    "weak_interpretation_raw",
    "strong_interpretation_raw",
]

InterpretationTypes = Literal[
    "weak",
    "strong",
]

ALL_PROPERTIES_COLUMNS: list[ReportedPropertyColumnType] = list(
    get_args(ReportedPropertyColumnType)
)
