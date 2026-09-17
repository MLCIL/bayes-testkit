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

# Figures dispatched through ``BBTTest.plot(kind=...)``.
PlotKindType = Literal["strong-posterior", "weak-posterior"]
PlotOrientationType = Literal["horizontal", "vertical"]

InterpretationTypes = Literal[
    "weak",
    "strong",
]

ALL_PROPERTIES_COLUMNS: list[ReportedPropertyColumnType] = list(
    get_args(ReportedPropertyColumnType)
)
