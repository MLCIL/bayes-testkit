"""Shared numerical helpers for the correlated and hierarchical t-tests.

Everything here is test-agnostic: resolving the ROPE and the Nadeau-Bengio
correlation, computing a highest-density interval from samples, turning a
three-way ROPE partition into the shared decision vocabulary, wrapping samples
into an :class:`arviz.InferenceData`, and the small printable ``summary``
object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arviz as az
import numpy as np
import pandas as pd

# Shared decision label vocabulary. The raw symbols mirror BBT's ``>``/``=``/
# ``?``, with an added ``<`` because the t-tests compare a *fixed* left vs.
# right pair (BBT orders its pairs strongest-first and so never needs ``<``).
DECISION_LEFT_BETTER = ">"
DECISION_RIGHT_BETTER = "<"
DECISION_EQUIVALENT = "="
DECISION_UNDETERMINED = "?"


def resolve_rope(rope: float | tuple[float, float]) -> tuple[float, float]:
    """Resolve a ROPE argument to an explicit ``(lo, hi)`` band on the difference scale.

    A scalar is interpreted as a half-width ``r`` giving the symmetric band
    ``[-r, r]`` (this is what keeps the CLI's ``--rope`` a consistent
    half-width shape across tests). A 2-tuple is used verbatim.

    Parameters
    ----------
    rope : float | tuple[float, float]
        Half-width ``r`` (scalar) or explicit ``(lo, hi)`` band.

    Returns
    -------
    tuple[float, float]
        The resolved ``(lo, hi)`` interval, with ``lo < hi``.
    """
    if isinstance(rope, tuple):
        lo, hi = rope
        if lo >= hi:
            raise ValueError(f"ROPE band must satisfy lo < hi, got ({lo}, {hi}).")
        return float(lo), float(hi)
    if rope < 0:
        raise ValueError(f"ROPE half-width must be non-negative, got {rope}.")
    return -float(rope), float(rope)


def resolve_fold_structure(fold_ids: pd.Series, fold_col: str) -> tuple[int, int]:
    """Derive the cross-validation structure from a fold-identifier column.

    Rows sharing a fold identifier are repeated runs of that same fold, so the
    number of distinct identifiers is the number of folds ``k`` and the number
    of times each recurs is the number of runs. The design must be balanced --
    every fold repeated the same number of times -- otherwise the correlation
    heuristic is not well defined.

    Parameters
    ----------
    fold_ids : pd.Series
        The fold identifier of each row.
    fold_col : str
        Name of the column, used in error messages.

    Returns
    -------
    tuple[int, int]
        The ``(n_folds, n_runs)`` of the design.

    Raises
    ------
    ValueError
        If the folds are not all repeated the same number of times.
    """
    counts = pd.Series(fold_ids).value_counts()
    run_counts = set(counts.to_numpy().tolist())
    if len(run_counts) != 1:
        offenders = counts[counts != counts.mode().iloc[0]].to_dict()
        raise ValueError(
            f"Unbalanced cross-validation design: every value in '{fold_col}' must "
            f"repeat the same number of times (one row per run). "
            f"Folds with a differing number of runs: {offenders}."
        )
    return len(counts), int(run_counts.pop())


def resolve_rho(rho: float | None, n_folds: int | None, n: int) -> float:
    """Resolve the correlation ``rho`` due to overlapping training sets.

    Follows the Nadeau-Bengio heuristic ``rho = n_test / n_total`` used by the
    correlated t-test (Benavoli et al. 2017, Eq. 5-6), which for k-fold
    cross-validation is ``1 / k``. Resolution priority:

    1. an explicit ``rho``, if the caller overrides it;
    2. ``1 / n_folds`` from the fold structure of the data;
    3. ``1 / n`` as a last resort, treating each of the ``n`` rows as its own
       fold of a single run.

    Parameters
    ----------
    rho : float | None
        Explicit correlation, if the caller overrides the derived value.
    n_folds : int | None
        Number of cross-validation folds derived from the data, if known.
    n : int
        Total number of scores.

    Returns
    -------
    float
        The resolved correlation in ``(0, 1)``.
    """
    if rho is not None:
        resolved = float(rho)
    elif n_folds is not None:
        resolved = 1.0 / n_folds
    else:
        resolved = 1.0 / n
    if not 0.0 < resolved < 1.0:
        raise ValueError(
            f"Resolved rho={resolved} is outside (0, 1). "
            f"Check the 'rho' override or the fold structure of the data."
        )
    return resolved


def decision_from_partition(
    p_left: float,
    p_rope: float,
    p_right: float,
    left_model: str,
    right_model: str,
    threshold: float,
) -> tuple[str, str]:
    """Turn a three-way ROPE partition into a decision label and raw symbol.

    The estimand is the mean difference ``mu = left - right`` (positive means
    the left model is better), so ``p_right = P(mu > r)`` votes for the left
    model and ``p_left = P(mu < -r)`` votes for the right model. A side is
    declared only if its probability exceeds ``threshold`` (Benavoli et al.
    2017, Section 3.2, uses 0.95); otherwise the decision is undetermined.

    Parameters
    ----------
    p_left, p_rope, p_right : float
        The three partition probabilities (should sum to one).
    left_model, right_model : str
        Names of the two algorithms being compared.
    threshold : float
        Probability a region must exceed to be declared.

    Returns
    -------
    tuple[str, str]
        ``(decision, decision_raw)`` where ``decision`` uses the human strings
        (``"<model> better"`` / ``"Equivalent"`` / ``"Unknown"``, matching BBT)
        and ``decision_raw`` is one of ``>`` / ``<`` / ``=`` / ``?``.
    """
    if p_right >= threshold:
        return f"{left_model} better", DECISION_LEFT_BETTER
    if p_left >= threshold:
        return f"{right_model} better", DECISION_RIGHT_BETTER
    if p_rope >= threshold:
        return "Equivalent", DECISION_EQUIVALENT
    return "Unknown", DECISION_UNDETERMINED


def samples_to_inference_data(
    posterior: dict[str, np.ndarray],
) -> az.InferenceData:
    """Wrap named 1-D sample arrays into an :class:`arviz.InferenceData`.

    Each array is treated as a single chain of draws, so the closed-form
    correlated t-test exposes exactly the same ``idata_`` endpoint as the
    sampled hierarchical test.

    Parameters
    ----------
    posterior : dict[str, np.ndarray]
        Mapping from variable name to a 1-D array of draws.

    Returns
    -------
    az.InferenceData
        Inference data with a single chain per variable.
    """
    reshaped = {
        name: np.asarray(draws)[np.newaxis, :] for name, draws in posterior.items()
    }
    return az.from_dict(posterior=reshaped)


@dataclass
class SummaryResult:
    """A small printable summary of a fitted t-test, statsmodels-flavoured.

    Attributes
    ----------
    title : str
        Heading naming the test and the comparison.
    estimand : str
        Human name of the point-estimated quantity (e.g. ``"mu"``).
    estimate : float
        Posterior point estimate of the estimand.
    hdi : tuple[float, float]
        Highest-density interval of the estimand.
    hdi_prob : float
        Mass of the reported HDI.
    rope : tuple[float, float]
        The ROPE band, on the difference scale.
    probabilities : dict[str, float]
        The named three-way ROPE probabilities.
    decision : str
        The verdict (shared decision vocabulary).
    extra : dict[str, str]
        Optional extra key/value lines rendered below the verdict.
    """

    title: str
    estimand: str
    estimate: float
    hdi: tuple[float, float]
    hdi_prob: float
    rope: tuple[float, float]
    probabilities: dict[str, float]
    decision: str
    extra: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        width = 66
        rule = "=" * width
        thin = "-" * width
        lines = [
            rule,
            self.title.center(width),
            rule,
            f"{self.estimand} (point estimate){self.estimate:>{width - len(self.estimand) - 17}.4f}",
            f"{int(self.hdi_prob * 100)}% HDI"
            f"{f'[{self.hdi[0]:.4f}, {self.hdi[1]:.4f}]':>{width - 7}}",
            f"ROPE (difference scale)"
            f"{f'[{self.rope[0]:.4f}, {self.rope[1]:.4f}]':>{width - 23}}",
            thin,
        ]
        for name, value in self.probabilities.items():
            lines.append(f"{name}{value:>{width - len(name)}.4f}")
        lines.append(thin)
        lines.append(f"Decision{self.decision:>{width - 8}}")
        for key, value in self.extra.items():
            lines.append(f"{key}{value:>{width - len(key)}}")
        lines.append(rule)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()
