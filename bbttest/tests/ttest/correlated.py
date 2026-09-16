"""Bayesian correlated t-test implementation for bbttest.tests.ttest."""

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bbttest.tests.common import BaseBayesianTest, validate_string

from ._stats import (
    SummaryResult,
    decision_from_partition,
    resolve_fold_structure,
    resolve_rho,
    resolve_rope,
    samples_to_inference_data,
)
from ._types import ALL_TTEST_COLUMNS, CORRELATED_PLOT_KINDS
from .plots import plot_correlated_posterior

_DEFAULT_COLUMNS: tuple[str, ...] = (
    "comparison",
    "estimate",
    "hdi_low",
    "hdi_high",
    "p_left",
    "p_rope",
    "p_right",
    "decision",
)


class CorrelatedTTest(BaseBayesianTest):
    r"""
    Bayesian correlated t-test estimator used for two-model single-dataset comparison [1]_.
    The model estimates the posterior distribution of the mean difference of scores
    between the two models, from the per-fold scores of a cross-validation experiment.

    The posterior is summarised as three probabilities that partition it and sum to one:
    the left model being practically better, the two models being practically equivalent,
    and the right model being practically better, where "practically" is delimited by the
    ROPE. The correlation induced by the overlapping training sets of cross-validation is
    accounted for by the Nadeau-Bengio heuristic, derived from the fold structure of the
    data passed to :meth:`fit`.

    Parameters
    ----------
    rope: float | tuple[float, float], default 0.01
        The region of practical equivalence, defined on the scale of the score
        difference. Differences of a smaller magnitude are treated as practically
        irrelevant.

            - a scalar is a half-width `r`, giving the band `[-r, r]`
            - a 2-tuple is used as an explicit `(lo, hi)` band

        Note that this differs from BBT, whose ROPE is defined on the probability
        scale, around 0.5.

    maximize: bool, default True
        Whether higher scores indicate better performance (e.g. accuracy/f1). If using a
        metric where the goal is to minimize the score (e.g. RMSE) set this to False, so
        that a positive mean difference still indicates the left model being better.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    idata_: arviz.InferenceData
        Analytic draws of the mean difference, exposed for the ArviZ toolchain.

    Examples
    --------
    >>> import pandas as pd
    >>> from bbttest import CorrelatedTTest
    >>> data = pd.DataFrame({
    ...     'fold': [1, 2, 3, 1, 2, 3],
    ...     'model_a': [0.81, 0.83, 0.80, 0.82, 0.84, 0.81],
    ...     'model_b': [0.78, 0.77, 0.79, 0.76, 0.78, 0.77]
    ... })
    >>> model = CorrelatedTTest(rope=0.01)
    >>> model.fit(data, fold_col='fold')
    >>> model.decision_table()

    Notes
    -----
    Under the matching prior, the posterior of the mean difference
    :math:`\mu = \text{left} - \text{right}` is the Student distribution

    .. math::
        p(\mu \mid \mathbf{x}) = St\!\left(\mu;\, n-1,\; \bar{x},\;
        \left(\tfrac{1}{n} + \tfrac{\rho}{1-\rho}\right)\hat{\sigma}^2\right),

    where :math:`\bar{x}` and :math:`\hat{\sigma}^2` are the sample mean and variance of
    the differences and :math:`\rho` is the Nadeau-Bengio correlation (Eq. 6). The
    reported probabilities are therefore computed in closed form, rather than sampled.

    References
    ----------
    .. [1] `Alessio Benavoli, Giorgio Corani, Janez Demsar, Marco Zaffalon
        "Time for a Change: a Tutorial for Comparing Multiple Classifiers Through
        Bayesian Analysis"
        Journal of Machine Learning Research 18 (2017): 1-36
        <http://jmlr.org/papers/v18/16-305.html>`_
    """

    ALL_TTEST_COLUMNS = ALL_TTEST_COLUMNS

    _DEFAULT_THRESHOLD = 0.95

    def __init__(
        self,
        rope: float | tuple[float, float] = 0.01,
        maximize: bool = True,
    ):
        self._rope = rope
        self._maximize = maximize
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
        fold_col: str | None = None,
        rho: float | None = None,
        draws: int = 50_000,
        random_seed: int | None = None,
    ) -> "CorrelatedTTest":
        """
        Fits the correlated t-test for a given result dataframe.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series | np.ndarray | Sequence[float]
            Dataframe containing the cross-validation scores of the two models, one row
            per score, with exactly two model columns holding the left and the right
            model, in that order, optionally alongside fold_col.
            A 1-D array or Series of precomputed differences (left - right) is also
            accepted, in which case fold_col is not available.
        fold_col : str | None, optional
            Column name for the cross-validation fold identifier. Rows sharing an
            identifier are repeated runs of the same fold, so the number of distinct
            identifiers gives the number of folds k, and the correlation is taken as
            rho = 1 / k. The design must be balanced, i.e. every fold repeated the same
            number of times. If None, every row is treated as a separate fold.
        rho : float | None, optional
            Correlation due to the overlapping training sets. If provided, it overrides
            the value derived from fold_col. Intended for resampling schemes other than
            k-fold cross-validation.
        draws : int, optional
            Number of analytic draws of the mean difference materialised into idata_.
            Does not affect the reported probabilities, which are computed in closed
            form. Defaults to 50000.
        random_seed : int | None, optional
            Seed for the analytic draws.

        Returns
        -------
        self : CorrelatedTTest
            Fitted CorrelatedTTest instance
        """
        diff, left, right, n_folds, n_runs = _extract_differences(
            data, fold_col, self._maximize
        )
        n = diff.shape[0]
        if n < 2:
            raise ValueError(f"Need at least 2 scores to estimate a variance, got {n}.")

        self._left = left
        self._right = right
        self._diff = diff
        self._n = n
        self._n_folds = n_folds
        self._n_runs = n_runs
        self._xbar = float(np.mean(diff))
        self._sigma2 = float(np.var(diff, ddof=1))
        self._rho_used = resolve_rho(rho, n_folds, n)

        self._df = n - 1
        self._loc = self._xbar
        self._scale = float(
            np.sqrt((1.0 / n + self._rho_used / (1.0 - self._rho_used)) * self._sigma2)
        )

        rng = np.random.default_rng(random_seed)
        if self._scale > 0:
            mu_draws = stats.t.rvs(
                df=self._df,
                loc=self._loc,
                scale=self._scale,
                size=draws,
                random_state=rng,
            )
        else:
            # Degenerate posterior (all differences identical): point mass.
            mu_draws = np.full(draws, self._loc)
        self._idata = samples_to_inference_data({"mu": mu_draws})

        self._fitted = True
        return self

    # -- diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict[str, float]:
        """Not applicable to this test, which has no sampler.

        Raises
        ------
        NotImplementedError
            Always. The posterior is a Student distribution in closed form, and the
            draws in :attr:`idata_` are independent draws from it rather than a Markov
            chain, so R-hat, effective sample size and divergences have no meaning
            here. The sampled tests -- :class:`~bbttest.HierarchicalTTest` and
            :class:`~bbttest.BBTTest` -- implement them.
        """
        raise NotImplementedError(
            "CorrelatedTTest has a closed-form posterior, so there is no sampler to "
            "diagnose: idata_ holds independent draws from a Student distribution, "
            "not a Markov chain. Convergence diagnostics apply to HierarchicalTTest "
            "and BBTTest."
        )

    # -- estimand queries ---------------------------------------------------

    def _probabilities(
        self, rope_band: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Closed-form three-way ROPE partition ``(p_left, p_rope, p_right)``."""
        lo, hi = rope_band
        if self._scale > 0:
            p_left = float(
                stats.t.cdf(lo, df=self._df, loc=self._loc, scale=self._scale)
            )
            p_right = float(
                stats.t.sf(hi, df=self._df, loc=self._loc, scale=self._scale)
            )
        else:
            p_left = float(self._loc < lo)
            p_right = float(self._loc > hi)
        p_rope = 1.0 - p_left - p_right
        return p_left, p_rope, p_right

    def _hdi(self, hdi_prob: float) -> tuple[float, float]:
        """Highest-density interval of ``mu`` (equal-tailed; the Student is symmetric)."""
        if self._scale == 0:
            return self._loc, self._loc
        tail = (1.0 + hdi_prob) / 2.0
        half = float(stats.t.ppf(tail, df=self._df)) * self._scale
        return self._loc - half, self._loc + half

    @property
    def comparison(self) -> str:
        """The row id for this comparison, e.g. ``"model_a vs model_b"``."""
        self._check_if_fitted()
        return f"{self._left} vs {self._right}"

    # -- reports ------------------------------------------------------------

    def decision_table(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        hdi_prob: float = 0.89,
        columns: Iterable[str] = _DEFAULT_COLUMNS,
        round_ndigits: int | None = 3,
    ) -> pd.DataFrame:
        """
        Construct the decision table for the fitted comparison.

        The table reports the posterior mean difference and its HDI, the three ROPE
        probabilities, and the resulting decision. The probabilities partition the
        posterior and sum to one, where p_left is the probability of the right model
        being practically better, and p_right of the left model being practically better.

        Parameters
        ----------
        threshold : float, optional
            The probability a ROPE region must exceed for the corresponding decision to
            be made, otherwise the decision is "Unknown". Defaults to 0.95, as used in
            [1]_.
        hdi_prob : float, optional
            Probability mass of the reported HDI. Defaults to 0.89.
        columns : Iterable[str], optional
            Columns to include in the table. See ALL_TTEST_COLUMNS for the available
            values.
        round_ndigits : int | None, optional
            Number of digits to round the numeric columns to. If None, no rounding is
            applied. Defaults to 3.

        Returns
        -------
        pd.DataFrame
            Single-row dataframe describing the comparison.
        """
        self._check_if_fitted()
        rope_band = resolve_rope(self._rope)
        p_left, p_rope, p_right = self._probabilities(rope_band)
        hdi_low, hdi_high = self._hdi(hdi_prob)
        decision, decision_raw = decision_from_partition(
            p_left, p_rope, p_right, self._left, self._right, threshold
        )

        row = {
            "comparison": self.comparison,
            "estimate": self._loc,
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
            "p_left": p_left,
            "p_rope": p_rope,
            "p_right": p_right,
            "decision": decision,
            "decision_raw": decision_raw,
        }
        columns = list(columns)
        for col in columns:
            if col not in row:
                raise ValueError(
                    f"Column {col} is not available in the decision table."
                )
        out = pd.DataFrame([{col: row[col] for col in columns}])
        if round_ndigits is not None:
            numeric = out.select_dtypes(include="number").columns
            out[numeric] = out[numeric].round(round_ndigits)
        return out

    def summary(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        hdi_prob: float = 0.89,
    ) -> SummaryResult:
        """
        Construct a printable summary of the fitted comparison.

        Parameters
        ----------
        threshold : float, optional
            The probability a ROPE region must exceed for the corresponding decision to
            be made. Defaults to 0.95.
        hdi_prob : float, optional
            Probability mass of the reported HDI. Defaults to 0.89.

        Returns
        -------
        SummaryResult
            Printable summary of the mean difference and its HDI, the three ROPE
            probabilities, and the decision.
        """
        self._check_if_fitted()
        rope_band = resolve_rope(self._rope)
        p_left, p_rope, p_right = self._probabilities(rope_band)
        decision, _ = decision_from_partition(
            p_left, p_rope, p_right, self._left, self._right, threshold
        )
        return SummaryResult(
            title=f"Bayesian correlated t-test: {self.comparison}",
            estimand="mu = E[left - right]",
            estimate=self._loc,
            hdi=self._hdi(hdi_prob),
            hdi_prob=hdi_prob,
            rope=rope_band,
            probabilities={
                f"P(mu < {rope_band[0]:.4g})  [{self._right} better]": p_left,
                "P(rope)  [equivalent]": p_rope,
                f"P(mu > {rope_band[1]:.4g})  [{self._left} better]": p_right,
            },
            decision=decision,
            extra={
                "rho": f"{self._rho_used:.4g}",
                "scores (folds x runs)": (
                    f"{self._n} ({self._n_folds} x {self._n_runs})"
                    if self._n_folds is not None
                    else str(self._n)
                ),
            },
        )

    def plot(
        self,
        kind: str = "posterior",
        hdi_prob: float = 0.89,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the posterior of the fitted comparison.

        Parameters
        ----------
        kind : str, optional
            The figure to draw. Defaults to `posterior`.

                - `posterior` - the posterior density, with the ROPE band shaded and the
                  HDI marked. See Fig. 4 in [1]_.
                - `hdi` - the HDIs of increasing probability mass. See Fig. 5 in [1]_.

        hdi_prob : float, optional
            Probability mass of the HDI marked on the posterior plot. Defaults to 0.89.
        ax : plt.Axes | None, optional
            Matplotlib Axes to plot on. If None, a new figure and axes will be created.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        plt.Axes
            The Axes the figure was drawn on.
        """
        self._check_if_fitted()
        validate_string(kind, CORRELATED_PLOT_KINDS, "kind")
        rope_band = resolve_rope(self._rope)
        return plot_correlated_posterior(
            df=self._df,
            loc=self._loc,
            scale=self._scale,
            rope_band=rope_band,
            hdi=self._hdi(hdi_prob),
            hdi_prob=hdi_prob,
            left_model=self._left,
            right_model=self._right,
            kind=kind,
            ax=ax,
            **kwargs,
        )


def _extract_differences(
    data: pd.DataFrame | pd.Series | np.ndarray | Sequence[float],
    fold_col: str | None,
    maximize: bool,
) -> tuple[np.ndarray, str, str, int | None, int | None]:
    """Extract per-fold differences, model names and fold structure from ``data``.

    Returns the difference vector (``left - right``, sign-flipped when
    minimising), the two model names, and the ``(n_folds, n_runs)`` derived from
    ``fold_col`` (both ``None`` when no fold column is available).
    """
    sign = 1.0 if maximize else -1.0
    if isinstance(data, pd.DataFrame):
        n_folds = n_runs = None
        if fold_col is not None:
            if fold_col not in data.columns:
                raise ValueError(
                    f"fold_col '{fold_col}' not found in the data columns."
                )
            n_folds, n_runs = resolve_fold_structure(data[fold_col], fold_col)
            data = data.drop(columns=[fold_col])
        if data.shape[1] != 2:
            raise ValueError(
                "CorrelatedTTest expects exactly two algorithm columns (left, "
                f"right); got {data.shape[1]}: {list(data.columns)}."
            )
        left, right = (str(c) for c in data.columns)
        diff = sign * (data.iloc[:, 0].to_numpy() - data.iloc[:, 1].to_numpy())
        return np.asarray(diff, dtype=float), left, right, n_folds, n_runs

    if fold_col is not None:
        raise ValueError(
            "fold_col requires a DataFrame input; pass a frame with the fold "
            "column alongside the two algorithm columns."
        )

    if isinstance(data, pd.Series):
        arr = sign * data.to_numpy(dtype=float)
        return np.asarray(arr, dtype=float), "left", "right", None, None

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(
                "CorrelatedTTest expects a two-column array (left, right); "
                f"got shape {arr.shape}."
            )
        return sign * (arr[:, 0] - arr[:, 1]), "left", "right", None, None
    if arr.ndim != 1:
        raise ValueError(
            f"Expected 1-D differences or 2-D scores, got shape {arr.shape}."
        )
    return sign * arr, "left", "right", None, None
