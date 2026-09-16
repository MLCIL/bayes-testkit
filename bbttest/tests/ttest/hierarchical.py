"""Bayesian hierarchical correlated t-test implementation for bbttest.tests.ttest."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bbttest.tests.common import BaseBayesianTest, hdi_from_samples, validate_string

from ._stats import (
    SummaryResult,
    decision_from_partition,
    resolve_fold_structure,
    resolve_rho,
    resolve_rope,
)
from ._types import ALL_TTEST_COLUMNS, DOF_PRIORS, HIERARCHICAL_PLOT_KINDS
from .model import _sample_hierarchical
from .plots import (
    plot_hierarchical_forest,
    plot_hierarchical_ppc,
    plot_hierarchical_simplex,
)

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


class HierarchicalTTest(BaseBayesianTest):
    r"""
    Bayesian hierarchical correlated t-test estimator used for two-model multi-dataset
    comparison [1]_. The model estimates the population mean difference of scores between
    the two models over the datasets, together with the per-dataset mean differences,
    from the per-fold scores of a cross-validation experiment run on each dataset.

    The decision is reported for the next, unseen dataset drawn from the same population.
    Each posterior draw yields a triple ``(theta_left, theta_rope, theta_right)`` -- the
    probabilities that on that next dataset the right model is practically better, the two
    are practically equivalent, or the left model is practically better, where
    "practically" is delimited by the ROPE. Following Section 4.3.2 of [1]_, the reported
    ``p_left``/``p_rope``/``p_right`` are the proportions of draws whose triple falls in
    each simplex region ``theta_i >= max(theta_j, theta_k)``, which is what the paper's
    Table 12 tabulates; they sum to one. The posterior means of the triple itself are
    available as the ``mean_theta_*`` columns of :meth:`decision_table`.

    The per-dataset means are estimated jointly rather than independently, so each is
    shrunk towards the population mean, by an amount depending on the evidence available
    in that dataset. See :meth:`per_dataset_table` and `plot(kind="forest")`. As in
    :class:`CorrelatedTTest`, the correlation induced by the overlapping training sets of
    cross-validation is accounted for by the Nadeau-Bengio heuristic, derived from the
    fold structure of the data passed to :meth:`fit`.

    Parameters
    ----------
    rope: float | tuple[float, float], default 0.01
        The region of practical equivalence, defined on the scale of the score
        difference. Differences of a smaller magnitude are treated as practically
        irrelevant.

            - a scalar is a half-width `r`, giving the band `[-r, r]`
            - a 2-tuple is used as an explicit `(lo, hi)` band

    dof_prior: str, default `hierarchical`
        The prior on the degrees of freedom of the Student distribution over the
        per-dataset means, which governs how tolerant the model is of datasets whose
        mean difference is far from the others.

            - `hierarchical` - Gamma with uniform hyper-priors on its shape and rate. See [1]_.
            - `kruschke` - fixed Gamma(1, 0.0345), balancing nearly normal and heavy-tailed distributions.
            - `juarez_steel` - fixed Gamma(2, 0.1), assigning larger prior probability to normal distributions.

    maximize: bool, default True
        Whether higher scores indicate better performance (e.g. accuracy/f1). If using a
        metric where the goal is to minimize the score (e.g. RMSE) set this to False, so
        that a positive mean difference still indicates the left model being better.

    mu0_bound: float, default 1.0
        Half-width of the uniform prior on the population mean difference,
        `mu_0 ~ U(-mu0_bound, mu0_bound)` (Eq. 18). The default of 1 is the paper's, and
        suits any measure bounded within +-1 (accuracy, AUC, precision, recall). For an
        unbounded metric, widen it; :meth:`fit` raises rather than silently sampling
        under a prior that excludes the observed data.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    idata_: arviz.InferenceData
        The sampled posterior, exposed for the ArviZ toolchain.

    Examples
    --------
    >>> import pandas as pd
    >>> from bbttest import HierarchicalTTest
    >>> data = pd.DataFrame({
    ...     'dataset': ['ds1', 'ds1', 'ds2', 'ds2'],
    ...     'fold': [1, 2, 1, 2],
    ...     'model_a': [0.81, 0.83, 0.75, 0.77],
    ...     'model_b': [0.78, 0.77, 0.74, 0.72]
    ... })
    >>> model = HierarchicalTTest(rope=0.01)
    >>> model.fit(data, dataset_col='dataset', fold_col='fold')
    >>> model.decision_table()

    Notes
    -----
    The estimator fits the hierarchical model

    .. math::
        \mathbf{x}_i \sim MVN(\mathbf{1}\mu_i, \Sigma_i), \quad
        \mu_i \sim t(\mu_0, \sigma_0, \nu), \quad
        \sigma_i \sim U(0, \bar\sigma),

    (Eq. 12-14) with PyMC. The headline estimand is the population mean difference
    :math:`\mu_0`, where a positive value indicates the left model being better. The
    per-dataset means :math:`\mu_i` are shrunk towards it, and the posterior predictive
    for a new dataset yields the three-way ROPE triple shown on the simplex.

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
        dof_prior: str = "hierarchical",
        maximize: bool = True,
        mu0_bound: float = 1.0,
    ):
        validate_string(dof_prior, DOF_PRIORS, "dof_prior")
        if mu0_bound <= 0:
            raise ValueError(f"mu0_bound must be positive, got {mu0_bound}.")
        self._rope = rope
        self._dof_prior = dof_prior
        self._maximize = maximize
        self._mu0_bound = mu0_bound
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        dataset_col: str = "dataset",
        fold_col: str | None = None,
        rho: float | None = None,
        **pymc_kwargs,
    ) -> "HierarchicalTTest":
        """
        Fits the hierarchical t-test for a given result dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the cross-validation scores of the two models, one row
            per score per dataset, with exactly two model columns holding the left and
            the right model, in that order, optionally alongside fold_col.
            Every dataset must have the same number of score rows.
        dataset_col : str, optional
            Column name for the dataset identifier. Defaults to "dataset".
        fold_col : str | None, optional
            Column name for the cross-validation fold identifier. Rows sharing an
            identifier are repeated runs of the same fold, so the number of distinct
            identifiers gives the number of folds k, and the correlation is taken as
            rho = 1 / k. The design must be balanced within each dataset, and identical
            across datasets. If None, every row is treated as a separate fold.
        rho : float | None, optional
            Correlation due to the overlapping training sets. If provided, it overrides
            the value derived from fold_col. Intended for resampling schemes other than
            k-fold cross-validation.
        **pymc_kwargs
            Sampler options passed to pm.sample (draws, tune, chains, cores,
            target_accept, random_seed).

        Returns
        -------
        self : HierarchicalTTest
            Fitted HierarchicalTTest instance
        """
        x, names, left, right, n_folds, n_runs = _build_dataset_matrix(
            data, dataset_col, fold_col, self._maximize
        )
        q, n = x.shape

        self._x = x
        self._dataset_names = names
        self._left = left
        self._right = right
        self._n_folds = n_folds
        self._n_runs = n_runs
        self._raw_means = x.mean(axis=1)
        self._rho_used = resolve_rho(rho, n_folds, n)

        # Eq. (14) and (19): the two scale bounds follow the paper. sigma_i is
        # bounded by 1000 times the *average* within-dataset sd; sigma_0 by 1000
        # times the sd of the per-dataset means. Both are deliberately loose --
        # Gelman (2006) notes the inference is insensitive once the bound is
        # large enough -- but they are not interchangeable, so they are kept
        # apart. The floors only guard the degenerate all-identical-scores case.
        sigma_bar = float(np.mean(np.std(x, axis=1, ddof=1)))
        s_xbar = float(np.std(self._raw_means, ddof=1)) if q > 1 else 0.0
        sigma_i_upper = 1000.0 * max(sigma_bar, 1e-6)
        sigma_0_upper = 1000.0 * max(s_xbar, 1e-6)

        # Eq. (18): mu_0 ~ U(-1, 1) works for measures bounded within +-1, which
        # is what the paper assumes. Anything else has to say so explicitly
        # rather than have the bound silently widened underneath it.
        mu0_bound = float(self._mu0_bound)
        largest_mean = float(np.max(np.abs(self._raw_means)))
        if largest_mean >= mu0_bound:
            raise ValueError(
                f"The prior on the population mean is U(-{mu0_bound}, {mu0_bound}),"
                f"but a dataset has a mean difference of {largest_mean:.4g}, "
                "which the prior excludes. This bound suits measures within +-1 such "
                "as accuracy or AUC; for an unbounded metric pass a wider "
                "'mu0_bound' to the constructor."
            )

        self._idata, self._pymc_model = _sample_hierarchical(
            x=x,
            rho=self._rho_used,
            dof_prior=self._dof_prior,
            mu0_bound=mu0_bound,
            sigma_0_upper=sigma_0_upper,
            sigma_i_upper=sigma_i_upper,
            **pymc_kwargs,
        )

        self._fitted = True
        self._warn_on_bad_diagnostics()
        return self

    # -- diagnostics --------------------------------------------------------

    def _diagnostic_vars(self) -> list[str]:
        """Restrict the diagnostics to the model's own parameters."""
        return ["mu_0", "sigma_0", "nu", "mu_i", "sigma_i"]

    def posterior_predictive_check(
        self,
        hdi_probs: Iterable[float] = (0.5, 0.9, 0.95, 1.0),
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Posterior predictive check of the fitted model.

        R-hat, ESS and divergences say whether the sampler explored the posterior;
        none of them says whether the model can reproduce the data. This replays the
        observed per-fold differences through the fitted model and reports, for each
        HDI mass, the proportion of datasets whose observed mean difference and
        within-dataset spread fall inside the HDI of the replicated ones.

        A proportion well below the HDI mass means the model is too narrow for the
        data it was fitted on -- typically a dataset whose folds disagree far more
        than the pooled scale allows.

        Parameters
        ----------
        hdi_probs : Iterable[float], optional
            HDI masses to evaluate. Defaults to ``(0.5, 0.9, 0.95, 1.0)``.
        random_seed : int | None, optional
            Seed for the replicated draws.

        Returns
        -------
        pd.DataFrame
            One row per HDI mass, with the proportion of datasets covered for the
            per-dataset ``mean`` and ``sd`` of the fold differences.
        """
        replicated = self._replicate_observations(random_seed)
        statistics = {
            "mean": (replicated.mean(axis=2), self._x.mean(axis=1)),
            "sd": (replicated.std(axis=2, ddof=1), self._x.std(axis=1, ddof=1)),
        }

        records = []
        for prob in hdi_probs:
            row: dict[str, float] = {"hdi": float(prob)}
            for label, (reps, observed) in statistics.items():
                inside = []
                for i in range(reps.shape[1]):
                    low, high = hdi_from_samples(reps[:, i], prob)
                    inside.append(low <= observed[i] <= high)
                row[label] = float(np.mean(inside))
            records.append(row)
        return pd.DataFrame.from_records(records)

    def _replicate_observations(self, random_seed: int | None) -> np.ndarray:
        """Replay the observed folds through the model, as ``(draws, datasets, folds)``."""
        import pymc as pm

        self._check_if_fitted()
        with self._pymc_model:
            ppc = pm.sample_posterior_predictive(
                self.idata_,
                var_names=["obs"],
                progressbar=False,
                random_seed=random_seed,
            )
        obs = ppc.posterior_predictive["obs"].to_numpy()
        return obs.reshape(-1, *obs.shape[2:])

    # -- posterior extraction ----------------------------------------------

    def _flat(self, var: str) -> np.ndarray:
        """Flatten a posterior variable across chains and draws."""
        arr = self.idata_.posterior[var].to_numpy()
        return arr.reshape(-1, *arr.shape[2:])

    def _next_dataset_triple(
        self, rope_band: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior samples of the next-dataset ROPE triple ``(theta_l, theta_e, theta_r)``.

        For each posterior draw of ``(mu_0, sigma_0, nu)``, the predictive
        distribution of a new dataset's mean difference is
        ``StudentT(nu, mu_0, sigma_0)``; the triple integrates it over
        ``(-inf, lo]``, ``[lo, hi]`` and ``[hi, inf)`` (Section 4.3.2).
        """
        lo, hi = rope_band
        mu_0 = self._flat("mu_0")
        sigma_0 = np.clip(self._flat("sigma_0"), 1e-12, None)
        nu = self._flat("nu")
        theta_l = stats.t.cdf((lo - mu_0) / sigma_0, df=nu)
        theta_r = stats.t.sf((hi - mu_0) / sigma_0, df=nu)
        theta_e = np.clip(1.0 - theta_l - theta_r, 0.0, 1.0)
        return theta_l, theta_e, theta_r

    def _region_probabilities(
        self, rope_band: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Probability of each simplex region, by counting posterior draws.

        Benavoli et al. (2017), Section 4.3.2 quantify the comparison "by
        counting the number of points that fall in the three regions" delimited
        by ``theta_i >= max(theta_j, theta_k)``.
        """
        theta_l, theta_e, theta_r = self._next_dataset_triple(rope_band)
        winner = np.argmax(np.column_stack([theta_l, theta_e, theta_r]), axis=1)
        return (
            float(np.mean(winner == 0)),
            float(np.mean(winner == 1)),
            float(np.mean(winner == 2)),
        )

    # -- reports ------------------------------------------------------------

    def decision_table(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        hdi_prob: float = 0.89,
        columns: Iterable[str] = _DEFAULT_COLUMNS,
        round_ndigits: int | None = 3,
    ) -> pd.DataFrame:
        """
        Construct the decision table for the population comparison.

        The table reports the population mean difference and its HDI, the three ROPE
        probabilities for the next dataset, and the resulting decision. Following
        Section 4.3.2 of [1]_, each probability is the proportion of posterior draws
        whose next-dataset triple falls in the corresponding simplex region
        ``theta_i >= max(theta_j, theta_k)`` -- the quantity reported in the paper's
        Table 12. The posterior means of the triple are available separately as the
        ``mean_theta_*`` columns. The per-dataset estimates are reported by
        :meth:`per_dataset_table`.

        Parameters
        ----------
        threshold : float, optional
            The probability a region must reach for the corresponding decision to be
            made, otherwise the decision is "Unknown". Defaults to 0.95, as used in
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
            Single-row dataframe describing the population comparison.
        """
        self._check_if_fitted()
        rope_band = resolve_rope(self._rope)
        p_left, p_rope, p_right = self._region_probabilities(rope_band)
        theta_l, theta_e, theta_r = self._next_dataset_triple(rope_band)
        mu_0 = self._flat("mu_0")
        hdi_low, hdi_high = hdi_from_samples(mu_0, hdi_prob)
        decision, decision_raw = decision_from_partition(
            p_left, p_rope, p_right, self._left, self._right, threshold
        )

        row = {
            "comparison": f"{self._left} vs {self._right} (population)",
            "estimate": float(mu_0.mean()),
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
            "p_left": p_left,
            "p_rope": p_rope,
            "p_right": p_right,
            "mean_theta_left": float(theta_l.mean()),
            "mean_theta_rope": float(theta_e.mean()),
            "mean_theta_right": float(theta_r.mean()),
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

    def rope_comparison_table(
        self,
        rope_values: Iterable[float | tuple[float, float]],
        threshold: float = _DEFAULT_THRESHOLD,
        round_ndigits: int | None = 3,
    ) -> pd.DataFrame:
        """
        Re-read the fitted posterior under several ROPEs.

        The ROPE is a property of the report, not of the likelihood: it delimits which
        differences are worth caring about, and enters only when the next-dataset
        predictive is integrated over the three regions. The posterior therefore does
        not have to be re-sampled to change it, and the sensitivity of the decision to
        that one subjective choice can be tabulated from a single fit.

        Parameters
        ----------
        rope_values : Iterable[float | tuple[float, float]]
            The ROPEs to evaluate, each a half-width or an explicit ``(lo, hi)`` band
            on the difference scale.
        threshold : float, optional
            The probability a region must reach for the corresponding decision to be
            made, otherwise the decision is "Unknown". Defaults to 0.95, as used in
            [1]_.
        round_ndigits : int | None, optional
            Number of digits to round the numeric columns to. If None, no rounding is
            applied. Defaults to 3.

        Returns
        -------
        pd.DataFrame
            One row per ROPE, with the resolved band, the three region probabilities
            and the decision they imply.
        """
        self._check_if_fitted()
        rows = []
        for rope in rope_values:
            band = resolve_rope(rope)
            p_left, p_rope, p_right = self._region_probabilities(band)
            decision, _ = decision_from_partition(
                p_left, p_rope, p_right, self._left, self._right, threshold
            )
            rows.append(
                {
                    "rope_low": band[0],
                    "rope_high": band[1],
                    "p_left": p_left,
                    "p_rope": p_rope,
                    "p_right": p_right,
                    "decision": decision,
                }
            )
        out = pd.DataFrame(rows)
        if round_ndigits is not None:
            numeric = out.select_dtypes(include="number").columns
            out[numeric] = out[numeric].round(round_ndigits)
        return out

    def per_dataset_table(
        self,
        hdi_prob: float = 0.89,
        round_ndigits: int | None = 3,
    ) -> pd.DataFrame:
        """
        Construct the table of per-dataset mean differences.

        Reports the shrunk estimate of each dataset's mean difference next to its raw
        sample mean, so that the magnitude of the shrinkage towards the population mean
        can be inspected.

        Parameters
        ----------
        hdi_prob : float, optional
            Probability mass of the reported HDI. Defaults to 0.89.
        round_ndigits : int | None, optional
            Number of digits to round the numeric columns to. If None, no rounding is
            applied. Defaults to 3.

        Returns
        -------
        pd.DataFrame
            Dataframe with one row per dataset, containing the raw mean, the shrunk mean
            and the HDI of the shrunk estimate.
        """
        self._check_if_fitted()
        mu_i = self._flat("mu_i")  # (samples, q)
        shrunk_mean = mu_i.mean(axis=0)
        hdis = np.array(
            [hdi_from_samples(mu_i[:, j], hdi_prob) for j in range(mu_i.shape[1])]
        )
        out = pd.DataFrame(
            {
                "dataset": self._dataset_names,
                "raw_mean": self._raw_means,
                "shrunk_mean": shrunk_mean,
                "hdi_low": hdis[:, 0],
                "hdi_high": hdis[:, 1],
            }
        )
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
            The probability a region must reach for the corresponding decision to be
            made, otherwise the decision is "Unknown". Defaults to 0.95.
        hdi_prob : float, optional
            Probability mass of the reported HDI. Defaults to 0.89.

        Returns
        -------
        SummaryResult
            Printable summary of the population mean difference and its HDI, the three
            next-dataset region probabilities, and the decision.
        """
        self._check_if_fitted()
        rope_band = resolve_rope(self._rope)
        p_left, p_rope, p_right = self._region_probabilities(rope_band)
        mu_0 = self._flat("mu_0")
        decision, _ = decision_from_partition(
            p_left, p_rope, p_right, self._left, self._right, threshold
        )
        return SummaryResult(
            title=f"Bayesian hierarchical correlated t-test: {self._left} vs {self._right}",
            estimand="mu_0 = population E[left - right]",
            estimate=float(mu_0.mean()),
            hdi=hdi_from_samples(mu_0, hdi_prob),
            hdi_prob=hdi_prob,
            rope=rope_band,
            probabilities={
                f"P(region left)  [{self._right} better]": p_left,
                "P(region rope)  [equivalent]": p_rope,
                f"P(region right) [{self._left} better]": p_right,
            },
            decision=decision,
            extra={
                "rho": f"{self._rho_used:.4g}",
                "datasets": str(len(self._dataset_names)),
            },
        )

    def plot(
        self,
        kind: str = "simplex",
        hdi_prob: float = 0.89,
        ax: plt.Axes | None = None,
        order: list[str] | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the posterior of the fitted comparison.

        Parameters
        ----------
        kind : str, optional
            The figure to draw. Defaults to `simplex`.

                - `simplex` - the ROPE triple for the next dataset, in barycentric
                  coordinates. See Fig. 11 in [1]_.
                - `forest` - the per-dataset mean differences, raw against shrunk.
                - `ppc` - the posterior predictive check: the HDI of the mean
                  difference the model replicates for each dataset, against the
                  mean actually observed. See :meth:`posterior_predictive_check`.

        hdi_prob : float, optional
            Probability mass of the per-dataset HDIs drawn on the forest plot.
            Defaults to 0.89.
        ax : plt.Axes | None, optional
            Matplotlib Axes to plot on. If None, a new figure and axes will be created.
        order : list[str] | None, optional
            Dataset order for the forest plot, bottom to top. Defaults to the order
            the datasets first appeared in the fitted data. Ignored for `simplex`.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        plt.Axes
            The Axes the figure was drawn on.
        """
        self._check_if_fitted()
        validate_string(kind, HIERARCHICAL_PLOT_KINDS, "kind")
        rope_band = resolve_rope(self._rope)
        if kind == "simplex":
            theta_l, theta_e, theta_r = self._next_dataset_triple(rope_band)
            return plot_hierarchical_simplex(
                theta_left=theta_l,
                theta_rope=theta_e,
                theta_right=theta_r,
                left_model=self._left,
                right_model=self._right,
                rope_band=rope_band,
                ax=ax,
                **kwargs,
            )
        if kind == "ppc":
            replicated = self._replicate_observations(kwargs.pop("random_seed", None))
            return plot_hierarchical_ppc(
                dataset_names=list(self._dataset_names),
                replicated_means=replicated.mean(axis=2),
                observed_means=self._x.mean(axis=1),
                ax=ax,
                **kwargs,
            )
        table = self.per_dataset_table(hdi_prob=hdi_prob, round_ndigits=None)
        if order is not None:
            names = list(order)
            if set(names) != set(table["dataset"]) or len(names) != len(table):
                raise ValueError("order must contain each fitted dataset exactly once.")
            table = table.set_index("dataset").loc[names].reset_index()
        return plot_hierarchical_forest(
            dataset_names=list(table["dataset"]),
            raw_means=table["raw_mean"].to_numpy(),
            shrunk_means=table["shrunk_mean"].to_numpy(),
            shrunk_hdi_low=table["hdi_low"].to_numpy(),
            shrunk_hdi_high=table["hdi_high"].to_numpy(),
            rope_band=rope_band,
            left_model=self._left,
            right_model=self._right,
            ax=ax,
            **kwargs,
        )


def _build_dataset_matrix(
    data: pd.DataFrame,
    dataset_col: str,
    fold_col: str | None,
    maximize: bool,
) -> tuple[np.ndarray, list[str], str, str, int | None, int | None]:
    """Build the ``(q, n)`` difference matrix and names from a wide frame.

    Returns the difference matrix (``left - right`` per fold, sign-flipped when
    minimising), the dataset names in first-seen order, the two model names, and
    the ``(n_folds, n_runs)`` derived from ``fold_col`` (both ``None`` when no
    fold column is given).
    """
    if dataset_col not in data.columns:
        raise ValueError(f"dataset_col '{dataset_col}' not found in the data columns.")
    if fold_col is not None and fold_col not in data.columns:
        raise ValueError(f"fold_col '{fold_col}' not found in the data columns.")

    reserved = {dataset_col} | ({fold_col} if fold_col is not None else set())
    alg_cols = [c for c in data.columns if c not in reserved]
    if len(alg_cols) != 2:
        raise ValueError(
            "HierarchicalTTest expects exactly two algorithm columns besides "
            f"{sorted(reserved)}; got {len(alg_cols)}: {alg_cols}."
        )
    left, right = (str(c) for c in alg_cols)
    sign = 1.0 if maximize else -1.0

    names: list[str] = []
    rows: list[np.ndarray] = []
    fold_structures: set[tuple[int, int]] = set()
    for name, group in data.groupby(dataset_col, sort=False):
        names.append(str(name))
        if fold_col is not None:
            fold_structures.add(resolve_fold_structure(group[fold_col], fold_col))
        diff = sign * (group[alg_cols[0]].to_numpy() - group[alg_cols[1]].to_numpy())
        rows.append(np.asarray(diff, dtype=float))

    n_folds = n_runs = None
    if fold_col is not None:
        if len(fold_structures) != 1:
            raise ValueError(
                "Every dataset must share the same cross-validation design; got "
                f"differing (n_folds, n_runs) across datasets: {sorted(fold_structures)}."
            )
        n_folds, n_runs = fold_structures.pop()

    fold_counts = {len(r) for r in rows}
    if len(fold_counts) != 1:
        raise ValueError(
            "Every dataset must have the same number of fold rows for the "
            f"hierarchical model; got fold counts {sorted(fold_counts)}."
        )
    n = fold_counts.pop()
    if n < 2:
        raise ValueError(f"Need at least 2 folds per dataset, got {n}.")

    return np.vstack(rows), names, left, right, n_folds, n_runs
