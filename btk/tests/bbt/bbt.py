import warnings
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from btk.tests.common import BaseBayesianTest, _validate_params, hdi_from_samples

from ._types import (
    ALL_PROPERTIES_COLUMNS,
    HyperPriorType,
    InterpretationTypes,
    PlotKindType,
    PlotOrientationType,
    ReportedPropertyColumnType,
    TieSolverType,
)
from .alg import _construct_win_table, _get_pwin, _hdi
from .model import _mcmcbbt_pymc
from .plots import plot_cdd_diagram, plot_strong_posterior, plot_weak_posterior


class BBTTest(BaseBayesianTest):
    """
    BBT model estimator used for multi-dataset multi-model comparison [1]_.
    The model estimates posterior probabilities for each pair of the model.

    Parameters
    ----------
    local_rope_value: float | None, default None
        Deprecated. Fills in for whichever of `local_rope_effect_size` and
        `absolute_tie_threshold` applies to the data, which made a single number mean two
        different things on two different scales. Pass the explicit parameter instead;
        this one will be removed in a future release.

    local_rope_effect_size: float | None, default None
        The local ROPE of [1]_, sec. 6.1, as a Cohen's d **effect size** -- a multiple of
        the spread of the scores, not a difference in the metric. It applies whenever the
        within-dataset spread is known: repeated rows per dataset, or `data_sd` passed to
        :meth:`fit`.

        With repeated rows (Eq. 6), per dataset::

            d = mean(score_a - score_b) over the folds
            s = sample sd(score_a - score_b) over the folds
            d >  local_rope_effect_size * s  => model A wins
            d < -local_rope_effect_size * s  => model B wins
            otherwise                        => tie

        With `data_sd` (Eq. 4) the spread is pooled across the two models instead::

            s = sqrt((sd_a^2 + sd_b^2) / 2)

        The paper argues for 0.4, or 0.2 with 10 repetitions of 10-fold cross-validation.
        If None, no ties are recorded.

    absolute_tie_threshold: float | None, default None
        A tie threshold **in the units of the metric**, applied when there is one score
        per model per dataset and no `data_sd` -- fixed-split designs, where no
        within-dataset spread exists to form an effect size from::

            score_a - score_b >  absolute_tie_threshold => model A wins
            score_a - score_b < -absolute_tie_threshold => model B wins
            otherwise                                   => tie

        This extends [1]_, which only defines the effect-size ROPE. It exists so that a
        difference of, say, 1e-4 does not count as a win. Choose it on the scale of the
        metric (e.g. 0.01 for accuracy). If None, no ties are recorded.

    tie_solver: str, defaults to `add`
        The strategy to handle ties when sampling the BBT model.

            - `add` - Adds 1 win to both players for each tie.
            - `spread` - Adds `ceil(ties / 2)` wins to both players.
            - `forget` - Ignores the ties.
            - `davidson` - Uses Davidson's method to handle ties in the BBT model. See [1]_.
        Note: we found inconsistencies in mathematical foundations of the `spread` method, which we still investigate.
            For the time being, we recommend using alternative methods such as `add`.

        Note: [1]_ uses `spread`, and reports `add`, `forget` and `spread` to fit equally
        well (sec. 6.2). We default to `add` because `spread` is not exactly
        representable: half a victory to each player is not an integer count, and only
        integer counts are valid observations of the Binomial likelihood. Rounding is
        therefore unavoidable, and rounding up -- as the paper does -- awards
        `ceil(ties / 2)` to *both* players, inflating the number of matches by one for
        every odd tie count. `add` needs no rounding. Its own cost is that each tie enters
        the match total twice, so it concentrates the posterior more than `spread` does;
        `forget` discards the ties entirely. All four remain available.

    hyper_prior: str, default `log_normal`
        The hyper prior distribution for `sigma`, the spread of the abilities: the
        log-normal of [1]_, Eq. (2), or the half-normal / half-Cauchy alternatives
        discussed in sec. 4.1.

    scale: float, default 1.0
        The scale parameter for the hyper prior distribution.

    maximize: bool, default True
        Whether higher scores indicate better performance (e.g. accuracy/f1). If using a metric where the goal is to
        minimize the score (e.g. RMSE) set this to False.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import pandas as pd
    >>> from btk import BBTTest
    >>> data = pd.DataFrame({
    ...     'dataset': ['ds1', 'ds2', 'ds3'],
    ...     'model_a': [0.8, 0.75, 0.9],
    ...     'model_b': [0.7, 0.8, 0.85],
    ...     'model_c': [0.6, 0.65, 0.7]
    ... })
    >>> # One score per dataset, so the tie rule is on the metric scale.
    >>> model = BBTTest(absolute_tie_threshold=0.01, tie_solver="add")
    >>> model.fit(data, dataset_col='dataset')
    >>> model.posterior_table(rope_value=(0.45, 0.55))

    References
    ----------
    .. [1] `Jacques Wainer
        "A Bayesian Bradley-Terry model to compare multiple ML algorithms on multiple data sets"
        Journal of Machine Learning Research 24 (2023): 1-34
        <http://jmlr.org/papers/v24/22-0907.html>`_
    """

    ALL_PROPERTIES_COLUMNS = ALL_PROPERTIES_COLUMNS

    _WEAK_INTERPRETATION_THRESHOLD = 0.95
    _STRONG_INTERPRETATION_BETTER_THRESHOLD = 0.70
    _STRONG_INTERPRETATION_EQUAL_THRESHOLD = 0.55

    _diagnostics_label = "BBT"

    @_validate_params
    def __init__(
        self,
        local_rope_value: float | None = None,
        tie_solver: TieSolverType = "add",
        hyper_prior: HyperPriorType = "log_normal",
        maximize: bool = True,
        scale: float = 1.0,
        local_rope_effect_size: float | None = None,
        absolute_tie_threshold: float | None = None,
    ):
        if local_rope_value is not None:
            warnings.warn(
                "'local_rope_value' is deprecated because it means an effect size on "
                "paired data but a difference in metric units on unpaired data. Pass "
                "'local_rope_effect_size' (a Cohen's d, e.g. 0.4) or "
                "'absolute_tie_threshold' (metric units, e.g. 0.01) explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._local_rope_value = local_rope_value
        self._local_rope_effect_size = local_rope_effect_size
        self._absolute_tie_threshold = absolute_tie_threshold
        self._tie_solver = tie_solver
        self._hyper_prior = hyper_prior
        self._maximize = maximize
        self._scale = scale
        self._fitted = False

    def _resolve_tie_rules(self) -> tuple[float | None, float | None]:
        """Resolve the two tie thresholds, honouring the deprecated alias.

        Returns ``(local_rope_effect_size, absolute_tie_threshold)``. The
        deprecated ``local_rope_value`` fills in for whichever is unset, which
        reproduces the old behaviour of one number serving both scales.
        """
        effect_size = self._local_rope_effect_size
        absolute = self._absolute_tie_threshold
        if self._local_rope_value is not None:
            if effect_size is None:
                effect_size = self._local_rope_value
            if absolute is None:
                absolute = self._local_rope_value
        return effect_size, absolute

    @property
    def _use_davidson(self) -> bool:
        """Whether to fit the Davidson tie extension.

        Derived on access rather than frozen in ``__init__`` so that
        ``set_params(tie_solver="davidson")`` actually takes effect.
        """
        return self._tie_solver == "davidson"

    @staticmethod
    def _get_interpretation_columns(
        interpretation: InterpretationTypes,
    ) -> ReportedPropertyColumnType:
        return (
            "weak_interpretation_raw"
            if interpretation == "weak"
            else "strong_interpretation_raw"
        )

    def fit(
        self,
        data: pd.DataFrame,
        data_sd: pd.DataFrame | None = None,
        dataset_col: str = "dataset",
        **pymc_kwargs,
    ):
        """
        Fits the BBT for a given result dataframes.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing scores for the models on the datasets.
            If data_sd is provided, this dataframe should contain mean scores per model per dataset.
            If multiple scores per model per dataset are provided, data_sd is ignored, and dataset_col is required.
        data_sd : pd.DataFrame | None, optional
            Dataframe containing standard deviations of the scores for the models on the datasets.
        dataset_col : str, optional
            Column name for the dataset identifier. Defaults to "dataset".
        **pymc_kwargs
            Sampler options forwarded to ``pm.sample`` (draws, tune, chains, cores,
            target_accept, random_seed, progressbar, ...).

        Returns
        -------
        self : BBTTest
            Fitted BBTTest instance

        Warns
        -----
        UserWarning
            If the sampler diverged or the chains did not mix. Wainer (2023), sec. 2.2
            stresses that convergence must be checked on every run; see
            :meth:`diagnostics`.
        """
        effect_size, absolute_threshold = self._resolve_tie_rules()
        self._win_table, self._algorithms = _construct_win_table(
            data=data,
            data_sd=data_sd,
            dataset_col=dataset_col,
            local_rope_effect_size=effect_size,
            absolute_tie_threshold=absolute_threshold,
            tie_solver=self._tie_solver,
            maximize=self._maximize,
        )

        self._fit_posterior, self._pymc_model = _mcmcbbt_pymc(
            table=self._win_table,
            use_davidson=self._use_davidson,
            hyper_prior=self._hyper_prior,
            scale=self._scale,
            **pymc_kwargs,
        )
        # Expose the posterior through the shared ``idata_`` endpoint as well;
        # ``_fit_posterior`` stays for backward compatibility.
        self._idata = self._fit_posterior

        self._fitted = True
        self._warn_on_bad_diagnostics()

        return self

    # -- diagnostics --------------------------------------------------------

    def _diagnostic_vars(self) -> list[str]:
        """Restrict the diagnostics to the model's own parameters."""
        var_names = ["beta", "sigma"]
        if self._use_davidson:
            var_names += ["nu", "sigmanu"]
        return var_names

    def posterior_predictive_check(
        self,
        hdi_probs: Sequence[float] = (0.5, 0.9, 0.95, 1.0),
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Posterior predictive check of the fitted model.

        Replays the observed win counts through the fitted model and reports, for each
        HDI mass, the proportion of the observed counts that fall inside the HDI of the
        replicated counts. This is the non-graphical form of the check in Wainer (2023),
        sec. 5.4: ideally the proportion inside the 90% HDI is at least 0.9.

        Parameters
        ----------
        hdi_probs : Sequence[float], optional
            HDI masses to evaluate. Defaults to the paper's ``(0.5, 0.9, 0.95, 1.0)``.
        random_seed : int | None, optional
            Seed for the replicated draws.

        Returns
        -------
        pd.DataFrame
            One row per HDI mass, with a ``wins`` proportion and, when the Davidson tie
            model is used, a ``ties`` proportion as well.
        """
        self._check_if_fitted()
        var_names = ["win1_obs"] + (["ties_obs"] if self._use_davidson else [])
        replicated_by_var = self._replicate_counts(var_names, random_seed)

        observed = {
            "win1_obs": self._win_table[:, 2],
            "ties_obs": self._win_table[:, 4],
        }
        labels = {"win1_obs": "wins", "ties_obs": "ties"}

        records = []
        for prob in hdi_probs:
            row: dict[str, float] = {"hdi": float(prob)}
            for var in var_names:
                replicated = replicated_by_var[var]
                inside = [
                    hdi_from_samples(replicated[:, k], prob)[0]
                    <= observed[var][k]
                    <= hdi_from_samples(replicated[:, k], prob)[1]
                    for k in range(replicated.shape[1])
                ]
                row[labels[var]] = float(np.mean(inside))
            records.append(row)
        return pd.DataFrame.from_records(records)

    def _replicate_counts(
        self, var_names: Sequence[str], random_seed: int | None
    ) -> dict[str, np.ndarray]:
        """Replay the observed counts through the model, flattened to ``(draws, matchups)``."""
        with self._pymc_model:
            ppc = pm.sample_posterior_predictive(
                self._fit_posterior,
                var_names=list(var_names),
                progressbar=False,
                random_seed=random_seed,
            )
        return {
            var: ppc.posterior_predictive[var]
            .to_numpy()
            .reshape(-1, len(self._win_table))
            for var in var_names
        }

    def plot_posterior_predictive(
        self,
        pairs: Sequence[tuple[str, str]],
        random_seed: int | None = None,
        axes: Sequence[plt.Axes] | None = None,
    ) -> np.ndarray:
        """Graphical posterior predictive check for selected pairs of models.

        For each pair, draws the histogram of the replicated win counts with the
        observed count as a vertical line (Wainer 2023, sec. 5.4). Wins are counted for
        the model that comes first in the win table, which the panel title names.

        Parameters
        ----------
        pairs : Sequence[tuple[str, str]]
            Pairs of model names to plot, one panel each, in any order within a pair.
        random_seed : int | None, optional
            Seed for the replicated draws.
        axes : Sequence[plt.Axes] | None, optional
            One axes per pair. If None, a new row of subplots is created.

        Returns
        -------
        np.ndarray
            The axes, one per pair.
        """
        self._check_if_fitted()
        replicated = self._replicate_counts(["win1_obs"], random_seed)["win1_obs"]
        if axes is None:
            _, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 3.6))
        axes = np.atleast_1d(axes)

        for ax, (a, b) in zip(axes, pairs, strict=True):
            lo, hi = sorted((self._algorithms.index(a), self._algorithms.index(b)))
            (k,) = np.flatnonzero(
                (self._win_table[:, 0] == lo) & (self._win_table[:, 1] == hi)
            )
            reps, observed = replicated[:, k], self._win_table[k, 2]
            n = self._win_table[k, 2] + self._win_table[k, 3]
            ax.hist(
                reps,
                bins=np.arange(reps.min(), reps.max() + 2) - 0.5,
                density=True,
                color="C0",
                alpha=0.6,
            )
            ax.axvline(observed, color="k", linewidth=2.5)
            ax.set_title(
                f"{a} vs {b}\nwins for {self._algorithms[lo]} (obs={observed}, n={n})",
                fontsize=9,
            )
            ax.set_xlabel("replicated wins")
        axes[0].set_ylabel("density")
        return axes

    # -- fitted data ---------------------------------------------------------

    @property
    def algorithms(self) -> list[str]:
        """The algorithm names, in the column order of the ``beta`` posterior.

        Needed to line up ``idata_.posterior["beta"]`` with the models it
        describes, since the posterior itself carries only positional indices.
        """
        self._check_if_fitted()
        return list(self._algorithms)

    @property
    def win_table(self) -> pd.DataFrame:
        """The win counts the model was actually fitted to.

        The scores go in, per-dataset win counts come out, and everything
        downstream is a function of this table alone -- which is what makes BBT
        metric-agnostic. Inspecting it is the quickest way to see what a tie rule
        did, and to spot pairs with too few decisive matches to support a claim.

        Returns
        -------
        pd.DataFrame
            One row per pair, with ``alg1``/``alg2`` and the ``wins1``, ``wins2``
            and ``ties`` counts between them.
        """
        self._check_if_fitted()
        return pd.DataFrame(
            {
                "alg1": [self._algorithms[int(row[0])] for row in self._win_table],
                "alg2": [self._algorithms[int(row[1])] for row in self._win_table],
                "wins1": self._win_table[:, 2].astype(int),
                "wins2": self._win_table[:, 3].astype(int),
                "ties": self._win_table[:, 4].astype(int),
            }
        )

    def pairwise_samples(self, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
        r"""Posterior draws of the pairwise probability for the given pairs.

        The Bradley-Terry probability that ``a`` beats ``b`` is
        ``w_a / (w_a + w_b)`` with ``w = exp(beta)``, i.e. the logistic of the
        ability difference. :meth:`posterior_table` summarises these draws;
        this returns them, for plotting a density or for any interval the
        summary does not cover.

        Parameters
        ----------
        pairs : Iterable[tuple[str, str]]
            Pairs of algorithm names. Each is read in the given order, so
            ``("a", "b")`` yields ``P(a > b)`` and ``("b", "a")`` its complement.

        Returns
        -------
        pd.DataFrame
            One column per pair, named ``"a > b"``, holding the posterior draws
            of that probability flattened across chains.
        """
        self._check_if_fitted()
        beta = self.idata_.posterior["beta"].to_numpy()
        beta = beta.reshape(-1, beta.shape[-1])
        index = {name: i for i, name in enumerate(self._algorithms)}

        columns = {}
        for left, right in pairs:
            unknown = [name for name in (left, right) if name not in index]
            if unknown:
                raise ValueError(
                    f"Unknown algorithms {unknown}; the fitted models are "
                    f"{sorted(index)}."
                )
            difference = beta[:, index[left]] - beta[:, index[right]]
            columns[f"{left} > {right}"] = 1.0 / (1.0 + np.exp(-difference))
        return pd.DataFrame(columns)

    @property
    def beta_ranking(self) -> dict[str, float]:
        r"""
        Get the $\beta$ values for each model.

        Beta values can be used for ranking the models globally from best to worst (higher beta indicates better performance).
        However, they do not have a direct probabilistic interpretation like the pairwise probabilities obtained from the posterior table.

        Returns
        -------
        dict[str, float]
            Dictionary mapping model names to their posterior mean beta values.
        """
        self._check_if_fitted()
        beta = self._fit_posterior.posterior["beta"].to_numpy()
        mean_beta = np.mean(beta.reshape(-1, beta.shape[-1]), axis=0)
        return dict(zip(self._algorithms, mean_beta, strict=True))

    def posterior_table(
        self,
        rope_value: tuple[float, float] = (0.45, 0.55),
        control_model: str | None = None,
        selected_models: Iterable[str] | None = None,
        columns: Iterable[ReportedPropertyColumnType] = (
            "mean",
            "delta",
            "above_50",
            "in_rope",
            "weak_interpretation",
        ),
        hdi_proba: float = 0.89,
        round_ndigits: int | None = 2,
    ) -> pd.DataFrame:
        """Compute posterior table containing sampling results for the fitted BBT model.

        Parameters
        ----------
        rope_value : tuple[float, float], optional
            Region of Practical Equivalence (ROPE). Defaults to (0.45, 0.55).
        control_model : str | None, optional
            Control model for comparison. Defaults to None.
        selected_models : Iterable[str] | None, optional
            Subset of models to include in the posterior table. Defaults to None.
        columns : Iterable[ReportedPropertyColumnType], optional
            Columns to include in the posterior table. Defaults to minimum set for weak interpretation.
        hdi_proba : float, optional
            Highest Density Interval probability. Defaults to 0.89.
        round_ndigits : int | None, optional
            Number of digits to round the results to. Defaults to 2.

        Returns
        -------
        pd.DataFrame
            Posterior table containing sampling results for the fitted BBT model.
        """
        self._check_if_fitted()

        samples, names = _get_pwin(
            bbt_result=self._fit_posterior,
            alg_names=self._algorithms,
            control=control_model,
            selected=list(selected_models) if selected_models is not None else None,
        )
        out_table = pd.DataFrame({"pair": names})
        pair_parts = out_table["pair"].str.split(">")
        out_table["left_model"] = pair_parts.str[0].str.strip()
        out_table["right_model"] = pair_parts.str[1].str.strip()
        out_table["median"] = np.median(samples, axis=0)
        out_table["mean"] = np.mean(samples, axis=0)
        out_table["above_50"] = np.mean(samples > 0.5, axis=0)
        out_table["in_rope"] = np.mean(
            (samples >= rope_value[0]) & (samples <= rope_value[1]), axis=0
        )
        out_table["weak_interpretation_raw"] = np.where(
            out_table["in_rope"] >= self._WEAK_INTERPRETATION_THRESHOLD,
            "=",
            np.where(
                out_table["above_50"] >= self._WEAK_INTERPRETATION_THRESHOLD,
                ">",
                "?",
            ),
        )
        out_table["weak_interpretation"] = np.where(
            out_table["weak_interpretation_raw"] == ">",
            out_table["left_model"] + " better",
            np.where(
                out_table["weak_interpretation_raw"] == "=",
                "Equivalent",
                "Unknown",
            ),
        )

        out_table["strong_interpretation_raw"] = np.where(
            out_table["mean"] > self._STRONG_INTERPRETATION_BETTER_THRESHOLD,
            ">",
            np.where(
                out_table["mean"] <= self._STRONG_INTERPRETATION_EQUAL_THRESHOLD,
                "=",
                "?",
            ),
        )
        out_table["strong_interpretation"] = np.where(
            out_table["strong_interpretation_raw"].str.endswith(">"),
            out_table["left_model"] + " better",
            np.where(
                out_table["strong_interpretation_raw"] == "=",
                "Equivalent",
                "Unknown",
            ),
        )

        hdi_values = _hdi(samples, hdi_proba)
        out_table["hdi_low"] = hdi_values[0]
        out_table["hdi_high"] = hdi_values[1]
        out_table["delta"] = out_table["hdi_high"] - out_table["hdi_low"]

        columns = list(columns)
        for col in columns:
            if col not in out_table.columns:
                raise ValueError(
                    f"Column {col} is not available in the posterior table. "
                    f"Available columns: {self.ALL_PROPERTIES_COLUMNS}."
                )

        if round_ndigits is not None:
            return out_table.round(round_ndigits)[["pair", *columns]]
        return out_table[["pair", *columns]]

    @_validate_params
    def rope_comparison_control_table(
        self,
        rope_values: Sequence[tuple[float, float]],
        control_model: str,
        selected_models: Sequence[str] | None = None,
        interpretation: InterpretationTypes = "weak",
        return_as_array: bool = False,
        join_char: str = ", ",
    ) -> pd.DataFrame:
        """
        Construct a table comparing models against predefined control models across multiple ROPEs.
        The output table contains N rows (one per ROPE) and 5 columns
        (rope value, better models, equivalent models, worse models, unknown models).

        Parameters
        ----------
        rope_values : Sequence[tuple[float, float]]
            List of ROPE tuples to evaluate.
        control_model : str
            Control model for comparison.
        selected_models : Sequence[str] | None, optional
            Subset of models to include. Defaults to None.
        interpretation : {"weak", "strong"}, optional
            Type of interpretation to use, see [1]_. Defaults to "weak".
        return_as_array : bool, optional
            Whether the individual cells should contain model names as list or joined into single string.
            Defaults to False.
        join_char : str, optional
            Character(s) used to join multiple model names in a single cell. Defaults to ", ".

        Returns
        -------
        pd.DataFrame
            Table comparing models against control models across multiple ROPEs.
        """
        self._check_if_fitted()
        records = []
        interpretation_col = self._get_interpretation_columns(interpretation)
        for rope in rope_values:
            posterior_df = self.posterior_table(
                rope_value=rope,
                control_model=control_model,
                selected_models=selected_models,
                columns=(
                    "left_model",
                    "right_model",
                    "weak_interpretation_raw",
                    "strong_interpretation_raw",
                ),
            )
            better_models: list[str] = []
            equivalent_models: list[str] = []
            worse_models: list[str] = []
            unknown_models: list[str] = []
            for _, row in posterior_df.iterrows():
                non_control_model = (
                    row["right_model"]
                    if row["left_model"] == control_model
                    else row["left_model"]
                )

                if row[interpretation_col] == ">":
                    if row["left_model"] == control_model:
                        worse_models.append(non_control_model)
                    else:
                        better_models.append(non_control_model)
                elif row[interpretation_col] == "=":
                    equivalent_models.append(non_control_model)
                elif row[interpretation_col] == "?":
                    unknown_models.append(non_control_model)
                else:
                    raise RuntimeError(
                        f"Unexpected interpretation value {row[interpretation_col]} in row {row['pair']}. Please report this as a bug."
                    )
            if not return_as_array:
                better_models_str = join_char.join(better_models)
                equivalent_models_str = join_char.join(equivalent_models)
                worse_models_str = join_char.join(worse_models)
                unknown_models_str = join_char.join(unknown_models)
                records.append(
                    {
                        "rope_value": rope,
                        "better_models": better_models_str,
                        "equivalent_models": equivalent_models_str,
                        "worse_models": worse_models_str,
                        "unknown_models": unknown_models_str,
                    }
                )
            else:
                records.append(
                    {
                        "rope_value": rope,
                        "better_models": better_models,
                        "equivalent_models": equivalent_models,
                        "worse_models": worse_models,
                        "unknown_models": unknown_models,
                    }
                )

        return pd.DataFrame.from_records(records)

    @_validate_params
    def plot(
        self,
        kind: PlotKindType = "strong-posterior",
        control_model: str | None = None,
        selected_pairs: Sequence[tuple[str, str]] | None = None,
        selected_models: Iterable[str] | None = None,
        hdi_prob: float = 0.89,
        rope_value: tuple[float, float] = (0.45, 0.55),
        orientation: PlotOrientationType = "horizontal",
        ax: plt.Axes | Sequence[plt.Axes] | None = None,
        **kwargs,
    ) -> plt.Axes | np.ndarray:
        """Plot the posterior of the fitted BBT model; shorthand for the ``plot_*`` methods.

        Arguments the chosen ``kind`` does not use are ignored, so one call
        signature serves every kind. See the dedicated methods for what each
        figure shows and what its arguments mean.

        Parameters
        ----------
        kind : str, optional
            The figure to draw. Defaults to `strong-posterior`.

                - `strong-posterior` - :meth:`plot_strong_posterior`; uses
                  ``hdi_prob``, ignores ``rope_value``.
                - `weak-posterior` - :meth:`plot_weak_posterior`; uses
                  ``rope_value``, ignores ``hdi_prob``.

        control_model : str | None, optional
            Compare every other model against this one. Defaults to None.
        selected_pairs : Sequence[tuple[str, str]] | None, optional
            Pairs to draw, each read in the given order. Defaults to None.
        selected_models : Iterable[str] | None, optional
            With ``control_model``, the subset of models to compare against it.
        hdi_prob : float, optional
            Probability mass of the HDIs. Defaults to 0.89.
        rope_value : tuple[float, float], optional
            Region of Practical Equivalence (ROPE). Defaults to (0.45, 0.55).
        orientation : str, optional
            `horizontal` or `vertical`. Defaults to `horizontal`.
        ax : plt.Axes | Sequence[plt.Axes] | None, optional
            One Axes for `strong-posterior`, two for `weak-posterior`. If None,
            a new figure is created.
        **kwargs
            Additional keyword arguments passed to the underlying plotting function.

        Returns
        -------
        plt.Axes | np.ndarray
            A single Axes for `strong-posterior`, an array of two for `weak-posterior`.

        See Also
        --------
        plot_strong_posterior : Posterior mean and HDI under the strong interpretation.
        plot_weak_posterior : P(pi > 0.5) and P(pi in ROPE) under the weak interpretation.
        """
        selection = {
            "control_model": control_model,
            "selected_pairs": selected_pairs,
            "selected_models": selected_models,
            "orientation": orientation,
            "ax": ax,
        }
        if kind == "strong-posterior":
            return self.plot_strong_posterior(**selection, hdi_prob=hdi_prob, **kwargs)
        if kind == "weak-posterior":
            return self.plot_weak_posterior(
                **selection, rope_value=rope_value, **kwargs
            )
        raise ValueError(f"Unsupported plot kind {kind!r}.")

    @_validate_params
    def plot_strong_posterior(
        self,
        control_model: str | None = None,
        selected_pairs: Sequence[tuple[str, str]] | None = None,
        selected_models: Iterable[str] | None = None,
        hdi_prob: float = 0.89,
        orientation: PlotOrientationType = "horizontal",
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        r"""Plot each pairwise probability under the strong interpretation.

        Draws the posterior mean of :math:`\pi` with its HDI for each comparison,
        coloured by the strong interpretation (Wainer 2023, sec. 8.3): `better`
        if :math:`E[\pi] > 0.70`, `equivalent` if :math:`0.45 \leq E[\pi] \leq 0.55`,
        `weaker` if :math:`E[\pi] < 0.30`, `no claim` otherwise. Shaded regions
        mark the three claims.

        Exactly one of ``control_model`` or ``selected_pairs`` must be given: with
        many models the full set of pairs is too large to read.

        Parameters
        ----------
        control_model : str | None, optional
            Compare every other model against this one, each read as
            ``P(model > control_model)``. Defaults to None.
        selected_pairs : Sequence[tuple[str, str]] | None, optional
            Pairs to draw, each read in the given order, so ``("a", "b")`` is
            ``P(a > b)``. Defaults to None.
        selected_models : Iterable[str] | None, optional
            With ``control_model``, the subset of models to compare against it.
            Defaults to all fitted models.
        hdi_prob : float, optional
            Probability mass of the HDIs. Defaults to 0.89.
        orientation : str, optional
            `horizontal` lays the comparisons along the x axis, best on the left;
            `vertical` lays them along the y axis, best on top. Defaults to
            `horizontal`.
        ax : plt.Axes | None, optional
            Matplotlib Axes to plot on. If None, a new figure and axes will be created.
        **kwargs
            Additional keyword arguments passed to the ``scatter`` call of the means.

        Returns
        -------
        plt.Axes
            The Axes the figure was drawn on.

        See Also
        --------
        plot_weak_posterior : The same comparisons under the weak interpretation.
        """
        samples, labels, value_label, subtitle = self._plot_samples(
            control_model, selected_pairs, selected_models
        )
        hdi_values = _hdi(samples, hdi_prob)
        return plot_strong_posterior(
            labels=labels,
            means=samples.mean(axis=0),
            hdi_low=hdi_values[0],
            hdi_high=hdi_values[1],
            better_threshold=self._STRONG_INTERPRETATION_BETTER_THRESHOLD,
            equal_threshold=self._STRONG_INTERPRETATION_EQUAL_THRESHOLD,
            hdi_prob=hdi_prob,
            value_label=value_label,
            orientation=orientation,
            subtitle=subtitle,
            ax=ax,
            **kwargs,
        )

    @_validate_params
    def plot_weak_posterior(
        self,
        control_model: str | None = None,
        selected_pairs: Sequence[tuple[str, str]] | None = None,
        selected_models: Iterable[str] | None = None,
        rope_value: tuple[float, float] = (0.45, 0.55),
        orientation: PlotOrientationType = "horizontal",
        ax: Sequence[plt.Axes] | None = None,
        **kwargs,
    ) -> np.ndarray:
        r"""Plot each pairwise probability under the weak interpretation.

        Draws two panels per comparison: :math:`P(\pi > 0.5)` and
        :math:`P(\pi \in \mathrm{ROPE})`, coloured by the weak interpretation
        (Wainer 2023, sec. 8.2): `equivalent` if
        :math:`P(\pi \in \mathrm{ROPE}) \geq 0.95`, otherwise `better` if
        :math:`P(\pi > 0.5) \geq 0.95`, `weaker` if :math:`P(\pi < 0.5) \geq 0.95`,
        `no claim` otherwise. Comparisons are ordered by :math:`E[\pi]`.

        Exactly one of ``control_model`` or ``selected_pairs`` must be given: with
        many models the full set of pairs is too large to read.

        Parameters
        ----------
        control_model : str | None, optional
            Compare every other model against this one, each read as
            ``P(model > control_model)``. Defaults to None.
        selected_pairs : Sequence[tuple[str, str]] | None, optional
            Pairs to draw, each read in the given order, so ``("a", "b")`` is
            ``P(a > b)``. Defaults to None.
        selected_models : Iterable[str] | None, optional
            With ``control_model``, the subset of models to compare against it.
            Defaults to all fitted models.
        rope_value : tuple[float, float], optional
            Region of Practical Equivalence (ROPE). Defaults to (0.45, 0.55).
        orientation : str, optional
            `horizontal` stacks the panels and lays the comparisons along the
            x axis, best on the left; `vertical` puts the panels side by side
            and lays the comparisons along the y axis, best on top. Defaults to
            `horizontal`.
        ax : Sequence[plt.Axes] | None, optional
            Exactly two Axes: ``ax[0]`` for :math:`P(\pi > 0.5)`, ``ax[1]`` for
            :math:`P(\pi \in \mathrm{ROPE})`. If None, a new figure is created.
        **kwargs
            Additional keyword arguments passed to both ``scatter`` calls.

        Returns
        -------
        np.ndarray
            The two Axes drawn on, in the order described for ``ax``.

        See Also
        --------
        plot_strong_posterior : The same comparisons under the strong interpretation.
        """
        samples, labels, value_label, subtitle = self._plot_samples(
            control_model, selected_pairs, selected_models
        )
        return plot_weak_posterior(
            labels=labels,
            means=samples.mean(axis=0),
            above_50=np.mean(samples > 0.5, axis=0),
            below_50=np.mean(samples < 0.5, axis=0),
            in_rope=np.mean(
                (samples >= rope_value[0]) & (samples <= rope_value[1]), axis=0
            ),
            threshold=self._WEAK_INTERPRETATION_THRESHOLD,
            rope_value=rope_value,
            value_label=value_label,
            orientation=orientation,
            subtitle=subtitle,
            ax=ax,
            **kwargs,
        )

    def _plot_samples(
        self,
        control_model: str | None,
        selected_pairs: Sequence[tuple[str, str]] | None,
        selected_models: Iterable[str] | None,
    ) -> tuple[np.ndarray, list[str], str, str | None]:
        """Posterior draws of ``pi`` per plotted comparison, with labels.

        Returns the ``(draws, comparisons)`` sample matrix, one label per
        comparison, the axis label defining ``pi`` and an optional subtitle.
        """
        self._check_if_fitted()
        if (control_model is None) == (selected_pairs is None):
            raise ValueError("Pass exactly one of control_model or selected_pairs.")
        if selected_pairs is not None and selected_models is not None:
            raise ValueError(
                "selected_models only applies with control_model; "
                "list the pairs in selected_pairs instead."
            )
        if selected_pairs is not None:
            pairs = list(selected_pairs)
            if not pairs:
                raise ValueError("selected_pairs must contain at least one pair.")
            draws = self.pairwise_samples(pairs)
            value_label = r"$\pi = P(\mathrm{left} \succ \mathrm{right})$"
            return draws.to_numpy(), list(draws.columns), value_label, None

        samples, names = _get_pwin(
            bbt_result=self._fit_posterior,
            alg_names=self._algorithms,
            control=control_model,
            selected=list(selected_models) if selected_models is not None else None,
        )
        # ``_get_pwin`` puts the better model on the left; re-orient every pair
        # as ``other > control`` so the control is the fixed reference.
        labels = []
        for k, name in enumerate(names):
            left, right = (part.strip() for part in name.split(">"))
            if left == control_model:
                samples[:, k] = 1.0 - samples[:, k]
                left = right
            labels.append(left)
        value_label = rf"$\pi = P(\mathrm{{model}} \succ$ {control_model}$)$"
        # Rows are named by the model alone, so say what they are compared to.
        return samples, labels, value_label, f"Control model: {control_model}"

    @_validate_params
    def plot_cdd_diagram(
        self,
        rope_value: tuple[float, float] = (0.45, 0.55),
        interpretation: InterpretationTypes = "weak",
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot critical difference diagram for the fitted BBT model."""
        self._check_if_fitted()
        posterior_df = self.posterior_table(
            rope_value=rope_value,
            columns=(
                "left_model",
                "right_model",
                "weak_interpretation_raw",
                "strong_interpretation_raw",
            ),
            round_ndigits=None,
        )
        interpretation_col = self._get_interpretation_columns(interpretation)
        # ``pos`` is the aggregated rank: 1 is the best algorithm, i.e. the
        # highest mean beta. ``_plot_cdd_diagram`` draws ``pos = 1`` at the
        # "better" end of the ruler, so the sort must be descending.
        models_df = pd.DataFrame(
            {
                "model": self._algorithms,
                "mean": self.beta_ranking.values(),
            }
        ).sort_values("mean", ascending=False)
        models_df["pos"] = range(1, len(models_df) + 1)

        return plot_cdd_diagram(
            models_df=models_df,
            posterior_df=posterior_df,
            interpretation_col=interpretation_col,
            ax=ax,
            **kwargs,
        )
