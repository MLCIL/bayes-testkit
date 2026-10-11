"""Bayesian signed-rank test implementation for btk.tests.signed_rank."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from btk.tests.common import (
    BaseBayesianTest,
    decision_from_partition,
    region_probabilities,
    resolve_rope,
    samples_to_inference_data,
)

from ._types import ALL_SIGNED_RANK_COLUMNS

_DEFAULT_COLUMNS: tuple[str, ...] = (
    "comparison",
    "p_left",
    "p_rope",
    "p_right",
    "decision",
)


class SignedRankTest(BaseBayesianTest):
    r"""
    Bayesian signed-rank test estimator used for two-model multi-dataset comparison [1]_.
    The model places a Dirichlet process prior on the distribution of the mean difference
    of scores between the two models over the datasets, and estimates its posterior
    nonparametrically, without assuming the distribution to be symmetric.

    Each posterior draw yields a triple ``(theta_left, theta_rope, theta_right)`` -- the
    probabilities that the average of two differences drawn from that distribution lies
    below, within or above the ROPE, i.e. that the right model is practically better, the
    two are practically equivalent, or the left model is practically better. Following
    Section 4.2.2 of [1]_, the reported ``p_left``/``p_rope``/``p_right`` are the
    proportions of draws whose triple falls in each simplex region
    ``theta_i >= max(theta_j, theta_k)``, which is what the paper's Table 9 tabulates;
    they sum to one. The posterior means of the triple itself are available as the
    ``mean_theta_*`` columns of :meth:`decision_table`.

    Parameters
    ----------
    rope: float | tuple[float, float], default 0.01
        The region of practical equivalence, defined on the scale of the score
        difference. Differences of a smaller magnitude are treated as practically
        irrelevant.

            - a scalar is a half-width `r`, giving the band `[-r, r]`
            - a 2-tuple is used as an explicit `(lo, hi)` band

    maximize: bool, default True
        Whether higher scores indicate better performance (e.g. accuracy/f1). If using a
        metric where the goal is to minimize the score (e.g. RMSE) set this to False, so
        that a positive difference still indicates the left model being better.

    prior_strength: float, default 0.5
        Strength `s` of the Dirichlet process prior, i.e. the weight of the prior
        pseudo-observation relative to that of a single dataset. The default of 0.5 is
        the paper's.

    prior_pseudo_obs: float, default 0.0
        Location `z_0` of the prior pseudo-observation, on the scale of the score
        difference. The default of 0 is the paper's, and places it inside the ROPE.
        `np.inf` and `-np.inf` place it in favour of the left and the right model
        respectively, which is how Section 4.3 of [1]_ checks the sensitivity of the
        conclusions to the prior.

    Attributes
    ----------
    fitted: bool
        Whether the model has been fitted.

    idata_: arviz.InferenceData
        Monte Carlo draws of the Dirichlet weights of the prior pseudo-observation and of
        each dataset, exposed for the ArviZ toolchain.

    Examples
    --------
    >>> import pandas as pd
    >>> from btk import SignedRankTest
    >>> data = pd.DataFrame({
    ...     'dataset': ['ds1', 'ds2', 'ds3', 'ds4'],
    ...     'model_a': [0.81, 0.75, 0.92, 0.66],
    ...     'model_b': [0.78, 0.74, 0.88, 0.61]
    ... })
    >>> model = SignedRankTest(rope=0.01)
    >>> model.fit(data, dataset_col='dataset')
    >>> model.decision_table()

    Notes
    -----
    Under a Dirichlet process prior with base measure :math:`\delta_{z_0}`, the
    posterior of the distribution of the difference :math:`z = \text{left} - \text{right}`
    is

    .. math::
        p(z) = w_0 \delta_{z_0}(z) + \sum_{j=1}^{q} w_j \delta_{z_j}(z), \quad
        (w_0, w_1, \ldots, w_q) \sim Dir(s, 1, \ldots, 1),

    where :math:`z_1, \ldots, z_q` are the mean differences on the :math:`q` datasets
    (Eq. 9). Each draw of the weights gives the triple

    .. math::
        \theta_l = \sum_{i=0}^{q} \sum_{j=0}^{q} w_i w_j I_{(-\infty, -2r)}(z_i + z_j),

    with :math:`\theta_e` and :math:`\theta_r` defined likewise over :math:`[-2r, 2r]`
    and :math:`(2r, \infty)` (Eq. 11).

    References
    ----------
    .. [1] `Alessio Benavoli, Giorgio Corani, Janez Demsar, Marco Zaffalon
        "Time for a Change: a Tutorial for Comparing Multiple Classifiers Through
        Bayesian Analysis"
        Journal of Machine Learning Research 18 (2017): 1-36
        <http://jmlr.org/papers/v18/16-305.html>`_
    """

    ALL_SIGNED_RANK_COLUMNS = ALL_SIGNED_RANK_COLUMNS

    _DEFAULT_THRESHOLD = 0.95

    def __init__(
        self,
        rope: float | tuple[float, float] = 0.01,
        maximize: bool = True,
        prior_strength: float = 0.5,
        prior_pseudo_obs: float = 0.0,
    ):
        if prior_strength <= 0:
            raise ValueError(f"prior_strength must be positive, got {prior_strength}.")
        self._rope = rope
        self._maximize = maximize
        self._prior_strength = prior_strength
        self._prior_pseudo_obs = prior_pseudo_obs
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        dataset_col: str = "dataset",
        draws: int = 50_000,
        random_seed: int | None = None,
    ) -> "SignedRankTest":
        """
        Fits the signed-rank test for a given result dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the scores of the two models, one or more rows per
            dataset, with exactly two model columns holding the left and the right model,
            in that order. The rows of a dataset, e.g. the folds of its cross-validation,
            are averaged into its mean difference.
        dataset_col : str, optional
            Column name for the dataset identifier. Defaults to "dataset".
        draws : int, optional
            Number of Monte Carlo draws of the Dirichlet weights. Defaults to 50000.
        random_seed : int | None, optional
            Seed for the Monte Carlo draws.

        Returns
        -------
        self : SignedRankTest
            Fitted SignedRankTest instance
        """
        z, left, right = _dataset_differences(data, dataset_col, self._maximize)

        self._z = z
        self._left = left
        self._right = right

        alpha = np.concatenate([[self._prior_strength], np.ones(len(z))])
        rng = np.random.default_rng(random_seed)
        self._weights = rng.dirichlet(alpha, size=draws)
        self._idata = samples_to_inference_data({"weights": self._weights})

        self._fitted = True
        return self

    def diagnostics(self) -> dict[str, float]:
        """Not applicable to this test, which has no sampler.

        Raises
        ------
        NotImplementedError
            Always. The weights in :attr:`idata_` are independent draws from a Dirichlet
            distribution rather than a Markov chain, so R-hat, effective sample size and
            divergences have no meaning here. The sampled tests --
            :class:`~btk.HierarchicalTTest` and :class:`~btk.BBTTest` -- implement them.
        """
        raise NotImplementedError(
            "SignedRankTest draws its posterior weights directly from a Dirichlet "
            "distribution, so there is no sampler to diagnose: idata_ holds independent "
            "draws, not a Markov chain. Convergence diagnostics apply to "
            "HierarchicalTTest and BBTTest."
        )

    def _triple(
        self, rope_band: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior samples of the ROPE triple ``(theta_l, theta_e, theta_r)`` (Eq. 11).

        Every pair of atoms of the posterior, the prior pseudo-observation included,
        adds the product of its weights to the region its average falls in.
        """
        lo, hi = rope_band
        atoms = np.concatenate([[self._prior_pseudo_obs], self._z])
        walsh = (atoms[:, None] + atoms[None, :]) / 2
        regions = (walsh < lo, (walsh >= lo) & (walsh <= hi), walsh > hi)
        theta_l, theta_e, theta_r = (
            np.sum((self._weights @ region) * self._weights, axis=1)
            for region in regions
        )
        return theta_l, theta_e, theta_r

    def decision_table(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        columns: Iterable[str] = _DEFAULT_COLUMNS,
        round_ndigits: int | None = 3,
    ) -> pd.DataFrame:
        """
        Construct the decision table for the fitted comparison.

        The table reports the three ROPE probabilities and the resulting decision.
        Following Section 4.2.2 of [1]_, each probability is the proportion of posterior
        draws whose triple falls in the corresponding simplex region
        ``theta_i >= max(theta_j, theta_k)`` -- the quantity reported in the paper's
        Table 9 -- where p_left is the probability of the right model being practically
        better, and p_right of the left model being practically better. The posterior
        means of the triple are available separately as the ``mean_theta_*`` columns.

        Parameters
        ----------
        threshold : float, optional
            The probability a region must reach for the corresponding decision to be
            made, otherwise the decision is "Unknown". Defaults to 0.95, as used in
            [1]_.
        columns : Iterable[str], optional
            Columns to include in the table. See ALL_SIGNED_RANK_COLUMNS for the
            available values.
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
        theta_l, theta_e, theta_r = self._triple(rope_band)
        p_left, p_rope, p_right = region_probabilities(theta_l, theta_e, theta_r)
        decision, decision_raw = decision_from_partition(
            p_left, p_rope, p_right, self._left, self._right, threshold
        )

        row = {
            "comparison": f"{self._left} vs {self._right}",
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


def _dataset_differences(
    data: pd.DataFrame,
    dataset_col: str,
    maximize: bool,
) -> tuple[np.ndarray, str, str]:
    """Extract the per-dataset mean differences and the model names from ``data``.

    Returns the mean difference of each dataset (``left - right``, sign-flipped
    when minimising) and the two model names.
    """
    if dataset_col not in data.columns:
        raise ValueError(f"dataset_col '{dataset_col}' not found in the data columns.")
    alg_cols = [c for c in data.columns if c != dataset_col]
    if len(alg_cols) != 2:
        raise ValueError(
            "SignedRankTest expects exactly two algorithm columns besides "
            f"'{dataset_col}'; got {len(alg_cols)}: {alg_cols}."
        )
    left, right = (str(c) for c in alg_cols)
    sign = 1.0 if maximize else -1.0
    diff = sign * (data[alg_cols[0]] - data[alg_cols[1]])
    z = diff.groupby(data[dataset_col]).mean().to_numpy(dtype=float)
    return z, left, right
