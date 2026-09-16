import logging as log
import warnings
from collections.abc import Generator, Iterable

import arviz as az
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bbttest.tests.common import hdi_from_samples

ALG1_COL = 2
ALG2_COL = 3
TIE_COL = 4

logger = log.getLogger(__name__)


UNNAMED_COLUMNS_WARNING_TEMPLATE = """Some algorithm names are unnamed. This may lead to issues in the win table construction.
Algorithm names extracted: {algorithms_names}
Dataset column: {dataset_col}
"""


def _gen_pairs(no_algs: int) -> Generator[tuple[int, int, int], None, None]:
    k = 0
    for i in range(no_algs):
        for j in range(i + 1, no_algs):
            yield (i, j, k)
            k += 1


def _construct_no_paired(
    data_mean: pd.DataFrame,
    alg_names: list[str],
    effect_size: float,
    data_sd: pd.DataFrame | None,
    absolute_threshold: float | None,
) -> np.ndarray:
    """Build the win table from one measure per algorithm per data set.

    Two tie rules are possible here, and they live on different scales:

    - with ``data_sd``, the local ROPE of Wainer (2023), Eq. (4), an *effect
      size* threshold ``effect_size * sqrt((s_i^2 + s_j^2) / 2)``;
    - without it, an *absolute* threshold in the units of the metric. This is
      not from the paper; it exists for fixed-split designs, where there is no
      within-data-set spread to form an effect size from and a difference of
      1e-4 would otherwise count as a win.
    """
    logger.debug("Using unpaired BBT test.")
    if data_sd is not None:
        logger.debug(
            "Tie rule: local ROPE effect size %s scaled by the pooled sd.", effect_size
        )
    elif absolute_threshold is not None:
        logger.debug(
            "Tie rule: absolute threshold %s in metric units.", absolute_threshold
        )
    else:
        logger.debug("No tie rule provided, no ties will be recorded.")
    no_algs = len(alg_names)
    no_pairs = no_algs * (no_algs - 1) // 2
    out_array = -1 * np.ones(
        (no_pairs, 5),  # alg_1, alg_2, 1_wins, 2_wins, ties
        dtype=np.int32,
    )
    for i, j, k in tqdm(
        _gen_pairs(no_algs),
        total=no_pairs,
        desc="Constructing win table",
        leave=False,
    ):
        i_name = alg_names[i]
        j_name = alg_names[j]
        deltas = (data_mean[i_name] - data_mean[j_name]).to_numpy(dtype=float)
        if data_sd is not None:
            # Eq. (3)-(4): Cohen's d pools by the *average* of the two
            # variances, so the threshold is d * sqrt((s_i^2 + s_j^2) / 2).
            th = effect_size * np.sqrt(
                (
                    np.power(data_sd[i_name].to_numpy(dtype=float), 2)
                    + np.power(data_sd[j_name].to_numpy(dtype=float), 2)
                )
                / 2.0
            )
        elif absolute_threshold is not None:
            th = absolute_threshold
        else:
            th = 0.0

        # A missing measure is neither a win nor a loss (Wainer 2023, sec. 5.7):
        # the pair simply does not play that match, so it must not fall through
        # into the tie count either.
        valid = np.isfinite(deltas)
        if isinstance(th, np.ndarray):
            valid &= np.isfinite(th)
            th = th[valid]
        deltas = deltas[valid]

        w1 = int(np.sum(deltas > th))
        w2 = int(np.sum(deltas < -th))
        ties = int(deltas.shape[0]) - w1 - w2
        out_array[k, :] = i, j, w1, w2, ties
    return out_array


def _construct_lrope(
    data: pd.DataFrame,
    alg_names: list[str],
    dataset_col: str | int,
    effect_size: float,
) -> np.ndarray:
    """Build the win table from repeated measures per algorithm per data set.

    Uses the paired local ROPE of Wainer (2023), Eq. (6): a data set is a win
    only when the mean of the per-fold differences exceeds ``effect_size``
    standard deviations of those differences.
    """
    logger.debug("Using paired BBT test.")
    no_algs = len(alg_names)
    no_pairs = no_algs * (no_algs - 1) // 2
    out_array = np.zeros(
        (no_pairs, 5),  # alg_1, alg_2, 1_wins, 2_wins, ties
        dtype=np.int32,
    )
    dataset_names = data[dataset_col].unique()
    for dataset_name in tqdm(
        dataset_names,
        total=len(dataset_names),
        desc="Constructing local ROPE win table",
        leave=False,
    ):
        data_subset = data[data[dataset_col] == dataset_name]
        for i, j, k in _gen_pairs(no_algs):
            i_name = alg_names[i]
            j_name = alg_names[j]
            out_array[k, 0] = i
            out_array[k, 1] = j

            deltas = (data_subset[i_name] - data_subset[j_name]).to_numpy(dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            if deltas.size == 0:
                # At least one of the two did not run on this data set: it is
                # neither a win, nor a loss, nor a tie (Wainer 2023, sec. 5.7).
                continue

            mean = float(np.mean(deltas))
            # Eq. (6): the paired effect size scales by the standard deviation
            # of the per-fold differences. A single fold leaves no spread to
            # scale by, so the comparison falls back to the sign of the mean.
            sd = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0

            win1 = int(mean > effect_size * sd)
            win2 = int(mean < -effect_size * sd)
            out_array[k, ALG1_COL] += win1
            out_array[k, ALG2_COL] += win2
            out_array[k, TIE_COL] += 1 - win1 - win2
    return out_array


def _solve_ties(table: np.ndarray, tie_solver: str) -> np.ndarray:
    if tie_solver == "davidson":
        return table
    if tie_solver == "spread":
        tie_val = np.ceil(table[:, TIE_COL] / 2).astype(int)
    elif tie_solver == "add":
        tie_val = table[:, TIE_COL]
    else:
        tie_val = 0
    table[:, ALG1_COL] += tie_val
    table[:, ALG2_COL] += tie_val
    return table


def _construct_win_table(
    data: pd.DataFrame,
    data_sd: pd.DataFrame | None,
    dataset_col: str | int | None,
    tie_solver: str,
    maximize: bool,
    local_rope_effect_size: float | None = None,
    absolute_tie_threshold: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Turn a results table into the win/loss/tie table the BBT model observes.

    Which tie rule applies is decided by the shape of the data, so the two
    thresholds are passed separately rather than sharing one number: they are
    measured on different scales and are not interchangeable.

    ==========================================  ==========================
    Input                                       Tie rule
    ==========================================  ==========================
    repeated rows per data set                  ``local_rope_effect_size``
    one row per data set, ``data_sd`` given     ``local_rope_effect_size``
    one row per data set, no ``data_sd``        ``absolute_tie_threshold``
    ==========================================  ==========================
    """
    # Extract algorithm names
    algorithms_names = data.columns.tolist()
    if isinstance(dataset_col, int):
        dataset_col = data.columns[dataset_col]
    if dataset_col is not None:
        algorithms_names.remove(dataset_col)

    data = data.copy()
    if not maximize:
        data.loc[:, algorithms_names] = -1 * data[algorithms_names]

    if any("Unnamed" in col for col in algorithms_names):
        warnings.warn(
            UNNAMED_COLUMNS_WARNING_TEMPLATE.format(
                algorithms_names=algorithms_names,
                dataset_col=dataset_col,
            ),
            UserWarning,
        )

    if dataset_col is None or data.shape[0] == data[dataset_col].nunique():
        table = _construct_no_paired(
            data_mean=data,
            effect_size=local_rope_effect_size or 0.0,
            data_sd=data_sd,
            alg_names=algorithms_names,
            absolute_threshold=absolute_tie_threshold,
        )
    else:
        table = _construct_lrope(
            data=data,
            effect_size=local_rope_effect_size or 0.0,
            dataset_col=dataset_col,
            alg_names=algorithms_names,
        )
    table = _solve_ties(
        table=table,
        tie_solver=tie_solver,
    )
    return table, algorithms_names


def _check_model_names(
    known: set[str],
    control: str | None,
    selected: Iterable[str] | None,
) -> None:
    """Reject unknown model names instead of quietly answering another question.

    A misspelled ``control`` used to fall through to the full all-pairs table,
    and a misspelled entry in ``selected`` surfaced as an error from pandas.
    """
    if control is not None and control not in known:
        raise ValueError(
            f"Unknown control_model {control!r}. Available models: {sorted(known)}."
        )
    if selected is not None:
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError(
                f"Unknown selected_models {unknown}. Available models: {sorted(known)}."
            )


def _get_pwin(
    bbt_result: az.InferenceData,
    alg_names: Iterable[str] | None = None,
    control: str | None = None,
    selected: Iterable[str] | None = None,
):
    def _pairwise_prob(strength_i, strength_j):
        return strength_i / (strength_i + strength_j)

    # Extract beta samples from InferenceData
    # PyMC stores samples in idata.posterior
    beta_samples = bbt_result.posterior["beta"].to_numpy()
    # Flatten chain and draw dimensions: (chains, draws, n_algs) -> (samples, n_algs)
    beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])

    n_algs = beta_samples.shape[1]
    # Order algorithms by mean strength (descending)
    mean_beta = np.mean(beta_samples, axis=0)
    order = np.argsort(-mean_beta)
    ordered_names = np.array(alg_names)[order]

    _check_model_names(set(ordered_names.tolist()), control, selected)

    # Exponentiate to get strengths (exp(beta))
    strengths = np.exp(beta_samples[:, order])

    # Filter by selected algorithms if specified
    if selected is not None:
        selected_set = set(selected)
        if control not in selected_set and control is not None:
            selected_set.add(control)

        indices = [i for i, name in enumerate(ordered_names) if name in selected_set]
        ordered_names = ordered_names[indices]
        strengths = strengths[:, indices]
        n_algs = len(indices)
        if n_algs < 2:
            raise ValueError(
                "At least two models are needed for a comparison; "
                f"selected_models resolved to {sorted(selected_set)}."
            )

    comparison_names = []

    # Generate comparisons
    if control is None:
        # All pairwise comparisons
        n_comparisons = n_algs * (n_algs - 1) // 2
        samples = np.empty((strengths.shape[0], n_comparisons))

        for i, j, k in _gen_pairs(n_algs):
            samples[:, k] = _pairwise_prob(strengths[:, i], strengths[:, j])
            comparison_names.append(f"{ordered_names[i]} > {ordered_names[j]}")
    else:
        # Comparisons with control algorithm
        control_idx = np.where(ordered_names == control)[0][0]
        n_comparisons = n_algs - 1
        samples = np.empty((strengths.shape[0], n_comparisons))

        k = 0
        # Comparisons where other algorithm is better than control
        for i in range(control_idx):
            samples[:, k] = _pairwise_prob(strengths[:, i], strengths[:, control_idx])
            comparison_names.append(
                f"{ordered_names[i]} > {ordered_names[control_idx]}"
            )
            k += 1

        # Comparisons where control is better than other algorithm
        for i in range(control_idx + 1, n_algs):
            samples[:, k] = _pairwise_prob(strengths[:, control_idx], strengths[:, i])
            comparison_names.append(
                f"{ordered_names[control_idx]} > {ordered_names[i]}"
            )
            k += 1

    return samples, comparison_names


def _hdi(samples: np.ndarray, hdi_prob: float = 0.89) -> np.ndarray:
    """Column-wise highest-density interval (see ``common.hdi_from_samples``)."""
    return np.apply_along_axis(
        lambda arr: np.array(hdi_from_samples(arr, hdi_prob)), 0, samples
    )
