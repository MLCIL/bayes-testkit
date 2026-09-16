import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._utils import _get_distribution_for_prior


def _build_bbt_model(
    player1: list[int],
    player2: list[int],
    win1: list[int],
    win2: list[int],
    ties: list[int] | None,
    hyp: str,
    scale: float,
    use_davidson: bool,
):
    """
    Build a PyMC version of the Bradley-Terry-Thurstone model.

    Args:
        player1: List of player 1 indices for each matchup.
        player2: List of player 2 indices for each matchup.
        win1: List of wins for player 1 in each matchup.
        win2: List of wins for player 2 in each matchup.
        ties: List of ties in each matchup. Required if use_davidson is True.
        hyp: The hyperprior for the sigma parameter.
        scale: The scale for the hyperprior.
        use_davidson: Whether to use the Davidson extension for ties.

    Returns
    -------
        A PyMC model object.
    """
    if use_davidson and ties is None:
        raise ValueError("'ties' must be provided when use_davidson is True.")

    # Convert to numpy arrays for easier manipulation
    p1_idx = np.array(player1, dtype=int)
    p2_idx = np.array(player2, dtype=int)
    w1 = np.array(win1, dtype=int)
    w2 = np.array(win2, dtype=int)

    # Find the number of unique players
    K = len(np.unique(np.concatenate((p1_idx, p2_idx))))

    # Transformed data
    n = w1 + w2
    if use_davidson:
        ties_arr = np.array(ties, dtype=int)
        nn = n + ties_arr

    with pm.Model() as model:
        # Hyperprior for sigma
        sigma = _get_distribution_for_prior(hyp, scale=scale)

        # Abilities for each player
        beta = pm.Normal("beta", mu=0, sigma=sigma, shape=K)

        # Player strengths
        w = pt.exp(beta)

        # Probabilities
        w_p1 = w[p1_idx]
        w_p2 = w[p2_idx]

        if use_davidson:
            sigmanu = pm.Lognormal("sigmanu", mu=0, sigma=0.5)
            nu = pm.Normal("nu", mu=0, sigma=sigmanu)

            # Davidson extension for ties
            aux = pt.exp(nu + (beta[p1_idx] + beta[p2_idx]) / 2.0)
            denominator = w_p1 + w_p2 + aux

            pwin = pm.Deterministic("pwin", w_p1 / denominator)
            ptie = pm.Deterministic("ptie", aux / denominator)

            # Likelihood. Both ``pwin`` and ``ptie`` are normalised over all
            # three outcomes (Wainer 2023, Eq. 7), so both are marginals of the
            # same Multinomial(nn, (pwin, plose, ptie)) and must use the *total*
            # number of matches ``nn``. Using ``n = w1 + w2`` here would pair an
            # unconditional probability with a total that excludes ties.
            pm.Binomial("win1_obs", n=nn, p=pwin, observed=w1)
            pm.Binomial("ties_obs", n=nn, p=ptie, observed=ties_arr)

        else:
            denominator = w_p1 + w_p2
            pwin = pm.Deterministic("pwin", w_p1 / denominator)

            # Likelihood
            pm.Binomial("win1_obs", n=n, p=pwin, observed=w1)

    # The posterior predictive check (Wainer 2023, sec. 5.4) replays the
    # *observed* variables through ``pm.sample_posterior_predictive``. Declaring
    # extra unobserved Binomial "replicate" variables here instead would leave
    # discrete free variables in the model, which forces ``pm.sample`` into a
    # NUTS + Metropolis compound step and drags those draws into every
    # convergence summary the user sees.
    return model


def _mcmcbbt_pymc(
    table: np.ndarray,
    use_davidson: bool,
    hyper_prior: str,
    scale: float,
    **kwargs,
) -> tuple[az.InferenceData, pm.Model]:
    """Build and sample the BBT model.

    Returns both the draws and the model, because the model is needed again for
    the posterior predictive check.
    """
    player1 = table[:, 0].tolist()
    player2 = table[:, 1].tolist()
    win1 = table[:, 2].tolist()
    win2 = table[:, 3].tolist()
    ties = table[:, 4].tolist() if use_davidson else None

    # Build the model
    model = _build_bbt_model(
        player1=player1,
        player2=player2,
        win1=win1,
        win2=win2,
        ties=ties,
        hyp=hyper_prior,
        scale=scale,
        use_davidson=use_davidson,
    )

    # Keep the pointwise log-likelihood so that az.waic / az.loo work on the
    # result -- Wainer (2023) sec. 6.2 compares the tie policies by WAIC.
    idata_kwargs = {"log_likelihood": True, **kwargs.pop("idata_kwargs", {})}

    # Sample from the model
    with model:
        fit = pm.sample(idata_kwargs=idata_kwargs, **kwargs)

    return fit, model
