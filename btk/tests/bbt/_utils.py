"""BBT-specific utilities."""

from pymc.distributions import HalfCauchy, HalfNormal, LogNormal


def _get_distribution_for_prior(prior: str, scale: float):
    """Build the hyper-prior for ``sigma``, the spread of the abilities.

    ``sigma`` is the scale of ``beta ~ Normal(0, sigma)`` and so must be
    positive. The alternatives to the log-normal of Wainer (2023), Eq. (2) are
    therefore the *half*-normal and half-Cauchy (sec. 4.1); an unbounded Normal
    or Cauchy puts mass on negative scales, for which the model's
    log-probability is ``-inf`` and sampling cannot even start.
    """
    match prior:
        case "log_normal":
            return LogNormal("sigma", mu=0, sigma=scale)
        case "cauchy":
            return HalfCauchy("sigma", beta=scale)
        case "normal":
            return HalfNormal("sigma", sigma=scale)
        case _:
            raise ValueError(f"Unsupported hyperprior: {prior}")
