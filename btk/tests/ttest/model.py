"""PyMC implementation of the Bayesian hierarchical correlated t-test.

Implements the hierarchical model of Benavoli et al. (2017), Eq. (12)-(14),
with the degrees-of-freedom priors of Section 4.3.1 (Corani et al. 2017). The
paper uses Stan; this is the PyMC translation, following the package's existing
``model.py`` conventions (kwargs filtered before reaching ``pm.sample``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    import arviz as az

# Hyper-prior bounds for the ``hierarchical`` degrees-of-freedom scheme, where
# nu ~ Gamma(alpha, beta) with uniform hyper-priors on the shape/rate (the
# recommended default; Corani et al. 2017).
_ALPHA_BOUNDS = (0.5, 5.0)
_BETA_BOUNDS = (0.05, 0.15)

# Fixed Gamma(shape, rate) parameters for the alternative dof priors.
_DOF_GAMMA_PARAMS = {
    "kruschke": (1.0, 0.0345),  # mean ~29, Kruschke (2013)
    "juarez_steel": (2.0, 0.1),  # mean 20, Juarez & Steel (2010)
}


def _correlation_matrix(n: int, rho: float) -> np.ndarray:
    """Compound-symmetry correlation matrix ``(1 - rho) I + rho J`` (Eq. 12)."""
    return (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))


def _build_hierarchical_model(
    x: np.ndarray,
    rho: float,
    dof_prior: str,
    mu0_bound: float,
    sigma_0_upper: float,
    sigma_i_upper: float,
) -> pm.Model:
    """Build the PyMC hierarchical correlated t-test model.

    Parameters
    ----------
    x : np.ndarray
        A ``(q, n)`` matrix of per-fold differences: ``q`` datasets, each with
        the same ``n`` scores (``left - right``).
    rho : float
        Correlation due to overlapping training sets.
    dof_prior : str
        Which degrees-of-freedom prior to use (``hierarchical``, ``kruschke``
        or ``juarez_steel``).
    mu0_bound : float
        Half-width of the uniform prior on the population mean ``mu_0``
        (Eq. 18; the paper uses 1).
    sigma_0_upper : float
        Upper bound of the uniform prior on ``sigma_0`` (Eq. 19; the paper uses
        ``1000 * sd(x_bar_i)``).
    sigma_i_upper : float
        Upper bound of the uniform prior on the per-dataset ``sigma_i``
        (Eq. 14; the paper uses ``1000 * mean_i(sigma_hat_i)``).

    Returns
    -------
    pm.Model
        The unsampled PyMC model.
    """
    q, n = x.shape
    corr = _correlation_matrix(n, rho)

    with pm.Model() as model:
        # Eq. (18)-(19) hyper-parameters: population mean and spread of mu_i.
        # The two standard deviations get their own bounds, as in the paper:
        # sigma_0 is scaled by the spread *between* datasets, sigma_i by the
        # average spread *within* them.
        mu_0 = pm.Uniform("mu_0", lower=-mu0_bound, upper=mu0_bound)
        sigma_0 = pm.Uniform("sigma_0", lower=0.0, upper=sigma_0_upper)

        # Degrees of freedom of the Student-t over the per-dataset means.
        if dof_prior == "hierarchical":
            alpha = pm.Uniform("alpha", *_ALPHA_BOUNDS)
            beta = pm.Uniform("beta", *_BETA_BOUNDS)
            nu = pm.Gamma("nu", alpha=alpha, beta=beta)
        else:
            g_alpha, g_beta = _DOF_GAMMA_PARAMS[dof_prior]
            nu = pm.Gamma("nu", alpha=g_alpha, beta=g_beta)

        # Eq. (13): per-dataset mean differences, shrunk towards mu_0.
        mu_i = pm.StudentT("mu_i", nu=nu, mu=mu_0, sigma=sigma_0, shape=q)

        # Eq. (14): per-dataset standard deviations.
        sigma_i = pm.Uniform("sigma_i", lower=0.0, upper=sigma_i_upper, shape=q)

        # Eq. (12): each dataset's fold vector is multivariate-normal with a
        # compound-symmetry covariance. Batched over the q datasets.
        cov = sigma_i[:, None, None] ** 2 * corr[None, :, :]
        mu_mat = mu_i[:, None] * pt.ones(n)
        pm.MvNormal("obs", mu=mu_mat, cov=cov, observed=x)

    return model


def _sample_hierarchical(
    x: np.ndarray,
    rho: float,
    dof_prior: str,
    mu0_bound: float,
    sigma_0_upper: float,
    sigma_i_upper: float,
    **kwargs,
) -> tuple[az.InferenceData, pm.Model]:
    """Build and sample the hierarchical model.

    Returns both the posterior and the model that produced it: the model is what
    ``pm.sample_posterior_predictive`` needs for the posterior predictive check.
    """
    model = _build_hierarchical_model(
        x=x,
        rho=rho,
        dof_prior=dof_prior,
        mu0_bound=mu0_bound,
        sigma_0_upper=sigma_0_upper,
        sigma_i_upper=sigma_i_upper,
    )
    idata_kwargs = {"log_likelihood": True, **kwargs.pop("idata_kwargs", {})}
    with model:
        return pm.sample(idata_kwargs=idata_kwargs, **kwargs), model
