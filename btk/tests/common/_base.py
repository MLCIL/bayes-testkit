"""Shared lifecycle contract for the Bayesian test estimators.

This formalises the conventions established by
:class:`~btk.tests.bbt.bbt.BBTTest` so that every estimator -- BBT and its
t-test siblings alike -- inherits the same outer shape: hyperparameters live in
``__init__``, ``fit`` returns ``self`` and sets the fitted flag, report methods
are guarded by ``_check_if_fitted`` and read off a stored
:class:`arviz.InferenceData`.

The base class is intentionally thin. It provides only what is genuinely
shared -- the fitted guard, the ``fitted`` property, scikit-learn-style
``get_params``/``set_params`` introspection, the public ``idata_`` endpoint for
the wider ArviZ toolchain, and the convergence diagnostics that every sampled
posterior has to be checked against. It deliberately says nothing about
estimands, ROPE scales, decision formats or plots, because those cannot be
shared faithfully across the tests.
"""

from __future__ import annotations

import warnings
from inspect import signature
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import arviz as az


class BaseBayesianTest:
    """Mixin providing the shared lifecycle for Bayesian test estimators.

    Sub-classes are expected to:

    - accept all hyperparameters in ``__init__`` and store each one as
      ``self._<param_name>`` (this is what makes :meth:`get_params` generic);
    - set ``self._fitted = True`` and store an :class:`arviz.InferenceData` on
      ``self._idata`` inside ``fit``;
    - return ``self`` from ``fit``.
    """

    _fitted: bool = False
    _idata: az.InferenceData | None = None

    #: Name used in the diagnostics warning, so the message reads naturally.
    _diagnostics_label: str = ""

    def _check_if_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("The model must be fitted before accessing this method.")

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._fitted

    @classmethod
    def _param_names(cls) -> list[str]:
        """Return constructor parameter names, sklearn-style (no ``self``/``**kwargs``)."""
        params = signature(cls.__init__).parameters.values()
        return [
            p.name
            for p in params
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]

    def get_params(self) -> dict[str, object]:
        """Get the hyperparameters of this estimator.

        Mirrors the scikit-learn convention. Each constructor parameter is read
        back from its ``self._<name>`` attribute, so this works for any
        sub-class that follows the storage convention above without bespoke
        plumbing. This is what lets a CLI set any hyperparameter generically.

        Returns
        -------
        dict[str, object]
            Mapping from parameter name to its currently stored value.
        """
        return {name: getattr(self, f"_{name}") for name in self._param_names()}

    def set_params(self, **params) -> BaseBayesianTest:
        """Set the hyperparameters of this estimator.

        Parameters
        ----------
        **params
            Hyperparameters to update, by constructor parameter name.

        Returns
        -------
        self
            The estimator instance, for chaining.
        """
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter '{name}' for estimator "
                    f"{type(self).__name__}. Valid parameters are {sorted(valid)}."
                )
            setattr(self, f"_{name}", value)
        return self

    @property
    def idata_(self) -> az.InferenceData:
        """The fitted posterior as an :class:`arviz.InferenceData`.

        This is the shared endpoint into the wider Bayesian toolchain: every
        estimator exposes the same attribute, so ``az.summary(test.idata_)``,
        ``az.plot_posterior``, ``az.plot_trace`` and the convergence
        diagnostics all work uniformly regardless of which test produced it.
        """
        self._check_if_fitted()
        # ``fit`` always sets ``_idata`` before flipping the fitted flag.
        return cast("az.InferenceData", self._idata)

    def to_inference_data(self) -> az.InferenceData:
        """Return the fitted posterior as an :class:`arviz.InferenceData`.

        Alias for the :attr:`idata_` attribute, provided for callers who prefer
        a method to a property.
        """
        return self.idata_

    # -- convergence diagnostics -------------------------------------------

    def _diagnostic_vars(self) -> list[str] | None:
        """Parameters whose R-hat and ESS :meth:`diagnostics` summarises.

        ``None`` means every variable in the posterior group. Sub-classes
        override this when only some of the sampled variables are the model's
        own parameters, or when the set depends on the hyperparameters.
        """
        return None

    def diagnostics(self) -> dict[str, float]:
        """Convergence diagnostics for the fitted chains.

        Wainer (2023), sec. 2.2: "It is important to run convergence diagnostics
        every time a Bayesian model is run, as they provide information on whether the
        samples generated by the algorithm are representative of the posterior
        distribution of the parameters, or if more steps of the MCMC algorithm need to
        be run."

        Returns
        -------
        dict[str, float]
            ``divergences`` (post-tuning divergent transitions), ``max_r_hat`` (largest
            split R-hat over the model parameters; should be below 1.01) and
            ``min_ess_bulk`` (smallest bulk effective sample size; a few hundred per
            chain is the usual bar).
        """
        import arviz as az

        self._check_if_fitted()
        idata = self.idata_
        summary = az.summary(idata, var_names=self._diagnostic_vars())

        divergences = 0
        sample_stats = getattr(idata, "sample_stats", None)
        if sample_stats is not None and "diverging" in sample_stats:
            divergences = int(sample_stats["diverging"].to_numpy().sum())

        return {
            "divergences": divergences,
            "max_r_hat": float(summary["r_hat"].max()),
            "min_ess_bulk": float(summary["ess_bulk"].min()),
        }

    def _warn_on_bad_diagnostics(self, hint: str = "") -> None:
        """Surface divergences and poor mixing, rather than leaving them in ``idata_``."""
        diag = self.diagnostics()
        problems = []
        if diag["divergences"] > 0:
            problems.append(f"  - {diag['divergences']} divergent transitions")
        if diag["max_r_hat"] > 1.01:
            problems.append(f"  - max R-hat {diag['max_r_hat']:.3f}, want below 1.01")
        if not problems:
            return
        label = self._diagnostics_label or type(self).__name__
        lines = [
            f"\n{label} sampling diagnostics look unreliable:",
            *problems,
            "",
            "The draws may not describe the posterior, so read the results with care.",
            "Fix: use more draws, or raise target_accept (e.g. 0.95 or 0.99).",
        ]
        if hint:
            lines.append(hint)
        lines.append(f"Numbers: {type(self).__name__}.diagnostics()")
        warnings.warn("\n".join(lines) + "\n", UserWarning, stacklevel=3)
