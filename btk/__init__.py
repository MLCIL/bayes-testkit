"""Bayes Test Kit: Bayesian model comparison tests for machine learning experiments."""

from ._pytensor_compat import (
    configure_pytensor_macos_linker as _configure_pytensor_macos_linker,
)

_configure_pytensor_macos_linker()

from .tests import (  # noqa: E402
    BBTTest,
    CorrelatedTTest,
    HierarchicalTTest,
    SignedRankTest,
)

__all__ = [
    "BBTTest",
    "CorrelatedTTest",
    "HierarchicalTTest",
    "SignedRankTest",
]
