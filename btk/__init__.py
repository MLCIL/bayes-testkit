"""Bayes Test Kit: Bayesian model comparison tests for machine learning experiments."""

from ._pytensor_compat import (
    configure_pytensor_macos_linker as _configure_pytensor_macos_linker,
)

_configure_pytensor_macos_linker()

from .tests import BBTTest, CorrelatedTTest, HierarchicalTTest  # noqa: E402

__all__ = [
    "BBTTest",
    "CorrelatedTTest",
    "HierarchicalTTest",
]
