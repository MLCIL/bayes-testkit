"""
Unit tests for the shared estimator infrastructure.

This module contains unit tests for the components in ``bbttest.tests.common``
shared by every test estimator: string parameter validation, the highest density
interval helper, and the ``BaseBayesianTest`` lifecycle contract.
"""

import numpy as np
import pytest

from bbttest.tests.common import (
    BaseBayesianTest,
    hdi_from_samples,
    validate_string,
)


class TestValidateString:
    """Test validate_string parameter validation."""

    def test_accepts_allowed_value(self):
        """Test that an allowed value is returned unchanged."""
        assert validate_string("a", ("a", "b"), "param") == "a"

    def test_rejects_unknown_value(self):
        """Test that a value outside the allowed set raises ValueError."""
        with pytest.raises(ValueError, match="Invalid value 'c' for parameter 'param'"):
            validate_string("c", ("a", "b"), "param")


class TestHdiFromSamples:
    """Test hdi_from_samples highest density interval computation."""

    def test_covers_target_mass(self):
        """Test that the interval contains approximately the requested probability mass."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100_000)
        low, high = hdi_from_samples(x, 0.9)
        assert np.mean((x >= low) & (x <= high)) == pytest.approx(0.9, abs=0.02)


class _DummyTest(BaseBayesianTest):
    """Minimal estimator exercising the shared lifecycle contract."""

    def __init__(self, alpha: float = 1.0, beta: str = "x"):
        self._alpha = alpha
        self._beta = beta
        self._fitted = False

    def fit(self):
        """Fit the dummy estimator, storing a stand-in for the posterior."""
        self._idata = "posterior"
        self._fitted = True
        return self


class TestBaseLifecycle:
    """Test the BaseBayesianTest lifecycle contract."""

    def test_fitted_flag_and_guard(self):
        """Test that the fitted guard raises before fitting and passes afterwards."""
        model = _DummyTest()
        assert not model.fitted
        with pytest.raises(RuntimeError, match="must be fitted"):
            model._check_if_fitted()
        model.fit()
        assert model.fitted
        model._check_if_fitted()  # no raise

    def test_get_params_reads_stored_attributes(self):
        """Test that get_params reads the constructor parameters back generically."""
        model = _DummyTest(alpha=2.0, beta="y")
        assert model.get_params() == {"alpha": 2.0, "beta": "y"}

    def test_set_params_updates_and_rejects_unknown(self):
        """Test that set_params updates known parameters and rejects unknown ones."""
        model = _DummyTest()
        model.set_params(alpha=3.0)
        assert model._alpha == 3.0
        with pytest.raises(ValueError, match="Invalid parameter 'gamma'"):
            model.set_params(gamma=1)

    def test_idata_guarded_then_available(self):
        """Test that idata_ raises on an unfitted model and is exposed after fitting."""
        model = _DummyTest()
        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = model.idata_
        model.fit()
        assert model.idata_ == "posterior"
        assert model.to_inference_data() == "posterior"
