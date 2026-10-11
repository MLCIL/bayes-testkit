"""
Unit tests for the shared estimator infrastructure.

This module contains unit tests for the components in ``btk.tests.common``
shared by every test estimator: string parameter validation, the highest density
interval helper, ROPE resolution, the decision rules, the simplex region
probabilities, the ArviZ wrapping, and the ``BaseBayesianTest`` lifecycle contract.
"""

import numpy as np
import pytest

from btk.tests.common import (
    BaseBayesianTest,
    decision_from_partition,
    hdi_from_samples,
    region_probabilities,
    resolve_rope,
    samples_to_inference_data,
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


class TestResolveRope:
    """Test resolve_rope ROPE band resolution."""

    def test_scalar_becomes_symmetric_band(self):
        """Test that a scalar half-width is expanded into a symmetric band."""
        assert resolve_rope(0.01) == (-0.01, 0.01)

    def test_zero_scalar_is_point(self):
        """Test that a zero half-width gives a degenerate band at zero."""
        assert resolve_rope(0.0) == (0.0, 0.0)

    def test_tuple_used_verbatim(self):
        """Test that an explicit band is used unchanged."""
        assert resolve_rope((-0.02, 0.03)) == (-0.02, 0.03)

    def test_negative_scalar_raises(self):
        """Test that a negative half-width raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            resolve_rope(-0.01)

    def test_inverted_tuple_raises(self):
        """Test that a band with lo >= hi raises ValueError."""
        with pytest.raises(ValueError, match="lo < hi"):
            resolve_rope((0.05, 0.01))


class TestDecisionFromPartition:
    """Test decision_from_partition threshold decision rule."""

    def test_left_model_better_when_p_right_high(self):
        """Test that a high p_right declares the left model better."""
        decision, raw = decision_from_partition(0.01, 0.02, 0.97, "A", "B", 0.95)
        assert decision == "A better"
        assert raw == ">"

    def test_right_model_better_when_p_left_high(self):
        """Test that a high p_left declares the right model better."""
        decision, raw = decision_from_partition(0.97, 0.02, 0.01, "A", "B", 0.95)
        assert decision == "B better"
        assert raw == "<"

    def test_equivalent_when_rope_high(self):
        """Test that a high p_rope declares the models equivalent."""
        decision, raw = decision_from_partition(0.01, 0.97, 0.02, "A", "B", 0.95)
        assert decision == "Equivalent"
        assert raw == "="

    def test_undetermined_when_none_exceed_threshold(self):
        """Test that no region exceeding the threshold gives an unknown decision."""
        decision, raw = decision_from_partition(0.4, 0.3, 0.3, "A", "B", 0.95)
        assert decision == "Unknown"
        assert raw == "?"


class TestRegionProbabilities:
    """Test region_probabilities simplex region counting."""

    def test_counts_draws_by_largest_component(self):
        """Test that each draw is assigned to the region of its largest component."""
        theta_l = np.array([0.6, 0.2, 0.1, 0.5])
        theta_e = np.array([0.3, 0.7, 0.2, 0.4])
        theta_r = np.array([0.1, 0.1, 0.7, 0.1])
        assert region_probabilities(theta_l, theta_e, theta_r) == (0.5, 0.25, 0.25)


class TestSamplesToInferenceData:
    """Test samples_to_inference_data ArviZ wrapping."""

    def test_single_chain_shape(self):
        """Test that samples are wrapped as a single chain of draws."""
        idata = samples_to_inference_data({"mu": np.arange(50.0)})
        assert idata.posterior["mu"].shape == (1, 50)


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
