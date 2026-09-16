"""
Unit tests for the shared t-test numerical helpers.

This module contains unit tests for the helpers in ``bbttest.tests.ttest._stats``
used by both t-test estimators: ROPE and correlation resolution, cross-validation
fold structure derivation, the decision rules, and the reporting helpers.
"""

import numpy as np
import pandas as pd
import pytest

from bbttest.tests.ttest._stats import (
    SummaryResult,
    decision_from_partition,
    resolve_fold_structure,
    resolve_rho,
    resolve_rope,
    samples_to_inference_data,
)


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


class TestResolveRho:
    """Test resolve_rho correlation resolution."""

    def test_explicit_rho_overrides_folds(self):
        """Test that an explicitly provided rho takes precedence over the fold count."""
        assert resolve_rho(0.2, n_folds=10, n=100) == 0.2

    def test_from_n_folds(self):
        """Test that rho is derived as 1 / n_folds when the fold count is known."""
        assert resolve_rho(None, n_folds=10, n=100) == pytest.approx(0.1)

    def test_falls_back_to_one_over_n(self):
        """Test that without a fold count each row is treated as a separate fold."""
        assert resolve_rho(None, n_folds=None, n=20) == pytest.approx(0.05)

    def test_out_of_range_raises(self):
        """Test that a resolved rho outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="outside"):
            resolve_rho(1.5, n_folds=None, n=10)


class TestResolveFoldStructure:
    """Test resolve_fold_structure cross-validation design derivation."""

    def test_repeated_runs_of_k_folds(self):
        """Test that repeated fold identifiers are read as runs of the same folds."""
        folds = pd.Series([1, 2, 3, 4, 5] * 3)
        assert resolve_fold_structure(folds, "fold") == (5, 3)

    def test_single_run(self):
        """Test that distinct fold identifiers are read as a single run."""
        folds = pd.Series(["a", "b", "c", "d"])
        assert resolve_fold_structure(folds, "fold") == (4, 1)

    def test_unbalanced_design_raises(self):
        """Test that folds repeated a differing number of times raise ValueError."""
        folds = pd.Series([1, 1, 2, 2, 3])  # fold 3 has only one run
        with pytest.raises(ValueError, match="Unbalanced cross-validation design"):
            resolve_fold_structure(folds, "fold")


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


class TestSamplesToInferenceData:
    """Test samples_to_inference_data ArviZ wrapping."""

    def test_single_chain_shape(self):
        """Test that samples are wrapped as a single chain of draws."""
        idata = samples_to_inference_data({"mu": np.arange(50.0)})
        assert idata.posterior["mu"].shape == (1, 50)


class TestSummaryResult:
    """Test SummaryResult rendering."""

    def test_str_contains_key_fields(self):
        """Test that the rendered summary contains the title, estimand and decision."""
        s = SummaryResult(
            title="t-test: A vs B",
            estimand="mu",
            estimate=-0.019,
            hdi=(-0.03, -0.01),
            hdi_prob=0.89,
            rope=(-0.01, 0.01),
            probabilities={"p_left": 0.95, "p_rope": 0.05, "p_right": 0.0},
            decision="B better",
        )
        text = str(s)
        assert "t-test: A vs B" in text
        assert "B better" in text
        assert "mu" in text
