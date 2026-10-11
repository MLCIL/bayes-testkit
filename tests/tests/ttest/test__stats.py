"""
Unit tests for the shared t-test numerical helpers.

This module contains unit tests for the helpers in ``btk.tests.ttest._stats``
used by both t-test estimators: correlation resolution, cross-validation fold
structure derivation, and the printable summary.
"""

import pandas as pd
import pytest

from btk.tests.ttest._stats import (
    SummaryResult,
    resolve_fold_structure,
    resolve_rho,
)


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
