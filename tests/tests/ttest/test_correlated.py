"""
Unit tests for CorrelatedTTest class.

This module contains unit tests for the CorrelatedTTest class, testing various
functionality including model fitting, fold structure resolution, decision table
generation, plotting, and parameter validation. The posterior is closed-form, so
these tests are fast and can assert exact numerical agreement with the values
reported in the paper.
"""

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bbttest import CorrelatedTTest


@pytest.fixture
def paper_example_diffs():
    """
    Create differences matching the paper's Eq. 6 example.

    Returns
    -------
    np.ndarray
        Score differences with sample mean -0.0194, sample sd 0.01583 and n = 100.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 100)
    base = (base - base.mean()) / base.std(ddof=1)
    return -0.0194 + 0.01583 * base


@pytest.fixture
def fitted_correlated(paper_example_diffs):
    """
    Create a fitted CorrelatedTTest model for testing.

    Parameters
    ----------
    paper_example_diffs : np.ndarray
        Score differences fixture.

    Returns
    -------
    CorrelatedTTest
        Fitted CorrelatedTTest model instance.
    """
    return CorrelatedTTest(rope=0.01).fit(paper_example_diffs, rho=0.1, random_seed=1)


class TestInitialization:
    """Test CorrelatedTTest initialization and parameter handling."""

    def test_defaults(self):
        """Test that default initialization values are set correctly."""
        t = CorrelatedTTest()
        assert t._rope == 0.01
        assert t._maximize
        assert not t.fitted

    def test_get_params_holds_only_hyperparameters(self):
        """Test that get_params exposes only the hyperparameters, not the data properties."""
        t = CorrelatedTTest(rope=0.02, maximize=False)
        assert t.get_params() == {"rope": 0.02, "maximize": False}
        t.set_params(rope=0.03)
        assert t._rope == 0.03

    def test_set_params_rejects_unknown(self):
        """Test that set_params raises ValueError for an unknown parameter."""
        with pytest.raises(ValueError, match="Invalid parameter"):
            CorrelatedTTest().set_params(nonsense=1)


class TestClosedForm:
    """Test the closed-form posterior against the values reported in the paper."""

    def test_scale_matches_paper(self):
        """Test that the Student posterior matches St(mu; 99, -0.0194, 0.000030)."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 100)
        base = (base - base.mean()) / base.std(ddof=1)
        t = CorrelatedTTest(rope=0.01).fit(-0.0194 + 0.01583 * base, rho=0.1)
        assert t._df == 99
        assert t._loc == pytest.approx(-0.0194, abs=1e-4)
        assert t._scale**2 == pytest.approx(0.000030, abs=2e-6)

    def test_probabilities_sum_to_one(self, fitted_correlated):
        """Test that the three ROPE probabilities partition the posterior."""
        row = fitted_correlated.decision_table(round_ndigits=None).iloc[0]
        assert row["p_left"] + row["p_rope"] + row["p_right"] == pytest.approx(1.0)

    def test_negative_mean_favours_right_model(self, fitted_correlated):
        """Test that a negative mean difference favours the right model."""
        row = fitted_correlated.decision_table(round_ndigits=None).iloc[0]
        assert row["p_left"] > row["p_right"]
        assert row["decision"] == "right better"


class TestFitInputs:
    """Test the accepted fit input formats and their validation."""

    def test_two_column_frame(self):
        """Test that a two-column frame is read as the left and right model scores."""
        df = pd.DataFrame({"a": [0.8, 0.82, 0.79, 0.81], "b": [0.7, 0.71, 0.72, 0.70]})
        t = CorrelatedTTest(rope=0.01).fit(df)
        assert t.comparison == "a vs b"
        assert t._xbar > 0  # a is better on average

    def test_precomputed_1d_differences(self):
        """Test that a 1-D array of precomputed differences is accepted."""
        t = CorrelatedTTest().fit(np.array([0.01, 0.02, 0.015, 0.005, 0.02]))
        assert t.fitted
        assert t._n == 5

    def test_maximize_false_flips_sign(self):
        """Test that maximize=False flips the sign for minimized metrics."""
        df = pd.DataFrame({"a": [0.1, 0.12, 0.11, 0.09], "b": [0.2, 0.22, 0.19, 0.21]})
        # a has lower error; with maximize=False, a should be favoured (mu > 0).
        t = CorrelatedTTest(rope=0.01, maximize=False).fit(df)
        assert t._xbar > 0

    def test_wrong_number_of_columns_raises(self):
        """Test that a frame with more than two model columns raises ValueError."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]})
        with pytest.raises(ValueError, match="exactly two algorithm columns"):
            CorrelatedTTest().fit(df)

    def test_too_few_scores_raises(self):
        """Test that fewer than two scores raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            CorrelatedTTest().fit(np.array([0.5]))


class TestFoldColumn:
    """Test that the correlation is derived from the fold structure of the data."""

    @staticmethod
    def _scores(n_folds, n_runs):
        """Build a scores frame with the given cross-validation design."""
        rng = np.random.default_rng(1)
        n = n_folds * n_runs
        return pd.DataFrame(
            {
                "fold": list(range(n_folds)) * n_runs,
                "a": rng.normal(0.85, 0.02, n),
                "b": rng.normal(0.83, 0.02, n),
            }
        )

    def test_rho_derived_from_fold_count(self):
        """Test that rho is derived as 1 / n_folds from the fold column."""
        t = CorrelatedTTest().fit(self._scores(10, 5), fold_col="fold")
        assert t._n_folds == 10
        assert t._n_runs == 5
        assert t._rho_used == pytest.approx(0.1)

    def test_fold_column_excluded_from_algorithms(self):
        """Test that the fold column is not treated as a model column."""
        t = CorrelatedTTest().fit(self._scores(4, 2), fold_col="fold")
        assert t.comparison == "a vs b"

    def test_explicit_rho_overrides_fold_column(self):
        """Test that an explicit rho takes precedence over the derived value."""
        t = CorrelatedTTest().fit(self._scores(10, 2), fold_col="fold", rho=0.25)
        assert t._n_folds == 10
        assert t._rho_used == pytest.approx(0.25)

    def test_without_fold_column_rho_is_one_over_n(self):
        """Test that without a fold column each row is treated as a separate fold."""
        t = CorrelatedTTest().fit(self._scores(5, 2).drop(columns="fold"))
        assert t._n_folds is None
        assert t._rho_used == pytest.approx(1 / 10)

    def test_unbalanced_folds_raise(self):
        """Test that an unbalanced cross-validation design raises ValueError."""
        df = pd.DataFrame(
            {"fold": [1, 1, 2, 3], "a": [0.8] * 4, "b": [0.7, 0.72, 0.71, 0.69]}
        )
        with pytest.raises(ValueError, match="Unbalanced cross-validation design"):
            CorrelatedTTest().fit(df, fold_col="fold")

    def test_missing_fold_column_raises(self):
        """Test that a fold_col absent from the data raises ValueError."""
        with pytest.raises(ValueError, match="fold_col 'nope' not found"):
            CorrelatedTTest().fit(self._scores(4, 2), fold_col="nope")

    def test_fold_col_with_array_input_raises(self):
        """Test that passing fold_col with a non-dataframe input raises ValueError."""
        with pytest.raises(ValueError, match="requires a DataFrame"):
            CorrelatedTTest().fit(np.array([0.1, 0.2, 0.3]), fold_col="fold")


class TestReports:
    """Test decision_table and summary method functionality."""

    def test_decision_table_has_required_columns(self, fitted_correlated):
        """Test that decision_table contains required columns."""
        cols = list(fitted_correlated.decision_table().columns)
        for c in ["comparison", "estimate", "hdi_low", "hdi_high", "decision"]:
            assert c in cols
        for c in ["p_left", "p_rope", "p_right"]:
            assert c in cols

    def test_decision_table_hdi_ordering(self, fitted_correlated):
        """Test that the point estimate lies within the reported HDI."""
        row = fitted_correlated.decision_table(round_ndigits=None).iloc[0]
        assert row["hdi_low"] <= row["estimate"] <= row["hdi_high"]

    def test_decision_table_invalid_column_raises(self, fitted_correlated):
        """Test that requesting an invalid column raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            fitted_correlated.decision_table(columns=["nonsense"])

    def test_threshold_controls_decision(self, paper_example_diffs):
        """Test that raising the threshold turns a borderline decision into Unknown."""
        t = CorrelatedTTest(rope=0.01).fit(paper_example_diffs, rho=0.1)
        strict = t.decision_table(threshold=0.999).iloc[0]["decision"]
        assert strict == "Unknown"

    def test_summary_is_printable(self, fitted_correlated):
        """Test that the summary renders as a string naming the test."""
        text = str(fitted_correlated.summary())
        assert "correlated t-test" in text.lower()

    def test_wider_rope_raises_p_rope(self, paper_example_diffs):
        """Test that widening the ROPE increases the equivalence probability."""
        narrow = CorrelatedTTest(rope=0.01).fit(paper_example_diffs, rho=0.1)
        wide = CorrelatedTTest(rope=0.05).fit(paper_example_diffs, rho=0.1)
        assert (
            wide.decision_table(round_ndigits=None).iloc[0]["p_rope"]
            > narrow.decision_table(round_ndigits=None).iloc[0]["p_rope"]
        )


class TestInferenceData:
    """Test the ArviZ InferenceData endpoint."""

    def test_idata_has_mu(self, fitted_correlated):
        """Test that the posterior exposes the mean difference variable."""
        assert "mu" in fitted_correlated.idata_.posterior.data_vars

    def test_idata_mean_matches_closed_form(self, fitted_correlated):
        """Test that the wrapped draws agree with the closed-form location."""
        mu = fitted_correlated.idata_.posterior["mu"].to_numpy()
        assert mu.mean() == pytest.approx(fitted_correlated._loc, abs=1e-3)

    def test_idata_before_fit_raises(self):
        """Test that idata_ raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = CorrelatedTTest().idata_


class TestPlots:
    """Test plot method functionality."""

    def test_posterior_plot_returns_axes(self, fitted_correlated):
        """Test that the posterior plot returns a matplotlib Axes."""
        ax = fitted_correlated.plot(kind="posterior")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_hdi_plot_returns_axes(self, fitted_correlated):
        """Test that the HDI plot returns a matplotlib Axes."""
        ax = fitted_correlated.plot(kind="hdi")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_invalid_kind_raises(self, fitted_correlated):
        """Test that an unsupported plot kind raises ValueError."""
        with pytest.raises(ValueError, match="Invalid value 'simplex'"):
            fitted_correlated.plot(kind="simplex")


class TestUnfittedGuards:
    """Test that methods raise errors when called on unfitted models."""

    def test_decision_table_requires_fit(self):
        """Test that decision_table() raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            CorrelatedTTest().decision_table()

    def test_summary_requires_fit(self):
        """Test that summary() raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            CorrelatedTTest().summary()


class TestDiagnosticsAreNotApplicable:
    """The closed-form posterior has no chain, so the diagnostics must not pretend."""

    def test_diagnostics_raises(self, fitted_correlated):
        """Inheriting the base implementation would report a meaningless NaN R-hat."""
        with pytest.raises(NotImplementedError, match="closed-form posterior"):
            fitted_correlated.diagnostics()
