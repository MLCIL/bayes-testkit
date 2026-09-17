"""
Unit tests for HierarchicalTTest class.

This module contains unit tests for the HierarchicalTTest class, testing various
functionality including model fitting, fold structure resolution, decision table
generation, per-dataset shrinkage, plotting, and parameter validation. The model
samples with PyMC, so the fitted model is a module-scoped fixture with a small
number of draws.
"""

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from btk import HierarchicalTTest

pytestmark = pytest.mark.filterwarnings(
    "ignore:HierarchicalTTest sampling diagnostics:UserWarning"
)


@pytest.fixture(scope="module")
def multi_dataset():
    """
    Create mock multi-dataset data with a clear signal for testing.

    Returns
    -------
    pd.DataFrame
        Mock 10-fold cross-validation scores over 8 datasets, where model_a is
        better than model_b by roughly 0.05.
    """
    rng = np.random.default_rng(7)
    rows = [
        {
            "dataset": f"ds{d}",
            "fold": fold,
            "model_a": 0.8 + effect + rng.normal(0, 0.02),
            "model_b": 0.8 + rng.normal(0, 0.02),
        }
        for d in range(8)
        for effect in [rng.normal(0.05, 0.01)]
        for fold in range(10)
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_hierarchical(multi_dataset):
    """
    Create a fitted HierarchicalTTest model for testing.

    Parameters
    ----------
    multi_dataset : pd.DataFrame
        Mock data fixture.

    Returns
    -------
    HierarchicalTTest
        Fitted HierarchicalTTest model instance.
    """
    return HierarchicalTTest(rope=0.01).fit(
        multi_dataset,
        dataset_col="dataset",
        fold_col="fold",
        draws=200,
        tune=200,
        chains=2,
        random_seed=42,
        progressbar=False,
    )


class TestFitInputs:
    """Test the accepted fit input format and its validation."""

    def test_missing_dataset_col_raises(self, multi_dataset):
        """Test that a dataset_col absent from the data raises ValueError."""
        with pytest.raises(ValueError, match="dataset_col 'nope' not found"):
            HierarchicalTTest().fit(
                multi_dataset, dataset_col="nope", draws=10, tune=10
            )

    def test_missing_fold_col_raises(self, multi_dataset):
        """Test that a fold_col absent from the data raises ValueError."""
        with pytest.raises(ValueError, match="fold_col 'nope' not found"):
            HierarchicalTTest().fit(multi_dataset, fold_col="nope", draws=10, tune=10)

    def test_wrong_number_of_algorithms_raises(self):
        """Test that a frame with more than two model columns raises ValueError."""
        df = pd.DataFrame(
            {"dataset": ["d", "d"], "a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]}
        )
        with pytest.raises(ValueError, match="exactly two algorithm"):
            HierarchicalTTest().fit(df, draws=10, tune=10)

    def test_unequal_folds_raises(self):
        """Test that datasets with differing numbers of score rows raise ValueError."""
        df = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2"],
                "a": [0.1, 0.2, 0.3],
                "b": [0.0, 0.1, 0.2],
            }
        )
        with pytest.raises(ValueError, match="same number of fold"):
            HierarchicalTTest().fit(df, draws=10, tune=10)

    def test_datasets_with_differing_cv_designs_raise(self):
        """Test that datasets sharing a row count but not a design raise ValueError."""
        # ds1 is 2-fold x 2 runs, ds2 is 4-fold x 1 run.
        df = pd.DataFrame(
            {
                "dataset": ["d1"] * 4 + ["d2"] * 4,
                "fold": [1, 1, 2, 2, 1, 2, 3, 4],
                "a": [0.8] * 8,
                "b": [0.7, 0.72, 0.71, 0.69, 0.7, 0.72, 0.71, 0.69],
            }
        )
        with pytest.raises(ValueError, match="same cross-validation design"):
            HierarchicalTTest().fit(df, fold_col="fold", draws=10, tune=10)


class TestFoldColumn:
    """Test that the correlation is derived from the fold structure of the data."""

    def test_rho_derived_from_fold_count(self, fitted_hierarchical):
        """Test that rho is derived as 1 / n_folds from the fold column."""
        assert fitted_hierarchical._n_folds == 10
        assert fitted_hierarchical._n_runs == 1
        assert fitted_hierarchical._rho_used == pytest.approx(0.1)

    def test_fold_column_excluded_from_algorithms(self, fitted_hierarchical):
        """Test that the fold column is not treated as a model column."""
        assert fitted_hierarchical._left == "model_a"
        assert fitted_hierarchical._right == "model_b"


class TestDecisionTable:
    """Test decision_table method functionality."""

    def test_single_population_row(self, fitted_hierarchical):
        """Test that the table holds a single row for the population comparison."""
        table = fitted_hierarchical.decision_table()
        assert len(table) == 1
        assert "population" in table.iloc[0]["comparison"]

    def test_partition_sums_to_one(self, fitted_hierarchical):
        """Test that the three region probabilities partition the posterior draws."""
        row = fitted_hierarchical.decision_table(round_ndigits=None).iloc[0]
        assert row["p_left"] + row["p_rope"] + row["p_right"] == pytest.approx(
            1.0, abs=1e-6
        )

    def test_region_probabilities_are_draw_proportions(self, fitted_hierarchical):
        """Benavoli sec. 4.3.2 / Table 12 counts draws per region, not means.

        The two summaries differ, so reporting one where the paper reports the
        other silently changes the published numbers.
        """
        rope_band = (-0.01, 0.01)
        theta_l, theta_e, theta_r = fitted_hierarchical._next_dataset_triple(rope_band)
        stacked = np.column_stack([theta_l, theta_e, theta_r])
        winner = np.argmax(stacked, axis=1)
        expected = [float(np.mean(winner == k)) for k in range(3)]

        row = fitted_hierarchical.decision_table(
            columns=[
                "p_left",
                "p_rope",
                "p_right",
                "mean_theta_left",
                "mean_theta_rope",
                "mean_theta_right",
            ],
            round_ndigits=None,
        ).iloc[0]
        assert [row["p_left"], row["p_rope"], row["p_right"]] == pytest.approx(expected)
        # The posterior means are still available, and are a different quantity.
        assert row["mean_theta_left"] == pytest.approx(float(theta_l.mean()))
        assert row["mean_theta_rope"] == pytest.approx(float(theta_e.mean()))
        assert row["mean_theta_right"] == pytest.approx(float(theta_r.mean()))

    def test_model_a_favoured(self, fitted_hierarchical):
        """Test that the genuinely better model is favoured by the decision."""
        # model_a is genuinely better, so mu_0 > 0 and p_right (left better) dominates.
        row = fitted_hierarchical.decision_table(round_ndigits=None).iloc[0]
        assert row["estimate"] > 0
        assert row["p_right"] > row["p_left"]
        assert row["decision"] == "model_a better"

    def test_threshold_can_withhold_a_decision(self, fitted_hierarchical):
        """A threshold no region reaches must leave the comparison undetermined."""
        row = fitted_hierarchical.decision_table(
            threshold=1.5, round_ndigits=None
        ).iloc[0]
        assert row["decision"] == "Unknown"

    def test_decision_table_has_required_columns(self, fitted_hierarchical):
        """Test that decision_table contains required columns."""
        cols = list(fitted_hierarchical.decision_table().columns)
        for c in ["comparison", "estimate", "hdi_low", "hdi_high", "decision"]:
            assert c in cols

    def test_decision_raw_follows_the_threshold(self, fitted_hierarchical):
        """Test that the raw decision symbol follows the threshold rule."""
        threshold = 0.95
        row = fitted_hierarchical.decision_table(
            threshold=threshold,
            columns=["decision", "decision_raw", "p_left", "p_rope", "p_right"],
            round_ndigits=None,
        ).iloc[0]
        expected = {
            "p_right": ">",
            "p_left": "<",
            "p_rope": "=",
        }
        declared = [
            symbol for name, symbol in expected.items() if row[name] >= threshold
        ]
        assert row["decision_raw"] == (declared[0] if declared else "?")


class TestRopeComparisonTable:
    """The ROPE is a property of the report, so it can be varied without refitting."""

    def test_one_row_per_rope(self, fitted_hierarchical):
        """Each requested ROPE gets a row, with its resolved band."""
        table = fitted_hierarchical.rope_comparison_table([0.01, (-0.05, 0.05)])
        assert len(table) == 2
        assert list(table["rope_low"]) == [-0.01, -0.05]
        assert list(table["rope_high"]) == [0.01, 0.05]

    def test_partition_sums_to_one(self, fitted_hierarchical):
        """The three regions partition the posterior draws for every ROPE."""
        table = fitted_hierarchical.rope_comparison_table(
            [0.005, 0.01, 0.05], round_ndigits=None
        )
        totals = table[["p_left", "p_rope", "p_right"]].sum(axis=1)
        assert np.allclose(totals, 1.0)

    def test_wider_rope_never_lowers_equivalence(self, fitted_hierarchical):
        """Widening the ROPE can only move mass into the equivalence region."""
        table = fitted_hierarchical.rope_comparison_table(
            [0.001, 0.01, 0.1], round_ndigits=None
        )
        assert table["p_rope"].is_monotonic_increasing

    def test_matches_the_decision_table_at_the_fitted_rope(self, fitted_hierarchical):
        """Passing the fitted ROPE reproduces the decision table's own numbers."""
        row = fitted_hierarchical.rope_comparison_table(
            [0.01], round_ndigits=None
        ).iloc[0]
        decision = fitted_hierarchical.decision_table(round_ndigits=None).iloc[0]
        assert row.p_rope == pytest.approx(decision.p_rope)
        assert row.decision == decision.decision

    def test_requires_fit(self):
        """The table is guarded like every other report."""
        with pytest.raises(RuntimeError):
            HierarchicalTTest().rope_comparison_table([0.01])


class TestPerDatasetTable:
    """Test per_dataset_table method functionality."""

    def test_one_row_per_dataset(self, fitted_hierarchical):
        """Test that the table holds one row per dataset with the required columns."""
        table = fitted_hierarchical.per_dataset_table()
        assert len(table) == 8
        assert list(table.columns) == [
            "dataset",
            "raw_mean",
            "shrunk_mean",
            "hdi_low",
            "hdi_high",
        ]

    def test_shrinkage_pulls_towards_population(self, fitted_hierarchical):
        """Test that the shrunk estimates are less spread out than the raw means."""
        table = fitted_hierarchical.per_dataset_table(round_ndigits=None)
        assert table["shrunk_mean"].std() <= table["raw_mean"].std() + 1e-9


class TestSummaryAndInferenceData:
    """Test summary rendering and the ArviZ InferenceData endpoint."""

    def test_summary_printable(self, fitted_hierarchical):
        """Test that the summary renders as a string naming the test."""
        text = str(fitted_hierarchical.summary())
        assert "hierarchical" in text.lower()

    def test_idata_has_population_params(self, fitted_hierarchical):
        """Test that the posterior exposes the population and per-dataset variables."""
        posterior_vars = list(fitted_hierarchical.idata_.posterior.data_vars)
        for v in ["mu_0", "sigma_0", "nu", "mu_i"]:
            assert v in posterior_vars


class TestPlots:
    """Test plot method functionality."""

    def test_simplex_returns_axes(self, fitted_hierarchical):
        """Test that the simplex plot returns a matplotlib Axes."""
        ax = fitted_hierarchical.plot(kind="simplex")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_forest_returns_axes(self, fitted_hierarchical):
        """Test that the forest plot returns a matplotlib Axes."""
        ax = fitted_hierarchical.plot(kind="forest")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_invalid_kind_raises(self, fitted_hierarchical):
        """Test that an unsupported plot kind raises ValueError."""
        with pytest.raises(ValueError, match="Invalid value 'posterior'"):
            fitted_hierarchical.plot(kind="posterior")


class TestUnfittedGuards:
    """Test that methods raise errors when called on unfitted models."""

    def test_decision_table_requires_fit(self):
        """Test that decision_table() raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            HierarchicalTTest().decision_table()

    def test_per_dataset_table_requires_fit(self):
        """Test that per_dataset_table() raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            HierarchicalTTest().per_dataset_table()


class TestDiagnosticsAndPPC:
    """Convergence and predictive checks, as required on every sampled model."""

    def test_diagnostics_reports_the_usual_three(self, fitted_hierarchical):
        """diagnostics() surfaces divergences, R-hat and ESS."""
        diag = fitted_hierarchical.diagnostics()
        assert set(diag) == {"divergences", "max_r_hat", "min_ess_bulk"}
        assert diag["divergences"] >= 0
        assert diag["max_r_hat"] > 0
        assert diag["min_ess_bulk"] > 0

    def test_diagnostics_requires_fit(self):
        """diagnostics() is guarded like every other report."""
        with pytest.raises(RuntimeError):
            HierarchicalTTest().diagnostics()

    def test_bad_diagnostics_warn_on_fit(self, multi_dataset):
        """A starved sampler is surfaced rather than left in idata_."""
        with pytest.warns(UserWarning, match="sampling diagnostics look unreliable"):
            HierarchicalTTest(rope=0.01).fit(
                multi_dataset,
                dataset_col="dataset",
                fold_col="fold",
                draws=15,
                tune=10,
                chains=2,
                random_seed=0,
                progressbar=False,
            )

    def test_posterior_predictive_check_table(self, fitted_hierarchical):
        """Coverage is a proportion per statistic, and widens with the HDI."""
        ppc = fitted_hierarchical.posterior_predictive_check(random_seed=0)
        assert list(ppc["hdi"]) == [0.5, 0.9, 0.95, 1.0]
        assert set(ppc.columns) == {"hdi", "mean", "sd"}
        for column in ("mean", "sd"):
            assert ppc[column].between(0.0, 1.0).all()
            assert ppc[column].is_monotonic_increasing

    def test_posterior_predictive_check_requires_fit(self):
        """The check is guarded like every other report."""
        with pytest.raises(RuntimeError):
            HierarchicalTTest().posterior_predictive_check()

    def test_ppc_plot_returns_axes(self, fitted_hierarchical):
        """plot(kind="ppc") draws one row per dataset."""
        ax = fitted_hierarchical.plot(kind="ppc", random_seed=0)
        assert isinstance(ax, plt.Axes)
        assert len(ax.get_yticklabels()) == len(fitted_hierarchical._dataset_names)
        plt.close("all")
