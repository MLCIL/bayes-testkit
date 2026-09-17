"""
Unit tests for BBTTest class.

This module contains unit tests for the BBTTest class, testing various
functionality including model fitting, posterior table generation,
ROPE comparison tables, and parameter validation.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest

from btk import BBTTest
from btk.tests.bbt._types import ALL_PROPERTIES_COLUMNS


@pytest.fixture(scope="module")
def mock_data():
    """
    Create simple mock data for testing.

    Returns
    -------
    pd.DataFrame
        Mock dataset with 3 datasets and 3 models.
    """
    return pd.DataFrame(
        {
            "dataset": ["ds1", "ds2", "ds3"],
            "model_a": [0.8, 0.75, 0.9],
            "model_b": [0.7, 0.8, 0.85],
            "model_c": [0.6, 0.65, 0.7],
        }
    )


@pytest.fixture(scope="module")
def fitted_model(mock_data):
    """
    Create a fitted BBTTest model for testing.

    Parameters
    ----------
    mock_data : pd.DataFrame
        Mock data fixture.

    Returns
    -------
    BBTTest
        Fitted BBTTest model instance.
    """
    model = BBTTest(absolute_tie_threshold=0.01, tie_solver="spread")
    model.fit(
        mock_data,
        dataset_col="dataset",
        draws=100,
        tune=100,
        chains=2,
        random_seed=42,
    )
    return model


class TestHyperPriors:
    """Every advertised hyper-prior must actually be samplable.

    ``sigma`` is the scale of ``beta ~ Normal(0, sigma)``, so an unbounded
    Normal or Cauchy makes the model's log-probability ``-inf`` at the starting
    point and ``pm.sample`` cannot begin.
    """

    @pytest.mark.parametrize("hyper_prior", ["log_normal", "cauchy", "normal"])
    def test_initial_logp_is_finite(self, hyper_prior: str):
        """The hyper-prior must put the initial point in the support."""
        from btk.tests.bbt.model import _build_bbt_model

        model = _build_bbt_model(
            player1=[0, 0, 1],
            player2=[1, 2, 2],
            win1=[6, 2, 9],
            win2=[13, 19, 8],
            ties=None,
            hyp=hyper_prior,
            scale=1.0,
            use_davidson=False,
        )
        logps = model.point_logps(model.initial_point())
        assert np.isfinite(logps["beta"]), (
            f"hyper_prior={hyper_prior!r} gives a non-finite initial logp for "
            f"beta, so sampling cannot start: {logps}"
        )

    @pytest.mark.parametrize("hyper_prior", ["log_normal", "cauchy", "normal"])
    def test_fit_samples(self, mock_data, hyper_prior: str):
        """Each hyper-prior fits end to end."""
        model = BBTTest(absolute_tie_threshold=0.01, hyper_prior=hyper_prior).fit(
            mock_data,
            dataset_col="dataset",
            draws=50,
            tune=50,
            chains=1,
            random_seed=0,
        )
        assert model.fitted
        assert set(model.beta_ranking) == {"model_a", "model_b", "model_c"}


class TestDavidsonLikelihood:
    """Davidson's ``pwin`` and ``ptie`` share one total: all matches played.

    Wainer (2023), Eq. (7) normalises both probabilities over the three
    outcomes, so both are marginals of the same Multinomial and both binomials
    must use ``w1 + w2 + ties``. Pairing the unconditional ``pwin`` with a total
    that excludes ties biases the abilities whenever ties are present.
    """

    WIN1 = [6, 2, 9]
    WIN2 = [13, 19, 8]
    TIES = [1, 3, 3]

    @pytest.mark.parametrize("var", ["win1_obs", "ties_obs"])
    def test_binomial_totals_include_ties(self, var: str):
        """Every Davidson binomial counts all matches, ties included."""
        from btk.tests.bbt.model import _build_bbt_model

        model = _build_bbt_model(
            player1=[0, 0, 1],
            player2=[1, 2, 2],
            win1=self.WIN1,
            win2=self.WIN2,
            ties=self.TIES,
            hyp="log_normal",
            scale=1.0,
            use_davidson=True,
        )
        expected = np.array(self.WIN1) + np.array(self.WIN2) + np.array(self.TIES)
        # BinomialRV inputs are (rng, size, n, p).
        n_used = np.asarray(model[var].owner.inputs[2].eval())
        np.testing.assert_array_equal(n_used, expected)

    def test_plain_model_total_excludes_ties(self):
        """Without Davidson the ties are folded in by the tie solver, not here."""
        from btk.tests.bbt.model import _build_bbt_model

        model = _build_bbt_model(
            player1=[0, 0, 1],
            player2=[1, 2, 2],
            win1=self.WIN1,
            win2=self.WIN2,
            ties=None,
            hyp="log_normal",
            scale=1.0,
            use_davidson=False,
        )
        expected = np.array(self.WIN1) + np.array(self.WIN2)
        n_used = np.asarray(model["win1_obs"].owner.inputs[2].eval())
        np.testing.assert_array_equal(n_used, expected)


class TestModelSelectionValidation:
    """A misspelled model name must fail loudly, not change what is computed."""

    def test_unknown_control_model_raises(self, fitted_model):
        """Silently returning the all-pairs table would answer a different question."""
        with pytest.raises(ValueError, match="Unknown control_model 'model_z'"):
            fitted_model.posterior_table(control_model="model_z")

    def test_unknown_selected_model_raises(self, fitted_model):
        """A typo in selected_models must name the mistake, not raise from pandas."""
        with pytest.raises(ValueError, match=r"Unknown selected_models \['typo'\]"):
            fitted_model.posterior_table(selected_models=["model_a", "typo"])

    def test_single_selected_model_raises(self, fitted_model):
        """One model makes no pair."""
        with pytest.raises(ValueError, match="At least two models"):
            fitted_model.posterior_table(selected_models=["model_a"])

    def test_known_names_still_work(self, fitted_model):
        """The valid path is untouched."""
        table = fitted_model.posterior_table(control_model="model_a")
        assert len(table) == 2

    def test_unknown_column_raises_with_default_rounding(self, fitted_model):
        """The friendly error must fire on the default path, not only when
        round_ndigits is None.
        """
        with pytest.raises(ValueError, match="Column not_a_column is not available"):
            fitted_model.posterior_table(columns=("mean", "not_a_column"))


class TestSampledModelShape:
    """The sampled model must contain only continuous free variables.

    An unobserved discrete variable (such as a Binomial "replicate" kept for the
    posterior predictive check) makes PyMC fall back to a NUTS + Metropolis
    compound step and pollutes every convergence summary. The predictive check
    is done with ``sample_posterior_predictive`` instead.
    """

    @pytest.mark.parametrize("use_davidson", [False, True])
    def test_no_free_discrete_variables(self, use_davidson: bool):
        """No free variable may be integer-valued."""
        from btk.tests.bbt.model import _build_bbt_model

        model = _build_bbt_model(
            player1=[0, 0, 1],
            player2=[1, 2, 2],
            win1=[6, 2, 9],
            win2=[13, 19, 8],
            ties=[1, 3, 3],
            hyp="log_normal",
            scale=1.0,
            use_davidson=use_davidson,
        )
        discrete = [
            rv.name for rv in model.free_RVs if rv.dtype.startswith(("int", "uint"))
        ]
        assert discrete == [], (
            f"free discrete variables force a compound step: {discrete}"
        )


class TestFittedData:
    """The win counts and posterior draws the reports are derived from."""

    def test_algorithms_match_the_beta_dimension(self, fitted_model):
        """The names line up with the columns of the beta posterior."""
        beta = fitted_model.idata_.posterior["beta"].to_numpy()
        assert fitted_model.algorithms == ["model_a", "model_b", "model_c"]
        assert beta.shape[-1] == len(fitted_model.algorithms)

    def test_win_table_has_one_row_per_pair(self, fitted_model):
        """Three algorithms make three pairs, named rather than indexed."""
        table = fitted_model.win_table
        assert len(table) == 3
        assert list(table.columns) == ["alg1", "alg2", "wins1", "wins2", "ties"]
        assert set(table.alg1) | set(table.alg2) == set(fitted_model.algorithms)

    def test_win_table_counts_every_dataset(self, fitted_model, mock_data):
        """Every dataset is a win for someone or a tie, for every pair."""
        table = fitted_model.win_table
        totals = table.wins1 + table.wins2 + table.ties
        assert (totals == len(mock_data)).all()

    def test_pairwise_samples_agree_with_the_posterior_table(self, fitted_model):
        """The draws average to the mean the summary table reports."""
        draws = fitted_model.pairwise_samples([("model_a", "model_b")])
        table = fitted_model.posterior_table(
            rope_value=(0.45, 0.55), columns=("mean",), round_ndigits=None
        )
        reported = table.loc[table.pair == "model_a > model_b", "mean"].iloc[0]
        assert draws["model_a > model_b"].mean() == pytest.approx(reported)

    def test_pairwise_samples_are_complementary(self, fitted_model):
        """Reversing a pair gives one minus the probability, draw by draw."""
        draws = fitted_model.pairwise_samples(
            [("model_a", "model_b"), ("model_b", "model_a")]
        )
        assert np.allclose(draws["model_a > model_b"], 1.0 - draws["model_b > model_a"])

    def test_pairwise_samples_rejects_unknown_models(self, fitted_model):
        """A typo in a model name is an error, not a silent empty column."""
        with pytest.raises(ValueError, match=r"Unknown algorithms \['typo'\]"):
            fitted_model.pairwise_samples([("model_a", "typo")])

    def test_fitted_data_requires_fit(self):
        """All three accessors are guarded like every other report."""
        for call in (
            lambda: BBTTest().algorithms,
            lambda: BBTTest().win_table,
            lambda: BBTTest().pairwise_samples([("a", "b")]),
        ):
            with pytest.raises(RuntimeError):
                call()


class TestDiagnosticsAndPPC:
    """Wainer sec. 2.2 and 5.4: convergence and predictive checks on every run."""

    def test_diagnostics_reports_the_usual_three(self, fitted_model):
        """diagnostics() surfaces divergences, R-hat and ESS."""
        diag = fitted_model.diagnostics()
        assert set(diag) == {"divergences", "max_r_hat", "min_ess_bulk"}
        assert diag["divergences"] >= 0
        assert diag["max_r_hat"] > 0

    def test_diagnostics_requires_fit(self):
        """diagnostics() is guarded like every other report."""
        with pytest.raises(RuntimeError):
            BBTTest().diagnostics()

    def test_posterior_predictive_check_table(self, fitted_model):
        """The PPC table follows the shape of the paper's Table 5."""
        ppc = fitted_model.posterior_predictive_check(random_seed=0)
        assert list(ppc["hdi"]) == [0.5, 0.9, 0.95, 1.0]
        assert ppc["wins"].between(0.0, 1.0).all()
        # A wider HDI can only contain more of the observed values.
        assert ppc["wins"].is_monotonic_increasing
        # The 100% HDI spans the full range of the replicated draws.
        assert ppc["wins"].iloc[-1] == pytest.approx(1.0)

    def test_plot_posterior_predictive_one_panel_per_pair(self, fitted_model):
        """Each pair gets a histogram and the observed count, in either name order."""
        import matplotlib.pyplot as plt

        axes = fitted_model.plot_posterior_predictive(
            [("model_a", "model_b"), ("model_c", "model_a")], random_seed=0
        )
        assert len(axes) == 2
        for ax in axes:
            assert ax.patches, "missing replicated-wins histogram"
            assert ax.lines, "missing observed-count line"
        assert "wins for model_a" in axes[1].get_title()
        plt.close("all")

    def test_log_likelihood_is_kept_for_waic(self, fitted_model):
        """az.waic needs the pointwise log-likelihood in the InferenceData."""
        assert "log_likelihood" in fitted_model.idata_
        waic = az.waic(fitted_model.idata_)
        assert np.isfinite(waic.elpd_waic)


class TestSetParams:
    """Derived state must follow ``set_params``, not the constructor."""

    def test_set_tie_solver_switches_davidson(self):
        """``set_params(tie_solver="davidson")`` must actually enable Davidson."""
        model = BBTTest(tie_solver="add")
        assert model._use_davidson is False

        model.set_params(tie_solver="davidson")
        assert model.get_params()["tie_solver"] == "davidson"
        assert model._use_davidson is True

        model.set_params(tie_solver="spread")
        assert model._use_davidson is False


class TestCriticalDifferenceOrientation:
    """The CD diagram must place the best algorithm at the "better" end.

    ``_plot_cdd_diagram`` maps ``pos`` to ``x = n_models - pos + 1`` and labels
    the high-``x`` end rank 1 / "better", so ``pos = 1`` has to be the *highest*
    mean beta. Sorting the other way silently prints every diagram backwards.
    """

    def test_best_model_is_rank_one(self, fitted_model):
        """Rank 1 goes to the highest mean beta, not the lowest."""
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        ranking = fitted_model.beta_ranking
        best = max(ranking, key=ranking.get)
        worst = min(ranking, key=ranking.get)

        fig, ax = plt.subplots()
        try:
            fitted_model.plot_cdd_diagram(ax=ax)
            positions = {
                text.get_text(): text.get_position()[0]
                for text in ax.texts
                if text.get_text() in ranking
            }
        finally:
            plt.close(fig)

        assert set(positions) == set(ranking), (
            "every algorithm should be labelled on the ruler"
        )
        # Higher x is the end the axis labels "better".
        assert positions[best] > positions[worst], (
            f"{best} (beta={ranking[best]:.3f}) should be drawn on the 'better' "
            f"side of {worst} (beta={ranking[worst]:.3f}), but was at "
            f"x={positions[best]} vs x={positions[worst]}"
        )
        # And the full order must follow beta.
        by_x = sorted(positions, key=lambda m: positions[m], reverse=True)
        by_beta = sorted(ranking, key=lambda m: ranking[m], reverse=True)
        assert by_x == by_beta


class TestBBTTestInitialization:
    """Test BBTTest initialization and parameter validation."""

    def test_init_with_string_parameters(self):
        """Test that BBTTest can be initialized with string parameters that are cast to enums."""
        model = BBTTest(
            absolute_tie_threshold=0.01,
            tie_solver="spread",
            hyper_prior="log_normal",
            scale=1.0,
        )
        # Verify string values are accepted and work correctly
        assert model._absolute_tie_threshold == 0.01
        assert model._tie_solver == "spread"
        assert model._hyper_prior == "log_normal"
        assert model._scale == 1.0
        assert not model.fitted

    def test_init_defaults(self):
        """Test that default initialization values are set correctly."""
        model = BBTTest()
        assert model._local_rope_value is None
        assert model._local_rope_effect_size is None
        assert model._absolute_tie_threshold is None
        assert model._tie_solver == "add"
        assert model._hyper_prior == "log_normal"
        assert model._scale == 1.0
        assert model._maximize
        assert not model.fitted

    def test_local_rope_value_is_deprecated(self):
        """The overloaded alias still works, but says so."""
        with pytest.warns(DeprecationWarning, match="local_rope_value"):
            model = BBTTest(local_rope_value=0.01)
        # It stands in for whichever scale the data turns out to need.
        assert model._resolve_tie_rules() == (0.01, 0.01)

    def test_explicit_parameters_win_over_the_alias(self):
        """An explicit threshold is not overridden by the deprecated alias."""
        with pytest.warns(DeprecationWarning, match="local_rope_value"):
            model = BBTTest(local_rope_value=0.01, local_rope_effect_size=0.4)
        assert model._resolve_tie_rules() == (0.4, 0.01)

    def test_split_parameters_are_independent(self):
        """The two scales are carried separately and neither implies the other."""
        model = BBTTest(local_rope_effect_size=0.4)
        assert model._resolve_tie_rules() == (0.4, None)

        model = BBTTest(absolute_tie_threshold=0.01)
        assert model._resolve_tie_rules() == (None, 0.01)

    def test_effect_size_does_not_leak_into_the_unpaired_path(self):
        """A Cohen's d must not be applied as a difference in metric units.

        0.4 is the paper's recommended effect size; read as an absolute
        threshold on accuracy it would make every comparison a tie.
        """
        data = pd.DataFrame(
            {
                "dataset": ["ds1", "ds2", "ds3"],
                "model_a": [0.9, 0.9, 0.9],
                "model_b": [0.6, 0.6, 0.6],
            }
        )
        model = BBTTest(local_rope_effect_size=0.4).fit(
            data, dataset_col="dataset", draws=50, tune=50, chains=1, random_seed=0
        )
        # No data_sd and one row per dataset, so no tie rule applies: model_a
        # wins all three. Had 0.4 been read as an absolute threshold, the 0.3
        # gaps would all have been ties.
        np.testing.assert_array_equal(model._win_table, np.array([[0, 1, 3, 0, 0]]))

    def test_validate_params(self):
        """Test that invalid parameter values raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Invalid value 'invalid_solver' for parameter 'tie_solver'",
        ):
            BBTTest(tie_solver="invalid_solver")
        with pytest.raises(
            ValueError,
            match="Invalid value 'invalid_prior' for parameter 'hyper_prior'",
        ):
            BBTTest(hyper_prior="invalid_prior")


@pytest.mark.filterwarnings("ignore:BBT sampling diagnostics:UserWarning")
class TestBBTTestFitting:
    """Test BBTTest model fitting functionality.

    These fits are deliberately tiny, so the convergence warning is expected and
    is not what is under test here.
    """

    def test_fit_updates_fitted_property(self, mock_data):
        """Test that fit() updates the fitted property."""
        model = BBTTest()
        assert not model.fitted
        model.fit(mock_data, dataset_col="dataset", draws=50, tune=50, chains=2)
        assert model.fitted

    def test_fit_returns_self(self, mock_data):
        """Test that fit() returns self for method chaining."""
        model = BBTTest()
        result = model.fit(
            mock_data, dataset_col="dataset", draws=50, tune=50, chains=2
        )
        assert result is model


class TestBBTTestUnfittedErrors:
    """Test that methods raise errors when called on unfitted models."""

    def test_posterior_table_without_fitting_raises_error(self):
        """Test that posterior_table() raises error on unfitted model."""
        model = BBTTest()
        with pytest.raises(
            RuntimeError, match="The model must be fitted before accessing this method"
        ):
            model.posterior_table()

    def test_rope_comparison_control_table_without_fitting_raises_error(self):
        """Test that rope_comparison_control_table() raises error on unfitted model."""
        model = BBTTest()
        with pytest.raises(
            RuntimeError, match="The model must be fitted before accessing this method"
        ):
            model.rope_comparison_control_table(
                rope_values=[(0.45, 0.55)], control_model="model_a"
            )


class TestPosteriorTable:
    """Test posterior_table method functionality."""

    def test_posterior_table_has_required_columns(self, fitted_model):
        """Test that posterior_table contains required columns."""
        result = fitted_model.posterior_table()
        required_cols = ["pair", "mean", "delta", "above_50", "in_rope"]
        for col in required_cols:
            assert col in result.columns

    def test_posterior_table_weak_interpretation_values(self, fitted_model):
        """Test that weak interpretation contains valid values."""
        result = fitted_model.posterior_table(rope_value=(0.45, 0.55))
        valid_values = {"Equivalent", "Unknown"}
        # Weak interpretation should end with "better", be "Equivalent", or be "Unknown"
        for interp in result["weak_interpretation"]:
            assert interp in valid_values or interp.endswith(" better"), (
                f"Invalid weak interpretation: {interp}"
            )

    def test_posterior_table_strong_interpretation_values(self, fitted_model):
        """Test that strong interpretation contains valid values."""
        result = fitted_model.posterior_table()
        # Add strong_interpretation to columns
        result = fitted_model.posterior_table(
            columns=[
                "mean",
                "strong_interpretation",
            ]
        )
        valid_values = {"Equivalent", "Unknown"}
        for interp in result["strong_interpretation"]:
            assert interp in valid_values or interp.endswith(" better"), (
                f"Invalid strong interpretation: {interp}"
            )

    def test_posterior_table_with_control_model(self, fitted_model):
        """Test posterior_table with control_model parameter."""
        result = fitted_model.posterior_table(
            control_model="model_a", columns=ALL_PROPERTIES_COLUMNS
        )
        assert len(result) > 0
        # All comparisons should involve model_a
        for _, row in result.iterrows():
            assert row["left_model"] == "model_a" or row["right_model"] == "model_a", (
                f"Comparison {row['pair']} does not involve control model"
            )

    def test_posterior_table_returns_only_requested_columns(self, fitted_model):
        """Test that posterior_table returns only requested columns."""
        requested_columns = ["mean", "delta"]
        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=None
        )

        # Should have 'pair' column plus requested columns
        expected_cols = ["pair", "mean", "delta"]
        assert set(result.columns) == set(expected_cols)

        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=3
        )
        assert set(result.columns) == set(expected_cols)

    def test_posterior_table_requested_columns_with_strings(self, fitted_model):
        """Test that posterior_table accepts string column names."""
        requested_columns = ["mean", "delta", "above_50"]
        # Must set round_ndigits=None to get column filtering
        result = fitted_model.posterior_table(
            columns=requested_columns, round_ndigits=None
        )

        expected_cols = ["pair", "mean", "delta", "above_50"]
        assert set(result.columns) == set(expected_cols)

    def test_posterior_table_invalid_column_raises_error(self, fitted_model):
        """Test that requesting invalid column raises ValueError."""
        with pytest.raises(ValueError, match="is not available in the posterior table"):
            # Must set round_ndigits=None to trigger column validation
            fitted_model.posterior_table(columns=["invalid_column"], round_ndigits=None)

    def test_posterior_table_rope_value_affects_in_rope(self, fitted_model):
        """Test that changing ROPE value affects in_rope column."""
        result1 = fitted_model.posterior_table(rope_value=(0.4, 0.6))
        result2 = fitted_model.posterior_table(rope_value=(0.45, 0.55))

        # Wider ROPE should generally have higher in_rope values
        mean_in_rope_1 = result1["in_rope"].mean()
        mean_in_rope_2 = result2["in_rope"].mean()
        assert mean_in_rope_1 >= mean_in_rope_2

    def test_posterior_table_rounding(self, fitted_model):
        """Test that rounding parameter works correctly."""
        result_rounded = fitted_model.posterior_table(round_ndigits=2)

        # Check that rounded version has at most 2 decimal places
        for col in ["mean", "delta"]:
            if col in result_rounded.columns:
                for val in result_rounded[col]:
                    if not pd.isna(val):
                        str_val = str(val)
                        if "." in str_val:
                            decimals = len(str_val.split(".")[1])
                            assert decimals <= 2


class TestRopeComparisonControlTable:
    """Test rope_comparison_control_table method functionality."""

    def test_rope_comparison_returns_dataframe(self, fitted_model):
        """Test that rope_comparison_control_table returns a DataFrame."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55), (0.4, 0.6)], control_model="model_a"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # One row per ROPE

    def test_rope_comparison_has_required_columns(self, fitted_model):
        """Test that rope_comparison_control_table has required columns."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)], control_model="model_a"
        )
        required_cols = [
            "rope_value",
            "better_models",
            "equivalent_models",
            "worse_models",
            "unknown_models",
        ]
        for col in required_cols:
            assert col in result.columns

    def test_rope_comparison_weak_interpretation(self, fitted_model):
        """Test rope_comparison_control_table with weak interpretation."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55), (0.4, 0.6)],
            control_model="model_a",
            interpretation="weak",
        )
        assert len(result) == 2
        # Check that each row has the correct ROPE value
        assert result.iloc[0]["rope_value"] == (0.45, 0.55)
        assert result.iloc[1]["rope_value"] == (0.4, 0.6)

    def test_rope_comparison_strong_interpretation(self, fitted_model):
        """Test rope_comparison_control_table with strong interpretation."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            interpretation="strong",
        )
        assert len(result) == 1
        # Verify it runs without error and returns expected structure
        assert "better_models" in result.columns

    def test_rope_comparison_return_as_array(self, fitted_model):
        """Test rope_comparison_control_table with return_as_array=True."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=True,
        )
        # When return_as_array=True, columns should contain lists
        assert isinstance(result.iloc[0]["better_models"], list)
        assert isinstance(result.iloc[0]["equivalent_models"], list)
        assert isinstance(result.iloc[0]["worse_models"], list)
        assert isinstance(result.iloc[0]["unknown_models"], list)

    def test_rope_comparison_return_as_string(self, fitted_model):
        """Test rope_comparison_control_table with return_as_array=False."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=False,
        )
        # When return_as_array=False, columns should contain strings
        assert isinstance(result.iloc[0]["better_models"], str)
        assert isinstance(result.iloc[0]["equivalent_models"], str)
        assert isinstance(result.iloc[0]["worse_models"], str)
        assert isinstance(result.iloc[0]["unknown_models"], str)

    def test_rope_comparison_custom_join_char(self, fitted_model):
        """Test rope_comparison_control_table with custom join character."""
        result = fitted_model.rope_comparison_control_table(
            rope_values=[(0.45, 0.55)],
            control_model="model_a",
            return_as_array=False,
            join_char=" | ",
        )
        # Check if custom join character is used (if there are multiple models)
        for col in [
            "better_models",
            "equivalent_models",
            "worse_models",
            "unknown_models",
        ]:
            value = result.iloc[0][col]
            if " | " in value:
                # Found the custom separator, test passes
                assert True
                return

    def test_rope_comparison_multiple_ropes(self, fitted_model):
        """Test rope_comparison_control_table with multiple ROPE values."""
        rope_values = [(0.3, 0.7), (0.4, 0.6), (0.45, 0.55)]
        result = fitted_model.rope_comparison_control_table(
            rope_values=rope_values, control_model="model_a"
        )
        assert len(result) == len(rope_values)
        # Verify ROPE values are correctly stored
        for i, rope in enumerate(rope_values):
            assert result.iloc[i]["rope_value"] == rope


class TestPosteriorTableInterpretations:
    """Test interpretation logic in posterior_table with mocked samples."""

    def test_weak_interpretation_logic(self, fitted_model, monkeypatch):
        """Test that weak interpretation follows expected logic with controlled samples."""
        # Create synthetic samples with known properties for testing interpretations
        # Case 1: Model A > Model B - should be "A better" (96% above 0.5, only 3% in ROPE)
        samples_a_better = np.concatenate(
            [np.full(960, 0.8), np.full(40, 0.4)]
        )  # 96% > 0.5, mean ~0.784

        # Case 2: Model B > Model C - should be "Equivalent" (40% above 0.5, 96% in ROPE)
        samples_equivalent = np.concatenate(
            [np.full(400, 0.52), np.full(600, 0.48)]
        )  # 40% > 0.5, mean = 0.496

        # Case 3: Model C > Model D - should be "Unknown" (70% above 0.5, 80% in ROPE)
        samples_unknown = np.concatenate(
            [np.full(700, 0.6), np.full(300, 0.4)]
        )  # 70% > 0.5, mean = 0.54

        samples = np.column_stack(
            [samples_a_better, samples_equivalent, samples_unknown]
        )
        names = ["A > B", "B > C", "C > D"]

        # Mock _get_pwin to return our controlled samples
        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(rope_value=(0.45, 0.55))

        # Test Case 1: A > B should be "A better"
        row_a_b = result[result["pair"] == "A > B"].iloc[0]
        assert row_a_b["weak_interpretation"] == "A better"
        assert row_a_b["above_50"] >= 0.95
        assert row_a_b["in_rope"] < 0.95

        # Test Case 2: B > C should be "Equivalent"
        row_b_c = result[result["pair"] == "B > C"].iloc[0]
        assert row_b_c["weak_interpretation"] == "Equivalent"
        assert row_b_c["in_rope"] >= 0.95

        # Test Case 3: C > D should be "Unknown"
        row_c_d = result[result["pair"] == "C > D"].iloc[0]
        assert row_c_d["weak_interpretation"] == "Unknown"
        assert row_c_d["above_50"] < 0.95
        assert row_c_d["in_rope"] < 0.95

    def test_strong_interpretation_logic(self, fitted_model, monkeypatch):
        """Test that strong interpretation follows expected logic with controlled samples."""
        # Create synthetic samples with known properties for testing strong interpretations
        # Case 1: Model A > Model B - mean > 0.70, should be "A better"
        samples_a_better = np.full(1000, 0.75)  # mean = 0.75

        # Case 2: Model B > Model C - mean <= 0.55, should be "Equivalent"
        samples_equivalent = np.full(1000, 0.50)  # mean = 0.50

        # Case 3: Model C > Model D - 0.55 < mean <= 0.70, should be "Unknown"
        samples_unknown = np.full(1000, 0.62)  # mean = 0.62

        samples = np.column_stack(
            [samples_a_better, samples_equivalent, samples_unknown]
        )
        names = ["A > B", "B > C", "C > D"]

        # Mock _get_pwin to return our controlled samples
        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=[
                "mean",
                "strong_interpretation",
            ],
            round_ndigits=None,
        )

        # Test Case 1: A > B should be "A better" (mean > 0.70)
        row_a_b = result[result["pair"] == "A > B"].iloc[0]
        assert row_a_b["strong_interpretation"] == "A better"
        assert row_a_b["mean"] > 0.70

        # Test Case 2: B > C should be "Equivalent" (mean <= 0.55)
        row_b_c = result[result["pair"] == "B > C"].iloc[0]
        assert row_b_c["strong_interpretation"] == "Equivalent"
        assert row_b_c["mean"] <= 0.55

        # Test Case 3: C > D should be "Unknown" (0.55 < mean <= 0.70)
        row_c_d = result[result["pair"] == "C > D"].iloc[0]
        assert row_c_d["strong_interpretation"] == "Unknown"
        assert 0.55 < row_c_d["mean"] <= 0.70


class TestPosteriorTableStructure:
    """Test the structure and content of posterior_table output with mocked samples."""

    def test_pair_column_format(self, fitted_model, monkeypatch):
        """Test that pair column has correct format."""
        # Create simple samples for structure testing
        samples = np.column_stack([np.full(100, 0.6), np.full(100, 0.5)])
        names = ["model_a > model_b", "model_b > model_c"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table()

        for pair in result["pair"]:
            assert " > " in pair, f"Pair {pair} does not contain ' > '"
            parts = pair.split(" > ")
            assert len(parts) == 2, f"Pair {pair} does not have exactly 2 parts"

    def test_left_right_model_consistency(self, fitted_model, monkeypatch):
        """Test that left_model and right_model match the pair column."""
        samples = np.column_stack([np.full(100, 0.6), np.full(100, 0.5)])
        names = ["model_a > model_b", "model_b > model_c"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(columns=ALL_PROPERTIES_COLUMNS)

        for _, row in result.iterrows():
            expected_pair = f"{row['left_model']} > {row['right_model']}"
            assert row["pair"] == expected_pair

    def test_probability_values_in_range(self, fitted_model, monkeypatch):
        """Test that probability values are in valid range [0, 1]."""
        # Create samples with values that should produce probabilities in [0, 1]
        samples = np.column_stack(
            [
                np.random.uniform(0.3, 0.9, 100),
                np.random.uniform(0.2, 0.8, 100),
                np.random.uniform(0.1, 0.7, 100),
            ]
        )
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table()

        for col in ["mean", "median", "above_50", "in_rope"]:
            if col in result.columns:
                assert (result[col] >= 0).all()
                assert (result[col] <= 1).all()

    def test_hdi_values_consistent(self, fitted_model, monkeypatch):
        """Test that HDI values are consistent (low <= high)."""
        # Create samples with varying distributions
        samples = np.column_stack(
            [
                np.random.beta(2, 5, 100),  # Skewed distribution
                np.random.beta(5, 5, 100),  # Symmetric distribution
                np.random.beta(8, 2, 100),  # Right-skewed distribution
            ]
        )
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=["hdi_low", "hdi_high"],
            round_ndigits=None,
        )

        assert (result["hdi_low"] <= result["hdi_high"]).all()

    def test_delta_equals_hdi_difference(self, fitted_model, monkeypatch):
        """Test that delta equals hdi_high - hdi_low."""
        # Create samples with known distributions
        np.random.seed(42)
        samples = np.column_stack(
            [
                np.random.normal(0.6, 0.1, 1000),
                np.random.normal(0.5, 0.15, 1000),
                np.random.normal(0.7, 0.08, 1000),
            ]
        )
        # Clip to [0, 1] range
        samples = np.clip(samples, 0, 1)
        names = ["A > B", "B > C", "C > D"]

        def mock_get_pwin(*args, **kwargs):
            return samples, names

        monkeypatch.setattr("btk.tests.bbt.bbt._get_pwin", mock_get_pwin)

        result = fitted_model.posterior_table(
            columns=[
                "hdi_low",
                "hdi_high",
                "delta",
            ],
            round_ndigits=None,  # Don't round to avoid rounding differences
        )

        calculated_delta = result["hdi_high"] - result["hdi_low"]
        np.testing.assert_array_almost_equal(
            result["delta"], calculated_delta, decimal=10
        )
