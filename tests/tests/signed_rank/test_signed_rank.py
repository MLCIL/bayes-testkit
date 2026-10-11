"""
Unit tests for SignedRankTest class.

This module contains unit tests for the SignedRankTest class, testing various
functionality including model fitting, the posterior ROPE triple, decision table
generation, and parameter validation. The posterior is drawn by plain Monte Carlo,
so these tests are fast and can check the values reported in the paper.
"""

import numpy as np
import pandas as pd
import pytest

from btk import SignedRankTest

TABLE_7 = np.array(
    [
        -1.939, -0.261, 0.467, -0.719, 2.000, -0.464, -1.014, -0.151, -7.269,
        -0.790, -2.600, 4.362, -0.614, 0.000, -0.625, -0.069, -0.212, -1.683,
        0.267, -3.242, -0.833, 0.000, -1.762, -1.863, -10.002, -0.343, -4.190,
        -2.434, -4.747, -3.548, 0.583, -10.043, -0.443, 1.333, -0.674, -3.922,
        -2.776, -0.688, -3.996, -0.338, -1.112, -3.284, -1.684, -0.699, -0.367,
        -5.600, -0.400, -16.909, -5.040, -1.809, 0.500, 0.143, -0.202, -0.682,
    ]
)  # fmt: skip


@pytest.fixture
def paper_data():
    """
    Create the per-dataset differences of Table 7 in the paper.

    Returns
    -------
    pd.DataFrame
        Mean accuracies (0-100) of nbc against a zero aode baseline on 54 datasets,
        so that nbc - aode reproduces Table 7.
    """
    return pd.DataFrame(
        {
            "dataset": [f"ds{i}" for i in range(len(TABLE_7))],
            "nbc": TABLE_7,
            "aode": 0.0,
        }
    )


@pytest.fixture
def fitted_signed_rank(paper_data):
    """
    Create a fitted SignedRankTest model for testing.

    Parameters
    ----------
    paper_data : pd.DataFrame
        Table 7 differences fixture.

    Returns
    -------
    SignedRankTest
        Fitted SignedRankTest model instance, with the paper's ROPE of 1%.
    """
    return SignedRankTest(rope=1.0).fit(paper_data, random_seed=1)


class TestInitialization:
    """Test SignedRankTest initialization and parameter handling."""

    def test_defaults(self):
        """Test that default initialization values are set correctly."""
        t = SignedRankTest()
        assert t._rope == 0.01
        assert t._maximize
        assert t._prior_strength == 0.5
        assert t._prior_pseudo_obs == 0.0
        assert not t.fitted

    def test_get_params_holds_only_hyperparameters(self):
        """Test that get_params exposes only the hyperparameters, not the data properties."""
        t = SignedRankTest(rope=0.02, prior_pseudo_obs=np.inf)
        assert t.get_params() == {
            "rope": 0.02,
            "maximize": True,
            "prior_strength": 0.5,
            "prior_pseudo_obs": np.inf,
        }

    def test_non_positive_prior_strength_raises(self):
        """Test that a prior strength of zero raises ValueError."""
        with pytest.raises(ValueError, match="prior_strength must be positive"):
            SignedRankTest(prior_strength=0.0)


class TestPaperResults:
    """
    Test the posterior against the values reported in the paper.

    Table 9 (z_0 = 0) is not checked: under Eq. 11 its values cannot be recovered
    from the printed Table 7 for any prior strength, while Tables 10 and 11 can.
    """

    @pytest.mark.parametrize(
        ("prior_pseudo_obs", "p_aode", "p_rope"),
        [(np.inf, 0.888, 0.112), (-np.inf, 0.904, 0.096)],
    )
    def test_region_probabilities_match_tables_10_and_11(
        self, paper_data, prior_pseudo_obs, p_aode, p_rope
    ):
        """Test that nbc vs aode reproduces the paper's robustness analysis."""
        t = SignedRankTest(rope=1.0, prior_pseudo_obs=prior_pseudo_obs)
        row = t.fit(paper_data, random_seed=0).decision_table().iloc[0]
        assert row["p_left"] == pytest.approx(p_aode, abs=0.01)
        assert row["p_rope"] == pytest.approx(p_rope, abs=0.01)
        assert row["p_right"] == pytest.approx(0.0, abs=0.01)

    def test_probabilities_sum_to_one(self, fitted_signed_rank):
        """Test that both the region probabilities and the mean triple sum to one."""
        row = fitted_signed_rank.decision_table(
            columns=SignedRankTest.ALL_SIGNED_RANK_COLUMNS, round_ndigits=None
        ).iloc[0]
        assert row["p_left"] + row["p_rope"] + row["p_right"] == pytest.approx(1.0)
        assert row["mean_theta_left"] + row["mean_theta_rope"] + row[
            "mean_theta_right"
        ] == pytest.approx(1.0)


class TestTriple:
    """Test the posterior ROPE triple of Eq. 11."""

    def test_matches_double_sum(self):
        """Test that the triple matches a direct evaluation of the double sum."""
        df = pd.DataFrame(
            {"dataset": list("abcde"), "x": [0.3, -0.2, 0.05, 1.2, -0.9], "y": 0.0}
        )
        t = SignedRankTest(rope=(-0.1, 0.4), prior_pseudo_obs=0.2).fit(
            df, draws=3, random_seed=0
        )
        theta_l, theta_e, theta_r = t._triple((-0.1, 0.4))

        atoms = np.concatenate([[0.2], df["x"].to_numpy()])
        for s, w in enumerate(t._weights):
            expected = np.zeros(3)
            for i in range(len(atoms)):
                for j in range(len(atoms)):
                    walsh = (atoms[i] + atoms[j]) / 2
                    region = 0 if walsh < -0.1 else 2 if walsh > 0.4 else 1
                    expected[region] += w[i] * w[j]
            assert (theta_l[s], theta_e[s], theta_r[s]) == pytest.approx(expected)

    def test_pseudo_obs_at_infinity_favours_left_model(self, paper_data):
        """Test that moving the pseudo-observation to +inf shifts mass to the left model."""
        cols = ["mean_theta_right"]
        centred = SignedRankTest(rope=1.0).fit(paper_data, random_seed=0)
        shifted = SignedRankTest(rope=1.0, prior_pseudo_obs=np.inf).fit(
            paper_data, random_seed=0
        )
        assert (
            shifted.decision_table(columns=cols).iloc[0, 0]
            > centred.decision_table(columns=cols).iloc[0, 0]
        )


class TestFitInputs:
    """Test the accepted fit input format and its validation."""

    def test_rows_averaged_per_dataset(self):
        """Test that repeated rows of a dataset are averaged into its mean difference."""
        df = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "a": [0.8, 0.9, 0.5, 0.7],
                "b": [0.7, 0.6, 0.6, 0.6],
            }
        )
        t = SignedRankTest().fit(df)
        assert sorted(t._z) == pytest.approx([0.0, 0.2])

    def test_maximize_false_flips_sign(self):
        """Test that maximize=False flips the sign for minimized metrics."""
        df = pd.DataFrame({"dataset": ["d1", "d2"], "a": [0.1, 0.12], "b": [0.2, 0.22]})
        t = SignedRankTest(maximize=False).fit(df)
        assert np.all(t._z > 0)

    def test_missing_dataset_col_raises(self, paper_data):
        """Test that a dataset_col absent from the data raises ValueError."""
        with pytest.raises(ValueError, match="dataset_col 'nope' not found"):
            SignedRankTest().fit(paper_data, dataset_col="nope")

    def test_wrong_number_of_columns_raises(self):
        """Test that a frame with more than two model columns raises ValueError."""
        df = pd.DataFrame(
            {"dataset": ["d1", "d2"], "a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]}
        )
        with pytest.raises(ValueError, match="exactly two algorithm columns"):
            SignedRankTest().fit(df)

    def test_random_seed_is_reproducible(self, paper_data):
        """Test that the same seed gives the same decision table."""
        a = SignedRankTest(rope=1.0).fit(paper_data, random_seed=3).decision_table()
        b = SignedRankTest(rope=1.0).fit(paper_data, random_seed=3).decision_table()
        pd.testing.assert_frame_equal(a, b)


class TestReports:
    """Test decision_table method functionality."""

    def test_decision_table_default_columns(self, fitted_signed_rank):
        """Test that decision_table reports the partition and the decision by default."""
        cols = list(fitted_signed_rank.decision_table().columns)
        assert cols == ["comparison", "p_left", "p_rope", "p_right", "decision"]

    def test_decision_table_all_columns(self, fitted_signed_rank):
        """Test that every advertised column is available."""
        table = fitted_signed_rank.decision_table(
            columns=SignedRankTest.ALL_SIGNED_RANK_COLUMNS
        )
        assert list(table.columns) == SignedRankTest.ALL_SIGNED_RANK_COLUMNS
        assert table.iloc[0]["comparison"] == "nbc vs aode"

    def test_decision_table_invalid_column_raises(self, fitted_signed_rank):
        """Test that requesting an invalid column raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            fitted_signed_rank.decision_table(columns=["estimate"])

    def test_threshold_controls_decision(self, fitted_signed_rank):
        """Test that raising the threshold turns a borderline decision into Unknown."""
        cols = ["decision", "decision_raw"]
        lenient = fitted_signed_rank.decision_table(threshold=0.85, columns=cols)
        strict = fitted_signed_rank.decision_table(threshold=0.95, columns=cols)
        assert lenient.iloc[0].tolist() == ["aode better", "<"]
        assert strict.iloc[0].tolist() == ["Unknown", "?"]

    def test_wider_rope_raises_p_rope(self, paper_data):
        """Test that widening the ROPE increases the equivalence probability."""
        narrow = SignedRankTest(rope=1.0).fit(paper_data, random_seed=0)
        wide = SignedRankTest(rope=3.0).fit(paper_data, random_seed=0)
        assert (
            wide.decision_table(round_ndigits=None).iloc[0]["p_rope"]
            > narrow.decision_table(round_ndigits=None).iloc[0]["p_rope"]
        )


class TestInferenceData:
    """Test the ArviZ InferenceData endpoint."""

    def test_idata_has_weights(self, fitted_signed_rank):
        """Test that the posterior exposes the weights of the pseudo-observation and datasets."""
        weights = fitted_signed_rank.idata_.posterior["weights"]
        assert weights.shape == (1, 50_000, len(TABLE_7) + 1)

    def test_idata_before_fit_raises(self):
        """Test that idata_ raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = SignedRankTest().idata_


class TestUnfittedGuards:
    """Test that methods raise errors when called on unfitted models."""

    def test_decision_table_requires_fit(self):
        """Test that decision_table() raises error on unfitted model."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            SignedRankTest().decision_table()


class TestDiagnosticsAreNotApplicable:
    """The Monte Carlo posterior has no chain, so the diagnostics must not pretend."""

    def test_diagnostics_raises(self, fitted_signed_rank):
        """Inheriting the base implementation would report a meaningless NaN R-hat."""
        with pytest.raises(NotImplementedError, match="no sampler to diagnose"):
            fitted_signed_rank.diagnostics()
