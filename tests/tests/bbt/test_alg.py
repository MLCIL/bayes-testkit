from io import StringIO

import numpy as np
import pandas as pd
import pytest

from btk.tests.bbt.alg import (
    _construct_win_table,
)

SCORES_1 = pd.DataFrame(
    {
        "alg1": [0.705, 0.7, 0.9],
        "alg2": [0.696, 0.7, 0.8],
        "alg3": [0.7, 0.75, 0.9],
    }
)


class TestConstructTable:
    """Test whether the win/tie/loss table is constructed correctly."""

    @pytest.mark.parametrize(
        "data, absolute_tie_threshold, maximize, expected_table",
        [
            (
                SCORES_1,
                None,
                True,
                np.array(
                    [
                        [0, 1, 2, 0, 1],  # alg1 vs alg2
                        [0, 2, 1, 1, 1],  # alg1 vs alg3
                        [1, 2, 0, 3, 0],  # alg2 vs alg3
                    ]
                ),
            ),
            (
                SCORES_1,
                0.01,
                True,
                np.array(
                    [
                        [0, 1, 1, 0, 2],  # alg1 vs alg2
                        [0, 2, 0, 1, 2],  # alg1 vs alg3
                        [1, 2, 0, 2, 1],  # alg2 vs alg3
                    ]
                ),
            ),
            (
                SCORES_1,
                0.01,
                False,
                np.array(
                    [
                        [0, 1, 0, 1, 2],  # alg1 vs alg2
                        [0, 2, 1, 0, 2],  # alg1 vs alg3
                        [1, 2, 2, 0, 1],  # alg2 vs alg3
                    ]
                ),
            ),
        ],
    )
    def test_construct_win_table(
        self,
        data: pd.DataFrame,
        absolute_tie_threshold: float | None,
        maximize: bool,
        expected_table: np.ndarray,
    ):
        """Test the construction of the win/tie/loss table."""
        # When
        result_table, alg_names = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            absolute_tie_threshold=absolute_tie_threshold,
            tie_solver="davidson",  # Keeps the ties in the table
            maximize=maximize,
        )

        # Then
        np.testing.assert_array_almost_equal(result_table, expected_table)

    def test_construct_win_table_paired_local_rope(self):
        """Test paired-path win/tie/loss construction for repeated datasets."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "alg1": [0.8, 0.9, 0.3, 0.2],
                "alg2": [0.7, 0.8, 0.4, 0.5],
            }
        )

        result_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=0.1,
            tie_solver="davidson",
            maximize=True,
        )

        expected_table = np.array(
            [
                [0, 1, 1, 1, 0],
            ]
        )
        np.testing.assert_array_equal(result_table, expected_table)

    def test_construct_win_table_paired_local_rope_three_algorithms(self):
        """Test paired local-ROPE construction with dataset column and 3 algorithms."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "alg1": [0.9, 0.8, 0.4, 0.3],
                "alg2": [0.8, 0.7, 0.3, 0.2],
                "alg3": [0.2, 0.1, 0.5, 0.4],
            }
        )

        result_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=0.1,
            tie_solver="davidson",
            maximize=True,
        )

        expected_table = np.array(
            [
                [0, 1, 2, 0, 0],  # alg1 > alg2 on both datasets
                [0, 2, 1, 1, 0],  # split decisions across datasets
                [1, 2, 1, 1, 0],  # split decisions across datasets
            ]
        )
        np.testing.assert_array_equal(result_table, expected_table)


class TestMissingMeasures:
    """A missing measure is neither a win, nor a loss, nor a tie.

    Wainer (2023), sec. 5.7: "the BBT model simply does not count it as a win or
    a loss for that algorithm in comparison to the others". Counting it as a tie
    would hand the absent algorithm free wins under the ``add`` and ``spread``
    policies.
    """

    MISSING = pd.DataFrame(
        {
            "dataset": ["d1", "d2", "d3", "d4"],
            "alg1": [0.9, 0.8, 0.7, 0.6],
            "alg2": [0.1, 0.2, np.nan, np.nan],
        }
    )

    @pytest.mark.parametrize(
        "tie_solver, expected",
        [
            # Only the two data sets where both algorithms ran are matches.
            ("davidson", np.array([[0, 1, 2, 0, 0]])),
            ("forget", np.array([[0, 1, 2, 0, 0]])),
            ("add", np.array([[0, 1, 2, 0, 0]])),
            ("spread", np.array([[0, 1, 2, 0, 0]])),
        ],
    )
    def test_unpaired_missing_measures_are_not_matches(
        self, tie_solver: str, expected: np.ndarray
    ):
        """Rows where either algorithm is absent are dropped from the pair."""
        table, _ = _construct_win_table(
            data=self.MISSING,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=None,
            absolute_tie_threshold=None,
            tie_solver=tie_solver,
            maximize=True,
        )
        np.testing.assert_array_equal(table, expected)

    def test_paired_missing_measures_are_not_matches(self):
        """A data set with no usable folds for a pair contributes nothing."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1", "d2", "d2"],
                "alg1": [0.9, 0.8, 0.4, 0.3],
                "alg2": [0.2, 0.1, np.nan, np.nan],
            }
        )

        table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=0.1,
            tie_solver="davidson",
            maximize=True,
        )

        # d1 is a win for alg1; d2 is not played at all.
        np.testing.assert_array_equal(table, np.array([[0, 1, 1, 0, 0]]))

    def test_fully_absent_algorithm_earns_nothing(self):
        """An algorithm that never ran must not accumulate wins."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d2"],
                "alg1": [0.9, 0.8],
                "alg2": [np.nan, np.nan],
            }
        )

        table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=None,
            absolute_tie_threshold=None,
            tie_solver="add",
            maximize=True,
        )

        np.testing.assert_array_equal(table, np.array([[0, 1, 0, 0, 0]]))


class TestLocalRopeFormulas:
    """The local ROPE thresholds must match Wainer (2023), Eq. (3)-(6)."""

    def test_unpaired_threshold_pools_average_variance(self):
        """Eq. (4): the threshold is d * sqrt((s_i^2 + s_j^2) / 2).

        Chosen so the two candidate formulas disagree: with s_i = s_j = 0.2 and
        d = 0.4 the paper's threshold is 0.08 and the difference of 0.10 is a
        win, whereas pooling by the *sum* gives 0.113 and would call it a tie.
        """
        mean = pd.DataFrame({"dataset": ["d1"], "alg1": [0.60], "alg2": [0.50]})
        sd = pd.DataFrame({"alg1": [0.20], "alg2": [0.20]})

        table, _ = _construct_win_table(
            data=mean,
            data_sd=sd,
            dataset_col="dataset",
            local_rope_effect_size=0.4,
            tie_solver="davidson",
            maximize=True,
        )

        np.testing.assert_array_equal(table, np.array([[0, 1, 1, 0, 0]]))

    def test_unpaired_threshold_still_ties_inside_the_rope(self):
        """A difference below the Eq. (4) threshold remains a tie."""
        mean = pd.DataFrame({"dataset": ["d1"], "alg1": [0.55], "alg2": [0.50]})
        sd = pd.DataFrame({"alg1": [0.20], "alg2": [0.20]})

        table, _ = _construct_win_table(
            data=mean,
            data_sd=sd,
            dataset_col="dataset",
            local_rope_effect_size=0.4,
            tie_solver="davidson",
            maximize=True,
        )

        # 0.05 < 0.08, so neither algorithm wins.
        np.testing.assert_array_equal(table, np.array([[0, 1, 0, 0, 1]]))

    def test_paired_threshold_uses_sample_standard_deviation(self):
        """Eq. (6) scales by the sample sd of the differences (ddof=1).

        The four differences below have mean 0.02, population sd 0.01 and sample
        sd 0.011547. With d = 1.9 the paper's threshold is 0.021939 -> tie,
        while the population sd would give 0.019 -> win.
        """
        data = pd.DataFrame(
            {
                "dataset": ["d1"] * 4,
                "alg1": [0.51, 0.53, 0.51, 0.53],
                "alg2": [0.50, 0.50, 0.50, 0.50],
            }
        )

        table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=1.9,
            tie_solver="davidson",
            maximize=True,
        )

        np.testing.assert_array_equal(table, np.array([[0, 1, 0, 0, 1]]))

    def test_paired_single_fold_falls_back_to_sign(self):
        """One usable fold leaves no spread, so the sign of the mean decides."""
        data = pd.DataFrame(
            {
                "dataset": ["d1", "d1"],
                "alg1": [0.9, np.nan],
                "alg2": [0.1, 0.2],
            }
        )

        table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col="dataset",
            local_rope_effect_size=0.4,
            tie_solver="davidson",
            maximize=True,
        )

        np.testing.assert_array_equal(table, np.array([[0, 1, 1, 0, 0]]))


class TestUserWarnings:
    """Test whether the correct warnings are raised."""

    def test_unnamed_columns(self):
        """Test whether a warning is raised when the dataset column is unnamed."""
        # Given - This simulated incorrect reading of a CSV file with an index

        CSV_CONTENT = """,alg1,alg2,alg3
        0,0.705,0.696,0.7
        1,0.7,0.7,0.75
        2,0.9,0.8,0.9
        """

        data = pd.read_csv(StringIO(CSV_CONTENT))
        # When / Then

        with pytest.warns(
            UserWarning,
            match="Some algorithm names are unnamed. This may lead to issues in the win table construction.",
        ):
            _construct_win_table(
                data=data,
                data_sd=None,
                dataset_col=None,  # This column is unnamed
                local_rope_effect_size=None,
                absolute_tie_threshold=None,
                tie_solver="davidson",
                maximize=True,
            )


class TestTieSolvers:
    """Test tie solver semantics for spread/add/forget strategies."""

    def test_add_solver_assigns_full_point_per_tie(self):
        """Each tie contributes 1 win to both algorithms in add mode."""
        data = pd.DataFrame(
            {
                "alg1": [0.7],
                "alg2": [0.7],
            }
        )

        add_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            absolute_tie_threshold=0.01,
            tie_solver="add",
            maximize=True,
        )

        expected = np.array([[0, 1, 1, 1, 1]])
        np.testing.assert_array_almost_equal(add_table, expected)

    def test_forget_solver_ignores_ties(self):
        """Forget mode should leave tie counts out of win totals."""
        data = pd.DataFrame(
            {
                "alg1": [0.7],
                "alg2": [0.7],
            }
        )

        forget_table, _ = _construct_win_table(
            data=data,
            data_sd=None,
            dataset_col=None,
            absolute_tie_threshold=0.01,
            tie_solver="forget",
            maximize=True,
        )

        expected = np.array([[0, 1, 0, 0, 1]])
        np.testing.assert_array_almost_equal(forget_table, expected)
