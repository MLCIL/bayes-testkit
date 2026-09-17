"""Golden fidelity test: reproduce the worked example of Wainer (2023).

The paper prints its "base results" in full -- 5 classifiers on 20 data sets
(Table 2) -- along with everything derived from them: the raw win/tie/loss table
(Table 3a), the table after the ``spread`` tie policy (Table 3b), and the
posterior summary (Table 4). That makes the example a complete, published
end-to-end fixture, and the only one available for checking that this package
computes the same thing the article does.

One caveat shapes how the checks are written. Table 2 is printed to three
decimals, and several comparisons are decided by digits that rounding has
discarded: ``lgbm`` and ``xgb`` are printed equal on six data sets, but the
paper's Table 3a records only three ties for that pair. So the printed data
determines the win table only for pairs it shows no ties for. The win-table test
therefore asserts exact equality where the published numbers determine the
answer, and containment where they do not -- every win the paper records must
come from a comparison the printed data either decides the same way or cannot
decide at all.

The posterior check side-steps the issue entirely by feeding the paper's *own*
Table 3b counts into the model, so Table 4 is reproduced from the same input the
paper used.

Reference
---------
Jacques Wainer, "A Bayesian Bradley-Terry model to compare multiple ML
algorithms on multiple data sets", JMLR 24 (2023): 1-34.
http://jmlr.org/papers/v24/22-0907.html
"""

import numpy as np
import pandas as pd
import pytest

from btk import BBTTest
from btk.tests.bbt.alg import _construct_win_table
from btk.tests.bbt.model import _mcmcbbt_pymc

ALGORITHMS = ["dt", "lda", "lgbm", "xgb", "svm"]

# Table 2: mean accuracy of the same 4-fold evaluation on each data set.
BASE_RESULTS = pd.DataFrame(
    [
        ("biomed", 0.837, 0.842, 0.876, 0.890, 0.886),
        ("breast", 0.931, 0.951, 0.964, 0.961, 0.957),
        ("breast_w", 0.940, 0.950, 0.961, 0.961, 0.961),
        ("buggyCrx", 0.790, 0.861, 0.867, 0.867, 0.861),
        ("clean1", 1.000, 1.000, 1.000, 1.000, 0.968),
        ("cmc", 0.455, 0.513, 0.525, 0.524, 0.544),
        ("colic", 0.761, 0.837, 0.815, 0.815, 0.641),
        ("corral", 1.000, 0.900, 1.000, 1.000, 1.000),
        ("credit_g", 0.668, 0.718, 0.766, 0.769, 0.724),
        ("diabetes", 0.714, 0.772, 0.747, 0.742, 0.758),
        ("ionosphere", 0.869, 0.866, 0.940, 0.932, 0.934),
        ("irish", 1.000, 0.740, 1.000, 1.000, 0.988),
        ("molecular_biology_promoters", 0.727, 0.689, 0.896, 0.887, 0.802),
        ("monk3", 0.975, 0.792, 0.980, 0.986, 0.964),
        ("prnn_crabs", 0.880, 1.000, 0.950, 0.935, 0.960),
        ("prnn_synth", 0.800, 0.852, 0.824, 0.828, 0.856),
        ("saheart", 0.626, 0.723, 0.660, 0.671, 0.712),
        ("threeOf9", 0.996, 0.809, 1.000, 0.998, 0.992),
        ("tokyo1", 0.902, 0.920, 0.928, 0.926, 0.931),
        ("vote", 0.929, 0.956, 0.945, 0.959, 0.956),
    ],
    columns=["dataset", *ALGORITHMS],
)

# Table 3a: wins for the first algorithm, wins for the second, and ties, before
# any tie policy is applied. Keyed by (alg1, alg2) in the paper's order.
TABLE_3A = {
    ("dt", "lda"): (6, 13, 1),
    ("dt", "lgbm"): (0, 17, 3),
    ("dt", "xgb"): (0, 17, 3),
    ("dt", "svm"): (5, 14, 1),
    ("lda", "lgbm"): (6, 13, 1),
    ("lda", "xgb"): (5, 14, 1),
    ("lda", "svm"): (5, 15, 0),
    ("lgbm", "xgb"): (9, 8, 3),
    ("lgbm", "svm"): (10, 9, 1),
    ("xgb", "svm"): (11, 8, 1),
}

# Table 3b: the same table once the ties have been added as half-victories,
# rounded up, to both algorithms.
TABLE_3B = {
    ("dt", "lda"): (7, 14),
    ("dt", "lgbm"): (2, 19),
    ("dt", "xgb"): (2, 19),
    ("dt", "svm"): (6, 15),
    ("lda", "lgbm"): (7, 14),
    ("lda", "xgb"): (6, 15),
    ("lda", "svm"): (5, 15),
    ("lgbm", "xgb"): (11, 10),
    ("lgbm", "svm"): (11, 10),
    ("xgb", "svm"): (12, 9),
}

# Table 4: mean P(a > b), the 89% HDI, above.50 and in.rope for ROPE
# [0.45, 0.55]. Pairs are printed best-first, as the package orders them.
TABLE_4 = {
    "xgb > lgbm": {
        "mean": 0.51,
        "low": 0.40,
        "high": 0.63,
        "above_50": 0.55,
        "in_rope": 0.52,
    },
    "xgb > svm": {
        "mean": 0.56,
        "low": 0.45,
        "high": 0.68,
        "above_50": 0.80,
        "in_rope": 0.36,
    },
    "xgb > lda": {
        "mean": 0.72,
        "low": 0.62,
        "high": 0.82,
        "above_50": 1.00,
        "in_rope": 0.00,
    },
    "xgb > dt": {
        "mean": 0.83,
        "low": 0.76,
        "high": 0.91,
        "above_50": 1.00,
        "in_rope": 0.00,
    },
    "lgbm > svm": {
        "mean": 0.55,
        "low": 0.45,
        "high": 0.67,
        "above_50": 0.77,
        "in_rope": 0.40,
    },
    "lgbm > lda": {
        "mean": 0.71,
        "low": 0.62,
        "high": 0.81,
        "above_50": 1.00,
        "in_rope": 0.01,
    },
    "lgbm > dt": {
        "mean": 0.83,
        "low": 0.75,
        "high": 0.90,
        "above_50": 1.00,
        "in_rope": 0.00,
    },
    "svm > lda": {
        "mean": 0.66,
        "low": 0.57,
        "high": 0.77,
        "above_50": 0.99,
        "in_rope": 0.05,
    },
    "svm > dt": {
        "mean": 0.79,
        "low": 0.71,
        "high": 0.88,
        "above_50": 1.00,
        "in_rope": 0.00,
    },
    "lda > dt": {
        "mean": 0.66,
        "low": 0.55,
        "high": 0.77,
        "above_50": 0.98,
        "in_rope": 0.06,
    },
}


def _win_table_as_dict(table: np.ndarray, names: list[str]) -> dict:
    return {
        (names[int(row[0])], names[int(row[1])]): (
            int(row[2]),
            int(row[3]),
            int(row[4]),
        )
        for row in table
    }


@pytest.fixture(scope="module")
def raw_table() -> dict:
    """Build the win/tie/loss table this package derives from the printed Table 2."""
    table, names = _construct_win_table(
        data=BASE_RESULTS,
        data_sd=None,
        dataset_col="dataset",
        tie_solver="davidson",  # keeps the tie column intact
        maximize=True,
    )
    return _win_table_as_dict(table, names)


class TestWinTable:
    """The win table is deterministic given the data, so it is checked exactly."""

    def test_every_pair_is_present(self, raw_table):
        """All ten pairs of the five classifiers are compared."""
        assert set(raw_table) == set(TABLE_3A)

    @pytest.mark.parametrize("pair", list(TABLE_3A))
    def test_all_twenty_data_sets_are_accounted_for(self, raw_table, pair):
        """Wins, losses and ties must sum to the 20 data sets."""
        assert sum(raw_table[pair]) == len(BASE_RESULTS)

    @pytest.mark.parametrize("pair", list(TABLE_3A))
    def test_table_3a_matches_where_the_printed_data_decides(self, raw_table, pair):
        """Exact agreement where Table 2 has the precision to decide the pair.

        Where it does not, every tie the printed data leaves undecided must be
        one the paper resolved: the paper's wins can exceed ours only by taking
        from our tie count, never by contradicting a decided comparison.
        """
        w1, w2, ties = raw_table[pair]
        p_w1, p_w2, p_ties = TABLE_3A[pair]

        if ties == 0:
            assert (w1, w2, ties) == (p_w1, p_w2, p_ties), (
                "no ties in the printed data, so the pair is fully determined"
            )
            return

        assert p_ties <= ties, (
            "an exact tie at full precision is still a tie once rounded, so the "
            "paper cannot have more ties than the printed data does"
        )
        assert w1 <= p_w1 <= w1 + ties
        assert w2 <= p_w2 <= w2 + ties
        # The extra wins the paper records are exactly the ties it could resolve.
        assert (p_w1 - w1) + (p_w2 - w2) == ties - p_ties

    def test_spread_policy_reproduces_table_3b(self):
        """``spread`` turns the paper's Table 3a into the paper's Table 3b."""
        for pair, (w1, w2, ties) in TABLE_3A.items():
            half = -(-ties // 2)  # ceil(ties / 2), "rounded up in the final"
            assert (w1 + half, w2 + half) == TABLE_3B[pair], pair


@pytest.mark.slow
class TestPosteriorTable:
    """Table 4, reproduced from the paper's own Table 3b win counts.

    Feeding the published counts in directly removes the rounding loss in
    Table 2, so any disagreement here is the model's, not the fixture's.
    """

    @pytest.fixture(scope="class")
    def posterior(self):
        """Fit the model on the paper's Table 3b counts and summarise it."""
        pairs = [(a, b) for i, a in enumerate(ALGORITHMS) for b in ALGORITHMS[i + 1 :]]
        win_table = np.array(
            [
                [
                    ALGORITHMS.index(a),
                    ALGORITHMS.index(b),
                    TABLE_3B[(a, b)][0],
                    TABLE_3B[(a, b)][1],
                    0,
                ]
                for a, b in pairs
            ],
            dtype=np.int32,
        )

        model = BBTTest(tie_solver="spread")
        model._win_table = win_table
        model._algorithms = list(ALGORITHMS)
        model._fit_posterior, model._pymc_model = _mcmcbbt_pymc(
            table=win_table,
            use_davidson=False,
            hyper_prior="log_normal",
            scale=1.0,
            draws=4000,
            tune=2000,
            chains=4,
            random_seed=20230907,
            progressbar=False,
        )
        model._idata = model._fit_posterior
        model._fitted = True

        return model.posterior_table(
            rope_value=(0.45, 0.55),
            columns=("mean", "hdi_low", "hdi_high", "above_50", "in_rope"),
            round_ndigits=None,
        ).set_index("pair")

    def test_pairs_and_ordering(self, posterior):
        """The aggregated ranking, and so the pair ordering, matches the paper."""
        assert list(posterior.index) == list(TABLE_4)

    @pytest.mark.parametrize("pair", list(TABLE_4))
    def test_row_matches_the_paper(self, posterior, pair):
        """Each summary statistic lands within Monte Carlo error of Table 4."""
        expected = TABLE_4[pair]
        row = posterior.loc[pair]
        assert row["mean"] == pytest.approx(expected["mean"], abs=0.02)
        assert row["hdi_low"] == pytest.approx(expected["low"], abs=0.03)
        assert row["hdi_high"] == pytest.approx(expected["high"], abs=0.03)
        assert row["above_50"] == pytest.approx(expected["above_50"], abs=0.05)
        assert row["in_rope"] == pytest.approx(expected["in_rope"], abs=0.05)
