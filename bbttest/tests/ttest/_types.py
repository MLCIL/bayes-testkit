"""Allowed option sets for the t-test estimators.

Following the scikit-learn style, categorical hyperparameters are typed as plain
``str`` and validated explicitly against these option tuples (via
``bbttest.tests.common.validate_string``) rather than encoded as ``Literal``
types.
"""

# Prior on the degrees-of-freedom parameter of the hierarchical model
# (Benavoli et al. 2017, Section 4.3.1; Corani et al. 2017). ``hierarchical``
# places uniform hyper-priors on the Gamma shape/rate, and is the paper's
# recommended default.
DOF_PRIORS = ("hierarchical", "kruschke", "juarez_steel")

# Role-level skeleton of ``decision_table`` (fixed across tests) plus the
# probability columns each t-test appends. ``comparison``/``estimate``/
# ``hdi_low``/``hdi_high``/``decision`` are the common ground; ``p_left``/
# ``p_rope``/``p_right`` form the three-way partition (summing to one) that the
# t-tests own but BBT does not.
TTEST_COLUMNS = (
    "comparison",
    "estimate",
    "hdi_low",
    "hdi_high",
    "p_left",
    "p_rope",
    "p_right",
    # Hierarchical test only: the posterior means of the next-dataset triple,
    # kept alongside the simplex-region probabilities in p_left/p_rope/p_right.
    "mean_theta_left",
    "mean_theta_rope",
    "mean_theta_right",
    "decision",
    "decision_raw",
)

# Natural plot for each test, dispatched through ``plot(kind=...)``.
CORRELATED_PLOT_KINDS = ("posterior", "hdi")
HIERARCHICAL_PLOT_KINDS = ("simplex", "forest", "ppc")

ALL_TTEST_COLUMNS: list[str] = list(TTEST_COLUMNS)
