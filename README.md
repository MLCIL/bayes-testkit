# Bayes Test Kit

---

Bayes Test Kit is a Python package for Bayesian Bradley-Terry model along with utilities for multi-algorithm multi-dataset statistical evaluation. It also ships the two Bayesian t-tests of Benavoli et al. (2017) for comparing **two** algorithms: the correlated t-test (one dataset) and the hierarchical correlated t-test (many datasets).

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Bayesian t-tests](#bayesian-t-tests)
- [License](#license)

## Installation

You can install Bayes Test Kit via pip:

```bash
pip install bayes-testkit
```

If needed, you can also install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/scikit-fingerprints/bbt-test
```

## Quickstart

To generate results from BBT model you need to first fit posterior MCMC samples. Bayes Test Kit supports unpaired (1 metric readout per algorithm per dataset) and paired (multiple metric readouts per algorithm per dataset) data.

For hands-on example of using the package, check out our example notebook: [01_simple_bbt_comparison.ipynb](examples/01_simple_bbt_comparison.ipynb).

### Unpaired posterior fitting

Start with single dataframe in shape (n_datasets, n_algorithms), optionally this dataframe can contain a dataset column:

```python
import pandas as pd

df = pd.DataFrame({
    "dataset": ["ds1", "ds2", "ds3"],
    "alg1": [0.8, 0.75, 0.9],
    "alg2": [0.7, 0.8, 0.85],
    "alg3": [0.9, 0.95, 0.88],
})
```

To generate data for BBT model, fit the `BBTTest` model with the dataframe

```python
from btk import BBTTest

model = BBTTest(
    absolute_tie_threshold=0.01, # What counts as a tie, in the units of your metric.
    # Here, a difference in score below 0.01 is a tie. Default is None (no ties).
).fit(
    df,
    dataset_col="dataset", # If dataset column is present, specify it here
)
```

#### Tie parameters
The two are on different scales and are not interchangeable. 
- Use `absolute_tie_threshold` when you have one score per model per dataset. It is a difference in the units of your metric. 
- Use `local_rope_effect_size` when you have repeated scores per dataset (or pass `data_sd`) — it is a Cohen's *d*, a multiple of the observed spread.

#### Evaluating BBT when reporting errors

By default BBT assumes that the goal of the evaluation is to maximize the metric (e.g. when reporting F1 score or AUROC). In cases, when metrics reported in the dataframe should be minimized (e.g. RMSE), you can set the parameter `maximize` in `BBTTest` to False:

```python
model = BBTTest(
    absolute_tie_threshold=0.01,
    maximize=False, # Set to False if the metric should be minimized
).fit(
    df,
    dataset_col="dataset",
)
```

### Paired posterior fitting

BBTTest model supports two variants of input data for paired case, either a single dataframe with multiple rows per algorithm per dataset, or a pair of dataframes, one defining mean performance per algorithm, and the second with standard deviations.

```python
import pandas as pd
from btk import BBTTest

df = pd.DataFrame({
    "dataset": ["ds1", "ds1", "ds1", "ds2", "ds2", "ds2", "ds3", "ds3", "ds3"],
    "alg1": [0.8, 0.82, 0.79, 0.75, 0.77, 0.74, 0.9, 0.91, 0.89],
    "alg2": [0.7, 0.72, 0.69, 0.8, 0.78, 0.81, 0.85, 0.86, 0.84],
    "alg3": [0.9, 0.92, 0.91, 0.95, 0.94, 0.96, 0.88, 0.87, 0.89],
})

model = BBTTest(
    local_rope_effect_size=0.4, # A Cohen's d, not a difference in metric units.
    # With repeated rows a dataset counts as a win only when the mean of the per-fold
    # differences exceeds 0.4 sample standard deviations of those differences
    # (Wainer 2023, Eq. 6). With `data_sd` instead, the spread is pooled across the two
    # models as sqrt((sd_a^2 + sd_b^2) / 2).
).fit(
    df,
    dataset_col="dataset",
)
```

### Generating BBT posterior statistics and interpretations

Once you obtained a fitted BBTTest model, you can generate statistic dataframe containing information about every hypothesis (i.e. every pair of algorithms). The table includes general statistics in form of mean and delta values, as well as probabilities of one algorithm being better than the other, or being tied. Additionally, by default the table contains weak and strong interpretations of the results based on ROPE values.

```python

stats_df = model.posterior_table(
    rope_value=(0.45, 0.55), # Defines ROPE of hypothesis for interpretations
    control_model="alg1", # If provided, only hypotheses comparing to control_model will be included
    selected_models=["alg2"], # If provided, only hypotheses comparing selected_models will be included
)

print(stats_df)

          pair  mean  delta  above_50  in_rope weak_interpretation
0  alg1 > alg2  0.63   0.53      0.75     0.19             Unknown
```

Additionally, you can generate multiple hypothesis interpretations regarding control model for different ROPE values:

```python
stats_df = model.rope_comparison_control_table(
    rope_values=[(0.4, 0.6), (0.45, 0.55), (0.48, 0.52)],
    control_model="alg1",
    interpretation="weak",
)

print(stats_df)

rope_value better_models equivalent_models worse_models unknown_models
0    (0.4, 0.6)                                                  alg3, alg1
1  (0.45, 0.55)                                                  alg3, alg1
2  (0.48, 0.52)                                                  alg3, alg1
```

## Bayesian t-tests

For comparing **two** algorithms, use `CorrelatedTTest` on the cross-validation folds of a single dataset:

```python
import pandas as pd
from btk import CorrelatedTTest

df = pd.DataFrame({
    "fold": [1, 2, 3, 4, 5],
    "alg1": [0.81, 0.83, 0.80, 0.82, 0.84],
    "alg2": [0.78, 0.77, 0.79, 0.76, 0.78],
})

model = CorrelatedTTest(rope=0.01).fit(df, fold_col="fold")
model.decision_table()
```

and `HierarchicalTTest` when the folds come from many datasets:

```python
from btk import HierarchicalTTest

df = pd.DataFrame({
    "dataset": ["ds1"] * 3 + ["ds2"] * 3 + ["ds3"] * 3,
    "fold": [1, 2, 3] * 3,
    "alg1": [0.81, 0.83, 0.80, 0.75, 0.77, 0.74, 0.90, 0.91, 0.89],
    "alg2": [0.78, 0.77, 0.79, 0.76, 0.75, 0.73, 0.85, 0.86, 0.84],
})

model = HierarchicalTTest(rope=0.01).fit(df, dataset_col="dataset", fold_col="fold")
model.decision_table()
```

Unlike BBT, `rope` here is a difference in the units of your metric.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
