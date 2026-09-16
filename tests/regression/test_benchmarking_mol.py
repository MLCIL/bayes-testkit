"""
Regression tests for BBTTest model using molecular embeddings benchmarking data.

This test suite validates the BBTTest model's weak interpretation results against
the ECFP baseline using molecular embeddings benchmarking data from the study
of pretrained molecular embedding models.

Notes
-----
The test data is adapted from the benchmarking study:

Praski, Mateusz, Jakub Adamczyk, and Wojciech Czech.
"Benchmarking pretrained molecular embedding models for molecular representation learning."
arXiv preprint arXiv:2508.06199 (2025).
https://arxiv.org/pdf/2508.06199

Data source:
https://github.com/scikit-fingerprints/benchmarking_molecular_models/blob/31779f16c004b3fb8aa555ecceb6b95ca71a1d7d/results/arxiv_preprint_2025_08.csv

The tests validate that the BBTTest model correctly identifies:
- Better performing models (vs ECFP baseline)
- Equivalent performing models (within ROPE)
- Unknown comparisons (insufficient evidence)
- Worse performing models

Test parameters:
- absolute_tie_threshold: 0.01 (one score per dataset)
- tie_solver: "add"
- MCMC sampling: 2000 draws, 1000 tune, 4 chains
"""

from pathlib import Path

import pandas as pd
import pytest

from bbttest import BBTTest


@pytest.fixture(scope="module")
def benchmarking_data():
    """
    Load benchmarking molecular data.

    Returns
    -------
    pd.DataFrame
        Molecular embeddings benchmarking results with columns for dataset
        and various model scores.
    """
    data_path = Path(__file__).parent.parent / "data" / "benchmarking_mol.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def fitted_model(benchmarking_data):
    """
    Fit BBTTest model with absolute_tie_threshold=0.01.

    Parameters
    ----------
    benchmarking_data : pd.DataFrame
        Benchmarking molecular data fixture.

    Returns
    -------
    BBTTest
        Fitted BBTTest model instance.
    """
    model = BBTTest(absolute_tie_threshold=0.01, tie_solver="add")
    model.fit(
        benchmarking_data,
        dataset_col="dataset",
        draws=2000,
        tune=1000,
        chains=4,
        random_seed=42,
    )
    return model


def _extract_interpretations(results):
    """
    Extract model interpretations from posterior table results.

    Parameters
    ----------
    results : pd.DataFrame
        Posterior table results with comparisons against ECFP_count.

    Returns
    -------
    dict
        Dictionary mapping model names to their weak interpretations.
    """
    interpretations = {}
    for _, row in results.iterrows():
        if row["left_model"] != "ECFP_count":
            interpretations[row["left_model"]] = row["weak_interpretation"]
        if row["right_model"] != "ECFP_count":
            # Invert interpretation when ECFP is on the left
            if row["weak_interpretation"] == f"{row['left_model']} better":
                interpretations[row["right_model"]] = "ECFP better"
            else:
                interpretations[row["right_model"]] = row["weak_interpretation"]
    return interpretations


# ---- SCORES FROM THE ARTICLE ----
ROPE_1_VALUE = (0.45, 0.55)
ROPE_1_BETTER_MODELS = ["CLAMP", "rmat_4M"]
ROPE_1_EQUIVALENT_MODELS = []
ROPE_1_UNKNOWN_MODELS = [
    "AtomPair_count",
    "CDDD",
    "ChemBERTa-10M-MTR",
    "mat_masking_2M",
    "molbert",
]
ROPE_1_WORSE_MODELS = [
    "ChemFM-3B",
    "ChemGPT-4.7M",
    "GEM",
    "GNN-GraphCL-sum",
    "GraphFP-CP",
    "GraphMVP_CP-max",
    "MoLFormer-XL-both-10pct",
    "SELFormer-Lite",
    "SimSon",
    "TT",
    "chemformer_mask",
    "coati",
    "grover_large",
    "mol2vec",
    "mol_r_tag_1024",
    "unimolv1",
    "unimolv2",
]

ROPE_2_VALUE = (0.4, 0.6)
ROPE_2_BETTER_MODELS = ["CLAMP", "rmat_4M"]
ROPE_2_EQUIVALENT_MODELS = [
    "AtomPair_count",
    "CDDD",
    "ChemBERTa-10M-MTR",
    "mat_masking_2M",
    "molbert",
]
ROPE_2_UNKNOWN_MODELS = []
ROPE_2_WORSE_MODELS = [
    "ChemFM-3B",
    "ChemGPT-4.7M",
    "GEM",
    "GNN-GraphCL-sum",
    "GraphFP-CP",
    "GraphMVP_CP-max",
    "MoLFormer-XL-both-10pct",
    "SELFormer-Lite",
    "SimSon",
    "TT",
    "chemformer_mask",
    "coati",
    "grover_large",
    "mol2vec",
    "mol_r_tag_1024",
    "unimolv1",
    "unimolv2",
]

ROPE_3_VALUE = (0.35, 0.65)
ROPE_3_BETTER_MODELS = ["CLAMP"]
ROPE_3_EQUIVALENT_MODELS = [
    "AtomPair_count",
    "CDDD",
    "ChemBERTa-10M-MTR",
    "MoLFormer-XL-both-10pct",
    "mat_masking_2M",
    "molbert",
    "rmat_4M",
]
ROPE_3_UNKNOWN_MODELS = []
ROPE_3_WORSE_MODELS = [
    "ChemFM-3B",
    "ChemGPT-4.7M",
    "GEM",
    "GNN-GraphCL-sum",
    "GraphFP-CP",
    "GraphMVP_CP-max",
    "SELFormer-Lite",
    "SimSon",
    "TT",
    "chemformer_mask",
    "coati",
    "grover_large",
    "mol2vec",
    "mol_r_tag_1024",
    "unimolv1",
    "unimolv2",
]

ROPE_4_VALUE = (0.3, 0.7)
ROPE_4_BETTER_MODELS = []
ROPE_4_EQUIVALENT_MODELS = [
    "AtomPair_count",
    "CDDD",
    "CLAMP",
    "ChemBERTa-10M-MTR",
    "MoLFormer-XL-both-10pct",
    "mat_masking_2M",
    "mol2vec",
    "molbert",
    "rmat_4M",
]
ROPE_4_UNKNOWN_MODELS = []
ROPE_4_WORSE_MODELS = [
    "ChemFM-3B",
    "ChemGPT-4.7M",
    "GEM",
    "GNN-GraphCL-sum",
    "GraphFP-CP",
    "GraphMVP_CP-max",
    "SELFormer-Lite",
    "SimSon",
    "TT",
    "chemformer_mask",
    "coati",
    "grover_large",
    "mol_r_tag_1024",
    "unimolv1",
    "unimolv2",
]


class TestWeakInterpretationAgainstECFP:
    """Test weak interpretation results against ECFP baseline for different ROPE values."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "rope,better_models,equivalent_models,unknown_models,worse_models",
        [
            (
                ROPE_1_VALUE,
                ROPE_1_BETTER_MODELS,
                ROPE_1_EQUIVALENT_MODELS,
                ROPE_1_UNKNOWN_MODELS,
                ROPE_1_WORSE_MODELS,
            ),
            (
                ROPE_2_VALUE,
                ROPE_2_BETTER_MODELS,
                ROPE_2_EQUIVALENT_MODELS,
                ROPE_2_UNKNOWN_MODELS,
                ROPE_2_WORSE_MODELS,
            ),
            (
                ROPE_3_VALUE,
                ROPE_3_BETTER_MODELS,
                ROPE_3_EQUIVALENT_MODELS,
                ROPE_3_UNKNOWN_MODELS,
                ROPE_3_WORSE_MODELS,
            ),
            (
                ROPE_4_VALUE,
                ROPE_4_BETTER_MODELS,
                ROPE_4_EQUIVALENT_MODELS,
                ROPE_4_UNKNOWN_MODELS,
                ROPE_4_WORSE_MODELS,
            ),
        ],
        ids=["rope_0.45_0.55", "rope_0.4_0.6", "rope_0.35_0.65", "rope_0.3_0.7"],
    )
    def test_weak_interpretation_for_rope(
        self,
        fitted_model,
        rope,
        better_models,
        equivalent_models,
        unknown_models,
        worse_models,
    ):
        """
        Test weak interpretation results for different ROPE values.

        Parameters
        ----------
        fitted_model : BBTTest
            Fitted BBTTest model fixture.
        rope : tuple of float
            Region of Practical Equivalence (ROPE) bounds.
        better_models : list of str
            Models expected to be better than ECFP.
        equivalent_models : list of str
            Models expected to be equivalent to ECFP.
        unknown_models : list of str
            Models with unknown comparison to ECFP.
        worse_models : list of str
            Models expected to be worse than ECFP.
        """
        results = fitted_model.posterior_table(
            rope_value=rope,
            control_model="ECFP_count",
            columns=["left_model", "right_model", "weak_interpretation"],
        )

        interpretations = _extract_interpretations(results)

        # Validate better models
        for model in better_models:
            assert interpretations[model] == f"{model} better", (
                f"Model {model} should be better than ECFP for ROPE {rope}"
            )

        # Validate equivalent models
        for model in equivalent_models:
            assert interpretations[model] == "Equivalent", (
                f"Model {model} should be equivalent to ECFP for ROPE {rope}"
            )

        # Validate unknown models
        for model in unknown_models:
            assert interpretations[model] == "Unknown", (
                f"Model {model} should have unknown comparison with ECFP for ROPE {rope}"
            )

        # Validate worse models
        for model in worse_models:
            assert interpretations[model] == "ECFP better", (
                f"Model {model} should be worse than ECFP for ROPE {rope}"
            )

    @pytest.mark.slow
    def test_multiple_rope_table(
        self,
        fitted_model,
    ):
        """
        Test posterior table with multiple ROPE values.

        Parameters
        ----------
        fitted_model : BBTTest
            Fitted BBTTest model fixture.
        """
        # Given
        expected_dataframe = pd.DataFrame(
            {
                "rope_value": [
                    ROPE_1_VALUE,
                    ROPE_2_VALUE,
                    ROPE_3_VALUE,
                    ROPE_4_VALUE,
                ],
                "better_models": [
                    sorted(ROPE_1_BETTER_MODELS),
                    sorted(ROPE_2_BETTER_MODELS),
                    sorted(ROPE_3_BETTER_MODELS),
                    sorted(ROPE_4_BETTER_MODELS),
                ],
                "equivalent_models": [
                    sorted(ROPE_1_EQUIVALENT_MODELS),
                    sorted(ROPE_2_EQUIVALENT_MODELS),
                    sorted(ROPE_3_EQUIVALENT_MODELS),
                    sorted(ROPE_4_EQUIVALENT_MODELS),
                ],
                "unknown_models": [
                    sorted(ROPE_1_UNKNOWN_MODELS),
                    sorted(ROPE_2_UNKNOWN_MODELS),
                    sorted(ROPE_3_UNKNOWN_MODELS),
                    sorted(ROPE_4_UNKNOWN_MODELS),
                ],
                "worse_models": [
                    sorted(ROPE_1_WORSE_MODELS),
                    sorted(ROPE_2_WORSE_MODELS),
                    sorted(ROPE_3_WORSE_MODELS),
                    sorted(ROPE_4_WORSE_MODELS),
                ],
            }
        )

        # When

        results = fitted_model.rope_comparison_control_table(
            rope_values=(ROPE_1_VALUE, ROPE_2_VALUE, ROPE_3_VALUE, ROPE_4_VALUE),
            control_model="ECFP_count",
            return_as_array=True,  # Return as array as order does not matter and we need to sort
        )
        for column in [
            "better_models",
            "equivalent_models",
            "unknown_models",
            "worse_models",
        ]:
            results[column] = results[column].apply(sorted)

        # Then
        # Order columns to match expected dataframe, ignore index
        results = results[expected_dataframe.columns]
        pd.testing.assert_frame_equal(
            results.reset_index(drop=True),
            expected_dataframe.reset_index(drop=True),
        )
