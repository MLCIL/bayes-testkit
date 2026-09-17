import matplotlib.pyplot as plt
import pandas as pd
import pytest

from btk.tests.bbt.plots._critical_difference import (
    _plot_cdd_diagram,
    assign_bar_position,
    get_bars_for_cdd,
)


@pytest.fixture
def models_df() -> pd.DataFrame:
    """Create a simple models DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["A", "B", "C", "D"],
            "pos": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def posterior_df() -> pd.DataFrame:
    """Create a posterior DataFrame with a known equivalence structure.

    A, B, C form a clique (all equivalent); D is isolated.
    """
    return pd.DataFrame(
        [
            {"left_model": "A", "right_model": "B", "interp": "="},
            {"left_model": "B", "right_model": "C", "interp": "="},
            {"left_model": "A", "right_model": "C", "interp": "="},
            {"left_model": "C", "right_model": "D", "interp": "<"},
        ]
    )


class TestGetBarsForCDD:
    """Test equivalence bar extraction from the posterior table."""

    def test_single_clique(
        self, models_df: pd.DataFrame, posterior_df: pd.DataFrame
    ) -> None:
        """Test that a single equivalence clique produces one bar spanning the correct positions."""
        bars = get_bars_for_cdd(
            posterior_df=posterior_df,
            models_df=models_df,
            interpretation_col="interp",
        )
        # Only one equivalence group: models A (pos 1), B (pos 2), C (pos 3)
        # so we expect a single bar spanning from 1 to 3.
        assert len(bars) == 1
        assert bars[0] == (1, 3)


class TestAssignBarPosition:
    """Test vertical bar positioning for the CDD plot."""

    def test_non_overlapping(self) -> None:
        """Check that non-overlapping bars are placed on the same row."""
        bars = [(0, 1), (2, 3), (4, 5)]
        positions = assign_bar_position(bars, min_distance=0)
        # All bars are disjoint; the greedy algorithm should be able to place
        # them all on the same row.
        assert len(positions) == len(bars)
        assert set(positions) == {0}

    def test_overlapping(self) -> None:
        """Check that overlapping bars are not placed on the same row."""
        # Bar 0 overlaps with bar 1, bar 1 overlaps with bar 2
        bars = [(0, 3), (2, 5), (4, 7)]
        positions = assign_bar_position(bars, min_distance=0)
        assert len(positions) == len(bars)
        # At least two rows are required for these overlapping intervals.
        assert max(positions) >= 1
        # Overlapping bars should not share the same row id.
        for i in range(len(bars)):
            for j in range(i + 1, len(bars)):
                s1, e1 = bars[i]
                s2, e2 = bars[j]
                if not (e1 <= s2 or e2 <= s1):
                    # Bars i and j overlap; they must be on different rows.
                    assert positions[i] != positions[j]


class TestPlotCDDDiagram:
    """Test the CDD diagram plotting function."""

    def test_smoke(self) -> None:
        """Ensure _plot_cdd_diagram runs without error and returns an Axes."""
        models_df = pd.DataFrame(
            {
                "model": ["A", "B", "C"],
                "pos": [1, 2, 3],
                "mean": [0.1, 0.2, 0.3],
            }
        )
        # A single bar spanning all three models on row 0
        bars = [(1, 3)]
        bars_positions = [0]
        fig, ax = plt.subplots()
        try:
            result_ax = _plot_cdd_diagram(
                models_df=models_df,
                bars=bars,
                bars_positions=bars_positions,
                ax=ax,
            )
        finally:
            plt.close(fig)
        assert isinstance(result_ax, plt.Axes)
