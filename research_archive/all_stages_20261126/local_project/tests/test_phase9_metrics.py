import numpy as np

from scripts.analyze_phase9_errors import average_precision


def test_average_precision_perfect_ranking() -> None:
    assert average_precision(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8])) == 1.0


def test_average_precision_tied_scores_and_no_positive() -> None:
    assert average_precision(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert average_precision(np.array([0, 0]), np.array([0.4, 0.2])) is None
