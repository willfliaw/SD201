from typing import List

import numpy as np


def precision_recall(
    expected_results: List[bool], actual_results: List[bool]
) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    precision = np.sum(np.logical_and(actual_results, expected_results)) / np.sum(
        actual_results
    )
    recall = np.sum(np.logical_and(actual_results, expected_results)) / np.sum(
        expected_results
    )

    return precision, recall


def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    precision, recall = precision_recall(expected_results, actual_results)
    return 2 * (precision * recall) / (precision + recall)
