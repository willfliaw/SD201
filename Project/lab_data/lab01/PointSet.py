from enum import Enum
from typing import List, Tuple

import numpy as np


def most_frequent_occurence(labels: List[bool]) -> bool:
    """Gets the most frequent occurence of a boolean array."""
    return np.sum(labels) > np.sum(~labels)


class FeaturesTypes(Enum):
    """Enumerate possible features types."""

    BOOLEAN = 0
    CLASSES = 1
    REAL = 2


class PointSet:
    """
    A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
        bestFeatureId : int = None
            The id of the best feature to split on.
        bestFeatureSplitValue : float = None
            The best splitting value of the best feature to split on.

    """

    def __init__(
        self,
        features: List[List[float]],
        labels: List[bool],
        types: List[FeaturesTypes],
    ):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        outputClass : bool
            The most frequent occurent class corresponding to the initial
            set of features and labels.
        """
        self.features = np.array(features)
        self.types = types
        self.labels = np.array(labels, dtype=bool)
        self.bestFeatureId = None
        self.bestFeatureSplitValue = None
        self.outputClass = most_frequent_occurence(self.labels)

    def get_gini(self) -> float:
        """
        Computes the Gini score of the set of points.

        Returns
        -------
        float
            The Gini score of the set of points.
        """

        return 1 - (np.sum(self.labels) ** 2 + np.sum(~self.labels) ** 2) / (
            self.labels.shape[0] ** 2
        )

    def get_best_gain(self, min_split_points=1) -> Tuple[int, float]:
        """
        Compute the feature along which splitting provides the best
        gain.

        Parameters
        ----------
        min_split_points : int (default 1)
            Minimum number of points associated with a feasible child
            node.

        Returns
        -------
        int
            The ID of the feature along which splitting the set
            provides the best Gini gain.
        float
            The best Gini gain achievable by splitting this set along
            one of its features.
        """
        bestGiniGain = 0
        selfGini = self.get_gini()

        for columnIndex, (column, columnType) in enumerate(
            zip(self.features.T, self.types)
        ):
            uniqueValues, uniqueValuesCounts = np.unique(column, return_counts=True)
            positivesUniqueValue = np.array(
                [
                    np.sum(self.labels[column == uniqueValue])
                    for uniqueValue in uniqueValues
                ]
            )

            if uniqueValues.shape[0] == 2:
                partitionsGiniSplit = (
                    column.shape[0]
                    - (
                        positivesUniqueValue[0] ** 2
                        + (uniqueValuesCounts[0] - positivesUniqueValue[0]) ** 2
                    )
                    / uniqueValuesCounts[0]
                    - (
                        positivesUniqueValue[1] ** 2
                        + (uniqueValuesCounts[1] - positivesUniqueValue[1]) ** 2
                    )
                    / uniqueValuesCounts[1]
                ) / column.shape[0]

                partitionsGiniGain = (selfGini - partitionsGiniSplit) * np.all(
                    (uniqueValuesCounts >= min_split_points), axis=0
                )

                giniGain = partitionsGiniGain

            if uniqueValues.shape[0] > 2:
                allPositives = np.sum(self.labels)

                if columnType == FeaturesTypes.REAL:
                    uniqueValuesCounts = np.cumsum(uniqueValuesCounts[:-1])
                    positivesUniqueValue = np.cumsum(positivesUniqueValue[:-1])

                partitionsGiniSplit = (
                    np.sum(
                        np.vstack(
                            (
                                np.full_like(uniqueValuesCounts, column.shape[0]),
                                -np.divide(
                                    np.power(positivesUniqueValue, 2)
                                    + np.power(
                                        uniqueValuesCounts - positivesUniqueValue, 2
                                    ),
                                    uniqueValuesCounts,
                                ),
                                -np.divide(
                                    np.power(allPositives - positivesUniqueValue, 2)
                                    + np.power(
                                        (column.shape[0] - uniqueValuesCounts)
                                        - (allPositives - positivesUniqueValue),
                                        2,
                                    ),
                                    (column.shape[0] - uniqueValuesCounts),
                                ),
                            )
                        ),
                        axis=0,
                    )
                    / column.shape[0]
                )

                partitionsGiniGain = (selfGini - partitionsGiniSplit) * np.all(
                    np.vstack(
                        (
                            uniqueValuesCounts,
                            column.shape[0] - uniqueValuesCounts,
                        )
                    )
                    >= min_split_points,
                    axis=0,
                )

                giniGain = np.max(partitionsGiniGain)

            if uniqueValues.shape[0] >= 2 and giniGain > bestGiniGain:
                bestGiniGain = giniGain
                self.bestFeatureId = columnIndex
                if self.types[self.bestFeatureId] == FeaturesTypes.BOOLEAN:
                    self.bestFeatureSplitValue = uniqueValues[0]
                elif self.types[self.bestFeatureId] == FeaturesTypes.CLASSES:
                    self.bestFeatureSplitValue = uniqueValues[
                        np.argmax(partitionsGiniGain)
                    ]
                elif self.types[self.bestFeatureId] == FeaturesTypes.REAL:
                    self.bestFeatureSplitValue = np.mean(
                        uniqueValues[
                            uniqueValues
                            <= uniqueValues[np.argmax(partitionsGiniGain) + 1]
                        ][-2:]
                    )

        if bestGiniGain == 0:
            return None, None
        return self.bestFeatureId, bestGiniGain

    def get_best_threshold(self) -> float:
        """
        Computes a refinement of the split value (needs get_best_gain
        to be called once before).
        """
        if self.bestFeatureId != None:
            if self.types[self.bestFeatureId] == FeaturesTypes.BOOLEAN:
                return None
            return self.bestFeatureSplitValue
