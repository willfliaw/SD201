from typing import List

import numpy as np
from PointSet import FeaturesTypes, PointSet


class Tree:
    """
    A binary splitted decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree.
        childs : Tuple
            A tuple of childs of the tree. On the first child, at index
            0, there is only one unique value of the best splitting
            feature for boolean and categorical features; there is only
            values smaller than the threshold for continous features.
        h : int (default 1)
            The maximum height of the tree.
        min_split_points : int (default 1)
            Minimum number of points associated with a feasible
                child node.
        beta : float (default 0)
            Minimum ratio of the quantity of set of examples to update the
            tree.
        changes : int
            Number of updates done after the last (re)built.
    """

    def __init__(
        self,
        features: List[List[float]],
        labels: List[bool],
        types: List[FeaturesTypes],
        h: int = 1,
        min_split_points: int = 1,
        beta: int = 0,
        changes: int = 0,
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
                The labels of the training points.
            types : List[FeaturesTypes]
                The types of the features.
            h : int (default 1)
                The maximum height of the tree.
            min_split_points : int (default 1)
                Minimum number of points associated with a feasible
                child node.
            beta : float (default 0)
                Minimum ratio of the quantity of set of examples to update the
                tree.
            changes : int (default 0)
                Number of updates done after the last (re)built.
        """
        self.points = PointSet(features, labels, types)
        self.childs = ()
        self.h = h
        self.min_split_points = min_split_points
        self.beta = beta
        self.changes = changes

        self.points.get_best_gain(self.min_split_points)
        if self.points.bestFeatureId != None:
            if self.points.types[self.points.bestFeatureId] in [
                FeaturesTypes.BOOLEAN,
                FeaturesTypes.CLASSES,
            ]:
                splitValueMask = (
                    self.points.features[:, self.points.bestFeatureId]
                    == self.points.bestFeatureSplitValue
                )
            elif self.points.types[self.points.bestFeatureId] == FeaturesTypes.REAL:
                splitValueMask = (
                    self.points.features[:, self.points.bestFeatureId]
                    < self.points.get_best_threshold()
                )

            self.childs = (
                [
                    Tree(
                        self.points.features[splitValueMask],
                        self.points.labels[splitValueMask],
                        self.points.types,
                        self.h - 1,
                        self.min_split_points,
                        self.beta,
                        0,
                    ),
                    Tree(
                        self.points.features[np.logical_not(splitValueMask)],
                        self.points.labels[np.logical_not(splitValueMask)],
                        self.points.types,
                        self.h - 1,
                        self.min_split_points,
                        self.beta,
                        0,
                    ),
                ]
                if h > 1
                else [
                    PointSet(
                        self.points.features[splitValueMask],
                        self.points.labels[splitValueMask],
                        self.points.types,
                    ),
                    PointSet(
                        self.points.features[np.logical_not(splitValueMask)],
                        self.points.labels[np.logical_not(splitValueMask)],
                        self.points.types,
                    ),
                ]
            )

    def descent(self, features: List[float], return_path=False, path=None):
        """
        Returns the points of a leave child, corresponding to the given
        features.
        """
        path = [self] if path == None else path

        if self.points.bestFeatureId != None:
            if self.points.types[self.points.bestFeatureId] in [
                FeaturesTypes.BOOLEAN,
                FeaturesTypes.CLASSES,
            ]:
                chosenChild = self.childs[
                    0
                    if features[self.points.bestFeatureId]
                    == self.points.bestFeatureSplitValue
                    else 1
                ]
            else:
                chosenChild = self.childs[
                    0
                    if features[self.points.bestFeatureId]
                    < self.points.bestFeatureSplitValue
                    else 1
                ]

            if isinstance(chosenChild, Tree):
                if return_path:
                    path.append(chosenChild)
                    return chosenChild.descent(features, return_path, path)
                return chosenChild.descent(features, return_path)
            elif isinstance(chosenChild, PointSet):
                if return_path:
                    return chosenChild, path
                return chosenChild

        if return_path:
            return self.points, path
        return self.points

    def decide(self, features: List[float]) -> bool:
        """
        Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
        features : List[float]
            The features of the unlabeled point.

        Returns
        -------
        bool
            The label of the unlabeled point,
            guessed by the Tree
        """
        return self.descent(features).outputClass

    def update_path(self, path) -> None:
        """
        Dynamic algorithm - FuDyADT: Foodie for Amatriciana Di Tonno
        """
        for indexPath, treePath in enumerate(path):
            treePath.changes += 1

            if treePath.changes >= treePath.beta * treePath.points.features.shape[0]:
                newTree = Tree(
                    treePath.points.features,
                    treePath.points.labels,
                    treePath.points.types,
                    treePath.h,
                    treePath.min_split_points,
                    treePath.beta,
                    0,
                )

                if indexPath != 0:
                    parentTree = path[indexPath - 1]

                    for indexChild, child in enumerate(parentTree.childs):
                        if child == treePath:
                            parentTree.childs[indexChild] = newTree

                elif indexPath == 0:
                    self = newTree

                break

    def add_training_point(self, features: List[float], label: bool) -> None:
        """
        Updates the tree by adding a new training point to the list of
        already existing training points wherever it is relevant in the
        tree.
        """
        chosenLeave, path = self.descent(features, return_path=True)
        chosenLeave.features = np.vstack((chosenLeave.features, features))
        chosenLeave.labels = np.vstack(
            (chosenLeave.labels.reshape(-1, 1), label), dtype=bool
        )

        self.update_path(path)

    def del_training_point(self, features: List[float], label: bool) -> None:
        """
        Removes a given training point from the list of training points
        wherever it is relevant in the tree.
        """
        chosenLeave, path = self.descent(features, return_path=True)

        chosenLeaveDataSet = np.hstack(
            (chosenLeave.features, chosenLeave.labels.reshape(-1, 1))
        )
        newData = np.hstack((features, label))

        chosenLeaveNewDataSet = np.delete(
            chosenLeaveDataSet, np.where(chosenLeaveDataSet == newData)[0][0], axis=0
        )
        chosenLeave.features = chosenLeaveNewDataSet[:, :-1]
        chosenLeave.labels = np.array(chosenLeaveNewDataSet[:, -1], dtype=bool)

        self.update_path(path)
