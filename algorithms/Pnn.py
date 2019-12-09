import numpy as np


class Knn():
    """k-Nearest Neighbor Classifier"""

    def __init__(self, k=1, distance=0):
        self.x_train = x
        self.y_train = y
        self.distance = self._set_distance(distance)
        if distance == 0:
            self.distance = np.abs  # absolute value
        elif distance == 1:
            self.distance = np.square  # square root
        else:
            raise Exception("Distance not defined.")
        self.k = k

    def train(self, x, y):
        """Train the classifier (here simply save training data)

        x -- feature vectors (N x D)
        y -- labels (N x 1)
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """Predict and return labels for each feature vector from x

        x -- feature vectors (N x D)
        """
        predictions = []  # placeholder for N labels

        # no. of classes = max label (labels starts from 0)
        nof_classes = np.amax(self.y_train) + 1

        # loop over all test samples
        for x_test in x:
            # array of distances between current test and all training samples
            distances = np.sum(self.distance(self.x_train - x_test), axis=1)

            # placeholder for labels votes
            votes = np.zeros(nof_classes, dtype=np.int)

            # find k closet neighbors and vote
            # argsort returns the indices that would sort an array
            # so indices of nearest neighbors
            # we take self.k first
            for neighbor_id in np.argsort(distances)[:self.k]:
                # this is a label corresponding to one of the closest neighbor
                neighbor_label = self.y_train[neighbor_id]
                # which updates votes array
                votes[neighbor_label] += 1

            # predicted label is the one with most votes
            predictions.append(np.argmax(votes))

        return predictions

    def _set_distance(self, distance):
        if distance == 0:
            self.distance = np.abs  # absolute value
        elif distance == 1:
            self.distance = np.square  # square root
        else:
            raise Exception("Distance not defined.")
