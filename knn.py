import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def represent(pred):
    # 0: 'setosa', 1: 'versicolor', 2: 'virginica'
    # also we can loop this to see all the predicted labels
    if pred[0] == 0.0:
        print("setosa")
    elif pred[0] == 1.0:
        print("versicolor")
    elif pred[0] == 2.0:
        print("virginica")


class KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # find distance of all x
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sort them and take only 'k' x
        k_distances = np.argsort(distance)[:self.n_neighbors]
        k_neighbor_labels = [self.y_train[i] for i in k_distances]
        # find the most common label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]