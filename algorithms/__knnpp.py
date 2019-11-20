import pandas as pd
import numpy as np
import operator
from peernet.PeerNet import PeerNet


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)


def knn(dataset, k):
    distances = {}
    length = dataset.shape[1]



    print(length)


def knn2(trainingSet, testInstance, k):
    distances = {}
    sort = {}

    length = testInstance.shape[1]
    print(length)

    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclidianDistance(testInstance, trainingSet.iloc[x], length)
    distances[x] = dist[0]

    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))  # by using it we store indices also
    sorted_d1 = sorted(distances.items())
    print(sorted_d[:5])
    print(sorted_d1[:5])

    neighbors = []

    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    counts = {"setosa": 0, "versicolor": 0, "virginica": 0}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1

    print(counts)
    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)
    return (sortedVotes[0][0], neighbors)


# testSet = [[1.4, 3.6, 3.4, 1.2]]
# test = pd.DataFrame(testSet)
# result, neigh = knn(iris, test, 4)  # here we gave k=4
# print("And the flower is:", result)
# print("the neighbors are:", neigh)


# Local functions -------------------------------------------------------------
def similarity(row1, row2, stype='euclidean'):
    """
    similarity measures: cosine, euclidean
    """
    if len(row1) == len(row2):
        if stype == 'cosine':
            pscalaire = 0.0
            normp1 = 0.0
            normp2 = 0.0
            for i in range(len(row1)):
                pscalaire += row1[i] * row2[i]
                normp1 += np.sqrt(row1[i] ** 2)
                normp2 += np.sqrt(row2[i] ** 2)
            return pscalaire / (normp1 * normp2)
        else:
            distance = 0.0
            for i in range(len(row1) - 1):
                distance += (float(row1[i].strip()) - float(row2[i].strip())) ** 2
            return np.sqrt(distance)

    else:
        raise ValueError('Euclidean similarity need vectors of the same size.')


import csv

if __name__ == '__main__':
    file = "../data/iris.csv"
    iris = pd.read_csv(file)
    algo = knn(iris, 4)
