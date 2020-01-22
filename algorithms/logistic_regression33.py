import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py

from peernet.message import request_model
from peernet.node import Node
from peernet.helpers import log
from peernet.protocol import RESPONSE_MODEL


# ------------------------- Logistic Regression Neural Network ----------------

class LogisticRegressionNN(object):
    """
    Logistic Regression Neural Network
    """

    def __init__(self, lr=0.01, epochs=50, threshold=0.5, debug=True):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.debug = debug
        self.costs_ = []
        self.w = None
        self.b_ = None

    def fit(self, X_, Y, node_name="", optimizer="", data=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        optimizer: optimizer type
        data: cl parameters
        Returns
        -------
        self : returns a trained LR model.
        """
        X = np.hstack((np.ones((X_.shape[0], 1)), X_.copy()))
        m = X.shape[0]
        n = X.shape[1]
        self._init_params(n)
        self._optimize(X, Y, optimizer, data)
        Y_pred = self.predict(X_)
        log("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred - Y)) * 100))

        return self

    def metrics(self, X_test, Y_test, metric='accuracy'):
        """
        Model metrics
        """
        Y_pred = self.predict(X_test)
        if metric == "accuracy":
            return 100 - np.mean(np.abs(Y_pred - Y_test)) * 100

        return self

    def cl_fit(self, T, X, Y, params):
        self.w = T
        # Optimization using gradient descent
        self.costs_ = []
        for i in range(self.epochs):
            # calculate gradients
            grads, cost = self.cl_propagate(X, Y, p=params)
            # get gradients
            dw = grads["dw"]
            db = grads["db"]
            # update rule
            self.w = self.w - (self.lr * dw)
            self.b_ = self.b_ - (self.lr * db)
            if i % 100 == 0:
                self.costs_.append(cost)
                if self.debug:
                    print(f"Cost after {i} epochs: {cost}")

        # Predictions on train data
        Y_pred = self.predict(X)
        log('success', f"{params['i']}:: CL :: Train accuracy: {100 - np.mean(np.abs(Y_pred - Y)) * 100} %")
        return self.w

    def predict(self, X_):
        """Predict using the LRNN classifier
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
             The input data.
        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        X = np.hstack((np.ones((X_.shape[0], 1)), X_.copy()))

        m = X.shape[0]
        Y_predict = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[1], 1)

        A = self._sigmoid(np.dot(self.w.T, X.T))

        for i in range(A.shape[1]):
            if A[0, i] <= self.threshold:
                Y_predict[0, i] = 0
            else:
                Y_predict[0, i] = 1

        return Y_predict

    def predict_one(self, X):
        Y_predict = None
        self.w = self.w.reshape(X.shape[0], 1)
        A = self._sigmoid(np.dot(self.w.T, X) + self.b_)
        if A <= self.threshold:
            Y_predict = 0
        else:
            Y_predict = 1

        return Y_predict

    def mp_cost(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        m = X.shape[0]
        A = self._sigmoid(np.dot(self.w.T, X.T))
        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))
        return np.squeeze(cost)

    # ------------------------------ Properties -------------------------------
    @property
    def costs(self):
        return self.costs_

    @property
    def learning_rate(self):
        return self.lr

    @property
    def parameters(self):
        return self.w

    @parameters.setter
    def parameters(self, params):
        self.w = params

    # ------------------------------ Local methods ----------------------------
    def _init_params(self, dimension):
        """
        Parameter initialization
        """
        self.w = np.zeros((dimension, 1))
        self.b_ = 0

    @staticmethod
    def _sigmoid(z):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def _propagate(self, X, Y):
        """
        Forward and back propagation
        """
        # num of training samples
        m = X.shape[0]

        # forward pass
        A = self._sigmoid(np.dot(self.w.T, X.T))
        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))
        # back propagation
        dw = (1 / m) * (np.dot(X.T, (A - Y).T))
        # db = (1 / m) * (np.sum(A - Y))
        db = 0

        cost = np.squeeze(cost)

        # gradient dictionary
        grads = {"dw": dw, "db": db}

        return grads, cost

    def cl_propagate(self, X, Y, p):
        """
        Forward and back propagation
        """

        # num of training samples
        m = X.shape[1]

        # forward pass
        A = self._sigmoid(np.dot(self.w.T, X) + self.b_)
        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))

        # forward pass over network
        i = p['i']
        ij = p['ij']
        T = p['Theta']
        Z = p['Z']
        cl = 1 / 2 * np.sum(np.array([w * np.linalg.norm(T[i] - T[j]) ** 2 for j, w in p['W'].items()]), axis=0)
        cl_cost = cl + p['mu'] * p['D'] * cost

        # back propagation
        dw_sol = (1 / m) * (np.dot(X, (A - Y).T))
        db_sol = (1 / m) * (np.sum(A - Y))

        # back propagation over network
        # todo dont forget to integrate the bais
        ti_tj = np.sum(np.array([w * (T[i] - T[j]) for j, w in p['W'].items()]), axis=0)
        dw = ti_tj + p['mu'] * p['D'] * dw_sol + p['A'][i] + p['rho'] * (T[i] - Z[i])
        db = db_sol

        # cost = np.squeeze(cost)
        cl_cost = np.squeeze(cl_cost)

        # gradient dictionary
        grads = {"dw": dw, "db": db}

        return grads, cl_cost

    def _optimize(self, X, Y, optimizer="", data=None):
        """
        Optimization using gradient descent
        """
        self.costs_ = []
        for i in range(self.epochs):
            # calculate gradients
            if optimizer == "CL":
                grads, cost = self.cl_propagate(X, Y, p=data)
            else:
                grads, cost = self._propagate(X, Y)

            # get gradients
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            self.w = self.w - (self.lr * dw)
            self.b_ = self.b_ - (self.lr * db)
            if i % 100 == 0:
                self.costs_.append(cost)
                if self.debug:
                    print("Cost after %i epochs: %f" % (i, cost))

        # gradient dict
        grads = {"dw": dw, "db": db}

        return grads

    # ------------------------- END Logistic Regression Neural Network --------


def load_dataset():
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def analysis(model, X_train, y_train, X_test, y_test):
    print('# Analysis:')
    print('## Learning rate:')
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for lr in learning_rates:
        print(f">> Learning rate: {lr}")
        m = model(lr=lr)
        m.fit(X_train, y_train)
        m.metrics(X_test, y_test)
        models[str(lr)] = m

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)].costs), label="α = " + str(models[str(i)].learning_rate))

    plt.ylabel('cost')
    plt.xlabel('Iterations')

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


if __name__ == '__main__':
    # Load dataset
    dataset = 3

    if dataset == 1:
        dataset = pd.read_csv('../data/breast_cancer.csv')
        X = dataset.iloc[:, 2:31].values
        Y = dataset.iloc[:, 1].values
        Y = np.where(Y == 'M', 0, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
        # X_train = X_train.reshape(X_train.shape[0], -1).T
        # X_test = X_test.reshape(X_test.shape[0], -1).T
        # Train model
        model = LogisticRegressionNN(debug=True)
        model.fit(X_train, y_train)
        accuracy = model.metrics(X_test, y_test)
        print(f"Accuracy: {round(accuracy, 2)}%.")

    elif dataset == 2:
        X, y, X_test, y_test, classes = load_dataset()
        # Reshape the training and test examples
        # X_flatten = X.reshape(X.shape[0], -1).T
        # X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
        # Standardize dataset.
        X = X / 255.
        X_test = X_test / 255.
        # Train model
        model = LogisticRegressionNN()
        model.fit(X, y)
        tacc = model.metrics(X, y)
        sacc = model.metrics(X_test, y_test)
        print(f"Train Accuracy: {round(tacc, 2)}%. | Test Accuracy: {round(sacc, 2)}%.")
        # index = 17
        # pic = X_test[:, index]
        # pic_y = y_test[0, index]
        # pred = model.predict_one(pic)
        # print(f"You predicted {classes[int(pic_y)].decode('utf-8')} as: {classes[int(pred)].decode('utf-8')}")
        # analysis(LogisticRegressionNN, X, y, X_test, y_test)
    elif dataset == 3:
        df = pd.read_csv('../data/sonar.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 60].values
        y = np.where(y == 'R', 0, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
        # X_train = X_train.reshape(X_train.shape[0], -1).T
        # X_test = X_test.reshape(X_test.shape[0], -1).T
        # Train model
        model = LogisticRegressionNN(debug=True)
        model.fit(X_train, y_train)
        accuracy = model.metrics(X_test, y_test)
        print(f"Accuracy: {round(accuracy, 2)}%.")
        index = 17
        sample = X_test[:, index]
        sample_y = y_test[index]
        pred = model.predict_one(sample)
        print(f"You predicted {sample_y} as: {pred}")
        # analysis(LogisticRegressionNN, X_train, y_train, X_test, y_test)
    elif dataset == 4:
        # accuracy: 98.08568824065634 %
        # Accuracy: 97.45 %.
        df = pd.read_csv('../data/banknotes.csv', sep=';')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
        # X_train = X_train.reshape(X_train.shape[0], -1).T
        # X_test = X_test.reshape(X_test.shape[0], -1).T
        # Train model

        print(X_train.shape)
        print('-----')
        model = LogisticRegressionNN(debug=True)
        model.fit(X_train, y_train)

        accuracy = model.metrics(X_test, y_test)
        print(f"Accuracy: {round(accuracy, 2)}%.")
        # index = 17
        # sample = X_test[:, index]
        # sample_y = y_test[index]
        # pred = model.predict_one(sample)
        # print(f"You predicted {sample_y} as: {pred}")
        # analysis(LogisticRegressionNN, X_train, y_train, X_test, y_test)
        # 97.45 %.
