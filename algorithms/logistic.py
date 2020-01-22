import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from sklearn import datasets
from sklearn.model_selection import train_test_split
import h5py

from peernet.message import request_model
from peernet.node import Node
from peernet.helpers import log
from peernet.protocol import RESPONSE_MODEL


# ------------------------- Logistic Regression Neural Network ----------------

class LogisticRegressionNN(object):

    def __init__(self, X, Y):
        self.costs_ = []
        self.w = None
        self.X = X
        self.Y = Y

    def fit(self):
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        initial_theta = np.zeros((self.X.shape[1], 1))
        cost = self.compute_cost(self.X, self.Y, initial_theta)
        gradient = self.compute_gradient(self.X, self.Y, initial_theta)
        print('Cost at initial theta (zeros): {0} \nGradient at initial theta (zeros): {1}'.format(cost, gradient))
        print("---------------")

        def f(theta):
            return self.compute_cost(self.X, self.Y, theta)

        def fprime(theta):
            return np.ndarray.flatten(self.compute_gradient(self.X, self.Y, theta))

        bfg = fmin_bfgs(f, initial_theta, fprime, disp=True, maxiter=400, full_output=True, retall=True)
        print(bfg[0])
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
        self.w_ = T
        # Optimization using gradient descent
        self.costs_ = []
        for i in range(self.epochs):
            # calculate gradients
            grads, cost = self.cl_propagate(X, Y, p=params)
            # get gradients
            dw = grads["dw"]
            db = grads["db"]
            # update rule
            self.w_ = self.w_ - (self.lr * dw)
            self.b_ = self.b_ - (self.lr * db)
            if i % 100 == 0:
                self.costs_.append(cost)
                if self.debug:
                    print(f"Cost after {i} epochs: {cost}")

        # Predictions on train data
        Y_pred = self.predict(X)
        log('success', f"{params['i']}:: CL :: Train accuracy: {100 - np.mean(np.abs(Y_pred - Y)) * 100} %")
        return self.w_

    def predict(self, X):
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
        m = X.shape[1]
        Y_predict = np.zeros((1, m))
        self.w_ = self.w_.reshape(X.shape[0], 1)

        A = self._sigmoid(np.dot(self.w_.T, X) + self.b_)

        for i in range(A.shape[1]):
            if A[0, i] <= self.threshold:
                Y_predict[0, i] = 0
            else:
                Y_predict[0, i] = 1

        return Y_predict

    def predict_one(self, X):
        Y_predict = None
        self.w_ = self.w_.reshape(X.shape[0], 1)
        A = self._sigmoid(np.dot(self.w_.T, X) + self.b_)
        if A <= self.threshold:
            Y_predict = 0
        else:
            Y_predict = 1

        return Y_predict

    # ------------------------------ Properties -------------------------------
    @property
    def costs(self):
        return self.costs_

    @property
    def learning_rate(self):
        return self.lr

    @property
    def parameters(self):
        return self.w_

    @parameters.setter
    def parameters(self, params):
        self.w_ = params

    # ------------------------------ Local methods ----------------------------
    @staticmethod
    def _sigmoid(z):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, Y, theta):
        m = X.shape[1]
        print(f"X: {X.shape} | Y: {Y.shape} | theta: {theta.shape}")
        A = self._sigmoid(np.dot(X, theta))
        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))
        print(cost)
        print()
        print()
        return cost

    def compute_gradient(self, X, Y, theta):
        m = X.shape[1]
        A = self._sigmoid(np.dot(X, theta))
        return (1 / m) * (np.dot((A - Y).T, X))

    def _propagate(self, X, Y):
        """
        Forward and back propagation
        """

        # num of training samples
        m = X.shape[1]

        # forward pass
        A = self._sigmoid(np.dot(self.w_.T, X) + self.b_)
        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))

        # back propagation
        dw = (1 / m) * (np.dot(X, (A - Y).T))
        db = (1 / m) * (np.sum(A - Y))

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
        A = self._sigmoid(np.dot(self.w_.T, X) + self.b_)
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
            self.w_ = self.w_ - (self.lr * dw)
            self.b_ = self.b_ - (self.lr * db)
            if i % 100 == 0:
                self.costs_.append(cost)
                if self.debug:
                    print("Cost after %i epochs: %f" % (i, cost))

        # gradient dict
        grads = {"dw": dw, "db": db}

        return grads

    # ------------------------- END Logistic Regression Neural Network --------
    def _opt(self, X, Y):

        # num of training samples
        m = X.shape[1]

        def cost(w_, X, Y):
            z = np.dot(w_.T, X) + self.b_
            A = 1 / (1 + np.exp(-z))
            A = self._sigmoid()

        cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))

        # back propagation
        dw = (1 / m) * (np.dot(X, (A - Y).T))
        db = (1 / m) * (np.sum(A - Y))

        cost = np.squeeze(cost)


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
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # X = X.reshape(X.shape[0], -1).T

    logistic_regression = LogisticRegressionNN(X, Y)
    # logistic_regression.plot_two_features()
    # logistic_regression.plot_three_features()

    logistic_regression.fit()
    dataset = 33
    if dataset == 1:
        X, y, X_test, y_test, classes = load_dataset()
        # Reshape the training and test examples
        X_flatten = X.reshape(X.shape[0], -1).T
        X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
        # Standardize dataset.
        X = X_flatten / 255.
        X_test = X_test_flatten / 255.
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
    elif dataset == 2:
        df = pd.read_csv('../data/sonar.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 60].values
        y = np.where(y == 'R', 0, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
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
