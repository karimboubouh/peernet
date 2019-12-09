import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py

from peernet.message import request_model
from peernet.node import Node
from peernet.helpers import log
from peernet.protocol import RESPONSE_MODEL


def mp_logistic_regression(node: Node):
    """
    Training a Logistic Regression Neural Network model in a P2P architecture
    :param node: Node
    :return: None
    """

    df = node.ldata

    # Prepare and split data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 60].values
    y = np.where(y == 'R', 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    # Training the solidarity model
    solitary_model = LogisticRegressionNN()
    solitary_model.fit(X_train, y_train)
    accuracy = solitary_model.metrics(X_test, y_test)
    log("success", f"{node.pname}: Solidarity model {accuracy}")

    # Initiate node for network training
    node.set_model(solitary_model)
    node.alpha = 0.5
    stop_conditin = 10

    # Model smoothing over network
    for i in range(stop_conditin):
        # Request a model from a random peer
        peer = node.get_random_peer()
        req = request_model(node)
        node.send(peer, req)
        res = node.wait_response(peer, RESPONSE_MODEL)
        peer_model = res['payload']['model']
        if peer_model is not None:
            node.update_models(peer["name"])
            # node.
        else:
            log(f'{node.pname}: {node.peers[i].get("Name", i)} has no model!')
            continue

        # Ensemble the current node model with the peer model
        ens_classifier = node.model.ensemble(peer_classifier)

        # Retrain and recalculate the accuracy of the ensemble model using local data
        ens_classifier.fit(X_train, y_train)
        node.model = ens_classifier
        accuracy = ens_classifier.metrics(X_test, y_test, metric='accuracy')
        log("success", f"{node.pname}: P[{node.peers[i].get('Name', i)}] Ensemble model {accuracy}")

        log('exception', f"END Range({i}) +++++++")

    log('success', f"{node.pname}: Final model accuracy: {accuracy}")


# ------------------------- Logistic Regression Neural Network ----------------

class LogisticRegressionNN(object):
    """
    Logistic Regression Neural Network
    """

    def __init__(self, lr=0.01, epochs=2000, threshold=0.5, debug=False):
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.debug = debug
        self.costs_ = []
        self.w_ = None
        self.b_ = None

    def fit(self, X, Y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained LR model.
        """
        self._init_params(X.shape[0])
        self._optimize(X, Y)
        Y_pred = self.predict(X)
        log('success', "Train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred - Y)) * 100))

        return self

    def metrics(self, X_test, Y_test, metric='accuracy'):
        """
        Model metrics
        """
        Y_pred = self.predict(X_test)
        if metric == "accuracy":
            return 100 - np.mean(np.abs(Y_pred - Y_test)) * 100

        return self

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
    def _init_params(self, dimension):
        """
        Parameter initialization
        """
        self.w_ = np.zeros((dimension, 1))
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

    def _optimize(self, X, Y):
        """
        Optimization using gradient descent
        """
        self.costs_ = []
        for i in range(self.epochs):
            # calculate gradients
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
    dataset = 2
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
        model.metrics(X_test, y_test)
        index = 17
        pic = X_test[:, index]
        pic_y = y_test[0, index]
        pred = model.predict_one(pic)
        print(f"You predicted {classes[int(pic_y)].decode('utf-8')} as: {classes[int(pred)].decode('utf-8')}")
        analysis(LogisticRegressionNN, X, y, X_test, y_test)
    elif dataset == 2:
        df = pd.read_csv('../data/sonar.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, 60].values
        y = np.where(y == 'R', 0, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
        # Train model
        model = LogisticRegressionNN(debug=True)
        model.fit(X_train, y_train)
        model.metrics(X_test, y_test)
        index = 17
        sample = X_test[:, index]
        sample_y = y_test[index]
        pred = model.predict_one(sample)
        print(f"You predicted {sample_y} as: {pred}")
        analysis(LogisticRegressionNN, X_train, y_train, X_test, y_test)
