import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from peernet.node import Node


class SoftMaxRegression(object):
    """
    Logistic Regression
    """

    def __init__(self, classes=10, lr=4, epochs=9, batch_size=128, beta=0.9, hidden_layers=64, debug=True, name=None):
        np.random.seed(138)
        self.lr = lr
        self.epochs = epochs
        self.debug = debug
        self.name = name
        self._costs = []
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.n_h = hidden_layers
        self.beta = beta
        self.batch_size = batch_size
        self.classes = classes

    def fit(self, X, Y):

        self.initialize(X.shape[0])
        self.optimize(X, Y)

        return self

    def collaborative_fit(self, node, neighbor):
        # Initialize parameters
        self.collaborative_initialize(node, neighbor)

        # Collaborative optimization
        self.collaborative_optimize(node, neighbor)
        return self.parameters

    def predict(self, X):
        cache = self.feed_forward(X)
        predictions = np.argmax(cache["A2"], axis=0)

        return predictions

    def test_cost(self, X, Y):
        cache = self.feed_forward(X)
        cost = self.loss(Y, cache["A2"])

        return cost

    def summary(self, X, Y, X_test, Y_test):
        prediction_train = self.predict(X)
        prediction_test = self.predict(X_test)

        labels_train = np.argmax(Y, axis=0)
        labels_test = np.argmax(Y_test, axis=0)

        d = {
            "costs": self._costs,
            "train_pred": prediction_train,
            "test_pred": prediction_test,
            "report": classification_report(prediction_test, labels_test),
            "train_acc": accuracy_score(prediction_train, labels_train),
            "test_acc": accuracy_score(prediction_test, labels_test),
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

        return d

    # ------------------------------ Properties -------------------------------
    @property
    def costs(self):
        return self._costs

    @property
    def learning_rate(self):
        return self.lr

    @property
    def parameters(self):
        return self.W

    @parameters.setter
    def parameters(self, params):
        self.W = params
        return self

    # ------------------------------ Local methods ----------------------------

    @classmethod
    def sigmoid(cls, z):

        s = 1.0 / (1.0 + np.exp(-z))

        return s

    def initialize(self, n_x):

        self.W1 = np.random.randn(self.n_h, n_x) * np.sqrt(1. / n_x)
        self.b1 = np.zeros((self.n_h, 1)) * np.sqrt(1. / n_x)

        self.W2 = np.random.randn(self.classes, self.n_h) * np.sqrt(1. / self.n_h)
        self.b2 = np.zeros((self.classes, 1)) * np.sqrt(1. / self.n_h)

        assert (self.W1.shape == (self.n_h, n_x))
        assert (self.W2.shape == (self.classes, self.n_h))
        assert (self.b1.shape == (self.n_h, 1))
        assert (self.b2.shape == (self.classes, 1))

        return self

    def propagate(self, X, Y, m_batch):

        cache = self.feed_forward(X)
        grads = self.back_propagate(X, Y, cache, m_batch)
        return grads, cache

    @classmethod
    def loss(cls, Y, A):
        m = Y.shape[1]
        cost = -1.0 / m * np.sum(np.multiply(Y, np.log(A)))
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def feed_forward(self, X):
        cache = {}
        cache["Z1"] = np.matmul(self.W1, X) + self.b1
        cache["A1"] = self.sigmoid(cache["Z1"])
        cache["Z2"] = np.matmul(self.W2, cache["A1"]) + self.b2
        cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

        return cache

    def back_propagate(self, X, Y, cache, m_batch):
        # print(f"{self.name} >> m_batch={m_batch}")
        dZ2 = cache["A2"] - Y
        dW2 = (1. / m_batch) * np.matmul(dZ2, cache["A1"].T)
        db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid(cache["Z1"]) * (1 - self.sigmoid(cache["Z1"]))
        dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
        db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads

    def optimize(self, X_train, Y_train):

        self._costs = []
        batches = -(-X_train.shape[1] // self.batch_size)
        dw, db = None, None

        V_dW1 = np.zeros(self.W1.shape)
        V_db1 = np.zeros(self.b1.shape)
        V_dW2 = np.zeros(self.W2.shape)
        V_db2 = np.zeros(self.b2.shape)

        for i in range(self.epochs):

            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            for j in range(batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, X_train.shape[1] - 1)
                X = X_train_shuffled[:, begin:end]
                Y = Y_train_shuffled[:, begin:end]
                m_batch = end - begin

                grads, cache = self.propagate(X, Y, m_batch)

                V_dW1 = (self.beta * V_dW1 + (1. - self.beta) * grads["dW1"])
                V_db1 = (self.beta * V_db1 + (1. - self.beta) * grads["db1"])
                V_dW2 = (self.beta * V_dW2 + (1. - self.beta) * grads["dW2"])
                V_db2 = (self.beta * V_db2 + (1. - self.beta) * grads["db2"])

                self.W1 = self.W1 - self.lr * V_dW1
                self.b1 = self.b1 - self.lr * V_db1
                self.W2 = self.W2 - self.lr * V_dW2
                self.b2 = self.b2 - self.lr * V_db2

            if self.debug:
                cache = self.feed_forward(X_train)
                train_cost = self.loss(Y_train, cache["A2"])
                self._costs.append(train_cost)
                # cache = self.feed_forward(X_test, params)
                # test_cost = self.loss(Y_test, cache["A2"])
                # print("Epoch {}: training cost = {}, test cost = {}".format(i + 1, train_cost, test_cost))
                print("{} >> Epoch {}: training cost = {}.".format(self.name, i + 1, train_cost))
            else:
                print(".", end=" ")
        print()

        return self

    # ------------------------------ collaborative methods --------------------
    def collaborative_initialize(self, node, neighbor):
        if neighbor:
            self.W = node.Theta[neighbor]
            self.b = np.zeros((1, 1))
        else:
            self.W = node.Theta[node.name]
            self.b = np.zeros((1, 1))

    @classmethod
    def collaborative_loss(cls, node, neighbor, Y, A):
        m = Y.shape[1]
        T = node.Theta
        if neighbor:
            i = neighbor
        else:
            i = node.name

        # Model forward pass
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
        # forward pass over network
        sigma = np.sum(np.array([w * np.linalg.norm(T[i] - T[j]) ** 2 for j, w in node.W.items()]), axis=0)
        cl_cost = 1 / 2 * sigma + node.mu * node.D * cost

        cl_cost = np.squeeze(cl_cost)
        assert (cl_cost.shape == ())

        return cl_cost

    def collaborative_gradient(self, node, neighbor, X, Y, A):
        m = X.shape[1]
        T = node.Theta
        if neighbor:
            i = neighbor
        else:
            i = node.name

        # back propagation
        dw = 1.0 / m * np.dot(X, (A - Y).T)
        db = 1.0 / m * np.sum(A - Y, axis=1, keepdims=True)

        # back propagation over network
        # TODO dont forget to integrate the bais

        if not neighbor:
            i = node.name
            sigma_Q = np.sum(np.array([w * (T[i] - T[j]) for j, w in node.W.items()]), axis=0)
            sigma_A = np.sum(np.array([node.A[i] + node.rho * (T[i] - node.Z[i]) for _ in node.Theta]), axis=0)
            cl_dw = sigma_Q + node.mu * node.D * dw + sigma_A
            cl_db = db
        else:
            i = neighbor
            sigma_Q1 = - np.sum(np.array([w * (T[i] - T[j]) for j, w in node.W.items()]), axis=0)
            sigma_Q2 = T[i] - T[node.name]
            sigma_A = np.sum(np.array([node.A[i] + node.rho * (T[i] - node.Z[i]) for _ in node.Theta]), axis=0)
            cl_dw = sigma_Q2 + node.mu * node.D * dw + sigma_A
            cl_db = db

        if i == 'w2':
            pass
            # print(
            # f"{node.pname} >> {neighbor} << db={np.sum(db)} | cl_dw={np.sum(cl_dw)} | sigma_Q={np.sum(sigma_Q)} | "
            # f"sigma_A={np.sum(sigma_A)}")
            # print(f"{node.pname} >> cl_dw={np.sum(cl_dw)}")

        # sigma = np.sum(np.array([w * (T[i] - T[j]) for j, w in node.W.items()]), axis=0)
        # cl_dw = sigma + node.mu * node.D * dw + node.A[i] + node.rho * (T[i] - node.Z[i])
        # cl_db = db

        assert (cl_dw.shape == self.W.shape)
        assert (cl_db.shape == self.b.shape)

        return {"dw": cl_dw, "db": cl_db}

    def collaborative_optimize(self, node: Node, neighbor):
        self._costs = []
        dw, db = None, None
        X = node.X
        Y = node.y

        for i in range(1):
            z = np.dot(self.W.T, X) + self.b
            A = self.sigmoid(z)
            cost = self.collaborative_loss(node, neighbor, Y, A)
            grads = self.collaborative_gradient(node, neighbor, X, Y, A)

            dw = grads["dw"]
            db = grads["db"]

            self.W = self.W - self.lr * dw
            self.b = self.b - self.lr * db
            if i % 100 == 0:
                self._costs.append(cost)
            # if i % 100 == 0:
            #     if self.debug:
            #         print(F"{self.name}: CL Epoch {i} Cost: {cost}")
            #     else:
            #         print(f'*', end='')

        grads = {"dw": dw, "db": db}

        return grads
