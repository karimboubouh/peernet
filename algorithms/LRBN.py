# import numpy as np
import numpy as np
from sklearn.metrics import accuracy_score

from peernet.node import Node


class LRBN(object):
    """
    Logistic Regression
    """

    def __init__(self, lr=4, epochs=9, batch_size=128, beta=0.9, threshold=0.5, debug=True, name=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.threshold = threshold
        self.debug = debug
        self.name = name
        self._costs = []
        self.W = None
        self.b = None

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
        """
        Predict whether the label is 0 or 1

        Arguments:
        X -- data of size (n_x, m)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1)
        """

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        self.W = self.W.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(self.W.T, X) + self.b)

        for i in range(A.shape[1]):
            if A[:, i] > self.threshold:
                Y_prediction[:, i] = 1
            elif A[:, i] <= self.threshold:
                Y_prediction[:, i] = 0

        assert (Y_prediction.shape == (1, m))

        return Y_prediction

    def test_cost(self, X, Y):
        z = np.dot(self.W.T, X) + self.b
        A = self.sigmoid(z)
        cost = self.loss(Y, A)
        return cost

    def summary(self, X, Y, X_test, Y_test):
        Y_prediction_test = self.predict(X_test)
        Y_prediction_train = self.predict(X)

        train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train - Y) * 100.0)
        test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test - Y_test) * 100.0)

        d = {
            "costs": self._costs,
            "train_pred": Y_prediction_train,
            "test_pred": Y_prediction_test,
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "W": self.W,
            "b": self.b,
            "lr": self.lr,
            "epochs": self.epochs
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

    # ------------------------------ Local methods ----------------------------

    @classmethod
    def sigmoid(cls, z):
        """
        Implement the sigmoid function

        Arguments:
        y -- a scalar (float)

        Return:
        s -- the sigmoid function evaluated on z
        """

        s = 1.0 / (1.0 + np.exp(-z))

        return s

    def initialize(self, dim):
        """
        Initialise the weights and the bias to tensors of dimensions (dim,1) for w and
        to 1 for b (a scalar)

        Arguments:
        dim -- a scalar (float)

        Return:
        self -- the class object
        """
        # np.random.seed(dim)
        self.W = np.random.randn(dim, 1) * 0.01
        self.b = np.zeros((1, 1))

        assert (self.W.shape == (dim, 1))
        assert (self.b.shape == (1, 1))

        return self

    def propagate(self, X, Y, m_batch):
        """
        Implement the cost function and its gradient for the propagation.
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px, number of examples)
        Y -- true "label" vector

        Return:
        cost -- negative log-likelihood cost for logistic regression
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        """

        z = np.dot(self.W.T, X) + self.b
        A = self.sigmoid(z)
        cost = self.loss(Y, A)
        grads = self.gradient(X, Y, A, m_batch)

        return grads, cost

    @classmethod
    def loss(cls, Y, A):
        m = Y.shape[1]
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def gradient(self, X, Y, A, m_batch):
        m = X.shape[1]
        dw = 1.0 / m_batch * np.dot(X, (A - Y).T)
        db = 1.0 / m_batch * np.sum(A - Y, axis=1, keepdims=True)

        assert (dw.shape == self.W.shape)
        assert (db.shape == self.b.shape)
        # assert (db.dtype == float)

        return {"dw": dw, "db": db}

    def optimize(self, X, Y):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (n_x, 1)
        b -- bias, a scalar
        X -- data of shape (n_x, m)
        Y -- true "label" vector (containing 0 if class 1, 1 if class 2), of shape (1, m)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        self -- the class object
        """
        self._costs = []
        batches = -(-X.shape[1] // self.batch_size)
        dw, db = None, None

        V_dW = np.zeros(self.W.shape)
        V_db = np.zeros(self.b.shape)

        for i in range(self.epochs):

            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            cost = None
            for j in range(batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, X.shape[1] - 1)
                X_ = X_shuffled[:, begin:end]
                Y_ = Y_shuffled[:, begin:end]
                m_batch = end - begin

                grads, cost = self.propagate(X_, Y_, m_batch)

                dw = grads["dw"]
                db = grads["db"]

                V_dW = self.beta * V_dW + (1. - self.beta) * dw
                V_db = self.beta * V_db + (1. - self.beta) * db

                self.W = self.W - self.lr * V_dW
                self.b = self.b - self.lr * V_db

            self._costs.append(cost)
            if self.debug:
                print(F"{self.name}: Epoch {i} Cost: {cost}")
            else:
                print('.', end='')

        grads = {"dw": dw, "db": db}

        return grads

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
            #  np.dot(T[i].T, (T[i] - T[j]))
            sigma_Q = np.sum(np.array([w * (T[i] - T[j]) for j, w in node.W.items()]), axis=0)
            sigma_A = np.sum(np.array([node.A[i] + node.rho * (T[i] - node.Z[i]) for _ in node.Theta]), axis=0)
            cl_dw = sigma_Q + node.mu * node.D * dw + sigma_A
            cl_db = db
        else:
            i = neighbor
            # sigma_Q1 = - np.sum(np.array([w * np.dot(T[i].T, (T[i] - T[j])) for j, w in node.W.items()]), axis=0)
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
            if np.sum(dw) > 500:
                print(f"CL::dw={np.sum(dw)}")
                continue

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

    def fit2(self, X, Y, X_test, Y_test):
        n_x, m = X.shape[0], X.shape[1]
        self.initialize(n_x)

        cost = 0
        for i in range(1000):
            Z = np.matmul(self.W.T, X) + self.b
            A = self.sigmoid(Z)

            cost = self.loss(Y, A)

            dW = (1 / m) * np.matmul(X, (A - Y).T)
            db = (1 / m) * np.sum(A - Y, axis=1, keepdims=True)

            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

            if i % 100 == 0:
                # print("Epoch", i, "cost: ", cost)
                print('.', end='')

        print("Final cost:", cost)

        Z = np.matmul(self.W.T, X_test) + self.b
        A = self.sigmoid(Z)

        predictions = (A > .5)[0, :]
        labels = (Y_test == 1)[0, :]

        return accuracy_score(predictions, labels)
