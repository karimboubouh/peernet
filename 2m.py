import numpy as np
from autograd import grad
from autograd.test_util import check_grads
import joblib

##

dataset = "./datasets/mnist.data"
X, y = joblib.load(dataset)
y = y.astype(np.int)

##

x_train = X[:10000]
y_train = y[:10000]
x_test = X[60000:]
y_test = y[60000:]
y_train = np.squeeze(y_train)
x_train = x_train[np.any([y_train == 1, y_train == 2], axis=0)]
y_train = y_train[np.any([y_train == 1, y_train == 2], axis=0)]

y_train = y_train - 1
y_train = y_train.reshape(-1, 1)
y_test = np.squeeze(y_test)

x_test = x_test[np.any([y_test == 1, y_test == 2], axis=0)]
y_test = y_test[np.any([y_test == 1, y_test == 2], axis=0)]
y_test = y_test - 1
y_test = y_test.reshape(-1, 1)
x_train = x_train / 255
x_test = x_test / 255
m = x_train.shape[0]

m_test = x_test.shape[0]
x_train, x_test = x_train.T, x_test.T
y_train, y_test = y_train.reshape(1, m), y_test.reshape(1, m_test)
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = x_train[:, shuffle_index], y_train[:, shuffle_index]

##
models = joblib.load('./models.data')
i = 0
theta_start = models[i]["W"]
m = X_train.shape[1]


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s


def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs.T, weights))


# def training_loss(weights):
#     # Training loss is the negative log-likelihood of the training labels.
#     preds = logistic_predictions(weights, inputs)
#     label_probabilities = preds * targets + (1 - preds) * (1 - targets)
#     print(-np.sum(np.log(label_probabilities)))
#     return (1 / m) * -np.sum(np.log(label_probabilities))


def training_loss(W):
    Z = np.matmul(W.T, X)
    A = sigmoid(Z)

    m = Y.shape[1]
    L = -(1. / m) * (np.sum(np.multiply(np.log(A), Y)) + np.sum(np.multiply(np.log(1 - A), (1 - Y))))

    return L


# Build a toy dataset.
X = X_train
Y = y_train

# Build a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Check the gradients numerically, just to be safe.
# weights = np.zeros((X_train.shape[0], 1))
W = theta_start

print(X_train.shape)

# check_grads(training_loss, modes=['rev'])(W)
print(W.shape)

# Optimize weights using gradient descent.
print("Initial loss:", training_loss(W))
for i in range(500):
    W -= training_gradient_fun(W) * 0.01

print("Trained loss:", training_loss(W))
