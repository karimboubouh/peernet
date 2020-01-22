import numpy as np
from sklearn.model_selection import train_test_split


def sonar(df):
    """Data pre-processing function for sonar dataset"""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def load_sonar(df):
    """Data pre-processing function"""

    X = df.iloc[:, :-1].values
    y = df.iloc[:, 60].values
    y = np.where(y == 'R', 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    return X_train, X_test, y_train, y_test


def breast_cancer(df):
    """Data pre-processing function for breast_cancer dataset"""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 60].values
    y = np.where(y == 'R', 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def load_banknotes(df):
    """Data pre-processing function for breast_cancer dataset"""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def load_mnist_12(data):
    x_train = data['x_train'].values
    y_train = data['y_train'].values
    x_test = data['x_test'].values
    y_test = data['y_test'].values

    # Extract 1 and 2 from train dataset
    y_train = np.squeeze(y_train)
    x_train = x_train[np.any([y_train == 1, y_train == 2], axis=0)]
    y_train = y_train[np.any([y_train == 1, y_train == 2], axis=0)]
    y_train = y_train - 1
    y_train = y_train.reshape(-1, 1)

    # Extract 1 and 2 from train dataset
    y_test = np.squeeze(y_test)
    x_test = x_test[np.any([y_test == 1, y_test == 2], axis=0)]
    y_test = y_test[np.any([y_test == 1, y_test == 2], axis=0)]

    y_test = y_test - 1
    y_test = y_test.reshape(-1, 1)

    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    m = x_train.shape[0]
    m_test = x_test.shape[0]

    x_train, x_test = x_train.T, x_test.T
    y_train, y_test = y_train.reshape(1, m), y_test.reshape(1, m_test)

    # train_one = (y_train == 0).sum()
    # train_two = (y_train == 1).sum()
    # test_one = (y_test == 0).sum()
    # test_two = (y_test == 1).sum()
    # print(f"Train >> One: {train_one} | Two: {train_two} <> Test  >> One: {test_one} | Two: {test_two}")

    return x_train, x_test, y_train, y_test


def load_mnist(data):
    x_train = data['x_train'].values
    y_train = data['y_train'].values
    x_test = data['x_test'].values
    y_test = data['y_test'].values

    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_test = x_train.T, x_test.T
    # one-hot encode labels
    digits = 10
    examples = y_train.shape[0]
    y_train = y_train.reshape(1, examples)
    y_train_new = np.eye(digits)[y_train.astype('int32')]
    y_train = y_train_new.T.reshape(digits, examples)

    examples = y_test.shape[0]
    y_test = y_test.reshape(1, examples)
    y_test_new = np.eye(digits)[y_test.astype('int32')]
    y_test = y_test_new.T.reshape(digits, examples)

    return x_train, x_test, y_train, y_test
