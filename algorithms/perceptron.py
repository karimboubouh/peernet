import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# ---------------------------------- Perceptron -------------------------------
from sklearn import metrics
from sklearn.model_selection import train_test_split


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=1000, shuffle=False, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)
        self.errors_ = []
        self.w_ = None
        self._models = []
        self.voting = False

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        if self.w_ is None:
            self.w_ = np.zeros(1 + X.shape[1])

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # self.w_[0] += update
                errors += int(update != 0.0)
                # errors += update
                # print(f"Xi={xi}, y={target}, prediction={self.predict(xi)}, update={update}, ERRORS={errors}")
            self.errors_.append(errors)
            if i % 100 == 0:
                print("Cost after %i iterations: %f" % (i, errors))

        y_pred = self.predict(X)
        print(f"Train Accuracy: {metrics.accuracy_score(y, y_pred)}")


        return self

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def metrics(self, X_test, y_test, metric='accuracy'):
        y_pred = self.predict(X_test)
        if type == 'all':
            return f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n" \
                   f"Confusion Matrix: {metrics.confusion_matrix(y_test, y_pred)}"
        elif type == 'confusion_matrix':
            return f"Confusion Matrix: {metrics.confusion_matrix(y_test, y_pred)}"
        elif type == 'accuracy':
            return f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred)}"
        else:
            return f"Test  Accuracy: {metrics.accuracy_score(y_test, y_pred)}"

    def plot_errors(self):
        """Plot the number of updates across epochs."""
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.tight_layout()
        plt.show()

    def ensemble(self, model, method='avg'):
        """type:avg|vote"""
        if not self._models:
            self._models.append(self.w_)
        # illustrative example
        if method == 'avg':
            # self.w_ = (np.array(self.w_) + np.array(model.w_)) / 2.0
            self.w_ = (np.array(self.w_) + np.array(model.w_)) / 2.0
        else:
            raise Exception("Unknown type")
        return self

    # ------------------------------ Local methods ----------------------------

    @staticmethod
    def _shuffle(X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _avg(self, w):
        arrays = [self.w_, w]
        mean = lambda x: sum(x) / float(len(x))
        transpose = [[item[i] for item in arrays] for i in range(len(arrays[0]))]
        return [[mean(j[i] for j in t if i < len(j)) for i in range(len(max(t, key=len)))] for t in transpose]


# ---------------------------------- END Perceptron ---------------------------


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=(cmap(idx),), edgecolor='black', marker=markers[idx],
                    label=cl)


if __name__ == '__main__':
    # Iris dataset
    # df = pd.read_csv('../data/iris.csv')
    # train = df.iloc[:, :-1].values
    # test = df.iloc[:, 4].values
    # test = np.where(test == 'Setosa', -1, 1)

    # Sonar dataset
    # df = pd.read_csv('../data/sonar.csv')
    # train = df.iloc[:, :-1].values
    # test = df.iloc[:, 60].values
    # # train = df.iloc[:, [0, 20]].values
    # test = np.where(test == 'R', -1, 1)

    # Breast cancer dataset
    df = pd.read_csv('../data/breast_cancer.csv')
    df.drop(df.columns[[-1, 0]], axis=1, inplace=True)
    featureMeans = list(df.columns[1:12])  # ['radius_mean', 'texture_mean', 'perimeter_mean']
    df.diagnosis = df.diagnosis.map({'M': -1, 'B': 1})
    X = df.loc[:, featureMeans].values
    y = df.loc[:, 'diagnosis'].values

    # correlationData = df[featureMeans].corr()
    # sns.pairplot(df[featureMeans].corr(), diag_kind='kde', height=2)
    # plt.show()
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(df[featureMeans].corr(), annot=True, square=True, cmap='coolwarm')
    # plt.show()
    np.random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Training the perceptron model and show its metrics
    ppn1 = Perceptron(eta=0.01, n_iter=1000, random_state=2)
    ppn1.fit(X_train, y_train)
    print(ppn1.metrics(X_test, y_test))
    # print("----------------------------------")
    # ppn2 = Perceptron(eta=0.01, n_iter=50)
    # ppn2.fit(X_train, y_train)
    # ppn2.metrics(X_test, y_test)
    # print("----------------------------------")
    # ppn1.ensemble(ppn2)
    # ppn1.fit(X_train, y_train)
    # ppn1.metrics(X_test, y_test)

    # ----------------------------
    """
    Notes: 
    1) What I would instead suggest is : say you get one output from the trained RandomForests - o1 and another output
    from the trained XGBoost - o2. You can train a simple LinearRegression on top of these to learn your final
    latitudes/longitudes. This will ensure that the weights are learnt in way that minimize the loss from each classifier.
    
    2) 
    """
