from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump, load
import pandas as  pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.classifier import EnsembleVoteClassifier

with open("../data/iris.csv") as file:
    data = pd.read_csv(file)
    msk = np.random.rand(len(data)) < 0.5
    df1 = data[msk]
    df2 = data[~msk]
    print(df1.shape)
    print(df2)
    X1 = df1.iloc[:, :-1].values
    y1 = df1.iloc[:, 4].values
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20)

    X2 = df2.iloc[:, :-1].values
    y2 = df2.iloc[:, 4].values
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20)

    c1 = KNeighborsClassifier(n_neighbors=2)
    c1.fit(X1_train, y1_train)

    c2 = KNeighborsClassifier(n_neighbors=6)
    c2.fit(X2_train, y2_train)

    y1_pred = c1.predict(X1_test)
    print("C1-----------------------------------")
    print(classification_report(y1_test, y1_pred))

    y2_pred = c2.predict(X2_test)
    print("C2-----------------------------------")
    print(classification_report(y2_test, y2_pred))

    c3 = EnsembleVoteClassifier(clfs=[c1, c2], weights=[1, 2], voting='soft')
    c3.fit(X2_train, y2_train)

    y3_pred = c3.predict(X1_test)
    print("C3-----------------------------------")
    print(classification_report(y1_test, y3_pred))

    c = EnsembleVoteClassifier(clfs=[c3, c2], weights=[1, 2], voting='soft')
    c.fit(X2_train, y2_train)

    y_pred = c.predict(X1_test)
    print("C-----------------------------------")
    print(classification_report(y1_test, y_pred))


# class BasicIrisClassifier:
#     def load(self):
#         iris = load_iris()
#         self.data = iris.data
#         self.target = iris.target
#         self.target_names = iris.target_names
#
#     def train(self):
#         data_train, data_test, target_train, target_test = train_test_split(self.data, self.target, test_size=0.3,
#                                                                             random_state=12)
#
#         self.classifier = KNeighborsClassifier()
#         self.classifier.fit(data_train, target_train)
#
#         target_pred = self.classifier.predict(data_test)
#         accuracy = metrics.accuracy_score(target_test, target_pred)
#
#         return accuracy
#
#     def predict(self, external_input_sample):
#         prediction_raw_values = self.classifier.predict(external_input_sample)
#         prediction_resolved_values = [self.target_names[p] for p in prediction_raw_values]
#         return prediction_resolved_values
#
#     def saveModel(self):
#         print(self.classifier)
#         dump(self.classifier, 'trained_iris_model.pkl')
#         dump(self.target_names, 'trained_iris_model_targetNames.pkl')
#
#     def loadModel(self):
#         self.classifier = load('trained_iris_model.pkl')
#         self.target_names = load('trained_iris_model_targetNames.pkl')
#
#
# # Using BasicIrisClassifier
# external_input_sample = [[5, 2, 4, 1], [6, 3, 5, 2], [5, 4, 1, 0.5]]
# basic_iris_classifier = BasicIrisClassifier()
#
# basic_iris_classifier.load()
#
# accuracy = basic_iris_classifier.train()
# print("Model Accuracy:", accuracy)
#
# prediction = basic_iris_classifier.predict(external_input_sample)
# print("Prediction for {0} => \n{1}".format(external_input_sample, prediction))
#
# basic_iris_classifier.saveModel()
#
# basic_iris_classifier.loadModel()
