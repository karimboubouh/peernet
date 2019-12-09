from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.utils import shuffle

import numpy as np
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()

X, y = shuffle(iris.data, iris.target)
X = X[:20]
y = y[:20]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# clf1 = LogisticRegression(random_state=1, multi_class="auto", solver='lbfgs', max_iter=1000)
# clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
# clf3 = GaussianNB()

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = KNeighborsClassifier(n_neighbors=2)

print('5-fold cross validation:\n')

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1], voting='soft')

labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))