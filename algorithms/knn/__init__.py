import pandas as  pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.classifier import EnsembleVoteClassifier

from peernet.helpers import log
from peernet.node import Node
from peernet.message import request_model
from peernet.protocol import RESPONSE_MODEL


def lp_knn(node: Node, k: int):
    df = node.ldata
    # Training a solidarity model
    X1 = df.iloc[:, :-1].values
    y1 = df.iloc[:, 4].values

    # Splitting data: train/test
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20)

    # Train a KNN sol_classifier
    sol_classifier = KNeighborsClassifier(n_neighbors=k)
    sol_classifier.fit(X1_train, y1_train)

    y_pred = sol_classifier.predict(X1_test)
    accuracy = accuracy_score(y1_test, y_pred)

    # Set node model
    node.model = sol_classifier
    log("success", f"Node({node.name}) Solidarity model accuracy : {accuracy}")

    for i in range(len(node.peers)):
        log(f'Range: {i}')
        peer = node.get_random_peer()
        req = request_model(node)
        node.send(peer, req)
        res = node.wait_response(peer, RESPONSE_MODEL)
        rem_classifier = res['payload']['model']
        if rem_classifier is None:
            continue
        ens_classifier = EnsembleVoteClassifier(clfs=[node.model, rem_classifier], weights=[2, 1], voting='soft')
        ens_classifier.fit(X1_train, y1_train)
        y_pred = ens_classifier.predict(X1_test)
        node.model = ens_classifier
        accuracy = accuracy_score(y1_test, y_pred)
        log("success", f"Node({node.name}) Ensemble model accuracy : {accuracy}")
        log('exception', '---------------------------------------------------')
