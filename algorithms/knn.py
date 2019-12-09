from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import VotingClassifier

from peernet.helpers import log
from peernet.node import Node
from peernet.message import request_model
from peernet.protocol import RESPONSE_MODEL


def lp_knn(node: Node, k: int):
    """
    Training a solidarity model
    :param node: Node
    :param k: int
    :return: None
    """
    df = node.ldata

    # Prepare and split data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Training a solidarity model: KNN classifier
    sol_classifier = KNeighborsClassifier(n_neighbors=k)
    sol_classifier.fit(X_train, y_train)

    # Measure the accuracy of the model
    y_pred = sol_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Set node model
    node.model = sol_classifier
    log("success", f"{node.pname}: Solidarity model accuracy : {accuracy}")

    for i in range(len(node.peers)):
        log(f'{node.pname}: Range({i}): {node.peers[i].get("Name", i)}')

        # Request a model from a random peer
        peer = node.get_random_peer()
        req = request_model(node)
        node.send(peer, req)
        res = node.wait_response(peer, RESPONSE_MODEL)
        peer_classifier = res['payload']['model']
        if peer_classifier is None:
            log(f'{node.pname}: {node.peers[i].get("Name", i)} has no model!')
            continue

        # Ensemble the current node model with the peer model
        # ens_classifier = EnsembleVoteClassifier(clfs=[node.model, peer_classifier], weights=[2, 1], voting='hard')
        ens_classifier = VotingClassifier(
            estimators=[('model', node.model), ('peer', peer_classifier)], weights=[1, 1], voting='soft'
        )

        # Train and calculate the accuracy of the ensemble model using local data
        ens_classifier.fit(X_train, y_train)
        y_pred = ens_classifier.predict(X_test)
        node.model = ens_classifier
        accuracy = accuracy_score(y_test, y_pred)

        log("success", f"{node.pname}: P[{node.peers[i].get('Name', i)}] Ensemble model accuracy : {accuracy}")
        log('exception', f"END Range({i}) +++++++")
    log('success', f"{node.pname}: Final model accuracy: {accuracy}")


"""
Clusters of nodes
Encrypted model to calculate proportions (ranges) of data
"""
