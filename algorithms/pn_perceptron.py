from sklearn.model_selection import train_test_split
from peernet.helpers import log
from peernet.node import Node
from peernet.message import request_model
from peernet.protocol import RESPONSE_MODEL
from algorithms.perceptron import Perceptron


def mprob_perceptron(node: Node):
    """
    Training a perceptron model in a P2P architecture
    :param node: Node
    :return: None
    """

    df = node.ldata

    # Prepare and split data
    df.drop(df.columns[[-1, 0]], axis=1, inplace=True)
    featureMeans = list(df.columns[1:11])
    df.diagnosis = df.diagnosis.map({'M': -1, 'B': 1})
    X = df.loc[:, featureMeans].values.astype(float)
    y = df.loc[:, 'diagnosis'].values
    # Split data to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Training the solitary model
    sol_classifier = Perceptron(eta=0.01, n_iter=100, random_state=0)
    sol_classifier.fit(X_train, y_train)
    accuracy = sol_classifier.metrics(X_test, y_test, metric='accuracy')

    # Set node model
    node.model = sol_classifier
    log("success", f"{node.pname}: Solidarity model {accuracy}")

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
        ens_classifier = node.model.ensemble(peer_classifier)

        # Retrain and recalculate the accuracy of the ensemble model using local data
        ens_classifier.fit(X_train, y_train)
        node.model = ens_classifier
        accuracy = ens_classifier.metrics(X_test, y_test, metric='accuracy')
        log("success", f"{node.pname}: P[{node.peers[i].get('Name', i)}] Ensemble model {accuracy}")

        log('exception', f"END Range({i}) +++++++")

    log('success', f"{node.pname}: Final model accuracy: {accuracy}")


def colab_perceptron():
    pass
