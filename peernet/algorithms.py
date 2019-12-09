import time

from peernet.message import request_information, exchange_model
from peernet.node import Node
from .helpers import log, wait_until, bold


def model_propagation(node: Node, model, pre=None, params=None):
    # Exchange necessary information if not already
    for neighbor in node.peers:
        if not neighbor.get("data_size", None):
            node.send(neighbor, request_information(node))

    # Set decentralized learning stop Condition
    node.stop_condition = len(node.peers)

    # Pre-processing step
    X_train, X_test, y_train, y_test = pre(node.ldata)

    # Training the solitary model
    m = model()
    m.fit(X_train, y_train)
    sol_accuracy = m.metrics(X_test, y_test)

    # Set node model
    node.solitary_model = m
    node.model = m
    log("success", f"{node.pname} >> Solitary model accuracy: {sol_accuracy}")

    # Wait for others to calculate solitary model
    s = 2
    log("info", f"Sleeping for {s} seconds...")
    time.sleep(s)

    # Smooth the model over network
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for Stop condition
    wait_until(finished, 10, 0.25, node)

    # Print results
    accuracy = node.model.metrics(X_test, y_test)
    log("result",
        f"{node.pname} >> Solitary model accuracy: {round(sol_accuracy)} % | Smooth model Accuracy: {round(accuracy)} %"
        f" | [{len(node.peers)} neighbors] [{len(node.ldata)} local data items]")


def collaborative_learning(node, model, pre=None):
    log(f"Model: {model}, pre: {pre}")


def finished(node):
    if node.stop_condition == 0:
        return True
    return False
