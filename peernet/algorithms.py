import random
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from peernet.message import request_information, exchange_model, exchange_sol_model, exchange_variables
import peernet.constants as sharedVars
from .helpers import log, wait_until, data_size, shuffle


def model_propagation(node, model, pre=None, params=None):
    # Set decentralized learning stop Condition
    node.stop_condition = sharedVars.STOP_CONDITION

    # Pre-processing step
    X_train, X_test, y_train, y_test = pre(node.ldata)

    # Set nodes test data
    node.X_test = X_test
    node.y_test = y_test
    # force None
    node.model, node.solitary_model = None, None

    # Training the solitary model
    min_epochs = 1
    max_epochs = 15
    # random.seed(100)
    # epochs = random.randrange(min_epochs, max_epochs)
    epochs = 9
    m = model(debug=params['debug'], name=node.name, epochs=epochs)
    # m = model(lr=4, epochs=9, debug=False, name=node.pname)

    m.fit(X_train, y_train)

    # Set node model
    node.solitary_model = m
    # sw = np.sum(node.solitary_model.parameters)
    node.model = m
    sharedVars.ll_done()

    # Local accuracy
    summary = m.summary(X_train, y_train, X_test, y_test)
    sol_accuracy = round(summary['test_acc'], 4)
    # sol_accuracy = m.metrics(X_test, y_test)
    log("success", f"{node.pname} >> Solitary model accuracy: {sol_accuracy}")

    # Add local cost to the list of costs (node.costs)
    node.costs = []
    # print(f"[{node.name}-{node.model.test_cost(X_test, y_test)}]", end="-")
    # node.costs.append(node.model.test_cost(X_test, y_test))
    node.costs.append(node.model.test_cost(X_test, y_test))
    wait_until(ll_finished, 600, 0.25)
    # TODO Mini  batch gradient descent.
    # print(f"{node.pname} [{len(node.peers)}] >> SModel accuracy: {sol_accuracy} with loss {node.costs} after {epochs} epochs.")
    # exit(0)

    # Exchange necessary information if not already
    if params['confidence']:
        for neighbor in node.peers:
            # if not neighbor.get("data_size", None):
            node.send(neighbor, request_information(node))
        # Wait until data received
        wait_until(data_received, 600, 0.25, node)

    # Smooth the model over network
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for Stop condition
    wait_until(finished, 600, 0.25, node)

    # Print results
    if params['show_results']:
        summary = node.model.summary(X_train, y_train, X_test, y_test)
        accuracy = summary['test_acc']
        # mw = np.sum(node.model.parameters)
        log("result",
            f"{node.pname} >> [C:{node.c}] Solitary model accuracy: {round(sol_accuracy, 4)} % | Smooth model Accuracy:"
            f" {round(accuracy, 4)} % | [{len(node.peers)} neighbors] [{data_size(node)} local data items]"
            )


def collaborative_learning(node, model, pre=None, params=None):
    # ---- STEP 1 :: ---- : Warm-start strategy using solitaryModel
    X_train, X_test, y_train, y_test = pre(node.ldata)
    node.X = X_train
    node.y = y_train
    node.X_test = X_test
    node.y_test = y_test
    node.model, node.solitary_model = None, None
    m = model(debug=params['debug'], node_name=node.pname, epochs=params['epochs'])
    m.fit(X_train, y_train)
    node.solitary_model = m
    sharedVars.ll_done()
    node.model = m
    summary = m.summary(X_train, y_train, X_test, y_test)
    sol_accuracy = round(summary['test_acc'], 4)
    sol_loss = node.model.test_cost(X_test, y_test)
    node.costs.append(sol_loss)
    sol_loss = round(sol_loss, 4)

    log("success", f"{node.pname} >> Solitary model accuracy: {sol_accuracy}")

    #  ---- STEP 2 :: ---- : Collaborative Learning setup
    # Set decentralized learning stop Condition
    node.stop_condition = sharedVars.STOP_CONDITION
    node.Theta[node.name] = m.parameters
    node.Z[node.name] = m.parameters
    node.A[node.name] = 0
    node.rho = 4
    node.mu = 0.5
    wait_until(ll_finished, 600, 0.25)
    # Share solitaryModel with neighbors
    for neighbor in node.peers:
        node.send(neighbor, exchange_sol_model(node))
    time.sleep(2)

    #  ---- STEP 3 :: ---- : Start the Collaborative Learning process
    neighbor = node.get_random_peer()
    node.update_primal(neighbor)

    #  ---- STEP 4 :: ---- : Collect Collaborative Learning results
    wait_until(finished, 600, 0.25, node)
    # Print results
    node.model.parameters = node.Theta[node.name]

    summary = m.summary(X_train, y_train, X_test, y_test)
    accuracy = round(summary['test_acc'], 4)
    loss = round(node.costs[-1], 4)
    log("result",
        f"{node.pname} >> [{node.stop_condition}] Solitary model accuracy: {sol_accuracy} % and loss: {sol_loss}"
        f"| Smooth model Accuracy [CL]: {accuracy} % and loss: {loss}| [{len(node.peers)} neighbors] [{data_size(node)}"
        f"local data items]")


def local_learning(node, model, pre=None, params=None):
    # Pre-processing step
    X_train, X_test, y_train, y_test = pre(node.ldata)

    # Training the solitary model
    m = model(debug=params['debug'], name=node.name, epochs=9)
    m.fit(X_train, y_train)

    # Summary of the training
    summary = m.summary(X_train, y_train, X_test, y_test)
    acc = round(summary['test_acc'], 4)

    # Set node model
    node.model = m

    # Log results
    log("result", f"{node.pname} >> Model Accuracy: {acc} % | [{len(node.peers)} neighbors] [{X_train.shape[1]} items]")


def one_node_ll(model, dataset, pre, params):
    # Pre-processing step
    X, y = joblib.load(f"./datasets/{dataset}")
    X_train = X[:60000]
    y_train = y[:60000]
    x_test = X[60000:]
    y_test = y[60000:]
    X_train, y_train = shuffle(X_train, y_train)

    data = {
        'x_train': pd.DataFrame(X_train, dtype=float),
        'y_train': pd.DataFrame(y_train, dtype=float),
        'x_test': pd.DataFrame(x_test[()], dtype=float),
        'y_test': pd.DataFrame(y_test[()], dtype=float),
    }
    X_train, X_test, y_train, y_test = pre(data)

    # Training the solitary model
    debug = params.get('debug', False)
    epochs = params.get('epochs', 9)
    m = model(debug=params['debug'], name='SOTA', epochs=epochs)
    m.fit(X_train, y_train)

    # Summary of the training
    summary = m.summary(X_train, y_train, X_test, y_test)
    train_acc = round(summary['train_acc'], 4)
    test_acc = round(summary['test_acc'], 4)

    # Log results
    log("result", f"SOTA >> Train Accuracy: {train_acc} % | Test Accuracy: {test_acc} % | [{X_train.shape[1]} items]")

    return test_acc


# ------------------------------ Local functions ------------------------------

def analysis(node, atype='confidence'):
    # print('# Analysis: ', end='')
    if atype == 'confidence':
        # print('Confidence')
        costs = node.costs
        plt.figure()
        plt.plot(np.squeeze(costs), label=f"Node({node.name})")
        plt.ylabel('cost')
        plt.xlabel('Iterations')
        legend = plt.legend(loc='upper right', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        plt.show()


def finished(node):
    if node.stop_condition == 0:
        return True
    # print(f"{node.pname} ++ {node.stop_condition}")
    return False


def ll_finished():
    if sharedVars.TRAINED_MODELS == 0:
        return True
    return False


def data_received(node):
    for peer in node.peers:
        if 'data_size' not in peer:
            return False
    return True
