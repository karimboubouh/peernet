import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from peernet.message import request_information, exchange_model, exchange_sol_model
import peernet.constants as c
from .datasets import fetch_mnist
from .helpers import log, wait_until, data_size, algo_from_dict


def model_propagation(node, wrapper, model, args=None):
    # todo Remove it
    node.stop_condition = c.STOP_CONDITION

    x_train, y_train, x_test, y_test = extract_data(node.ldata)

    algorithm_args = args.get('algorithm', None)
    params = args.get('protocol', None)
    behavior = args.get('behavior', None)

    if node.byzantine:
        m = random_model(wrapper(model), (10, x_train.shape[1]))
    else:
        # Local training
        m = wrapper(model, **algorithm_args)
        m.train(x_train, y_train)
    m.evaluate(x_test, y_test)

    # Set node model and announce finish local train
    node.solitary_model = m
    node.model = copy.deepcopy(m)
    c.ll_done()

    # Local accuracy
    ll_score = node.solitary_model.summary()['test_score']
    if params.get('results', None):
        log("result", f"{node.pname} >> Finishes Local training with Test Accuracy: {ll_score}")
    node.costs = []
    # node.costs.append(ll_score)

    # Wait for nodes to finish the local learning step
    wait_until(ll_finished, 600, 0.5)
    # Exchange necessary information if not already
    if params.get('confidence', None):
        for neighbor in node.peers:
            node.send(neighbor, request_information(node))
        wait_until(data_received, 600, 0.5, node)

    # Start the MP algorithm
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for the model reaches a target accuracy
    # todo wait until done : node.reached_target_accuracy == True
    wait_until(finished, 600, 0.5, node)

    # evaluate the final model
    node.model.evaluate(x_test, y_test)

    # print(f"{node.pname} pre={np.sum(node.solitary_model.weights)} | post = {np.sum(node.model.weights)}")

    # Print results
    if params.get('results', None):
        mp_score = node.model.summary()['test_score']
        log("result", f"{node.pname} >> [C:{node.c}] Local Test Accuracy: {ll_score} %| Smooth Model Accuracy:"
                      f" {mp_score} % | [{len(node.peers)} neighbors] [{data_size(node)} local data items]")


def controlled_model_propagation(node, wrapper, model, args=None):
    node.stop_condition = c.STOP_CONDITION

    x_train, y_train, x_test, y_test = extract_data(node.ldata)
    algorithm_args = args.get('algorithm', None)
    params = args.get('protocol', None)

    # Local training
    m = wrapper(model, **algorithm_args)
    m.train(x_train, y_train)
    if node.byzantine:
        m = random_model(wrapper(model), (10, x_train.shape[1]))
    else:
        # Local training
        m.train(x_train, y_train)
    m.evaluate(x_test, y_test)

    # Set node model and announce finish local train
    node.solitary_model = m
    node.model = copy.deepcopy(m)
    c.ll_done()

    # Local accuracy
    ll_score = node.solitary_model.summary()['test_score']
    if params.get('results', None):
        log("result", f"{node.pname} >> Finishes Local training with Test Accuracy: {ll_score}")
    node.costs = []
    # node.costs.append(ll_score)

    # Wait for nodes to finish the local learning step
    wait_until(ll_finished, 600, 0.5)

    # Start the MP algorithm
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for the model reaches a target accuracy
    # todo wait until done : node.reached_target_accuracy == True
    wait_until(finished, 600, 2, node)

    # evaluate the final model
    node.model.evaluate(x_test, y_test)

    # print(f"{node.pname} pre={np.sum(node.solitary_model.weights)} | post = {np.sum(node.model.weights)}")

    # Print results
    if params.get('results', None):
        mp_score = node.model.summary()['test_score']
        if c.CONFIDENCE_MEASURE == 'max':
            cf = 1 / np.max(list(node.cf.values()))
        else:
            cf = 1 / np.mean(list(node.cf.values()))
        log("result", f"{node.pname} >> [CF:{cf}] Local Test Accuracy: {ll_score} %| Smooth Model Accuracy:"
                      f" {mp_score} % | [{len(node.peers)} neighbors] [{data_size(node)} local data items]")


def fair_model_propagation(node, model, pre=None, params=None):
    # Pre-processing step
    X_train, X_test, y_train, y_test = pre(node.ldata)

    # Set nodes test data
    node.X_test = X_test
    node.y_test = y_test

    # Training the local model
    m = model()
    m.fit(X_train, y_train)
    m.evaluate(X_test, y_test)

    # Set node model
    node.solitary_model = node.model = m

    # declare that node finished the local learning
    c.ll_done()

    # Local accuracy
    score = m.summary()['test_score']
    # sol_accuracy = m.metrics(X_test, y_test)
    log("success", f"{node.pname} >> Local model score: {score}")

    # Add local cost to the list of costs (node.costs)
    node.costs = []
    node.costs.append(score)

    # Wait for nodes to finish the local learning step
    wait_until(ll_finished, 600, 0.5)

    # Exchange necessary information if not already
    if params['confidence']:
        for neighbor in node.peers:
            node.send(neighbor, request_information(node))
        # Wait until data received
        wait_until(data_received, 600, 0.5, node)

    # Start the MP algorithm
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for the model reaches a target accuracy
    wait_until(finished, 600, 0.5, node)

    # Print results
    if params['show_results']:
        accuracy = node.model.summary()['test_score']
        log("result", f"{node.pname} >> [C:{node.c}] Solitary model score: {score} %| Smooth model score:"
                      f" {accuracy} % | [{len(node.peers)} neighbors] [{data_size(node)} local data items]")


def ___model_propagation(node, model, pre=None, params=None):
    # Set decentralized learning stop Condition
    node.stop_condition = c.STOP_CONDITION

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
    c.ll_done()

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
    wait_until(ll_finished, 600, 0.5)
    # TODO Mini  batch gradient descent.
    # print(f"{node.pname} [{len(node.peers)}] >> SModel accuracy: {sol_accuracy} with loss {node.costs} after {epochs} epochs.")
    # exit(0)

    # Exchange necessary information if not already
    if params['confidence']:
        for neighbor in node.peers:
            # if not neighbor.get("data_size", None):
            node.send(neighbor, request_information(node))
        # Wait until data received
        wait_until(data_received, 600, 0.5, node)

    # Smooth the model over network
    neighbor = node.get_random_peer()
    exchange = exchange_model(node)
    node.send(neighbor, exchange)

    # Wait for Stop condition
    wait_until(finished, 600, 0.5, node)

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
    c.ll_done()
    node.model = m
    summary = m.summary(X_train, y_train, X_test, y_test)
    sol_accuracy = round(summary['test_acc'], 4)
    sol_loss = node.model.test_cost(X_test, y_test)
    node.costs.append(sol_loss)
    sol_loss = round(sol_loss, 4)

    log("success", f"{node.pname} >> Solitary model accuracy: {sol_accuracy}")

    #  ---- STEP 2 :: ---- : Collaborative Learning setup
    # Set decentralized learning stop Condition
    node.stop_condition = c.STOP_CONDITION
    node.Theta[node.name] = m.parameters
    node.Z[node.name] = m.parameters
    node.A[node.name] = 0
    node.rho = 4
    node.mu = 0.5
    wait_until(ll_finished, 600, 0.5)
    # Share solitaryModel with neighbors
    for neighbor in node.peers:
        node.send(neighbor, exchange_sol_model(node))
    time.sleep(2)

    #  ---- STEP 3 :: ---- : Start the Collaborative Learning process
    neighbor = node.get_random_peer()
    node.update_primal(neighbor)

    #  ---- STEP 4 :: ---- : Collect Collaborative Learning results
    wait_until(finished, 600, 0.5, node)
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


def one_node_ll(algorithm, dataset, pre, args):
    wrapper, model = algo_from_dict(algorithm)
    # fetch and preprocess data
    x_train, y_train, x_test, y_test = fetch_mnist(train_size=6000, pre=pre)

    # Training
    m = wrapper(model, **args)
    m.train(x_train, y_train)
    m.evaluate(x_test, y_test)

    # Summary of the training
    summary = m.summary()

    # Log results
    log("result", f"SOTA >> Train Accuracy: {summary['train_score']} % | "
                  f"Test Accuracy: {summary['test_score']} % | [{x_train.shape[0]} items]")

    return m


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


def extract_data(data: dict):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test


def random_model(model, shape):
    a = shape[0] if shape[0] > 1 else 2
    b = shape[1]
    rX = np.random.rand(a, b)
    rY = np.array([str(x) for x in range(a)])
    model.train(rX, rY)

    return model


def finished(node):
    # if node.stop_condition == 0:
    if not node.check_exchange():
        return True
    # if node.name == 'w3':
    #     print(f"{node.pname} ++ {node.stop_condition}")
    return False


def ll_finished():
    if c.TRAINED_MODELS == 0:
        return True
    return False


def data_received(node):
    for peer in node.peers:
        if 'data_size' not in peer:
            return False
    return True
