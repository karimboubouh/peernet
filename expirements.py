"""
    Benchmark experiments
    ~~~~~~~~~~~~~~~~~~~~~
    Experiments used to benchmark the different setting of both MP and CL
"""
import time
import numpy as np

from peernet.PeerNet import PeerNet
import peernet.constants as sharedVars
from peernet.helpers import save
from peernet.constants import DATASETS_FOLDER
import plots


# ------------------------- Model Propagation / Iterations x Cost -------------

def experiment_mp_iter(model, config, network, dataset, confidence, analysis):
    """Model Propagation / Iterations x Cost"""

    # init PeerNet with a configuration option
    p2p = PeerNet(config)

    # Setup the P2P network and the communication between neighbors
    p2p.network(network).init()

    # Load and randomly distribute training samples between nodes
    p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3])

    start_time = time.time()
    iterations = analysis['iterations']
    sharedVars.STOP_CONDITION = iterations[1]

    p2p.train(
        model=model,
        pre=dataset[4],
        algorithm="MP",
        params={'confidence': confidence, 'debug': False, 'show_results': False},
        analysis=analysis['type']
    )
    print(f"\nSWITCH {sharedVars.STOP_CONDITION} done in {time.time() - start_time} seconds")
    # Plotting
    x = range(iterations[0], iterations[1] + 1, iterations[2])
    y = p2p.results
    # print(y)
    save(f"OLD_results/mp_iterations_{confidence}", (x, y))
    labels = {'x': "Iterations", 'y': "Cost", 'title': "Iterations X Cost"}
    # plots.iterations(x, y, labels)
    # ddd
    start_time = time.time()
    iterations = analysis['iterations']
    sharedVars.STOP_CONDITION = iterations[1]

    p2p.train(
        model=model,
        pre=dataset[4],
        algorithm="MP",
        params={'confidence': not confidence, 'debug': False, 'show_results': False},
        analysis=analysis['type']
    )
    print(f"\nSWITCH {sharedVars.STOP_CONDITION} done in {time.time() - start_time} seconds")
    # Plotting
    x = range(iterations[0], iterations[1] + 1, iterations[2])
    y = p2p.results
    # print(y)
    save(f"OLD_results/mp_iterations_{not confidence}", (x, y))
    labels = {'x': "Iterations", 'y': "Cost", 'title': "Iterations X Cost"}
    # plots.iterations(x, y, labels)

    fileA = "./results/mp_iterations_True"
    fileB = "./results/mp_iterations_False"
    info = {
        'xlabel': "Iterations",
        # 'ylabel': "Test accuracy",
        'ylabel': "Test cost",
        'title': "MP with and without confidence w.r.t.the number of iterations."
    }
    plots.file_mp_iter(fileA, fileB, info)


# ------------------------- Model Propagation / Data unbalancedness x Cost ----

def experiment_mp_unbalancedness(model, config, network, dataset, confidence, analysis):
    """Model Propagation / Data unbalancedness x Cost"""

    p2p = PeerNet(config)

    # Setup the P2P network and the communication between neighbors
    p2p.network(network).init()

    # Train the model using one of the approaches: "MP", "CL" or "LL"
    sharedVars.STOP_CONDITION = analysis['SC']
    a = p2p.epsilon
    while p2p.epsilon <= 1:
        print(f"SWITCH: {p2p.epsilon}")
        start_time = time.time()
        p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3],
                         data_distribution="uniform")
        show_results = p2p.epsilon == 1
        p2p.train(
            model=model,
            pre=dataset[4],
            algorithm="MP",
            params={'confidence': confidence, 'debug': False, 'show_results': False},
            analysis=analysis['type']
        )
        print(f"\nSWITCH done in {time.time() - start_time} seconds")

    b = p2p.epsilon
    # Plotting
    x = np.arange(0, 1 + sharedVars.EPSILON_STEP, sharedVars.EPSILON_STEP)
    y = p2p.results
    save(f"OLD_results/mp_epsilon_{analysis['SC']}_{confidence}", (x, y))

    # Train the model using one of the approaches: "MP", "CL" or "LL"
    sharedVars.STOP_CONDITION = analysis['SC']

    # new
    p2p.epsilon = 0.0
    p2p.results = []
    a = p2p.epsilon
    while p2p.epsilon <= 1:
        print(f"SWITCH: {p2p.epsilon}")
        start_time = time.time()
        p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3],
                         data_distribution="uniform")
        show_results = p2p.epsilon == 1
        p2p.train(
            model=model,
            pre=dataset[4],
            algorithm="MP",
            params={'confidence': not confidence, 'debug': False, 'show_results': False},
            analysis=analysis['type']
        )
        print(f"\nSWITCH done in {time.time() - start_time} seconds")

    b = p2p.epsilon
    # Plotting
    x = np.arange(0, 1 + sharedVars.EPSILON_STEP, sharedVars.EPSILON_STEP)
    y = p2p.results
    save(f"OLD_results/mp_epsilon_{analysis['SC']}_{not confidence}", (x, y))
    # plots
    fileA = f"./results/mp_epsilon_{analysis['SC']}_{confidence}"
    fileB = f"./results/mp_epsilon_{analysis['SC']}_{not confidence}"
    info = {
        'xlabel': "Width ε",
        'ylabel': "Cost",
        'title': "MP with and without confidence w.r.t. data unbalancednesss."
    }
    plots.file_mp_iter(fileA, fileB, info)


def experiment_mp_data(model, config, network, dataset, confidence, analysis):
    """Model Propagation / Data unbalancedness x Cost"""

    p2p = PeerNet(config)
    p2p.network(network).init()
    output = []
    for i in [0.0, 1.0]:  # np.arange(0.9, 1.01, 0.1)
        results = []
        for j in range(10):
            p2p.epsilon = i
            p2p.results = []
            print(f"RUN: {i}:{j}")
            start_time = time.time()
            sharedVars.STOP_CONDITION = 100
            data = f"{DATASETS_FOLDER}/{dataset[0]}"
            p2p.load_dataset(data, df=dataset[1], min_samples=dataset[2], sep=dataset[3], data_distribution="uniform")
            p2p.train(
                model=model,
                pre=dataset[4],
                algorithm="MP",
                params={'confidence': confidence, 'debug': False, 'show_results': False},
                analysis='data'
            )
            results.append(p2p.results)
            print(f"\nRUN done in {time.time() - start_time} seconds")
        rr = np.mean(results)
        print(f"SWITCH {i} :: +++++++++++++++++++++ DONE R= {rr}")
        output.append(rr)

    print("Experiment done ++")
    print(output)


# ------------------------- Model Propagation / Data unbalancedness x Cost ----

def experiment_mp_graph_sparsity(model, config, network, dataset, confidence, analysis):
    # Train the model using one of the approaches: "MP", "CL" or "LL"
    p = 0.0
    a = p
    results = []
    while p <= 1:
        print(f"SWITCH: {p}")
        sharedVars.STOP_CONDITION = analysis['SC']
        start_time = time.time()
        p2p = PeerNet(config)
        p2p.network(network, p).init()
        p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3])
        show_results = p == 0.9  # and False
        print(f"PEERS: {sum([len(node.peers) for node in p2p.nodes])}")
        p2p.train(
            model=model,
            pre=dataset[4],
            algorithm="MP",
            params={'confidence': confidence, 'debug': False, 'show_results': show_results, 'epochs': 200},
            analysis=analysis['type']
        )
        print(f"\nSWITCH {p} done in {time.time() - start_time} seconds")
        results.append(p2p.results)
        print(f"COST={p2p.results}")
        p2p.stop()
        p = round(p + 0.1, 1)
    b = p
    # Plotting
    x = np.arange(a, b, 0.1)
    y1 = results
    print(y1)

    save(f"OLD_results/mp_sparsity_{analysis['SC']}_{confidence}", (x, y1))
    # labels = {'x': "Graph Sparsity", 'y': "Cost", 'title': "Graph Sparsity X Cost"}
    # plots.iterations(x, y1, labels)


# ------------------------- Model Propagation / Measure x Communication ----

def experiment_mp_communication(model, config, network, dataset, confidence, analysis):
    start_time = time.time()
    sharedVars.STOP_CONDITION = analysis['SC']
    p2p = PeerNet(config)
    p = 0.94
    p2p.network(network, p).init()
    p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3])
    p2p.train(
        model=model,
        pre=dataset[4],
        algorithm="MP",
        params={'confidence': confidence, 'debug': False, 'show_results': False},
        analysis=analysis['type']
    )
    print(f"\nDone in {time.time() - start_time} seconds")
    print(f"Number of communication with {p} nodes for {analysis['SC']} == {p2p.results}")
    # p2p.stop()


# ------------------------- Collaborative Learning / Iterations x Cost -------------

def experiment_cl_iter(model, config, network, dataset, analysis):
    """Collaborative Learning / Iterations x Cost."""

    p2p = PeerNet(config)

    p2p.network(network).init()

    data = f"{DATASETS_FOLDER}/{dataset[0]}"
    p2p.load_dataset(data, df=dataset[1], min_samples=dataset[2], sep=dataset[3])

    start_time = time.time()
    iterations = analysis['iterations']
    sharedVars.STOP_CONDITION = iterations[1]
    p2p.train(
        model=model,
        pre=dataset[4],
        algorithm="CL",
        params={'debug': False, 'show_results': True, 'epochs': 100},
        analysis=analysis['type']
    )
    print(f"\nSWITCH {sharedVars.STOP_CONDITION} done in {time.time() - start_time} seconds")

    # Plotting
    x = range(iterations[0], iterations[1] + 1, iterations[2])
    y = p2p.results
    print("X:")
    print(x)
    print("Y:")
    print(y)
    save(f"OLD_results/cl_iterations_iter_{iterations[1]}", (x, y))
    labels = {'x': "Iterations", 'y': "Cost", 'title': "Iterations X Cost"}
    plots.iterations(x, y, labels)


# ------------------------- Local Learning / Accuracy -------------------------

def experiment_ll(model, config, network, dataset, confidence, analysis):
    """
    @param model: Used machine learning model
    @param config: configuration file or a tuple in the form (Number of nodes, INIT_PORT)
    @param network: random or  static
    @param dataset: tuple (dataset_file, df?, min_samples, separator, pre_processing)
    @param confidence: use confidence or not
    @param analysis:
    """
    # init PeerNet with a configuration option
    p2p = PeerNet(config)

    # Setup the P2P network and the communication between neighbors
    p2p.network(network).init()

    # Load and randomly distribute training samples between nodes
    p2p.load_dataset(f"./datasets/{dataset[0]}", df=dataset[1], min_samples=dataset[2], sep=dataset[3])

    # Train the model using one of the approaches: "MP", "CL" or "LL"
    p2p.train(
        model=model,
        pre=dataset[4],
        algorithm="LL",
        params={'confidence': confidence},
        analysis=analysis['type']
    )
