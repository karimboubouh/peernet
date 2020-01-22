from algorithms.SoftMaxRegression import SoftMaxRegression
from peernet.helpers import bold
from peernet.network.arch import random_network, static_network
from algorithms.LogisticRegression import LogisticRegression
import expirements
from lib.graph.generator.erdos_renyi import ErdosRenyi
from pre_processing import load_banknotes, load_mnist_12, load_mnist


def model_propagation():
    iterations = {'type': "iterations", 'iterations': (0, 50, 1)}
    unbalancedness = {'type': "unbalancedness", "SC": 20}
    sparsity = {'type': "sparsity", "SC": 10}
    communication = {'type': "communication", "SC": 100}

    # for i in range(55, 105, 5):
    mnist_12__mp = {
        "model": LogisticRegression,
        "config": (10, 40000),  # number of clients
        "network": ErdosRenyi,
        "dataset": ["mnist.data", True, 0, ';', load_mnist_12],
        "confidence": True,
        "analysis": sparsity
    }
    # Experiment MP x Iterations analysis
    # expirements.experiment_mp_iter(**mnist_12__mp)
    # expirements.experiment_mp_unbalancedness(**mnist_12__mp)
    # expirements.experiment_mp_data(**mnist_12__mp)
    expirements.experiment_mp_graph_sparsity(**mnist_12__mp)
    # expirements.experiment_mp_communication(**mnist_12__mp)


def collaborative_learning():
    iterations = {'type': "iterations", 'iterations': (0, 10, 1)}
    empty = {'type': None}

    mnist_12__mp = {
        "model": LogisticRegression,
        "config": (10, 40000),
        "network": random_network,
        "dataset": ["mnist.data", True, 0, ';', load_mnist_12],
        "analysis": iterations
    }

    # Experiment
    expirements.experiment_cl_iter(**mnist_12__mp)


def local_learning():
    mnist_12__mp = {
        "model": LogisticRegression,
        "config": (10, 40000),
        "network": random_network,
        "dataset": ["mnist.data", True, 0, ';', load_mnist],
        "confidence": True,
        "analysis": {'type': "none"}
    }

    # Experiment MP x Iterations analysis
    expirements.experiment_ll(**mnist_12__mp)


if __name__ == '__main__':
    print(bold('Starting simulation ...'))
    model_propagation()
    # collaborative_learning()
    # local_learning()
    print(bold('Done ...'))

# OPENBLAS_NUM_THREADS=1
# GOTO_NUM_THREADS=1
# OMP_NUM_THREADS=1
