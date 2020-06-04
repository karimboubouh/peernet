import numpy as np
import random
import plots
import mp_expirements as mp_exp
from peernet.algorithms import one_node_ll
from peernet.preprocessing import pre_mnist


def sota():
    """State Of the Art Accuracy"""
    config = {
        'algorithm': {'wrapper': 'sklearn', 'model': 'logistic'},
        'dataset': 'mnist.data',
        'pre': pre_mnist,
        'args': {'solver': 'saga', 'tol': 1e-1, 'C': 1e4}
    }
    m = one_node_ll(**config)


def experiment():
    config = {
        'nodes': 10,
        'topology': 'random',  # (static, random, ErdosRenyi)
        'data': {'dataset': 'mnist.data', 'pre': pre_mnist, 'iid': True, 'balancedness': 1},
        'algorithm': {'wrapper': 'sklearn', 'model': 'logistic'},
        'protocol': 'CDPL',  # MP or CDPL
        'args': {
            'behavior': {'Byzantine': 2, 'model': 'random'},  # Byzantine: -1: no byzantine
            'algorithm': {'solver': 'saga', 'tol': 1e-1, 'C': 1e4},
            'protocol': {'confidence': True, 'results': False}
        }
    }

    # communication rounds experiment
    # file = mp_exp.communication_rounds(**config)
    # plots.figure(file, config)

    # byzantine resilience
    file = mp_exp.byzantine(**config)
    plots.figure(file, config)

    # contribution_factor
    # file = mp_exp.contribution_factor(**config)
    # plots.contribution_factor(file)

    # Byzantine detection precision
    # file = mp_exp.byzantine_metrics(**config)
    # plots.byzantine_metrics(file)


if __name__ == '__main__':
    # Fix seed
    np.random.seed(0)
    random.seed(0)

    # run a given experiment
    experiment()

    # plot the result of experiments including MP/CDPL
    # plots.plot(50, "byzantine")

    # plot accuracy of one node
    # sota()
