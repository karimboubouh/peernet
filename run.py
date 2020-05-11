import numpy as np
import random
import plots
import mp_expirements as mp_exp
from peernet.algorithms import one_node_ll
from peernet.preprocessing import pre_mnist


def experiment():
    config = {
        'nodes': 20,
        'topology': 'random',  # (static, random, ErdosRenyi)
        'data': {'dataset': 'mnist.data', 'pre': pre_mnist, 'iid': True, 'balancedness': 1},
        'algorithm': {'wrapper': 'sklearn', 'model': 'logistic'},
        'protocol': 'CMP',  # CMP or MP
        'args': {
            'behavior': {'Byzantine': -1, 'model': 'random'},  # Byzantine: -1: no byzantine
            'algorithm': {'solver': 'saga', 'tol': 1e-1, 'C': 1e4},
            'protocol': {'confidence': True, 'results': True}
        }
    }
    # communication_rounds
    file = mp_exp.communication_rounds(**config)
    # contribution_factor
    # file = mp_exp.contribution_factor(**config)
    # data unbalancedness
    # file = mp_exp.data_unbalancedness(**config)
    # graph_sparsity
    # file = mp_exp.graph_sparsity(**config)
    #  byzantine_metrics
    # file = mp_exp.byzantine_metrics(**config)


def plot(nodes, analysis):
    N = f"results/{analysis}_{nodes}_N"  # MP without confidence
    C = f"results/{analysis}_{nodes}_C"  # MP with confidence
    F = f"results/{analysis}_{nodes}_F"  # CMP with confidence
    plots.communication_rounds(N, C, F)


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    experiment()
    # plot(20, "communication_rounds")
    # sota()


def ppp():
    # path = "/Users/boubouhkarim/Desktop/Final/5_CR-NON-IID/Data/20"
    # N = f"{path}/communication_rounds_NON_IID20_N_0.3"
    # C = f"{path}/communication_rounds_NON_IID20_C_0.3"
    # F = f"{path}/communication_rounds_NON_IID20_F_0.3"
    # plots.communication_rounds(N, C, F, ylabel="Loss")  # , ylabel="Loss"
    analysis = "byzantine"
    A = f"results/{analysis}_{20}_N"
    B = f"results/{analysis}_{20}_C"
    C = f"results/{analysis}_{20}_F"
    plots.communication_rounds(A, B, C, ylabel="F1 score")  # Recall


def sota():
    """State Of the Art Accuracy"""
    config = {
        'algorithm': {'wrapper': 'sklearn', 'model': 'logistic'},
        'dataset': 'mnist.data',
        'pre': pre_mnist,
        'args': {'solver': 'saga', 'tol': 1e-1, 'C': 1e4}
    }
    m = one_node_ll(**config)


def defaults():
    # >> SOTA ---------------------------------------------------------------------
    # 60000 > Train Accuracy: 0.9307 % | Test Accuracy: 0.9255 % | [60000 items]
    # 10000 > Train Accuracy: 0.9307 % | Test Accuracy: 0.9094 % | [10000 items]
    # 5000  > Train Accuracy: 0.94 % | Test Accuracy: 0.903 % | [5000 items]
    # 1000  > Train Accuracy: 0.938 % | Test Accuracy: 0.8622 % | [1000 items]
    # 500   > Train Accuracy: 0.972 % | Test Accuracy: 0.8214 % | [500 items]
    # 100   > Train Accuracy: 0.99 %  | Test Accuracy: 0.677 % | [100 items]

    # config template

    full_list_config = {
        'nodes': 20,
        'topology': 'random',  # (static, ErdosRenyi)
        'data': {'dataset': 'mnist.data', 'pre': pre_mnist, 'iid': True, 'balanced': False, 'epsilon': 0.1},
        'algorithm': {'wrapper': 'sklearn', 'model': 'logistic'},
        'args': {
            'algorithm': {'solver': 'saga', 'tol': 1e-1, 'C': 1e4},
            'protocol': {'confidence': True, 'results': True}
        },
        'verbose': 0
    }
