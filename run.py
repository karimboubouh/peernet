import mp_expirements as mp_exp
from peernet.algorithms import one_node_ll
from peernet.pre_processing import load_mnist_12
from algorithms.LogisticRegression import LogisticRegression


def model_propagation():
    # stop condition == target accuracy
    config = {
        'nodes': 2,
        'topology': 'random',  # (static, ErdosRenyi)
        'data': {'dataset': 'mnist.data', 'iid': True, 'balanced': False},
        'algorithm': {'model': 'logistic', 'confidence': True},
        'verbose': 0
    }
    mp_exp.communication_rounds(**config)


def sota():
    config = {
        'model': LogisticRegression,
        'dataset': 'mnist.data',
        'pre': load_mnist_12,
        'params': {'debug': True, 'epochs': 9}
    }
    one_node_ll(**config)


if __name__ == '__main__':
    # model_propagation()
    sota()
