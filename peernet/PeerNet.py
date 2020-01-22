import csv
import h5py
import joblib
import threading
import time
import random
import pandas as pd
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename

from . import algorithms
from .node import Node
from .pnconfig import PNConfig
from .helpers import log, sample, bold, save, shuffle, sample_xy, random_number, data_size
from .constants import ALGORITHMS, DEBUG_LEVEL, EPSILON_STEP
import peernet.constants as sharedVars


# Class PeerNet ---------------------------------------------------------

class PeerNet:
    verbose = 4
    def __init__(self, config_file):
        # Initialization
        self.config = PNConfig(config_file)
        self.nodes = self.config.get_nodes()  # type: List[Node]
        self.epsilon = 0.0
        self.results = []
        random.seed(100)
        np.random.seed(100)

    def init(self):
        for node in self.nodes:
            # instantiate node
            node.init()
            # Start the threads
            node.start()

        for node in self.nodes:
            status = node.connect_neighbors()
            log('info', f"{node.pname} >> Connection status: {status}")

        return self

    def stop(self):
        # Stop the nodes
        print('Stoping nodes')
        for node in self.nodes:
            node.stop()
        for node in self.nodes:
            node.join()
        print('Nodes joined')

    def network(self, network_strategy, p=None):
        """Network architecture """
        if p is not None:
            er = network_strategy(self.nodes, p, seed=False)
            er.generate_graph()
        else:
            network_strategy(self.nodes)

        return self

    @staticmethod
    def sleep(s):
        log('info', f"Sleep for {s} second(s)...")
        time.sleep(s)

    def load_dataset(self, dataset, df=False, min_samples=1, dtype=float, sep=',', data_distribution="random"):
        """
        Load a dataset file and split it randomly with random size
        over the list of nodes in the PeerNet.
        :param dataset: str
        :param df: DataFrame
        :param min_samples: int minimum number of training samples per node
        :param dtype: float/int data types
        :param sep: separator
        :param data_distribution: random or uniform
        :return: bool
        """
        if dataset.endswith('.csv'):
            with open(dataset) as file:
                data = list(csv.DictReader(file, delimiter=sep))
                for i, node in enumerate(self.nodes):
                    if data_distribution == "uniform":
                        # np.random.seed(i)
                        mu = len(data) / 2
                        sigma = len(data) * self.epsilon
                        n = min_samples - 1
                        while n < min_samples or n > len(data):
                            n = int(np.rint(np.random.normal(mu, sigma)))
                    else:
                        # random.seed(i)
                        n = random.randrange(min_samples, len(data))

                    if pd:
                        node.ldata = pd.DataFrame(sample(data, n), dtype=dtype)
                    else:
                        node.ldata = sample(data, n)
            log('info', "Data loaded and randomly distributed over the nodes")
        elif basename(dataset) == "MNIST.hdf5":
            start_time = time.time()
            data = h5py.File(dataset, 'r')
            # size = len(data[list(data.keys())[0]])
            for i, node in enumerate(self.nodes):
                n = random.randrange(min_samples, len(data['x_train']))
                np_sample = lambda x, rn: x[np.random.choice(x.shape[0], rn, replace=False), :]
                ldata = {
                    'x_train': pd.DataFrame(np_sample(data['x_train'][()], n), dtype=dtype),
                    'y_train': pd.DataFrame(np_sample(data['y_train'][()], n), dtype=dtype),
                    'x_test': pd.DataFrame(data['x_test'][()], dtype=dtype),
                    'y_test': pd.DataFrame(data['y_test'][()], dtype=dtype)
                }
                node.ldata = ldata
                print('.', end='')
            data.close()
            print("\n--- %s seconds ---" % (time.time() - start_time))
        elif basename(dataset) == "mnist.data":
            start_time = time.time()
            X, y = joblib.load(dataset)
            X_train = X[:60000]
            y_train = y[:60000]
            x_test = X[60000:]
            y_test = y[60000:]
            # shuffle training data
            X_train, y_train = shuffle(X_train, y_train)
            data_len = X_train.shape[0]

            for i, node in enumerate(self.nodes):
                n = random_number(min_samples, data_len, epsilon=self.epsilon, distribution=data_distribution, seed=i)
                rX, ry = sample_xy(X_train, y_train, n, i)
                ldata = {
                    'x_train': pd.DataFrame(rX, dtype=dtype),
                    'y_train': pd.DataFrame(ry, dtype=dtype),
                    'x_test': pd.DataFrame(x_test[()], dtype=dtype),
                    'y_test': pd.DataFrame(y_test[()], dtype=dtype),
                }

                node.ldata = ldata
                print('.', end='')
                log('info',
                    f"{node.pname}: x_train: {ldata['x_train'].shape} | y_train: {ldata['y_train'].shape} | "
                    f"x_test: {ldata['x_test'].shape} | y_test: {ldata['y_test'].shape} | "
                    )
            print(f"\nData distributed in {time.time() - start_time} seconds.")
        else:
            raise Exception("Dataset not supported.")
        return self

    def train(self, **kwargs):
        sharedVars.TRAINED_MODELS = len(self.nodes)
        model = kwargs.get('model', None)
        pre_processing = kwargs.get('pre', None)
        algorithm = kwargs.get('algorithm', "MP")
        params = kwargs.get('params', {})
        analysis = kwargs.get('analysis', None)
        if not model:
            log('exception', f"No model provided")
            raise Exception(f"No model provided")
        if algorithm not in ALGORITHMS:
            log('exception', f"Unknown algorithm provided")
            raise Exception(f"Unknown algorithm provided")
        target = None
        if algorithm == "MP":
            target = algorithms.model_propagation
        elif algorithm == "CL":
            target = algorithms.collaborative_learning
        elif algorithm == "LL":
            target = algorithms.local_learning
        else:
            return

        self.run_training(target, model, pre_processing, params)
        self.analysis(analysis)
        self.epsilon = round(self.epsilon + EPSILON_STEP, 2)
        self.reset_nodes()

    def reset_nodes(self):
        for node in self.nodes:
            # node.model = None
            node.solitary_model = None
            node.costs = []
            node.models = {}
            node.c = 1 + sharedVars.M_CONSTANT

    def analysis(self, atype):
        if atype == 'iterations':
            # costs = [node.costs[-1] for node in self.nodes]
            # self.results.append(np.mean(costs))
            self.results = np.sum([node.costs for node in self.nodes], axis=0)
            for node in self.nodes:
                node.costs = []
        elif atype == 'unbalancedness':
            costs = [node.costs[-1] for node in self.nodes]
            lll = np.sum([len(node.costs) for node in self.nodes])
            print(f"\nEPSI: COST[{lll}]: {np.mean(costs)} ")
            self.results.append(np.sum(costs))
            for node in self.nodes:
                node.costs = []
        elif atype == 'data':
            self.results = np.sum([node.costs[-1] for node in self.nodes])
            for node in self.nodes:
                node.costs = []

        elif atype == 'sparsity':
            costs = [node.costs[-1] for node in self.nodes]
            self.results = np.mean(costs)
        elif atype == 'confidence':
            mix = True
            if mix:
                costs = [np.array(node.costs) for node in self.nodes]
                costs = [np.sum(k) for k in zip(*costs)]
                save(f'./save/s_confidence_{self.epsilon}', np.squeeze(costs))
                self.results = np.squeeze(costs)
                for node in self.nodes:
                    node.costs = []
                # iterations = range(1, len(costs) + 1)
                # plt.figure()
                # plt.plot(iterations, np.squeeze(costs), label=f"MP with confidence")
                # plt.ylabel('cost')
                # plt.xlabel('Iterations')
                # plt.legend(loc='upper right', shadow=True)
                # plt.show()
            else:
                fig, axs = plt.subplots(3, 3)
                fig.suptitle('MP without confidence')
                for i, node in enumerate(self.nodes):
                    # axs.flat[i].set_title(f"Node({node.name})")
                    axs.flat[i].plot(np.squeeze(node.costs), label=f"Node({node.name})")
                    axs.flat[i].set(xlabel='Iterations', ylabel='Cost')
                    axs.flat[i].legend(loc='upper right', shadow=True)
                plt.show()
        elif atype == 'epsilon':
            costs = [node.costs[-1] for node in self.nodes]
            self.results = np.sum(costs)
        elif atype == 'communication':
            c = np.sum(node.message_count_send for node in self.nodes)
            self.results = c
        elif atype == 'cvsd':
            self.results = [node.accuracy for node in self.nodes]

    def info(self, show="peers", verbose=False):
        log("Printing information...")
        if show == "peers":
            print(f"List of nodes ({len(self.nodes)})")
            for node in self.nodes:
                print(f"{node.pname}: has {data_size(node)} local data items, and {len(node.peers)} neighbors.")
                if verbose:
                    for peer in node.peers:
                        log(f"{peer}")
                    log("---")

    # -----------------------------------------------------------------------------

    def run_training(self, target, model, pre_processing, params):
        threads = []
        if DEBUG_LEVEL:
            print(bold('Training STARTS...'))
            print()

        for node in self.nodes:
            t = threading.Thread(target=target, args=(node, model, pre_processing, params,), daemon=True)
            threads.append(t)
            t.start()
        for index, thread in enumerate(threads):
            thread.join()

        if DEBUG_LEVEL:
            print()
            print(bold('Training DONE'))

# Definitions
# Node: a object in the network that can communicate with other nodes
# Peer: a representation of a node for the other nodes, a dict that contains
#       the basic information of a node (eg., host, port, weight)
