import csv
# import h5py
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
from .helpers import log, sample, bold, save, shuffle, sample_xy, random_number, data_size, algo_from_dict
from .constants import PROTOCOLS, DEBUG_LEVEL, EPSILON_STEP
from .datasets import fetch_mnist, fetch_cifar, iid_sample, noniid_sample, mnist_iid_sample, mnist_iid, mnist_noniid
import peernet.constants as c


# Class PeerNet ---------------------------------------------------------

class PeerNet:
    protocol = "MP"
    verbose = 4

    def __init__(self, config_file):
        # Initialization
        self.config = PNConfig(config_file)
        self.nodes = self.config.get_nodes()  # type: List[Node]
        self.epsilon = 0.0
        self.results = []
        c.TRAINED_MODELS = len(self.nodes)
        # random.seed(100)
        # np.random.seed(100)

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
        for node in self.nodes:
            node.stop()
        for node in self.nodes:
            node.join()
        print(bold("System shutdown"))

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

    def load_data(self, dataset: str = None, pre=None, iid=True, balancedness=1.0, epsilon=None):
        start_time = time.time()
        if 'mnist' in dataset.lower():
            data = fetch_mnist(pre=pre)
            x_train, y_train, x_test, y_test = data
            if iid:
                nodes_data = mnist_iid(x_train, y_train, len(self.nodes), balancedness)
            else:
                nodes_data = mnist_noniid(x_train, y_train, len(self.nodes), balancedness)
        elif 'cifar' in dataset.lower():
            data = fetch_cifar(pre=pre)
            x_train, y_train, x_test, y_test = data
            nodes_data = mnist_iid_sample(x_train, y_train, nodes=len(self.nodes), balanced=balancedness)
        else:
            raise ValueError(f" {dataset} not supported.")

        for i, node in enumerate(self.nodes):
            x = nodes_data
            node.ldata = {
                'x_train': nodes_data[i][0],
                'y_train': nodes_data[i][1],
                'x_test': x_test,
                'y_test': y_test,
            }

        x_train, y_train, x_test, y_test = data
        # for i, node in enumerate(self.nodes):
        #     if iid:
        #         x, y = iid_sample(x_train, y_train, balanced=balanced, epsilon=epsilon)
        #     else:
        #         x, y = noniid_sample(x_train, y_train, balanced=balanced, epsilon=epsilon)
        #
        #     node.ldata = {
        #         'x_train': x,
        #         'y_train': y,
        #         'x_test': x_test,
        #         'y_test': y_test,
        #     }

        end_time = round(time.time() - start_time, 4)
        print(f"\nData distributed in {end_time} seconds.")

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

    def _compile(self, **kwargs):
        algorithm = kwargs.get('algorithm', None)
        protocol = kwargs.get('protocol', "MP")
        args = kwargs.get('args', None)
        analysis = kwargs.get('analysis', None)
        wrapper, model = algo_from_dict(algorithm)
        target = PeerNet.protocol_target(protocol)
        for node in self.nodes:
            node.protocol = protocol
            node.use_cf = args['protocol']['confidence']
        return wrapper, model, target, args, analysis

    def byzantine(self, behavior):
        if behavior['Byzantine'] > 0:
            byz = np.random.choice(range(len(self.nodes)), behavior['Byzantine'], replace=False)
            print(f"Byzantine nodes: {byz}")
            for i, node in enumerate(self.nodes):
                if i in byz:
                    node.byzantine = True
            for node in self.nodes:
                cm_node = list(np.sort([int(peer['name'][1:]) for peer in node.peers]))
                for i, p in enumerate(cm_node):
                    if p in byz:
                        cm_node[i] = 1
                    else:
                        cm_node[i] = 0
                node.cm_true = cm_node

    def train(self, **kwargs):
        wrapper, model, target, args, analysis = self._compile(**kwargs)

        print(bold('Starting training...'))
        start_time = time.time()
        threads = []
        for node in self.nodes:
            t = threading.Thread(target=target, args=(node, wrapper, model, args,), daemon=True)
            threads.append(t)
            t.start()

        for index, thread in enumerate(threads):
            thread.join()
        end_time = round(time.time() - start_time, 4)
        print(bold(f"\nTraining done in {end_time} seconds."))

        self.analysis(analysis)  # measure="std"

    def train2(self, **kwargs):
        c.TRAINED_MODELS = len(self.nodes)
        model = kwargs.get('model', None)
        pre_processing = kwargs.get('pre', None)
        algorithm = kwargs.get('algorithm', "MP")
        params = kwargs.get('params', {})
        analysis = kwargs.get('analysis', None)
        if not model:
            log('exception', f"No model provided")
            raise Exception(f"No model provided")
        if algorithm not in PROTOCOLS:
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
            node.c = 1 + c.M_CONSTANT

    def analysis(self, atype, measure="mean"):
        if atype == 'communication_rounds':
            length = max([len(node.costs) for node in self.nodes])
            for node in self.nodes:
                node.costs.extend([np.mean(node.costs)] * (length - len(node.costs)))
            if measure == "mean":
                self.results = np.mean([node.costs for node in self.nodes], axis=0)
            elif measure == "std":
                self.results = np.std([node.costs for node in self.nodes], axis=0)
            elif measure == "max":
                self.results = np.max([node.costs for node in self.nodes], axis=0)
        elif atype == 'byzantine':
            costs = np.array([node.costs for node in self.nodes if node.byzantine is False])
            self.results = np.mean(costs, axis=0)
        elif atype == 'byzantine_metrics':
            precision = np.mean([node.cm['precision'][-1] for node in self.nodes if node.byzantine is False], axis=0)
            recall = np.mean([node.cm['recall'][-1] for node in self.nodes if node.byzantine is False], axis=0)
            f_score = np.mean([node.cm['f_score'][-1] for node in self.nodes if node.byzantine is False], axis=0)
            accuracy = np.mean([node.costs[-1] for node in self.nodes if node.byzantine is False], axis=0)
            # self.results = accuracy, precision, recall, f_score
            print(f"Accuracy={accuracy} | precision={precision} | recall={recall} | f_score={f_score}")
        elif atype == 'contribution_factor':
            blength = max([len(node.bans) for node in self.nodes])
            ilength = max([len(node.ignores) for node in self.nodes])
            for node in self.nodes:
                node.bans.extend([node.bans[-1]] * (blength - len(node.bans)))
                node.ignores.extend([node.ignores[-1]] * (ilength - len(node.ignores)))
            self.results.append(np.sum([node.bans for node in self.nodes], axis=0))
            self.results.append(np.sum([node.ignores for node in self.nodes], axis=0))
        elif atype == 'data_unbalancedness':
            costs = [node.costs[-1] for node in self.nodes]
            self.results = np.sum(costs)
        elif atype == 'graph_sparsity':
            costs = [node.costs[-1] for node in self.nodes]
            self.results = np.sum(costs)
        elif atype == 'iterations':
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
        else:
            pass

    def info(self, show="peers", verbose=False):
        log("Printing information...")
        if show == "peers":
            print(f"List of nodes ({len(self.nodes)})")
            for node in self.nodes:
                p = [peer['name'] for peer in node.peers]
                print(f"{node.pname}: has {data_size(node)} data items, and {len(node.peers)} neighbors: {p}")
                if verbose:
                    for peer in node.peers:
                        log(f"{peer}")
                    log("---")

    # -----------------------------------------------------------------------------

    @staticmethod
    def protocol_target(protocol):
        if protocol not in PROTOCOLS:
            raise SystemExit(log('exception', "Unknown protocol."))
        if protocol == "MP":
            return algorithms.model_propagation
        elif protocol == "CMP":
            return algorithms.controlled_model_propagation
        elif protocol == "CL":
            return algorithms.collaborative_learning
        else:
            return algorithms.local_learning

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
