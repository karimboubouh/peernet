import csv
import threading
import time
import random
import pandas as pd
from typing import List

from . import algorithms
from .node import Node
from .pnconfig import PNConfig
from .helpers import log, sample, bold
from .constants import ALGORITHMS


# Class PeerNet ---------------------------------------------------------

class PeerNet:
    def __init__(self, config_file):
        # Initialization
        self.config = PNConfig(config_file)
        self.nodes = self.config.get_nodes()  # type: List[Node]

    def init(self):
        timeout_ms = self.config.get_timeoutms()
        for node in self.nodes:
            # instantiate node
            node.init()
            # Start the threads
            node.start()

        for node in self.nodes:
            status = node.connect_neighbors()
            log('info', f"{node.pname} >> Connection status: {status}")

        return self

    def network(self, network_strategy):
        """Network architecture """
        network_strategy(self.nodes)
        return self

    @staticmethod
    def sleep(s):
        log('info', f"Sleep for {s} second(s)...")
        time.sleep(s)

    def load_dataset(self, dataset, df=False, min_samples=1, dtype=float):
        """
        Load a dataset file and split it randomly with random size
        over the list of nodes in the PeerNet.
        :param dataset: str
        :param df: DataFrame
        :param min_samples: int minimum number of training samples per node
        :param dtype: float/int data types
        :return: bool
        """
        with open(dataset) as file:
            data = list(csv.DictReader(file))
            for i, node in enumerate(self.nodes):
                # Fix the randomness for experimentation purposes.
                random.seed(i)
                n = random.randrange(min_samples, len(data))
                if pd:
                    node.ldata = pd.DataFrame(sample(data, n), dtype=dtype)
                else:
                    node.ldata = sample(data, n)
        log('info', "Data loaded and randomly distributed over the nodes")
        return self

    def train(self, **kwargs):
        model = kwargs.get('model', None)
        pre_processing = kwargs.get('pre', None)
        algorithm = kwargs.get('algorithm', "MP")
        params = kwargs.get('params', {})
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
            return
        threads = []
        for node in self.nodes:
            t = threading.Thread(target=target, args=(node, model, pre_processing, params,), daemon=True)
            threads.append(t)
            t.start()
        for index, thread in enumerate(threads):
            thread.join()
        print(bold('Training DONE'))

    def info(self, show="peers"):
        log("Printing information...")
        self.sleep(1)
        if show == "peers":
            log(f"List of nodes ({len(self.nodes)}): {self.nodes}")
            for node in self.nodes:
                log(f"{node.pname}: has {len(node.ldata)} local data items, and {len(node.peers)} neighbors.")
                for peer in node.peers:
                    log(f"{peer}")
                log("---")
# -----------------------------------------------------------------------------


# Definitions
# Node: a object in the network that can communicate with other nodes
# Peer: a representation of a node for the other nodes, a dict that contains
#       the basic information of a node (eg., host, port, weight)
