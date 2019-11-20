import csv
import threading
import time
import random
import pandas as pd
from typing import List

from .network.arch import full_network
from .node import Node
from .pnconfig import PNConfig
from .helpers import log, sample, bold


# Class PeerNet ---------------------------------------------------------

class PeerNet:
    def __init__(self, name, config_file):
        # Initialization
        self.name = name
        self.config = PNConfig(config_file)
        self.nodes = self.config.get_nodes()  # type: List[Node]
        # Network architecture: defines the list of peers for each node.
        # Strategy: Full network
        full_network(self.nodes)

    def init(self):
        timeout_ms = self.config.get_timeoutms()
        for node in self.nodes:
            # instantiate node
            node.init()
            # Start the threads
            node.start()

    def sleep(self, s):
        time.sleep(s)

    def load_dataset(self, dataset, df=False):
        """
        Load a dataset file and split it randomly with random size
        over the list of nodes in the PeerNet.
        :param dataset: str
        :return: bool
        """
        with open(dataset) as file:
            data = list(csv.DictReader(file))
            for node in self.nodes:
                n = random.randrange(1, len(data))
                if pd:
                    node.ldata = pd.DataFrame(sample(data, n))
                else:
                    node.ldata = sample(data, n)

    def train(self, model, **kwargs):
        threads = []
        for node in self.nodes:
            t = threading.Thread(target=model, args=(node, *kwargs,), daemon=True)
            threads.append(t)
            t.start()
        for index, thread in enumerate(threads):
            thread.join()
        print(bold('DONE'))

    def info(self):
        # log('info', f"PeerNet: {self.name}")
        # log('info', f"List of nodes ({len(self.nodes)}): {self.nodes}")
        for node in self.nodes:
            # log('info', f"Node({node.name}): has {len(node.ldata)} local data items.")
            pl = []
            for p in node.peers:
                pl.append((p['name'], p['connected']))
            log('exception', f"Node({node.name}): {pl}")

# -----------------------------------------------------------------------------
# Definitions
# Node: a object in the network that can communicate with other nodes
# Peer: a representation of a node for the other nodes, a dict that contains
#       the basic information of a node (eg., host, port, weight)
