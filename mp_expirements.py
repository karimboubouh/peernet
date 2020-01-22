import numpy as np
from pydoc import locate
import peernet.constants as c
from peernet.PeerNet import PeerNet
from peernet.network.arch import random_network


def communication_rounds(nodes: int = 10, topology: str = 'random', data=None, algorithm=None, verbose: int = 2):
    """Model Propagation Experiment: communication rounds X Accuracy"""
    # set verbosity level
    PeerNet.verbose = verbose
    # instantiate PeerNet
    p2p = PeerNet((nodes, c.START_PORT))
    # load topology class
    topology_class = locate(c.TOPOLOGY[topology])
    # setup topology
    p2p.network(topology_class)
    # init the system
    p2p.init()
    # Load and randomly distribute training samples between nodes
    dataset = data['dataset']
    p2p.load_dataset(f"./datasets/{dataset}")
    # start training
    model = algorithm['model']
    confidence = algorithm['confidence']
    p2p.train(
        model=locate(c.ALGORITHM[model]),
        pre=locate(c.PRE_PROCESSING[dataset]),
        algorithm="MP",
        params={'confidence': confidence, 'debug': False, 'show_results': True},
        analysis="iterations"
    )
