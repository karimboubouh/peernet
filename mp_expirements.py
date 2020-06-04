import gc
import time
from pydoc import locate
import numpy as np
import peernet.constants as c
from peernet.PeerNet import PeerNet
from peernet.helpers import save, bold


def communication_rounds(nodes, topology, data, algorithm, protocol, args=None):
    """Experiment: communication rounds X Accuracy"""
    print(f"Nodes: {nodes} | Protocol: {protocol} | confidence: {args['protocol']['confidence']}")
    analysis = "communication_rounds"
    # analysis = "byzantine_metrics"
    # analysis = "byzantine"
    # analysis = "ban_epsilon"
    y = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis)
    x = range(1, len(y) + 1, 1)
    file_name = generate_filename(nodes, protocol, args, analysis)
    save(file_name, (x, y))
    return file_name


def byzantine(nodes, topology, data, algorithm, protocol, args=None):
    """Experiment: communication rounds X Accuracy"""
    print(f"Nodes: {nodes} | Protocol: {protocol} | confidence: {args['protocol']['confidence']}")
    analysis = "byzantine"
    y = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis)
    x = range(1, len(y) + 1, 1)
    file_name = generate_filename(nodes, protocol, args, analysis)
    save(file_name, (x, y))
    return file_name


def byzantine_metrics(nodes, topology, data, algorithm, protocol, args=None):
    """Experiment: communication rounds X metrics"""
    print(f"Protocol: {protocol} | confidence: {args['protocol']['confidence']}")
    analysis = "byzantine_metrics"
    info = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis)
    accuracy, precision, recall, f_score = info
    x = range(1, len(accuracy) + 1, 1)
    file_name = generate_filename(nodes, protocol, args, analysis)
    save(file_name, (x, info))
    return file_name


def contribution_factor(nodes, topology, data, algorithm, protocol, args=None):
    """Model Propagation Experiment: communication rounds X ignored nodes"""
    print(f"Protocol: {protocol} | confidence: {args['protocol']['confidence']}")
    analysis = "contribution_factor"
    info = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis)
    y = info[0]
    z = info[1]
    x = range(1, len(y) + 1, 1)
    file_name = f"results/contribution_factor_{nodes}_BI"
    save(file_name, (x, y, z))

    return file_name


def data_unbalancedness(nodes: int = 10, topology='random', data=None, algorithm=None, protocol=None, args=None):
    """Model Propagation Experiment: width X Loss"""
    c.ACCURACY_METRIC = "loss"
    analysis = "data_unbalancedness"
    x = np.arange(0.0, 1.1, 0.1)
    y = []
    print(f"Protocol: {protocol} | confidence: {args['protocol']['confidence']}")
    for i in x:
        i = round(i, 2)
        print(bold(f"DATA UNBALANCEDNESS >> balancedness={i}"))
        data['balancedness'] = i
        r = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis)
        print(f"RESULT OF b={i} >> sum(loss) = {r}")
        y.append(r)
        time.sleep(1)
    file_name = generate_filename(nodes, protocol, args, analysis)
    save(file_name, (x, y))

    return file_name


def graph_sparsity(nodes, topology, data, algorithm, protocol, args=None):
    """Model Propagation Experiment: width X Loss"""
    c.ACCURACY_METRIC = "loss"
    analysis = "graph_sparsity"
    x = np.arange(1, 1.1, 0.1)
    y = []
    for i in x:
        i = round(i, 2)
        print(bold(f"GRAPH SPARSITY >> p={i}"))
        r = abs_exp(nodes, topology, data, algorithm, protocol, args, analysis, p=i)
        print(f"RESULT OF p={i} >> COST = {r}")
        y.append(r)
        time.sleep(2)
    file_name = generate_filename(nodes, protocol, args, analysis)
    save(file_name, (x, y))

    return file_name


# ------------------------------ Internal functions ---------------------------

def abs_exp(nodes, topology, data, algorithm, protocol, args, analysis) -> list:
    # instantiate PeerNet
    p2p = PeerNet((nodes, c.START_PORT))
    # load topology class
    topology_class = locate(c.TOPOLOGY[topology])
    # setup topology
    p2p.network(topology_class)
    # init the P2P system
    p2p.init()
    # Load and randomly distribute training samples among nodes
    p2p.load_data(**data)
    p2p.byzantine(args['behavior'])
    # p2p.info()
    # start training
    p2p.train(
        algorithm=algorithm,
        protocol=protocol,
        args=args,
        analysis=analysis
    )
    p2p.stop()
    r = p2p.results
    del p2p
    p2p = None
    gc.collect()
    return r


def generate_filename(n, protocol, args, analysis, balancedness=None):
    confidence = args.get('protocol', {}).get('confidence')
    if protocol == "CDPL" and confidence:
        # ctype = f"F__{c.EPSILON_FAIRNESS}"
        ctype = "F"
    elif protocol == "MP" and confidence:
        ctype = "C"
    else:
        ctype = "N"
    if balancedness:
        ctype = f"{ctype}_{balancedness}"

    return f"results/{analysis}_{n}_{ctype}"
