from typing import List
import numpy as np
from ..node import Node
from ..constants import NETWORK_SEED


def full_network(nodes: List[Node]):
    # Each node peers with the full network
    for node in nodes:
        for n in nodes:
            if node.name != n.name:
                node.peers += [
                    {
                        'name': f"{n.name}", 'shost': n.host, 'sport': n.port, 'chost': n.host, 'cport': n.port,
                        'ctype': "Out", 'weight': 1, 'conn': None, 'connected': False
                    }]


def get_node(nodes, name):
    for node in nodes:
        if node.name == name:
            return node


def static_network(nodes: List[Node]):
    # A network with 8 nodes
    if len(nodes) != 8:
        raise Exception("Static network!")

    arch = {
        'w1': {'w2': 3, 'w4': 2},
        'w2': {'w1': 3, 'w5': 1, 'w7': 1},
        'w3': {'w5': 4},
        'w4': {'w1': 2, 'w7': 2},
        'w5': {'w2': 1, 'w3': 4, 'w6': 2},
        'w6': {'w5': 2, 'w7': 3},
        'w7': {'w2': 1, 'w4': 2, 'w6': 3, 'w8': 4},
        'w8': {'w7': 4},
    }

    for name, peers in arch.items():
        node = get_node(nodes, name)
        for peer, weight in peers.items():
            n = get_node(nodes, peer)
            node.peers += [
                {
                    'name': f"{n.name}", 'shost': n.host, 'sport': n.port, 'chost': n.host, 'cport': n.port,
                    'ctype': "Out", 'weight': weight, 'conn': None, 'connected': False
                }]


def random_network(nodes: List[Node]):
    nodes_size = len(nodes)
    np.random.seed(NETWORK_SEED)
    for node in nodes:

        number_neighbors = 0
        while len(nodes) <= number_neighbors or number_neighbors < 1:
            number_neighbors = int(np.random.laplace(np.log(nodes_size), 1, 1))

        neighbors = np.random.choice(list(x for x in nodes if x is not node), number_neighbors, replace=False)

        probabilities = np.random.dirichlet(np.ones(number_neighbors), size=10)
        for i, neighbor in enumerate(neighbors):
            node.peers.append({
                'name': f"{neighbor.name}", 'shost': neighbor.host, 'sport': neighbor.port, 'chost': None,
                'cport': None, 'ctype': 'Out', 'weight': probabilities[0][i], 'conn': None, 'connected': False
            })


def rrr_network(nodes: List[Node]):
    nodes_size = len(nodes)
    for node in enumerate(nodes):
        number_neighbors = np.random.randint(1, nodes_size)
        neighbors = np.random.choice(list(x for x in nodes if x is not node), number_neighbors, replace=False)
        probabilities = np.random.dirichlet(np.ones(number_neighbors), size=1)
        for i, neighbor in enumerate(neighbors):
            node.peers.append({
                'name': f"{neighbor.name}", 'shost': neighbor.host, 'sport': neighbor.port, 'chost': None,
                'cport': None, 'ctype': 'Out', 'weight': probabilities[0][i], 'conn': None, 'connected': False
            })


def r_network(nodes: List[Node]):
    nodes_size = len(nodes)
    a = np.random.randint(0, 2, (nodes_size, nodes_size))
    m = np.tril(a) + np.tril(a, -1).T
    np.fill_diagonal(m, 0)
    for i, node in enumerate(nodes):
        line = m[i]
        print(line)
        print("33")
        for index, j in enumerate(line):
            if j == 1:
                node.peers.append({
                    'name': f"{nodes[index].name}", 'shost': nodes[index].host, 'sport': nodes[index].port,
                    'chost': None,
                    'cport': None, 'ctype': 'Out', 'weight': 1000, 'conn': None, 'connected': False
                })

    # for i, node in enumerate(nodes):
    #     number_neighbors = np.random.randint(1, nodes_size)
    #     defined_neighbors = []
    #     defined_neighbors_proba = []
    #     prev = nodes[:i]
    #     for n in prev:
    #         exist = next((peer for peer in n.peers if peer["name"] == n.name), None)
    #         if exist:
    #             defined_neighbors.append(n)
    #             defined_neighbors_proba.append(exist["weight"])
    #
    #     number_remain_neighbors = number_neighbors - len(defined_neighbors)
    #     neighbors = defined_neighbors + list(np.random.choice(
    #         list((x for x in list(set(nodes) - set(defined_neighbors)) if x is not node)), number_remain_neighbors,
    #         replace=False))
    #
    #     probabilities = defined_neighbors_proba + list(np.random.dirichlet(np.ones(number_remain_neighbors),
    #                                                                        size=1 - sum(defined_neighbors_proba)))
    #     for j, neighbor in enumerate(neighbors):
    #         node.peers.append({
    #             'name': f"{neighbor.name}", 'shost': neighbor.host, 'sport': neighbor.port, 'chost': None,
    #             'cport': None, 'ctype': 'Out', 'weight': probabilities[0][j], 'conn': None, 'connected': False
    #         })
