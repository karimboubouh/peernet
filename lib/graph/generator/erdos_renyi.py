# https://www.youtube.com/watch?v=Vuv4v_4n7sM

"""
Erdos Renyi Random graph generator
ErdosRenyi(nodes: List, n: int, p: float[0,1])

# Step 1: take n, i.e total number of nodes
# Step 2: take p, i.e probability of adding an edge between two nodes
# Step 3: take an empty graph (that implements to list to return list of nodes) and apply step 1 and 2 for all peers
          of nodes.

--> display graph
--> draw graph degree distribution
https://www.youtube.com/watch?v=0SdzPJksV3Q
"""
from typing import List
from peernet.pnconfig import PNConfig
import random


class ErdosRenyi(object):

    def __init__(self, nodes: List, p: float, seed=True):
        self.nodes = nodes
        self.n = len(nodes)
        self.p = p
        if seed:
            random.seed(len(nodes))

    def generate_graph(self, force_connected=True):
        for i, inode in enumerate(self.nodes):
            for j, jnode in enumerate(self.nodes):
                r = random.random()  # edge probability
                if i < j and r <= self.p:
                    inode.peers.append({
                        'name': f"{jnode.name}", 'shost': jnode.host, 'sport': jnode.port, 'chost': None,
                        'cport': None, 'ctype': 'Out', 'weight': r, 'conn': None, 'connected': False
                    })
                    jnode.peers.append({
                        'name': f"{inode.name}", 'shost': inode.host, 'sport': inode.port, 'chost': None,
                        'cport': None, 'ctype': 'Out', 'weight': r, 'conn': None, 'connected': False
                    })

        if force_connected:
            self.connect_graph()

        return self

    def connect_graph(self):
        excluded = []
        for i, node in enumerate(self.nodes):
            if len(node.peers) == 0:
                excluded.append(node)
                available = list(set(self.nodes) - set(excluded))
                try:
                    rnode, r = random.choice(available), random.random()
                    excluded.append(rnode)
                    node.peers.append({
                        'name': f"{rnode.name}", 'shost': rnode.host, 'sport': rnode.port, 'chost': None,
                        'cport': None, 'ctype': 'Out', 'weight': r, 'conn': None, 'connected': False
                    })
                    rnode.peers.append({
                        'name': f"{node.name}", 'shost': node.host, 'sport': node.port, 'chost': None,
                        'cport': None, 'ctype': 'Out', 'weight': r, 'conn': None, 'connected': False
                    })
                except IndexError as e:
                    print("Very small graph to support forcing connectivity between nodes!")

    def get_nodes(self):
        return self.nodes


if __name__ == '__main__':
    for ii in range(11):
        print(ii)
        config = PNConfig("conf.yaml")
        nodes = config.get_nodes()
        er = ErdosRenyi(nodes, p=ii / 10)
        er.generate_graph()
        for node in nodes:
            print(node.pname, '--', [peer['name'] for peer in node.peers])
        print("------------------------------------------------- P = ", ii / 10)
