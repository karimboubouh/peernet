from typing import List

from ..node import Node


def full_network(nodes: List[Node]):
    # Each node peers with the full network
    for node in nodes:
        for n in nodes:
            if node.name != n.name:
                node.peers += [
                    {'name': f"{n.name}", 'host': n.host, 'port': n.port, 'weight': 1, 'conn': None,
                     'connected': False}]
