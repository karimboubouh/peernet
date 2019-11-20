from peernet.helpers import log
from peernet.node import Node
from peernet.message import request_hello
from peernet.protocol import RESPONSE_HELLO


def hello(node):
    """
    Each node of the PeerNet sends a hello request to one random peer from its
    peers list, and wait for a hello response from that peer.
    :type node: Node
    :rtype: None
    """
    peer = node.get_random_peer()
    if peer is not None:
        req = request_hello(node)
        node.send(peer, req)
        res = node.wait_response(peer, RESPONSE_HELLO)
        # TODO look for future objects to handle waiting
        log(f"{node.pname}: Received -> {res['payload']}")
    else:
        log('error', "No peer selected !")