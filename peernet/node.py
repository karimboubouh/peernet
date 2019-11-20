import random
import socket
import time
import threading
import traceback
from typing import List

from .protocol import *
from .node_connection import NodeConnection
from .helpers import log, bold, unique_id, create_tcp_socket, peer_id


# Class Node
# Implements a node that is able to connect to other nodes and is able to accept connections from other nodes.
# After instantiation, the node creates a TCP/IP server with the given port.
#
class Node(threading.Thread):

    # Node class constructor
    def __init__(self, node_config, timeout_ms, callback):
        super(Node, self).__init__()
        self.id = unique_id(10)
        self.name = node_config['name']
        # Server details
        self.host = node_config['host']
        self.port = node_config['port']
        self.ldata = {}
        self.model = None
        self.timeout_ms = timeout_ms
        # Events are send back to the given callback
        self.callback = callback
        # List of node peers (ex:[{host, port, weight, [node, conn, ...]}...])
        self.peers = []
        # Nodes (Threads) that have established a connection with this node N->(self)
        # (ex: [NodeConnection])
        self.nodesIn = []  # type: List[NodeConnection]
        # Nodes (Threads) that this nodes is connected to (self)->N
        self.nodesOut = []  # type: List[NodeConnection]
        # When this flag is set, the node will stop and close
        self.terminate_flag = threading.Event()
        # Responses
        self.responses = {}
        # Debugging
        self.debug = False
        # statistical attributes
        self.message_count_send = 0
        self.message_count_recv = 0

    def __str__(self):
        return f"Node({self.name})"  # can be more readable

    def __repr__(self):
        return f"Node({self.name})"

    def init(self):
        # setup default peers
        # self._init_peers()
        # Start the TCP/IP server
        self._init_server()

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                log('info', f"{self.pname}: Waiting for incoming connections ...")
                conn, add = self.sock.accept()
                tmp_peer = {'name': add[1], 'host': add[0], 'port': add[1], 'tmp': True}
                thread_client = NodeConnection(self, conn, tmp_peer, self.callback, False)
                thread_client.start()
                self.nodesIn.append(thread_client)
                # self._set_peer_node(client_peer, thread_client)
                self.event_node_connected(tmp_peer, thread_client)
            except socket.timeout as e:
                log('exception', f"{self.pname}: Socket timeout!\n{e}")
            except Exception as e:
                log('exception', f"{self.pname}: Exception!\n{e}")
                traceback.print_exc()
            time.sleep(0.01)

        log('info', f"{self.pname}: Terminating connections ...")
        for t in self.nodesIn:
            t.stop()
        for t in self.nodesOut:
            t.stop()
        time.sleep(1)
        for t in self.nodesIn:
            t.join()
        for t in self.nodesOut:
            t.join()
        self.sock.close()
        log('info', f"{self.pname}: Stopped.")

    # Stop the thread
    def stop(self):
        self.terminate_flag.set()

    # Setup a new connection thread
    def new_connection___(self, connection, client_address, callback):
        return NodeConnection(self, connection, client_address, callback)

    # Make a connection with a node
    def connect(self, peer, info=None):
        if peer['host'] == self.host and peer['port'] == self.port:
            log('warning', f"Cannot connect with yourself!")
            return None
        if peer.get('connected') is True:
            return peer['conn']
        return self._new_connection(peer, info)

    # Disconnect with a node. It sends a last message to the node!
    def disconnect(self, node):
        if node in self.nodesOut:
            # node.send(self.create_message({"type": "message", "message": "Terminate connection"}))
            # todo create_message(type, message)
            node.stop()
            node.join()  # When this is here, the application is waiting and waiting
            del self.nodesOut[self.nodesOut.index(node)]

    # Send a message to a node.
    def send(self, peer, message):
        self._clean_peers()
        if self._is_peer_ready(peer):
            try:
                if peer['conn'] is None:
                    peer['conn'] = self.connect(peer)
                peer['conn'].send(message)
            except Exception as e:
                log('exception', f"{self.pname}: -> Send: Error while sending message to Node({peer['name']})\n{e}")
                traceback.print_exc()
        else:
            log('error', f"{self.pname}: -> Send: Could not send the message, Node({peer['name']}) is not known")

    # Broadcast a message to all peers.
    def broadcast(self, message, include='all'):
        if include == 'all':
            log('info', f"{self.pname}: broadcasts a message to all peers.")
            for n in self.peers:
                self.send(n['node'], message)
        elif include == 'in':
            log('info', f"{self.pname}: broadcasts a message to all inbound nodes.")
            for n in self.nodesIn:
                self.send(n, message)
        elif include == 'out':
            log('info', f"{self.pname}: broadcasts a message to all outbound nodes.")
            for n in self.nodesOut:
                self.send(n, message)
        elif type(include) == list:
            log('info', f"{self.pname}: broadcasts a message to a sample of nodes.")
            for n in include:
                self.send(n, message)
        else:
            raise Exception(f"Wrong type for argument 'include': ({type(include)})!")

    def get_model(self):
        return self.model

    # Local functions ---------------------------------------------------------

    # Initialize the default set of peers
    def _init_peers(self):
        log('info', self.peers)

    # Creates the TCP/IP server
    def _init_server(self):
        log('info', f"{self.pname}: Starting server on ({self.host}:{self.port})")
        self.sock = create_tcp_socket()
        self.sock.bind(('', self.port))
        self.sock.settimeout(self.timeout_ms)
        self.sock.listen(10)

    # Clean the array list of terminated nodes.
    def _clean_peers(self):
        for n in self.nodesIn:
            if n.terminate_flag.is_set():
                self.event_node_inbound_closed(n)
        for n in self.nodesOut:
            if n.terminate_flag.is_set():
                self.event_node_outbound_closed(n)

    # Check if a node is known for this node
    def _is_peer_ready(self, peer):
        for p in self.peers:
            if p['host'] == peer['host'] and p['port'] == peer['port']:
                if peer['connected']:
                    return True
                else:
                    return self.connect(peer)
        return False

    # update node's peers list with new peer if it doesn't exist already
    def _update_peers(self, client_address):
        host = client_address[0]
        port = client_address[1]
        for p in self.peers:
            if p['host'] == host and p['port'] == port:
                return p
        peer = {'name': f"{host}:{port}", 'host': host, 'port': port, 'weight': 0, 'conn': None, 'connected': False}
        # TODO: add name
        self.peers.append(peer)
        return peer

    # update peer information
    def _update_peer(self, peer, **kwargs):
        for p in self.peers:
            if p['host'] == peer['host'] and p['port'] == peer['port']:
                p.update(kwargs)
                return p
        return False

    # update peer's node attribute
    def _set_peer_node(self, peer, node):
        for p in self.peers:
            if p['host'] == peer['host'] and p['port'] == peer['port']:
                p['conn'] = node
                return p
        return None

    def _new_connection(self, peer, info):
        log('info', f"{self.pname}: Establishing a connection with Node({peer.get('name')})...")
        try:
            # The node plays the role of a client
            # creates a client socket and connects to a peer
            sock = create_tcp_socket()
            sock.settimeout(self.timeout_ms)
            sock.connect((peer['host'], peer['port']))
            # Create a communication thread and add it to nodesOut
            thread_client = NodeConnection(self, sock, peer, self.callback, True)
            thread_client.start()
            self.nodesOut.append(thread_client)
            peer = self._update_peer(peer, conn=thread_client, connected=True)
            self.event_connected_with_node(peer)
            return thread_client
        except Exception as e:
            log('exception', f"{self.pname}: -> Connect: Could not connect with Node({peer['name']})\n{e}")
            traceback.print_exc()
        return None

    # Events ------------------------------------------------------------------
    def event_node_inbound_closed(self, node):
        log('event', f"Event: node({node.name}) inbound closed")
        if self.callback is not None:
            self.callback(NODE_INBOUND_CLOSED, self, node, {})
        node.join()  # block until the node is terminated
        del self.nodesIn[self.nodesIn.index(node)]
        # todo remove from peers

    def event_node_outbound_closed(self, node):
        log('event', f"Event: node({node.name}) outbound closed")
        if self.callback is not None:
            self.callback(NODE_OUTBOUND_CLOSED, self, node, {})
        node.join()  # block until the node is terminated
        del self.nodesOut[self.nodesOut.index(node)]
        # todo remove from peers

    def event_connected_with_node(self, peer):
        log('event', f"Event Out: {self.pname}: Connected with Node({peer['name']}).")
        if self.callback is not None:
            self.callback(CONNECTED_WITH_NODE, self, peer['conn'], {})

    def event_node_connected(self, peer, thread_client):
        pname = bold(f"Node({peer['name']})")
        log('event', f"Event In: {pname} connected with Node({self.name}).")
        if self.callback is not None:
            self.callback(NODE_CONNECTED, self, thread_client, {})

    def event_node_message(self, node, data):
        log('event', f"Event Message: Node({node.name}) received ({data['mtype']} from Node({data['sender']['name']}).")
        if self.callback is not None:
            self.callback(NODE_MESSAGE, self, node, data)

    # Getters and setters -----------------------------------------------------
    @property
    def pname(self):
        return bold(f"Node({self.name})")

    def get_message_count_send(self):
        return self.message_count_send

    def get_message_count_recv(self):
        return self.message_count_recv

    def get_peer_by_address(self, client_address):
        host = client_address[0]
        port = client_address[1]
        for peer in self.peers:
            if peer['host'] == host and peer['port'] == port:
                return peer
        return None

    def get_random_peer(self):
        log('info', f"{self.pname}: Select a random peer")
        peer = random.choice(self.peers)
        if self.connect(peer) is not None:
            time.sleep(0.01)  # wait for message exchange to take place
            return peer
        else:
            return None

    def wait_response(self, peer, mtype):
        # TODO use threading built it  wait functions
        i = 0
        while True:
            try:
                if i % 1000:
                    print('.', end='')
                if self.responses[peer_id(peer)][mtype] is not None:
                    return self.responses[peer_id(peer)][mtype]
            except:
                pass

# ----------------------------------------------------------------------------------------------------------------------
