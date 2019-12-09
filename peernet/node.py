import random
import socket
import time
import threading
import traceback
from typing import List

import numpy as np

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
        # Identification
        self.id = unique_id(10)
        self.name = node_config['name']
        self.ldata = {}
        # Server details
        self.host = node_config['host']
        self.port = node_config['port']
        self.timeout_ms = timeout_ms
        # Algorithm details
        self.solitary_model = None
        self.model = None
        self.models = {}
        self.W = {}
        self.D = 0.0  # represent the value Dii
        self.c = 1  # Confidence values
        self.alpha = 0.5
        self.stop_condition = 10
        # Implementation details
        self.excluded_peers = []
        self.lock = threading.RLock()
        # Events are send back to the given callback
        self.callback = callback
        # List of node peers (ex:[{shost, sport, weight, [node, conn, ...]}...])
        self.peers = []
        # Nodes (Threads) that have established a connection with this node N->(self)
        # (ex: [NodeConnection])
        self.nodesIn = []
        # Nodes (Threads) that this nodes is connected to (self)->N
        self.nodesOut = []
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
        self.update_weights()
        # self._init_peers()
        # Start the TCP/IP server
        self._init_server()

    def run(self):
        while not self.terminate_flag.is_set():
            try:
                log('info', f"{self.pname}: Waiting for incoming connections ...")
                conn, add = self.sock.accept()
                tmp_peer = {'name': add[1], 'chost': add[0], 'cport': add[1], 'ctype': "In"}
                thread_client = NodeConnection(self, conn, tmp_peer, self.callback)
                thread_client.start()
                self.nodesIn.append(thread_client)  # TODO Remove
                # self._set_peer_node(client_peer, thread_client) # TODO Remove
                self.event_node_connected(tmp_peer, thread_client)
            except socket.timeout as e:
                log('exception', f"{self.pname}: Socket timeout!\n{e}")
            except Exception as e:
                log('exception', f"{self.pname}: Exception!\n{e}")
                traceback.print_exc()
            # time.sleep(0.01)

        log('info', f"{self.pname}: Terminating connections ...")
        # TODO handle this via peers list
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

    # Connect node with its neighbors
    def connect_neighbors(self):
        status = {}
        for peer in self.peers:
            if peer['shost'] == self.host and peer['sport'] == self.port:
                status.update({peer['name']: False})
            if peer.get('connected') is not True:
                if not self._connection(peer):
                    status.update({peer['name']: False})
            status.update({peer['name']: True})
        return status

    # Make a connection with a node
    def connect(self, peer, info=None):
        if peer['shost'] == self.host and peer['sport'] == self.port:
            log('warning', f"Cannot connect with yourself!")
            return None
        if peer.get('connected') is True:
            return peer['conn']
        return self._connection(peer, info)

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

    # Calculate model updates over network
    def calculate_update(self, data):
        with self.lock:
            self.stop_condition -= 1
            sigma = np.zeros((self.model.parameters.shape[0], 1))
            for name, model in self.models.items():
                sigma += self.W[name] / self.D * model.parameters
            log("success", f"{self.pname}: Model to be updated {sum(self.model.parameters)}")

            self.model.parameters = (self.alpha + (1 - self.alpha) * self.c) ** -1 * (
                    self.alpha * sigma + (1 - self.alpha) * self.c * self.solitary_model.parameters)
            log("success", f"{self.pname}: Model updated {sum(self.model.parameters)}")

    def get_model(self):
        return self.model

    def update_models(self, index, model):
        self.models[index] = model

    def check_exchange(self):
        update = self.stop_condition > 0
        return update

    def exclude_peer(self, peer):
        if peer not in self.excluded_peers:
            self.excluded_peers.append(peer)

    # Local functions ---------------------------------------------------------

    # Initialize the default set of peers
    def _init_peers(self):
        log('info', self.peers)

    def update_weights(self):
        self.D = 0
        for peer in self.peers:
            w = peer["weight"]
            name = peer["name"]
            self.W[name] = w
            self.D += w

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
        if peer['connected']:
            return True
        else:
            return self.connect(peer)

    # update peer's node attribute
    def _set_peer_node(self, peer, node):
        # TODO update it or remove
        for p in self.peers:
            if p['host'] == peer['host'] and p['port'] == peer['port']:
                p['conn'] = node
                return p
        return None

    def _connection(self, peer, info=None):
        log('info', f"{self.pname}: Establishing a connection with Node({peer.get('name')})...")
        try:
            # The node plays the role of a client
            # creates a client socket and connects to a peer
            sock = create_tcp_socket()
            sock.settimeout(self.timeout_ms)
            sock.connect((peer['shost'], peer['sport']))
            # Create a communication thread and add it to nodesOut
            thread_client = NodeConnection(self, sock, peer, self.callback)
            thread_client.start()
            upeer = next((p for p in self.peers if p["shost"] == peer['shost'] and p["sport"] == peer['sport']), {})
            if upeer:
                upeer.update({'conn': thread_client, 'connected': True})
                peer = upeer
            else:
                log('exception', f"{self.pname}: -> Connect: Unknown node({peer['name']})")
                return False
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

    def get_random_peer(self, ignore_excluded=False):
        log('info', f"{self.pname}: Select a random peer")
        peers_list = []
        if ignore_excluded:
            for peer in self.peers:
                if not next((e for e in self.excluded_peers if e["name"] == peer['name']), False):
                    peers_list.append(peer)
        else:
            peers_list = self.peers
        if len(peers_list) > 0:
            peer = random.choice(peers_list)
            if self.connect(peer) is not None:
                time.sleep(0.1)  # wait for message exchange to take place
                return peer
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
