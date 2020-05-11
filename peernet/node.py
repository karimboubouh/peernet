import copy
import random
import socket
import time
import threading
import traceback
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from .constants import M_CONSTANT, TEST_SAMPLE, CF_THRESHOLD, EPSILON_FAIRNESS, ACCURACY_METRIC
from .message import exchange_variables
from .protocol import *
from .node_connection import NodeConnection
from .helpers import log, bold, unique_id, create_tcp_socket, peer_id, _p, sample_xy
from .updates import mp_update, cl_update_primal, cl_update_secondary, cl_update_dual, cmp_update


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
        self.byzantine = False
        self.ldata = {}
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        # Server details
        self.host = node_config['host']
        self.port = node_config['port']
        self.timeout_ms = timeout_ms
        # Algorithm details
        self.protocol = "MP"
        self.use_cf = True
        self.solitary_model = None
        self.model = None
        self.models = {}
        self.costs = []
        self.bans = []
        self.ignores = []
        self.cm = {'precision': [], 'recall': [], 'f_score': []}
        self.cm_true = []

        self.W = {}
        self.D = 0.0  # represent the value Dii
        self.c = 1 + M_CONSTANT  # Confidence values
        self.cf = {}  # contribution factor
        self.ff = {}  # fairness control
        self.alpha = 0.9
        # CL
        self.Theta = {}
        self.Z = {}
        self.A = {}
        self.rho = 1
        self.mu = 0.5
        self.stop_condition = None
        self.target_accuracy = 0
        # Implementation details
        self.excluded_peers = []
        self.lock = threading.RLock()
        # Events are send back to the given callback
        self.callback = callback
        # List of node peers (ex:[{shost, sport, weight, [node, conn, ...]}...])
        self.peers = []
        self.banned = []
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
        self.message_count_ignr = 0
        self.accuracy = 0
        # random.seed(100)
        # np.random.seed(100)

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
        self.sock.settimeout(None)
        while not self.terminate_flag.is_set():
            try:
                log('info', f"{self.pname}: Waiting for incoming connections ...")
                conn, add = self.sock.accept()
                tmp_peer = {'name': add[1], 'chost': add[0], 'cport': add[1], 'ctype': "In"}
                if not self.terminate_flag.is_set():
                    thread_client = NodeConnection(self, conn, tmp_peer, self.callback)
                    thread_client.start()
                    self.nodesIn.append(thread_client)  # TODO Remove
                    # self._set_peer_node(client_peer, thread_client) # TODO Remove
                    self.event_node_connected(tmp_peer, thread_client)

            except socket.timeout as e:
                log('exception', f"{self.pname}: Socket timeout!\n{e}")
                pass
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
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.port))

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
        # self._clean_peers()
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

    # CMP::Calculate model updates over network
    def cmp_calculate_update(self, data):
        with self.lock:
            if self.check_exchange():
                self.stop_condition -= 1
                if self.model is not None:
                    model = copy.deepcopy(self.model)
                    cmp_update(self)
                    if self.model is None:
                        self.model = model
                    x_test, y_test = self.ldata['x_test'], self.ldata['y_test']
                    cost = self.model.evaluate(x_test, y_test)
                    if ACCURACY_METRIC == "loss":
                        self.costs.append(self.model.summary()['test_loss'])
                    elif ACCURACY_METRIC == "precision":
                        self.costs.append(self.model.summary()['precision'])
                    elif ACCURACY_METRIC == "recall":
                        self.costs.append(self.model.summary()['recall'])
                    elif ACCURACY_METRIC == "f1_score":
                        self.costs.append(self.model.summary()['f1_score'])
                    else:
                        self.costs.append(cost)
                        self.target_accuracy = cost
                    self.bans.append(len(self.banned))
                    self.ignores.append(self.message_count_ignr)
                    if not self.byzantine and len(self.cm_true) > 0:
                        self.calculate_cm()

    # MP::Calculate model updates over network
    def mp_calculate_update(self, data):

        with self.lock:
            if self.check_exchange():
                self.stop_condition -= 1
                if self.model is not None:
                    model = copy.deepcopy(self.model)
                    # Perform model propagation updates
                    # mp_update(self, parameter="W")
                    # mp_update(self, parameter="b")
                    mp_update(self)
                    if self.model is None:
                        self.model = model
                    x_test, y_test = self.ldata['x_test'], self.ldata['y_test']
                    cost = self.model.evaluate(x_test, y_test)
                    if ACCURACY_METRIC == "loss":
                        self.costs.append(self.model.summary()['test_loss'])
                    elif ACCURACY_METRIC == "precision":
                        self.costs.append(self.model.summary()['precision'])
                    elif ACCURACY_METRIC == "recall":
                        self.costs.append(self.model.summary()['recall'])
                    elif ACCURACY_METRIC == "f1_score":
                        self.costs.append(self.model.summary()['f1_score'])
                    else:
                        self.costs.append(cost)
                        self.target_accuracy = cost
                    self.bans.append(len(self.banned))
                    self.ignores.append(self.message_count_ignr)
                    if not self.byzantine:
                        self.calculate_cm()
                    raise


    # CL::Calculate primal variables (minimize arg min L(models, Z, A)).
    def update_primal(self, neighbor, respond=True):
        # update primal variables if SC > 0
        cl_update_primal(self, neighbor)
        # exchange variables with neighbor
        if respond:
            update = self.check_exchange()
            self.send(neighbor, exchange_variables(self, neighbor, respond=update))

    def fairness_control(self, name, p_j):
        pct_change = (p_j - self.ff[name]) / self.ff[name] if self.ff[name] > 0 else p_j
        if pct_change >= EPSILON_FAIRNESS:
            self.ff[name] = p_j
        else:
            log("info", f"{self.pname} banned node {name} --- {p_j} - {self.ff[name]} = {p_j - self.ff[name]}")
            self.banned.append(name)

    def evaluate_model(self, data):
        """
        Evaluate the received model, return True if it's valid False otherwise.
        @param data:
        @return:
        """
        name = data['sender']['name']
        model = data['payload']['model']
        test_sample = sample_xy(self.ldata['x_test'], self.ldata['y_test'], TEST_SAMPLE)
        p_i = self.model.evaluate(*test_sample)
        p_j = model.evaluate(*test_sample)
        c_j = p_j / p_i
        try:
            self.cf[name] = 0.2 * self.cf[name] + 0.8 * c_j
        except KeyError:
            self.cf[name] = c_j

        log("info", f"{self.pname} :: {name} | p_j={p_j} & p_i ={p_i} & CF={c_j}")

        if self.cf[name] >= CF_THRESHOLD:
            return True
        else:
            self.fairness_control(name, p_j)
            return False

    def get_model(self):
        if self.byzantine:
            random_model = copy.deepcopy(self.model)
            s = self.model.weights.shape
            random_model.weights = np.random.rand(s[0], s[1])
            random_model.evaluate(self.ldata['x_test'], self.ldata['y_test'])
            # print(f"Byz{self.pname} >> summary={random_model.summary()}")
            return random_model
        else:
            return self.model

    def update_models(self, index, model):
        self.models[index] = model

    def calculate_cm(self):
        inx = list(np.sort([int(peer['name'][1:]) for peer in self.peers]))
        banned = np.sort([int(b[1:]) for b in self.banned])
        cm_pred = [0] * len(self.cm_true)
        for b in banned:
            cm_pred[inx.index(b)] = 1

        precision = precision_score(self.cm_true, cm_pred, average="macro", zero_division=0)
        recall = recall_score(self.cm_true, cm_pred, average="macro", zero_division=0)
        f_score = f1_score(self.cm_true, cm_pred, average="macro", zero_division=0)
        self.cm['precision'].append(precision)
        self.cm['recall'].append(recall)
        self.cm['f_score'].append(f_score)

        # print(f"{self.pname} >> precision={precision} | recall={recall} | f1_score={f_score} | "
        #       f"true={self.cm_true} & pred={cm_pred}")

    def check_exchange(self):
        # if self.target_accuracy >= TARGET_ACCURACY:
        #     return False
        if self.stop_condition <= 0:
            return False

        return True

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
            # initialize fairness factor
            self.ff[name] = 0

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
        log('event',
            f"Event Message: Node({node.name}) received ({_p(data['mtype'])} from Node({data['sender']['name']}).")
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

    def get_random_peer(self, ignore_excluded=True):
        log('info', f"{self.pname}: Select a random peer")
        peers_list = []
        if ignore_excluded:
            excluded_names = [peer['name'] for peer in self.excluded_peers]
            peers_list = [peer for peer in self.peers if peer['name'] not in excluded_names]
            # print(f"{self.pname} peers_list >> {[p['name'] for p in peers_list ]}")
        else:
            peers_list = self.peers
        if len(peers_list) > 0:
            random.seed(self.stop_condition)
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
