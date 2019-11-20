import socket
import time
import pickle
import threading
import traceback

from . import protocol
from .helpers import log, unique_id, peer_id
from .message import response_hello, response_model, request_exchange, response_exchange


# Class NodeConnection
# Implements the connection that is made with a node.
# Both inbound and outbound nodes are created with this class.
# Events are send when data is coming from the node
# Messages could be sent to this node.


class NodeConnection(threading.Thread):

    def __init__(self, server, sock, peer, callback):
        super(NodeConnection, self).__init__()
        self.id = unique_id(10)
        self.server = server
        self.sock = sock
        if peer.get('ctype', '') == "In":
            self.host = peer['chost']
            self.port = peer['cport']
            log('success', f"{self.server.pname}: [S] Connection started with  Node({peer['name']})")
        else:
            self.host = peer['shost']
            self.port = peer['sport']
            log('success', f"{self.server.pname}: [C] Connection accepted from Node({peer['name']})")
        self.peer = peer
        self.callback = callback
        self.terminate_flag = threading.Event()
        self.lock = threading.RLock()

    def send(self, message):
        try:
            # In case of big messages use a header and send the message in chunks
            blob = pickle.dumps(message)
            self.sock.sendall(blob)
        except socket.error as e:
            self.terminate_flag.set()
            log('exception', f"{self.server.pname} NodeConnection: Unexpected error!\n{e}")

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        # Timeout, so the socket can be closed when it is dead!
        self.sock.settimeout(10.0)
        # Send init message to exchange necessary nodes information in case of InNode
        if self.peer.get('ctype', '') == "In":
            self.send(request_exchange(self.server))

        while not self.terminate_flag.is_set():
            try:
                buffer = self.sock.recv(65536)
                self.handle_request(buffer)
                self.server.message_count_recv = self.server.message_count_recv + 1
            except socket.timeout:
                pass
            except:
                traceback.print_exc()
                self.terminate_flag.set()
                log('exception', f"{self.server.pname}: NodeConnection: Socket has been terminated!")
            time.sleep(0.01)
        self.sock.settimeout(None)
        self.sock.close()
        log('info', f"{self.server.pname}: NodeConnection: Stopped")

    def handle_request(self, buffer):
        if not self.check_message(buffer):
            log('error', f"{self.server.pname}: NodeConnection: Message is damaged)")
            return False
        data = pickle.loads(buffer)
        self.server.event_node_message(self.server, data)
        if data is not None:
            try:
                # requests ----------------------------------------
                if data['mtype'] == protocol.REQUEST_EXCHANGE:
                    self.do_exchange(data)
                elif data['mtype'] == protocol.REQUEST_HELLO:
                    self.do_hello(data)
                elif data['mtype'] == protocol.REQUEST_MODEL:
                    self.do_model(data)
                # responses ---------------------------------------
                elif data['mtype'] == protocol.RESPONSE_EXCHANGE:
                    self.update_info(data)
                elif data['mtype'] == protocol.RESPONSE_HELLO:
                    self.handle_response(protocol.RESPONSE_HELLO, data)
                elif data['mtype'] == protocol.RESPONSE_MODEL:
                    self.handle_response(protocol.RESPONSE_MODEL, data)
                else:
                    log('error',
                        f"{self.server.pname}: NodeConnection: Message type ({data['mtype']}) is not supported")
                    return False
            except:
                log('error', f"{self.server.pname}: NodeConnection: Exception while processing message!")
        else:
            log('error', f"{self.server.pname}: NodeConnection: Message empty!")

    def check_message(self, buffer):
        if buffer is not None:
            return True
        else:
            log('error', f"{self.server.pname}: NodeConnection: Message is empty or damaged)")
            return False

    def do_hello(self, data):
        message = response_hello(self.server)
        self.send(message)

    def do_model(self, data):
        message = response_model(self.server)
        self.send(message)

    def do_exchange(self, data):
        message = response_exchange(self.server)
        self.send(message)

    def update_info(self, data):
        s = data['sender']
        upeer = next((p for p in self.server.peers if p["shost"] == s['shost'] and p["sport"] == s['sport']), {})
        if upeer:
            upeer.update({'name': s['name'], 'shost': s['shost'], 'sport': s['sport'], 'conn': self, 'connected': True})
            self.peer = upeer
        else:
            s.update({'chost': self.host, 'cport': self.port, 'conn': self, 'connected': True})
            self.server.peers.append(s)
            self.peer = s

    def handle_response(self, mtype, data):
        pid = peer_id(self.peer)
        with self.lock:
            if pid not in self.server.responses:
                self.server.responses[pid] = {}
            self.server.responses[pid].update({mtype: data})
