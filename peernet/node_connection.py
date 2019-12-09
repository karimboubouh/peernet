import socket
import time
import random
import pickle
import threading
import traceback

from . import protocol
from .helpers import log, unique_id, peer_id
from .message import response_hello, exchange_model, request_subscribe, response_subscribe, response_information


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
            self.send(request_subscribe(self.server))

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
                if data['mtype'] == protocol.REQUEST_SUBSCRIBE:
                    self.do_subscribe(data)
                elif data['mtype'] == protocol.REQUEST_INFORMATION:
                    self.do_exchange_info(data, request=True)
                elif data['mtype'] == protocol.REQUEST_HELLO:
                    self.do_hello(data)
                elif data['mtype'] == protocol.EXCHANGE_MODEL:
                    self.do_exchange_model(data)
                # responses ---------------------------------------
                elif data['mtype'] == protocol.RESPONSE_SUBSCRIBE:
                    self.update_info(data)
                elif data['mtype'] == protocol.RESPONSE_INFORMATION:
                    self.do_exchange_info(data, request=False)
                elif data['mtype'] == protocol.RESPONSE_HELLO:
                    self.handle_response(protocol.RESPONSE_HELLO, data)
                elif data['mtype'] == protocol.RESPONSE_MODEL:
                    self.handle_response(protocol.RESPONSE_MODEL, data)
                else:
                    log('error',
                        f"{self.server.pname}: NodeConnection: Message type ({data['mtype']}) is not supported")
                    return False
            except Exception as e:
                log('error', f"{self.server.pname}: NodeConnection: Exception while processing message! {e}")
                traceback.print_exc()
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

    def do_exchange_model(self, data):
        # update models list
        self.server.update_models(data['sender']['name'], data['payload']['model'])
        # Check if we still need to exchange models
        update = self.server.check_exchange()
        # Send back my model if neighbor's respond is true
        if data['payload']['respond']:
            self.send(exchange_model(self.server, respond=update))
        else:
            self.server.exclude_peer(self.peer)
        # update while not stop condition
        if update:
            self.server.calculate_update(data)
            if self.server.check_exchange():
                # todo wait for a random time before performing the next iteration!
                neighbor = self.server.get_random_peer(ignore_excluded=True)
                exchange = exchange_model(self.server)
                self.server.send(neighbor, exchange)

    def do_subscribe(self, data):
        message = response_subscribe(self.server)
        self.send(message)

    def do_exchange_info(self, data, request=False):
        # Send back information if its a request message
        if request:
            message = response_information(self.server)
            self.send(message)
        # Update neighbor information
        s = data['sender']
        upeer = next((p for p in self.server.peers if p["shost"] == s['shost'] and p["sport"] == s['sport']), {})
        if upeer:
            upeer.update({'data_size': data['payload']['data_size']})
            self.peer = upeer
            try:
                self.server.c = len(self.server.ldata) / max(peer.get("data_size", 0) for peer in self.server.peers)
            except ZeroDivisionError:
                log('exception', f"{self.server.pname}: Max data sizes equal to zero")
                self.server.c = 1

    def update_info(self, data):
        s = data['sender']
        upeer = next((p for p in self.server.peers if p["shost"] == s['shost'] and p["sport"] == s['sport']), {})
        if upeer:
            upeer.update({'name': s['name'], 'shost': s['shost'], 'sport': s['sport'], 'ctype': "In", 'conn': self,
                          'connected': True})
            self.peer = upeer
        else:
            rw = random.uniform(0, 1)
            s.update(
                {'chost': self.host, 'cport': self.port, 'ctype': "In", 'weight': rw, 'conn': self, 'connected': True}
            )
            self.server.peers.append(s)
            self.peer = s
            self.server.update_weights()

    def handle_response(self, mtype, data):
        pid = peer_id(self.peer)
        with self.lock:
            if pid not in self.server.responses:
                self.server.responses[pid] = {}
            self.server.responses[pid].update({mtype: data})
