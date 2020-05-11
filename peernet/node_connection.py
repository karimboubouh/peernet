import _pickle
import socket
import time
import random
import pickle
import threading
import traceback

from . import protocol
from .constants import M_CONSTANT, SOCK_TIMEOUT
from .helpers import log, unique_id, peer_id, data_size, find_peer
from .updates import cl_update_secondary, cl_update_dual
from .message import response_hello, exchange_model, request_subscribe, response_subscribe, response_information, \
    exchange_variables, banned_node


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
            log('info', f"{self.server.pname}: [S] Connection started with  Node({peer['name']})")
        else:
            self.host = peer['shost']
            self.port = peer['sport']
            log('info', f"{self.server.pname}: [C] Connection accepted from Node({peer['name']})")
        self.peer = peer
        self.callback = callback
        self.terminate_flag = threading.Event()
        self.lock = threading.RLock()
        # random.seed(100)

    def send(self, message):
        try:
            # In case of big messages use a header and send the message in chunks
            blob = pickle.dumps(message)
            self.sock.sendall(blob)
            if message['mtype'] not in [10, 11]:
                self.server.message_count_send += 1
        except socket.error as e:
            self.terminate_flag.set()
            log('exception', f"{self.server.pname} NodeConnection: Unexpected error!\n{e}")
        except Exception as e:
            print(f"EXCEPTION {e}")
            traceback.print_exc()

    def stop(self):
        self.terminate_flag.set()

    def run(self):
        # Timeout, so the socket can be closed when it is dead!
        self.sock.settimeout(SOCK_TIMEOUT)
        # Send init message to exchange necessary nodes information in case of InNode
        if self.peer.get('ctype', '') == "In":
            self.send(request_subscribe(self.server))

        while not self.terminate_flag.is_set():
            try:
                # buffer = self.sock.recv(165536)
                buffer = self.sock.recv(500000)
                self.handle_request(buffer)
                self.server.message_count_recv = self.server.message_count_recv + 1
            except pickle.UnpicklingError as e:
                log('exception', f"{self.server.pname}: NodeConnection: Corrupted message. {len(buffer)}")
            except OverflowError as e:
                log('exception', f"{self.server.pname}: OverflowError. {len(buffer)}")
            except KeyError as e:
                log('exception', f"{self.server.pname}: NodeConnection: KeyError")
            except socket.timeout:
                pass
            except:
                traceback.print_exc()
                self.terminate_flag.set()
                log('exception', f"{self.server.pname}: NodeConnection: Socket has been terminated!")

        self.sock.settimeout(None)
        self.sock.close()
        log('info', f"{self.server.pname}: NodeConnection: Stopped")

    def handle_request(self, buffer):
        if not self.check_message(buffer):
            log('error', f"{self.server.pname}: NodeConnection: Message is damaged)")
            return False
        if buffer == b'':
            return False
        try:
            data = pickle.loads(buffer)
        except EOFError as e:
            log("error", f"handle_request exception: {e}")
            print(len(buffer))
            print(buffer)
            return
            # traceback.print_exc()
            # data = None
        except ValueError as e:
            log("error", f"handle_request exception: {e}")
            return False

        if data is not None:
            self.server.event_node_message(self.server, data)
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
                elif data['mtype'] == protocol.EXCHANGE_SOL_MODEL:
                    self.do_exchange_sol_model(data)
                elif data['mtype'] == protocol.BANNED_NODE:
                    self.do_got_banned(data)
                # responses ---------------------------------------
                elif data['mtype'] == protocol.RESPONSE_SUBSCRIBE:
                    self.update_info(data)
                elif data['mtype'] == protocol.RESPONSE_INFORMATION:
                    self.do_exchange_info(data, request=False)
                elif data['mtype'] == protocol.RESPONSE_HELLO:
                    self.handle_response(protocol.RESPONSE_HELLO, data)
                elif data['mtype'] == protocol.RESPONSE_MODEL:
                    self.handle_response(protocol.RESPONSE_MODEL, data)
                elif data['mtype'] == protocol.EXCHANGE_VARIABLES:
                    self.do_cl_update(data)
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

    def do_exchange_sol_model(self, data):
        # update models, Z, and A
        index = data['sender']['name']
        model = data['payload']['model']
        self.server.Theta[index] = model.parameters
        self.server.Z[index] = model.parameters
        self.server.A[index] = 0
        # in case of new peers
        self.server.update_weights()

    def do_exchange_model(self, data):
        # Deal with synchrony issues
        if self.server.model is None or self.server.solitary_model is None:
            self.server.stop_condition = 0
            self.send(exchange_model(self.server, respond=False))
            return

        if self.server.protocol == "MP":
            self.do_mp_exchange_model(data)
        elif self.server.protocol == "CMP":
            self.do_cmp_exchange_model(data)
        else:
            raise ValueError("Protocol not recognized.")

    def do_mp_exchange_model(self, data):
        if data['payload']['model'] is not None:
            # update models list
            self.server.update_models(data['sender']['name'], data['payload']['model'])
            if data['payload']['respond']:
                self.send(exchange_model(self.server, respond=self.server.check_exchange()))
            # else:
            #     self.server.exclude_peer(self.peer)
            # update while not stop condition
            if self.server.check_exchange():
                self.server.mp_calculate_update(data)
                # time.sleep(random.randint(10, 100) / 1000)
                if self.server.check_exchange():
                    neighbor = self.server.get_random_peer(ignore_excluded=True)  # ignore_excluded=True
                    if neighbor:
                        exchange = exchange_model(self.server)
                        self.server.send(neighbor, exchange)
                    else:
                        self.server.stop_condition = 0
                        print("FFFFFF")
                        return
        else:
            self.server.exclude_peer(find_peer(self.server, data))
            neighbor = self.server.get_random_peer(ignore_excluded=True)
            if neighbor:
                exchange = exchange_model(self.server)
                self.server.send(neighbor, exchange)
            else:
                print("DDDDDD")
                self.server.stop_condition = 0
                return

    def do_cmp_exchange_model(self, data):
        # if node is banned ignore exchange operation
        banned = False
        name = data['sender']['name']
        if name in self.server.banned:
            log("warning", f"{self.server.pname} blocked collaboration with node {name}")
            banned = True
            # exchange_model(self.server, respond=self.server.check_exchange())
            self.send(banned_node(self.server))
            self.server.message_count_ignr += 1

        # if contribution factor of node is less then a threshold ignore
        valid = self.server.evaluate_model(data)
        if not banned and not valid:
            log("warning", f"{self.server.pname} ignored one exchange with node {name} | cf={self.server.cf[name]}")
            self.server.message_count_ignr += 1
            if data['payload']['respond']:
                self.send(exchange_model(self.server, respond=self.server.check_exchange()))

        if not banned and valid and data['payload']['model'] is not None:
            # Send back my model if neighbor's respond is true
            if data['payload']['respond']:
                self.send(exchange_model(self.server, respond=self.server.check_exchange()))

            # update models list
            self.server.update_models(data['sender']['name'], data['payload']['model'])

            # update while not stop condition
            if self.server.check_exchange():
                self.server.cmp_calculate_update(data)
                # time.sleep(random.randint(10, 100) / 1000)
                if self.server.check_exchange():
                    neighbor = self.server.get_random_peer(ignore_excluded=True)
                    if neighbor:
                        exchange = exchange_model(self.server)
                        self.server.send(neighbor, exchange)
                    else:
                        self.server.stop_condition = 0
                        print("FFFFFF")
                        return
        else:
            self.server.exclude_peer(find_peer(self.server, data))
            neighbor = self.server.get_random_peer(ignore_excluded=True)  # ignore_excluded=True
            if neighbor:
                exchange = exchange_model(self.server)
                self.server.send(neighbor, exchange)
            else:
                print("DDDDDD")
                self.server.stop_condition = 0
                return

    def do_got_banned(self, data):
        peer = find_peer(self.server, data)
        self.server.exclude_peer(peer)
        self.server.stop_condition -= 1
        self.server.costs.append(self.server.costs[-1])
        if self.server.check_exchange():
            neighbor = self.server.get_random_peer(ignore_excluded=True)
            if neighbor:
                self.server.send(neighbor, exchange_model(self.server))
            else:
                self.server.stop_condition = 0

    def do_cl_update(self, data):
        # CURRENT ROUND if SC > 0
        with self.server.lock:
            peer = find_peer(self.server, data)
            if self.server.check_exchange():
                respond = data['payload']['respond']
                if not respond:
                    self.server.exclude_peer(peer)
                # Step 1: Minimize node.Theta
                self.server.update_primal(peer, respond=respond)
                # Step 2: Update secondary variables.
                cl_update_secondary(self.server, data)
                # Step 3: Update dual variables.
                cl_update_dual(self.server, data)
            else:
                self.server.send(peer, exchange_variables(self.server, peer, respond=False))

            # NEXT ROUND if SC > 0
            if self.server.check_exchange():
                neighbor = self.server.get_random_peer(ignore_excluded=True)
                if neighbor:
                    self.server.update_primal(neighbor)

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
                max_size = max(peer.get("data_size", 0) for peer in self.server.peers)
                self.server.c = M_CONSTANT + data_size(self.server) / max_size
            except ZeroDivisionError:
                log('exception', f"{self.server.pname}: Max data sizes equal to zero")
                print(f"{self.server.pname}: Max data sizes equal to zero")
                self.server.c = M_CONSTANT + 1
                print(f"{self.server.pname}:EXCEPT:C={self.server.c}")

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
