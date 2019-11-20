from threading import Thread, Lock
import logging

LOGGER = logging.getLogger(__name__)


class Sender(Thread):
    def __init__(self, host, port, socket_timeout_ms):
        # Initialization
        super(Sender, self).__init__()
        self.host = host
        self.port = port
        self.socket_timeout_ms = socket_timeout_ms
        self.have_state = False
        self.lock = Lock()
        # Logger
        LOGGER.info(f"Starting Sender thread. listening on {host}:{port}...")
        # Create server socket
        self.sock = _create_tcp_socket()
        self.sock.bind((host, port))
        self.sock.listen(TCP_SOCKET_SERVER_LISTEN)

    def _handle_request(self, client_sock):
        try:
            # The socket is blocking
            LOGGER.debug("RxThread: receiving message fd=%d", client_sock.fileno())
            message_type, _, _ = recv_message(client_sock)
            assert message_type == MESSAGE_TYPE_FETCH_PARAMETERS

            # send the result
            if not self.have_state:
                send_message(client_sock, MESSAGE_TYPE_FETCH_PARAMETERS)
            else:
                with self.lock:
                    send_message(client_sock, MESSAGE_TYPE_FETCH_PARAMETERS, self.state, self.payload)

        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning("Other end had a timeout, socket closed")
            self._unregister_fd(client_sock.fileno())
            client_sock.close()

        except:
            LOGGER.exception("Error handling request (closing socket, client will retry)")
            self._unregister_fd(client_sock.fileno())
            client_sock.close()

    def _handle_client_event(self, events, conn):
        pass

    def _handle_new_connection(self, events, _):
        pass

    def run(self):
        LOGGER.info("Sender: run()")
        try:
            while True:
                # Blocking
                events = self.efd.poll()
                for fd, events in events:
                    cb, args = self.fds[fd]
                    cb(events, args)
        finally:
            sock_fd = self.sock.fileno()
            fds = list(self.fds.keys())
            for fd in fds:
                if fd != sock_fd:
                    _, sock = self.fds[fd]
                    self.efd.unregister(fd)
                    sock.close()
            self.efd.unregister(sock_fd)
            self.sock.close()
            self.efd.close()

        LOGGER.info("TxThread: Exiting...")

    def shutdown(self):
        # TODO(guyz): Implement using eventfd...
        raise NotImplementedError
