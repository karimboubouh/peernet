import socket
import random
import string

from .constants import *


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def unique_id(length=6):
    """Generate a random string of letters and digits """
    ld = string.ascii_letters + string.digits
    return ''.join(random.choice(ld) for i in range(length))


def peer_id(peer):
    return f"{peer['shost']}:{peer['sport']}"


def sample(data, n):
    """
    Takes a dictionary and return n random samples.
    :param data: Dict
    :param n: int
    :rtype: list
    """
    return random.sample(data, n)


def log(mtype, message=''):
    if mtype == 'exception' and DEBUG_LEVEL > 0:
        print(f"\033[31mEXCEPTION >>\033[0m {message}")
        return
    if mtype == 'error' and DEBUG_LEVEL > 1:
        print('\033[31m', "ERROR >>    ", '\033[0m', message)
        return
    if mtype == 'event' and DEBUG_LEVEL > 1:
        print(f"\033[35mEVENT     >>\033[0m {message}")
        return
    if mtype == 'warning' and DEBUG_LEVEL > 2:
        print('\033[33m', "WARNING >>  ", '\033[0m', message)
        return
    if mtype == 'success' and DEBUG_LEVEL > 1:
        print(f"\033[32mSUCCESS   >>\033[0m {message}")
        return
    if mtype == 'info' and DEBUG_LEVEL > 3:
        print(f"\033[34mINFO      >>\033[0m {message}")
        return
    if message == '' and DEBUG_LEVEL > 0:
        print(f"\033[31mDEBUG     >> {mtype} {message}\033[0m")
        return

def bold(text):
    return f"\033[1m{text}\033[0m"
