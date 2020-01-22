from . import protocol
from .helpers import data_size


def request_subscribe(node):
    return {
        'mtype': protocol.REQUEST_SUBSCRIBE,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
    }


def response_subscribe(node):
    return {
        'mtype': protocol.RESPONSE_SUBSCRIBE,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
    }


def request_information(node):
    return {
        'mtype': protocol.REQUEST_INFORMATION,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'data_size': data_size(node)}
    }


def response_information(node):
    return {
        'mtype': protocol.RESPONSE_INFORMATION,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'data_size': data_size(node)}
    }


def exchange_model(node, respond=True):
    return {
        'mtype': protocol.EXCHANGE_MODEL,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'model': node.get_model(), 'respond': respond}
    }


def exchange_sol_model(node):
    return {
        'mtype': protocol.EXCHANGE_SOL_MODEL,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'model': node.solitary_model}
    }


def exchange_variables(node, neighbor, respond=True):
    name = neighbor['name']
    return {
        'mtype': protocol.EXCHANGE_VARIABLES,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {
            'model_i': node.Theta[node.name],
            'model_j': node.Theta[node.name],
            'A_i': node.A[node.name],
            'A_j': node.A[name],
            'respond': respond
        }
    }


# Request messages ------------------------------------------------------------

def request_exchange(node):
    return {
        'mtype': protocol.REQUEST_EXCHANGE,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'data_size': data_size(node)}
    }


def request_hello(node):
    return {
        'mtype': protocol.REQUEST_HELLO,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': f"Hello World! from Node({node.name})"
    }


def request_model(node):
    return {
        'mtype': protocol.REQUEST_MODEL,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'model': node.get_model()}
    }


# Response messages -----------------------------------------------------------

def response_exchange(node):
    return {
        'mtype': protocol.RESPONSE_EXCHANGE,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'data_size': data_size(node)}
    }


def response_hello(node):
    return {
        'mtype': protocol.RESPONSE_HELLO,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': protocol.NO_ERROR,
        'payload': f"Hi from Node({node.name})"
    }


def response_model(node):
    error = protocol.NO_ERROR if node.get_model() is not None else protocol.NO_MODEL
    return {
        'mtype': protocol.RESPONSE_MODEL,
        'sender': {'name': node.name, 'shost': node.host, 'sport': node.port},
        'status': error,
        'payload': {'model': node.get_model()}
    }
