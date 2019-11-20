from . import protocol


# Request messages ------------------------------------------------------------

def request_exchange(node):
    return {
        'mtype': protocol.REQUEST_EXCHANGE,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'accept': True, 'info': None}
    }


def request_hello(node):
    return {
        'mtype': protocol.REQUEST_HELLO,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': protocol.NO_ERROR,
        'payload': f"Hello World! from Node({node.name})"
    }


def request_model(node):
    return {
        'mtype': protocol.REQUEST_MODEL,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'model': node.get_model()}
    }


# Response messages -----------------------------------------------------------

def response_exchange(node):
    return {
        'mtype': protocol.RESPONSE_EXCHANGE,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': protocol.NO_ERROR,
        'payload': {'accept': True, 'info': None}
    }


def response_hello(node):
    return {
        'mtype': protocol.RESPONSE_HELLO,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': protocol.NO_ERROR,
        'payload': f"Hi from Node({node.name})"
    }


def response_model(node):
    error = protocol.NO_ERROR if node.get_model() is not None else protocol.NO_MODEL
    return {
        'mtype': protocol.RESPONSE_MODEL,
        'sender': {'name': node.name, 'host': node.host, 'port': node.port},
        'status': error,
        'payload': {'model': node.get_model()}
    }
