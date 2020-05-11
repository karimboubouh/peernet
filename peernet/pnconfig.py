import yaml
# from .node import Node
from peernet.constants import SOCK_TIMEOUT
from peernet.node import Node
from peernet.helpers import log


# Class PNConfig ---------------------------------------------------------
class PNConfig:
    def __init__(self, param):
        self.config = {}
        if isinstance(param, tuple):
            self.process_tuple(param)
        else:
            self.process_yaml(param)

    def get_nodes(self):
        nodes = []
        for entry in self.config['nodes']:
            nodes += [Node(entry, self.get_timeoutms(), None)]
        return nodes

    def get_timeoutms(self):
        return self.config['timeout_ms']

    def get_debug(self):
        return self.config['debug']

    def process_yaml(self, config_file):
        yaml_ = yaml.load(open(config_file, 'rt'), Loader=yaml.FullLoader)
        self.config = {}
        for c in yaml_:
            k = list(c.keys())[0]
            self.config[k] = c[k]

    def process_tuple(self, tuple_):
        if len(tuple_) != 2:
            log('exception', f"Enter a config file or a tuple in the form of (Number of nodes, INIT_PORT)")
        n, port = tuple_
        self.config['timeout_ms'] = SOCK_TIMEOUT
        self.config['nodes'] = []
        for i in range(n):
            self.config['nodes'].append({'name': f"w{i}", 'host': '', 'port': port + i})


if __name__ == '__main__':
    pass
    # # generate conf file
    # filename = "../myconf.yaml"
    # n = 100
    # port_start = 45000
    # name = 'w'
    # nodes = []
    # for i in range(n):
    #     x = f"name: {name}{i + 1}, host: localhost, port: {port_start + i}"
    #     nodes.append(x)
    # with open(filename, 'w') as file:
    #     documents = yaml.dump({'nodes': nodes}, file)
