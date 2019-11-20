import yaml
from .node import Node

# Class PNConfig ---------------------------------------------------------
class PNConfig:
    def __init__(self, config_file):
        self.yaml = yaml.load(open(config_file, 'rt'), Loader=yaml.FullLoader)
        self.config = {}
        for c in self.yaml:
            k = list(c.keys())[0]
            self.config[k] = c[k]

    def get_nodes(self):
        nodes = []
        for entry in self.config['nodes']:
            nodes += [Node(entry, self.get_timeoutms(), None)]
        return nodes

    def get_timeoutms(self):
        return self.config['timeout_ms']

    def get_debug(self):
        return self.config['debug']
