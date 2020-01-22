# START_PORT
START_PORT = 15000
# TOPOLOGY
TOPOLOGY = {
    'random': 'peernet.network.arch.random_network',
    'static': 'peernet.network.arch.static_network',
    'ErdosRenyi': 'lib.graph.generator.erdos_renyi.ErdosRenyi'
}
# ML ALGORITHMS
ALGORITHM = {
    'logistic': 'algorithms.LogisticRegression.LogisticRegression'
}
PRE_PROCESSING = {
    'mnist.data': 'peernet.pre_processing.load_mnist_12',
    'mnist': 'peernet.pre_processing.load_mnist'
}
# DEBUG_LEVEL: 0 - NO DEBUG | 1 -'EXCEPTION' | 2 -'ERROR' | 3 -'WARNING' | 4 -'INFO' and 'SUCCESS'
DEBUG_LEVEL = 4

# Supported algorithms
ALGORITHMS = ["MP", "CL", "LL"]

# datasets folder
DATASETS_FOLDER = "./datasets"

# MP
M_CONSTANT = 0

TCP_SOCKET_BUFFER_SIZE = 8 * 1024 * 1024
TCP_SOCKET_SERVER_LISTEN = 10

# ------------------------------ Seed variables -------------------------------
SHUFFLE_SEED = 100
NETWORK_SEED = 100

# ------------------------------ Shared variables -----------------------------
STOP_CONDITION = 100
ITERATION_STEP = 10
EPSILON_STEP = 0.05
TRAINED_MODELS = 0


def nextSC(step=10):
    global STOP_CONDITION
    STOP_CONDITION += step


def ll_done():
    global TRAINED_MODELS
    TRAINED_MODELS -= 1
