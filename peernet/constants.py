# DEBUG_LEVEL: 0 - NO DEBUG | 1 -'EXCEPTION' | 2 -'ERROR' | 3 -'WARNING' | 4 -'INFO' and 'SUCCESS'
DEBUG_LEVEL = 1
# START_PORT
START_PORT = 45000
# TOPOLOGY
TOPOLOGY = {
    'random': 'peernet.network.arch.random_network',
    'static': 'peernet.network.arch.static_network',
    'ErdosRenyi': 'lib.graph.generator.erdos_renyi.ErdosRenyi'
}
# ML ALGORITHMS
ALGORITHM = {
    'logistic': 'sklearn.linear_model.LogisticRegression',
    'svm': 'sklearn.svm.SVC',
    'logistic1': 'algorithms.LogisticRegression.LogisticRegression'
}
PRE_PROCESSING = {
    'mnist.data': 'peernet.pre_processing.load_mnist_12',
    'mnist': 'peernet.pre_processing.load_mnist'
}
# Model wrappers
WRAPPERS = {
    'sklearn': 'peernet.wrappers.scikit_learn.SklearnWrapper',
    'keras': None,
    'pytorch': None
}

# Supported algorithms


# datasets folder
DATASETS_FOLDER = "./datasets"

TCP_SOCKET_BUFFER_SIZE = 500000
TCP_SOCKET_SERVER_LISTEN = 10

# ------------------------------ Seed variables -------------------------------
SHUFFLE_SEED = 100
NETWORK_SEED = 100

# ------------------------------ Shared variables -----------------------------
ITERATION_STEP = 10
EPSILON_STEP = 0.05
TRAINED_MODELS = 0
TARGET_ACCURACY = 0.91
M_CONSTANT = 0
SOCK_TIMEOUT = 10


def nextSC(step=10):
    global STOP_CONDITION
    STOP_CONDITION += step


def ll_done():
    global TRAINED_MODELS
    TRAINED_MODELS -= 1


# ------------------------------  DATASETS Related ----------------------------
MNIST_PATH = "./datasets/mnist.data"
CIFAR_PATH = "./datasets/cifar.data"
BALANCED_DATA_SIZE = 30000
# ------------------------------  ALGORITHM Related ---------------------------
STOP_CONDITION = 100
TEST_SAMPLE = 1000
CF_THRESHOLD = 0.8  # 0.75
EPSILON_FAIRNESS = -0.1  # range[-1,1] | chosen values (0.2 ,0.1, 0, -0.1, -0.2)
CONFIDENCE_MEASURE = "max"  # mean or max (same behavior)
ACCURACY_METRIC = "accuracy"  # accuracy, loss, precision, recall or f1_score
# ------------------------------  PROTOCOL Related ----------------------------
PROTOCOLS = ["MP", "CDPL", "CL", "LL"]
