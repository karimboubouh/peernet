import socket
import random
import string
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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


def load(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("Writing to file", filename)
    return


def sample(data, n, numpy_array=False):
    """
    Takes a dictionary and return n random samples.
    :param data: Dict
    :param n: int
    :param numpy_array: Boolean
    :rtype: list
    """
    if numpy_array:
        mask = np.random.choice(data.shape[0], n, replace=False)
        if data.ndim == 1:
            return data[mask]
        else:
            return data[mask, :]
    else:
        return random.sample(data, n)


def sample_xy(X, y, n, seed):
    # np.random.seed(seed)
    mask = np.random.choice(X.shape[0], n, replace=False)
    return X[mask, :], y[mask]


def random_number(min_size, max_size, epsilon=0.0, distribution="random", seed=0):
    n = 0
    if distribution == "uniform":
        # np.random.seed(seed)
        mu = max_size / 2
        sigma = max_size * epsilon
        n = min_size - 1
        while n < min_size or n > max_size:
            n = int(np.rint(np.random.normal(mu, sigma)))
        # print(f"MEAN={mu} | STD={sigma} | n={n}")
    else:
        # random.seed(seed + 55)
        n = random.randrange(min_size, max_size)

    return n


def shuffle(X, y):
    # np.random.seed(SHUFFLE_SEED)
    shuffle_index = np.random.permutation(X.shape[0])
    return X[shuffle_index], y[shuffle_index]


def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


def data_size(node, attribute="x_train"):
    if isinstance(node.ldata, dict):
        return node.ldata[attribute].shape[0]
    else:
        return len(node.ldata)


def find_peer(node, data):
    peer_name = data['sender']['name']
    for peer in node.peers:
        if peer['name'] == peer_name:
            return peer
    print(f"{node.pname} :: Unable to find peer {peer_name}")
    return None


def _p(protocol_id):
    if protocol_id == 10:
        return "REQUEST_SUBSCRIBE"
    elif protocol_id == 11:
        return "RESPONSE_SUBSCRIBE"
    elif protocol_id == 12:
        return "REQUEST_INFORMATION"
    elif protocol_id == 13:
        return "RESPONSE_INFORMATION"
    elif protocol_id == 14:
        return "EXCHANGE_MODEL"
    elif protocol_id == 15:
        return "EXCHANGE_SOL_MODEL"
    elif protocol_id == 16:
        return "EXCHANGE_VARIABLES"
    elif protocol_id == 11:
        return


def wait_until(predicate, timeout, period=0.25, *args, **kwargs):
    start_time = time.time()
    mustend = start_time + timeout
    while time.time() < mustend:
        if predicate(*args, **kwargs):
            log("info", f"{predicate} finished after {time.time() - start_time} seconds.")
            return True
        time.sleep(period)
    print(bold("TIME OUT SYSTEM !!!!!"))
    return False


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
    if mtype == 'success' and DEBUG_LEVEL > 2:
        print(f"\033[32mSUCCESS   >>\033[0m {message}")
        return
    if mtype == 'result' and DEBUG_LEVEL > -1:
        print(f"\033[32mRESULT    >> {message}\033[0m")
        return
    if mtype == 'info' and DEBUG_LEVEL > 3:
        print(f"\033[34mINFO      >>\033[0m {message}")
        return
    if message == '' and DEBUG_LEVEL > 0:
        print(f"\033[31mDEBUG     >> {mtype} {message}\033[0m")
        return


def bold(text):
    return f"\033[1m{text}\033[0m"

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users
