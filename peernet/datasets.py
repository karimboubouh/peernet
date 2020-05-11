import joblib
import numpy as np
import scipy.stats as stats

from .constants import MNIST_PATH, BALANCED_DATA_SIZE
from .helpers import shuffle, random_int, sample_xy, log


# ------------------------- FETCH DATASET -------------------------------------

def fetch_mnist(pre=None, train_size=60000):
    X, y = joblib.load(MNIST_PATH)
    x_train = X[:train_size]
    y_train = y[:train_size]
    x_test = X[train_size:]
    y_test = y[train_size:]
    x_train, y_train = shuffle(x_train, y_train)
    data = x_train, y_train, x_test, y_test
    if pre:
        data = pre(data)

    return data


def fetch_cifar(pre=None, train_size=60000):
    return None


# ------------------------- OPERATIONS ON MNIST DATASET -----------------------

def mnist_iid(x_train, y_train, num_users, balancedness, lower=None, upper=None):
    """
    Sample I.I.D. client data from MNIST dataset
    @param x_train:
    @param y_train:
    @param num_users:
    @param balancedness:
    @param upper:
    @param lower:
    @return: dict of user images
    """
    dataset = np.hstack((x_train, y_train.reshape(-1, 1)))
    if balancedness <= 0:
        balancedness = .00001
    if balancedness >= 1:
        balancedness = 1
    mu = int(len(dataset) / num_users)
    if not lower:
        lower = 100  # 1 / 2 * mu
    if not upper:
        upper = 2 * mu - lower  # 3 / 2 * mu
    sigma = balancedness * (upper - lower) / 2  # balancedness * mu / 2
    cpm = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    num_items = list(np.round(cpm.rvs(num_users)).astype(np.int))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        if i == num_users - 1 or len(all_idxs) < num_items[i]:
            num_items[i] = len(all_idxs)
        mask = np.random.choice(all_idxs, num_items[i], replace=False)
        if mask.size < lower:
            to_add = int(lower - mask.size)
            entries = np.random.choice(dataset.shape[0], to_add, replace=False)
            mask = np.append(mask, entries).astype(int)
        client_data = dataset[mask, :]
        dict_users[i] = (client_data[:, :-1], client_data[:, -1])
        all_idxs = list(set(all_idxs) - set(mask))
        print('.', end="")

    return dict_users


def mnist_noniid(x_train, y_train, num_users, balancedness, lower=1, upper=10):
    """
    Sample non-I.I.D client data from MNIST dataset.
    @param x_train:
    @param y_train:
    @param num_users: number of training images
    @param upper: max shard per user
    @param lower: min shard per user
    @param balancedness: data balancedness between 0 and 1
    @returns a dict of clients with each clients assigned
    certain number of training images
    """

    dataset = np.hstack((x_train, y_train.reshape(-1, 1)))
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards = num_users * 2
    num_imgs = int(len(dataset) / num_shards)
    # num_shards, num_imgs = 100, 600  #  1200, 50 #
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset[:, -1].astype(int)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    if balancedness <= 0:
        balancedness = .00001
    if balancedness >= 1:
        balancedness = 1
    mu = np.round(num_shards / num_users)
    sigma = balancedness * mu
    cpm = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    num_items = list(np.round(cpm.rvs(num_users)).astype(np.int))
    num_items = np.around(num_items / sum(num_items) * num_shards)
    random_shard_size = num_items.astype(int)
    log("info", f"shards of size {num_imgs} are divided among {num_users} users as fallows:\n{random_shard_size}")

    # Assign each client one sample from each label
    digits = np.zeros(10)
    for i in range(10):
        digits[i] = np.where(labels == i)[0][0]
    for i in range(num_users):
        dict_users[i] = np.concatenate((dict_users[i], digits), axis=0).astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0).astype(int)

        random_shard_size = random_shard_size - 1

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
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0).astype(int)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0).astype(int)

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
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0).astype(int)

    for i in range(num_users):
        user_data = dataset[dict_users[i], :]
        dict_users[i] = (user_data[:, :-1], user_data[:, -1])
        print('.', end="")

    return dict_users


def mnist_iid_sample(x, y, nodes, balancedness=1.0):
    data_per_client = []
    data = np.hstack((x, y.reshape(-1, 1)))
    if balancedness >= 1.0:
        chunk = int(data.shape[0] / nodes)
        for i in range(nodes):
            mask = np.random.choice(data.shape[0], chunk, replace=False)
            node_data = data[mask, :]
            data_per_client.append((node_data[:, :-1], node_data[:, -1]))
            data = np.delete(data, tuple(mask), axis=0)
            print('.', end='')
    else:
        chunks = balancedness ** np.linspace(0, nodes - 1, nodes)
        chunks /= np.sum(chunks)
        chunks = 0.1 / nodes + (1 - 0.1) * chunks
        chunks = [np.floor(chunk * data.shape[0]).astype('int') for chunk in chunks]
        np.random.shuffle(chunks)
        for i in range(nodes):
            if chunks[i] > data.shape[0]:
                chunks[i] = data.shape[0]
            mask = np.random.choice(data.shape[0], chunks[i], replace=False)
            node_data = data[mask, :]
            data_per_client.append((node_data[:, :-1], node_data[:, -1]))
            data = np.delete(data, tuple(mask), axis=0)
            print('.', end='')

    return data_per_client


def iid_sample(x, y, balanced=True, epsilon=None):
    # todo add option where each node's data is unique
    n = 0
    if balanced:
        n = BALANCED_DATA_SIZE
    else:
        size = x.shape[0]
        n = random_int(size, min_size=0, epsilon=epsilon)

    return sample_xy(x, y, n)


def noniid_sample(x, y, balanced=True, epsilon=None):
    n = 0
    if balanced:
        n = BALANCED_DATA_SIZE
    else:
        size = x.shape[0]
        n = random_int(size, min_size=0, epsilon=epsilon)

    return sample_xy(x, y, n)
