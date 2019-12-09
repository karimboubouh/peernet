import numpy as np
from sklearn.model_selection import train_test_split

from algorithms.hello import hello
from peernet.PeerNet import PeerNet
from peernet.network.arch import random_network
from algorithms.logistic_regression import LogisticRegressionNN


def pre_processing(df):
    """Data pre-processing function"""

    X = df.iloc[:, :-1].values
    y = df.iloc[:, 60].values
    y = np.where(y == 'R', 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    return X_train, X_test, y_train, y_test


def main():
    p2p = PeerNet("conf.yaml")

    # Setup the P2P network and the communication between neighbors
    p2p.network(random_network).init()

    # Load and randomly distribute training samples between nodes
    p2p.load_dataset("./data/sonar.csv", df=True, min_samples=0)

    # Train the model using one of the approaches: "MP", "CL" or "LL"
    p2p.train(
        model=LogisticRegressionNN,
        pre=pre_processing,
        algorithm="MP",
        params={"K": 5}
    )

    # Show information about the nodes
    p2p.info()


main()

# ---- Tasks
# TODO Select random peers with probability π_j if Agent_j \in Neighbors
# # TODO Define a static model for analysis
# # TODO (plus some small constant in the case where mi = 0).
# # TODO where µ > 0 is a trade-off parameter
# ---- Remarks
# R1: With t -> infinity: the algorithm converge to optimal model
