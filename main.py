from algorithms.hello import hello
from peernet.PeerNet import PeerNet
from algorithms.knn import lp_knn


def main():
    p2p = PeerNet("KNN PeerNet", "conf.yaml")
    # Initialize the P2P network
    p2p.init()
    # Load dataset
    # p2p.load_dataset("./data/iris.csv", df=True)
    # Train a model
    p2p.train(hello)
    # wait
    # p2p.sleep(0.5)
    print("-----------------------------------------------------------------------------------------------------------")
    p2p.info()


main()
