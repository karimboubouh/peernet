"""
    Experiment Plots
    ~~~~~~~~~~~~~~~~~~~~~
    Experiments used to benchmark the different setting of both MP and CL
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from peernet.helpers import load, save
from scipy.interpolate import make_interp_spline, BSpline


def data_distribution(nodes):
    if isinstance(nodes[0].ldata, dict):
        dist = [node.ldata['x_train'].shape[0] for node in nodes]
        plt.xlim(0, 60000)
    else:
        dist = [len(node.ldata) for node in nodes]
    sns.set(color_codes=True)
    dd = sns.distplot(dist)
    dd.set(xlabel='Data size', ylabel='Frequency', title="Data distribution of node's datasets")
    plt.show()


def neighbors_distribution(nodes):
    dist = [len(node.peers) for node in nodes]
    sns.set(color_codes=True)
    dd = sns.distplot(dist)
    dd.set(xlabel='Number of neighbors', ylabel='Frequency', title="Data distribution of neighbors")
    plt.show()


def iterations(x, y, labels):
    plt.title(labels['title'])
    plt.plot(x, y)
    plt.xlabel(labels['x'])
    plt.ylabel(labels['y'])
    plt.show()


def file_mp_iter(fileA, fileB, info):
    xa, ya = load(fileA)
    xb, yb = load(fileB)

    plt.figure()
    # plt.title(info['title'])
    plt.plot(xa, np.squeeze(ya), label=f"MP with confidence")
    plt.plot(xb, np.squeeze(yb), label=f"MP without confidence")
    plt.ylabel(info['ylabel'])
    plt.xlabel(info['xlabel'])
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def communication_rounds(N, C, F, ylabel="Accuracy", xlabel="Communication rounds"):
    xN, yN = load(N)
    xC, yC = load(C)
    xF, yF = load(F)
    plt.figure()
    plt.plot(xN, np.squeeze(yN), label=f"MP without confidence")
    plt.plot(xC, np.squeeze(yC), label=f"MP with confidence")
    plt.plot(xF, np.squeeze(yF), label=f"MP with contribution factor")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best', shadow=True)
    plt.show()


def byzantine_metrics(file):
    x, y = load(file)
    accuracy, precision, recall, f_score = y
    plt.figure()
    plt.plot(x, np.squeeze(precision), label=f"Precision")
    plt.plot(x, np.squeeze(recall), label=f"Recall")
    plt.plot(x, np.squeeze(f_score), label=f"F1 score")
    plt.ylabel("")  # "STD accuracy"
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def byz_metrics():
    x = list(range(1, 16))
    accuracy = [0.8869346938775512, 0.8883187499999999, 0.8918936170212766, 0.8865130434782608, 0.8861777777777776,
                0.8893181818181819, 0.8880023255813952, 0.8835880952380951, 0.8889853658536587, 0.88609,
                0.8883717948717947, 0.8829973684210528, 0.8820945945945947, 0.8820945945945947, 0.882611428571428, ]
    precision = [1.0]*15
    recall = [1.0]*15
    f_score = [1.0]*15
    y = accuracy, precision, recall, f_score
    save("byzantine_metrics_50_new_banned_1_15", (x, y))

    plt.figure()
    plt.plot(x, np.squeeze(precision), label=f"Precision")
    plt.plot(x, np.squeeze(recall), label=f"Recall")
    plt.plot(x, np.squeeze(f_score), label=f"F1 score")
    plt.ylabel("")
    plt.xlabel("Number of misbehaving nodes")
    plt.legend(loc='best', shadow=True)
    plt.show()


def graph_sparsity(N, C, F):
    xN, yN = load(N)
    yN = np.poly1d(np.polyfit(xN, yN, 5))(xN)
    xC, yC = load(C)
    yC = np.poly1d(np.polyfit(xC, yC, 5))(xC)
    xF, yF = load(F)
    yF = np.poly1d(np.polyfit(xF, yF, 5))(xF)
    plt.figure()
    plt.plot(xN, np.squeeze(yN), label=f"MP without confidence")
    plt.plot(xC, np.squeeze(yC), label=f"MP with confidence")
    plt.plot(xF, np.squeeze(yF), label=f"MP with contribution factor")
    plt.ylabel("Loss")
    plt.xlabel("CR")
    plt.legend(loc='best', shadow=True)
    plt.show()


def contribution_factor(F):
    x, y, z = load(F)
    plt.figure()
    plt.plot(x, np.squeeze(y), label=f"Banned nodes")
    plt.plot(x, np.squeeze(z), label=f"Temporary ignored nodes")
    plt.ylabel("Number of nodes")
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def banned_nodes(A, B, C, D):
    xA, yA = load(A)
    # xB, yB = load(B)
    # xC, yC = load(C)
    xD, yD = load(D)
    plt.figure()
    plt.plot(xA, np.squeeze(yA), label="Ban threshold $\epsilon=-0.05$")
    # plt.plot(xB, np.squeeze(yB), label="Ban threshold $\epsilon=-0.1$")
    # plt.plot(xC, np.squeeze(yC), label="Ban threshold $\epsilon=-0.15$")
    plt.plot(xD, np.squeeze(yD), label="Ban threshold $\epsilon=-0.2$")
    plt.ylabel("Accuracy")
    # plt.ylabel("STD accuracy")
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def ignored_nodes(A, B, C):
    xA, yA = load(A)
    xB, yB = load(B)
    xC, yC = load(C)
    plt.figure()
    plt.plot(xA, np.squeeze(yA), label="$CF_{th}=0.0$")
    plt.plot(xB, np.squeeze(yB), label="$CF_{th}=0.5$")
    plt.plot(xC, np.squeeze(yC), label="$CF_{th}=1.0$")
    plt.ylabel("Accuracy")
    # plt.ylabel("STD accuracy")
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def file_mp_itt(info):
    plt.figure()
    plt.title(info['title'])

    x = range(1, 101, 10)
    y_True = [23.80368138, 23.52593669, 23.50443649, 23.58219447, 23.55076273, 23.48020406,
              23.52051572, 23.52177749, 23.52161982, 23.51322253]
    y_False = [24.00486419, 24.44446922, 24.31556928, 24.34722569, 24.30652399, 24.33799417, 24.34685562, 24.34855124,
               24.35180307, 24.35942388]
    plt.plot(x, np.squeeze(y_True), label=f"MP with confidence")
    plt.plot(x, np.squeeze(y_False), label=f"MP without confidence")

    plt.ylabel(info['ylabel'])
    plt.xlabel(info['xlabel'])
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def file_mp_dense(info):
    plt.figure()
    plt.title(info['title'])

    x = np.arange(0, 1.1, 0.1)
    y_10_True = [0.027423279137122607, 0.02667724879147525, 0.033611347593196304, 0.03512237835181435,
                 0.04579236000624156, 0.08435176423205408, 0.08087801334978603, 0.13846205285545857,
                 0.15982682060059455, 0.24523217736516206, 0.1759090702367231]
    y_10_False = [0.030754575986556447, 0.03032223949832174, 0.026535763871424735, 0.03772503842564531,
                  0.037759506620966374, 0.10056036400543225, 0.11759669496885994, 0.12027872967394952,
                  0.08160015863702974, 0.19208787835645616, 0.20952840439881365]

    y_50_True = [0.027987786065690794, 0.11896590653605027, 0.2933899666669796, 0.3992139607202846, 0.4978003849954703,
                 0.5295687951729843, 0.5517126794357448, 0.5946363705257265, 0.6101507075683855, 0.636946920042505,
                 0.6420862759163994]
    y_50_False = [0.028418500293239213, 0.09269316435122082, 0.26703097829972644, 0.414719915942214, 0.4857704993859452,
                  0.5228590704553963,
                  0.5521480187459062, 0.5549814908832963, 0.6001085104246616, 0.619623493198325, 0.6506496282046205]

    y_100_True = [0.02593647524334251, 0.2416670844156414, 0.480860136479544, 0.5845701029033701, 0.6150450625299485,
                  0.6496488213565965, 0.6409474220848587, 0.6605176709299572, 0.6669131473563273, 0.673843786782845,
                  0.6700383911968143]
    y_100_False = [0.03445213748904238, 0.2332939360612752, 0.4618440893557946, 0.5564832810732301, 0.6159375100345745,
                   0.6363208191434693, 0.6415533129395767, 0.6470687784595163, 0.662503404159375, 0.6711421611269919,
                   0.6631067005300595]
    plt.plot(x, np.squeeze(y_10_True), label=f"MP with confidence")
    plt.plot(x, np.squeeze(y_10_False), label=f"MP without confidence")

    plt.ylabel(info['ylabel'])
    plt.xlabel(info['xlabel'])
    plt.legend(loc='lower right', shadow=True)
    plt.show()


def file_mp_data(info):
    plt.figure()
    plt.title(info['title'])
    # x, y = load(file)
    # iteration = 10
    # with_conf = []
    # without_conf = []
    # for epsilon in y['with']:
    #     with_conf.append(epsilon[iteration])
    # for epsilon in y['without']:
    #     without_conf.append(epsilon[iteration])
    x = np.arange(0, 1.1, 0.1)
    # y_0 = [8.693634064298132, 8.696129094238113, 8.693606087826042, 8.693041854858253, 8.70142955796378,
    #        8.679316006615487, 8.706036668231263, 8.698727025946152, 8.69798212170908, 8.70708454206522,
    #        8.736174292697054, 8.72158224607622, 8.709241940174762, 8.701223571168146, 8.696983965848572,
    #        8.74459726502001, 8.69403225559236, 8.70289269544868, 8.715333576774613, 8.72910343176976, 8.695207230717628]
    # y_10_False = [24.137693879745434, 24.174437934371145, 23.463571835361343, 24.475213157975436, 22.878461248616187,
    #               21.76769097456413, 23.902577122523198, 24.01757867593464, 23.81533726810062, 25.60462525794604,
    #               24.42390254458309, 25.60393943346981, 24.99583480605197, 24.570409896521717, 24.850017763683045,
    #               23.346442148339978, 24.044657930497923, 23.48593662963243, 25.812377243641368, 23.770622268780972,
    #               22.888498981981527]
    # y_10_True = [23.30308385144564, 23.845038394889016, 24.351380207901887, 23.72320087647418, 24.121738752266303,
    #              23.843166713633387, 15.933693732169115, 25.339593023823213, 25.451061844070875, 25.389804479660796,
    #              25.748081531000683, 25.374749949420487, 25.189830868265968, 24.109263722215953, 25.10274103766194,
    #              25.42533477503209, 25.016040501936068, 24.66706208245583, 23.53237594654569, 25.017627889839854,
    #              23.61864400866455]
    y_50_True = [49.58670973696354, 49.58131056760499, 49.57512690355329, 49.57143516382096, 49.5941393631749,
                 49.61301338255653, 49.59007844946932, 49.58020304568528, 49.6148592524227, 49.58952468850946,
                 49.57932625749884]
    y_50_False = [49.5757268112598, 49.58020304568528, 49.562990309183206, 49.580249192431936, 49.61176742039686,
                  49.619289340101524, 49.57757268112598, 49.57125057683433, 49.591508998615595, 49.561513613290266,
                  49.575542224273185]
    plt.plot(x, np.squeeze(y_50_True), label=f"MP with confidence")
    plt.plot(x, np.squeeze(y_50_False), label=f"MP without confidence")

    plt.ylabel(info['ylabel'])
    plt.xlabel(info['xlabel'])
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def communication():
    plt.figure()
    # Com x Nodes
    # x = range(5, 105, 5)
    # y_10 = [124, 243, 365, 514, 649, 798, 941, 1089, 1289, 1314, 1407, 1559, 1725, 1886, 2021, 2363, 2471, 2605, 2822,
    #         2981]
    # y_50 = [519, 1051, 1578, 2117, 2658, 3225, 3764, 4307, 4931, 5459, 6015, 6635, 7201, 7747, 8338, 8908, 9410, 10019,
    #         10681, 11176]
    # y_100 = [1021, 2046, 3078, 4120, 5164, 6226, 7268, 8328, 9457, 10508, 11506, 12663, 13738, 14826, 15891, 16948,
    #          17982, 19071, 20210, 21263]
    # Com x Sparsity
    x = np.arange(0, 1.1, 0.1)
    y_10 = [2082, 3078, 3619, 3879, 4025, 4035, 4176, 4228, 4242, 4241, 4264]
    y_50 = [10089, 11370, 12982, 14453, 15505, 16595, 17389, 18035, 18610, 19006, 19302]
    y_100 = [20084, 21430, 23131, 24804, 26367, 27754, 29087, 30413, 31616, 32478, 33121]

    plt.plot(x, y_10, label=f"10 iterations")
    plt.plot(x, y_50, label=f"50 iterations")
    plt.plot(x, y_100, label=f"100 iterations")
    plt.ylabel("Number of communications")
    plt.xlabel("Number of nodes")
    plt.xlabel("Graph density")
    plt.legend(loc='upper right', shadow=True)
    plt.show()


if __name__ == '__main__':
    byz_metrics()
    exit(0)
    # Iterations
    fileA = "./results/mp_iterations_True"
    fileB = "./results/mp_iterations_False"
    info = {
        'xlabel': "Iterations",
        # 'ylabel': "Test accuracy",
        'ylabel': "Test cost",
        'title': "MP with and without confidence w.r.t.the number of iterations."
    }
    # file_mp_iter(fileA, fileB, info)
    # file_mp_itt(info)
    # unbalancednesss
    # fileA = "results/mp_epsilon_50_True"
    # fileB = "results/mp_epsilon_50_False"
    # file = "exp/mp_epsilon_100_nodes"
    info = {
        'xlabel': "Width ε",
        'ylabel': "Cost",
        'title': "MP with and without confidence w.r.t. data unbalancednesss."
    }
    # file_mp_data(file, info)
    # file_mp_data(info)
    # Sparsity
    """
    In mathematics, a dense graph is a graph in which the number of edges is close to the maximal number of edges.
    The opposite, a graph with only a few edges, is a sparse graph. The distinction between sparse and dense graphs is
    rather vague, and depends on the context.
    """
    fileA = "results/mp_sparsity_10_True"
    fileB = "results/mp_sparsity_10_False"
    info = {
        'xlabel': "Graph density",
        'ylabel': "Test Cost",
        'title': "MP with and without confidence w.r.t. graph  sparsity."
    }
    file_mp_dense(info)
    # file_mp_iter(fileA, fileB, info)
    # x = np.arange(0, 1.05, 0.1)
    # ya = [0.5548954658872594, 0.5851961842291964, 0.8127301963307406]
    # yb = []
    # plt.figure()
    # plt.plot(x, np.squeeze(ya), label=f"MP with confidence")
    # # plt.plot(x, np.squeeze(yb), label=f"MP without confidence")
    # plt.ylabel("cost")
    # plt.xlabel("width")
    # plt.legend(loc='upper right', shadow=True)
    # plt.show()
    # communication()
