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


def figure(f, c):
    protocol = c['protocol']
    if protocol == "MP":
        if c["args"]["protocol"]["confidence"]:
            protocol += "with confidence"
        else:
            protocol += "without confidence"

    xa, ya = load(f)
    plt.figure()
    plt.plot(xa, np.squeeze(ya), label=protocol)
    plt.legend(loc='lower right', shadow=True)
    plt.show()


def communication_rounds(N, C, F, ylabel="Test accuracy", xlabel="Communication rounds"):
    xN, yN = load(N)
    xC, yC = load(C)
    xF, yF = load(F)

    # yN = np.poly1d(np.polyfit(xN, yN, 5))(xN)
    # yC = np.poly1d(np.polyfit(xC, yC, 5))(xC)
    # yF = np.poly1d(np.polyfit(xF, yF, 5))(xF)

    plt.figure()
    plt.plot(xN, np.squeeze(yN), '--', label=f"MP without confidence")
    plt.plot(xC, np.squeeze(yC), '-.', label=f"MP with confidence")
    plt.plot(xF, np.squeeze(yF), label=f"CDPL")
    plt.rc('legend', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.legend(loc='best', shadow=True)
    plt.show()


def byzantine_metrics(file):
    x, y = load(file)
    accuracy, precision, recall, f_score = y
    plt.figure()
    plt.plot(x, np.squeeze(precision), '-.', label=f"Precision")
    plt.plot(x, np.squeeze(recall), ':', label=f"Recall")
    plt.plot(x, np.squeeze(f_score), '-', label=f"F1 score")
    plt.ylabel("")  # "STD accuracy"
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def plot(nodes, analysis):
    N = f"results/{analysis}_{nodes}_N"  # MP without confidence
    C = f"results/{analysis}_{nodes}_C"  # MP with confidence
    F = f"results/{analysis}_{nodes}_F"  # CDPL with confidence
    communication_rounds(N, C, F)


def byz_metrics():
    # Data unbalancedness
    x = np.arange(0, 1.04, 0.04)
    a = [20.97522270704118, 21.217673384474576, 20.876454631019296, 21.07826777996672, 21.349780447038157,
         21.393942478768825, 21.532803030065228, 22.657205686154647, 22.29665183020926, 23.027300121823586,
         22.815068476884967, 25.73017628615337, 23.944403332203606, 24.04844875188948, 23.31627348386516,
         25.974550614373058, 23.780625897295366, 24.44231845263808, 25.896590814062524, 24.464179877845577,
         24.831684693585732, 25.468797224581778, 25.438998259407548, 26.469847090726958, 24.18340451113953,
         26.848631982188593]
    b = [20.834537506078362, 21.433614423111706, 22.0139343141721, 22.74375941314483, 22.905046101844015,
         23.129726106403552, 23.35440611096309, 23.918826666255757, 25.65756239676712, 26.420161782205664,
         27.000853838280477, 24.592526709428405, 26.60440273322162, 30.25508491601425, 29.988037864725424,
         26.338992220441718, 28.337370576934077, 29.428795411552308, 29.621630315260447, 30.64375491902934,
         27.363785140923586, 32.34280675793754, 27.237769001749943, 25.916013654476387, 27.107113341901684,
         30.72455463578075]
    c = [21.40099975594494, 22.381069892432663, 22.0139343141721, 21.537095279233423, 22.649897579106515,
         22.535125479758758, 22.144784384827553, 23.30744757975546, 24.50233010582157, 24.183163729342905,
         23.967503769894346, 24.251919436403554, 24.71006291947645, 24.510613321449696, 24.499526616436576,
         25.366167460744677, 24.8212616425998, 24.599566315818226, 23.148352572042786, 25.52913222563778,
         25.586813645708357, 25.824018236341036, 27.8224110516684, 26.550603292850663, 27.892855528971932,
         25.531583204434444]

    # Graph sparsity
    a1 = [15.686966777270044, 16.010875259847474, 19.316491140967777, 21.57121087522782, 22.47766401440054,
          22.842883932879676, 27.634236873516045, 28.35572477967493, 30.766254622484155, 31.78035720014011,
          31.52420093188252, 34.76508207143912, 35.49499611906782, 35.36667615833082, 38.331093734563865,
          40.10024479773742, 40.35297210086058, 39.86334952207632, 42.41329244243512, 43.13970940637516,
          42.10771534859562, 44.11206701013951, 42.27749177807368, 42.538003148177026, 42.94904806697376,
          42.4739993299398, ]
    b2 = [15.344067528659018, 14.703704230041932, 18.101967916102446, 23.790547395495615, 26.218572077454148,
          28.431676973387724, 32.455284586574635, 39.691267270606566, 38.15690411263445, 43.581695474951275,
          43.71638207876443, 48.148541429589145, 44.916286537338856, 48.4026884300349, 53.69972405076261,
          52.71665533661079, 59.408636208264234, 60.03378808584248, 64.64492768338604, 63.260399932357075,
          65.33017469288968, 65.77508494973269, 64.47297896379644, 63.371741518683294, 66.13253708221916,
          62.58157533473175, ]
    c3 = [15.707690213668181, 15.098622551213772, 18.579555214862864, 20.923859528130784, 22.50205320014141,
          24.40904152186545, 27.417010791385295, 30.218323560555007, 31.990325279434156, 35.02001299334454,
          33.555830402865034, 36.67699782981615, 37.96788150302173, 38.06627913014085, 41.08966669323188,
          42.55031007470284, 44.68463634463381, 45.15483678247006, 44.823045978069224, 46.43927294489728,
          45.638875187028745, 47.2930852687084, 46.58889430287625, 45.95703884879189, 44.427697092408366,
          45.5605535372266, ]

    f = [14.879170795699618, 15.144682579117047, 14.990330106270667, 15.182636406344445, 14.944507067911577,
         15.397161710368124, 16.00506915433506, 14.234978263497363, 15.963074843091158, 16.345626592844443,
         15.63737320615790]
    c = [14.290162621326573, 16.11264965334899, 16.642167400277998, 16.829890009970676, 17.122654607689057,
         20.51235593842737, 16.561278432687946, 21.58827265876344, 17.02554882177909, 22.090620759919176,
         19.31665368788403]
    n = [14.570599684609773, 14.86373844206828, 15.101173542879739, 14.153226244863541, 14.313678725978708,
         16.73495917528062, 15.219717354289013, 15.439753880550942, 15.101713512852335, 14.503741257420867,
         15.156022787187679]
    x = np.arange(0, 1.01, 0.1)
    save("data_unbalancedness_50_NCF", (x, n, c, f))

    x, a, b, c = load("data_unbalancedness_50_NCF")

    a = np.poly1d(np.polyfit(x, a, 5))(x)
    b = np.poly1d(np.polyfit(x, b, 5))(x)
    c = np.poly1d(np.polyfit(x, c, 5))(x)

    plt.figure()
    plt.plot(x, np.squeeze(a), '-.', label=f"MP without confidence")
    plt.plot(x, np.squeeze(b), '--', label=f"MP with confidence")
    plt.plot(x, np.squeeze(c), '-', label=f"MP with contribution factor")
    plt.ylabel("Total loss")
    plt.xlabel("Graph density")
    # plt.xlabel("Width $\epsilon$")
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


def contribution_factor(F, ylabel="Test accuracy"):
    x, y, z = load(F)
    plt.figure()
    plt.plot(x, np.squeeze(y), '-.', label=f"Banned peers")
    plt.plot(x, np.squeeze(z), '-', label=f"Ignored exchanges")
    plt.ylabel("")
    plt.xlabel("Communication rounds")
    plt.legend(loc='best', shadow=True)
    plt.show()


def banned_nodes(A, B, C, D):
    xA, yA = load(A)
    xB, yB = load(B)
    xC, yC = load(C)
    xD, yD = load(D)
    plt.figure()
    plt.plot(xA, np.squeeze(yA), label="Ban threshold $\epsilon=-0.0$")
    # plt.plot(xB, np.squeeze(yB), label="Ban threshold $\epsilon=-0.1$")
    plt.plot(xC, np.squeeze(yC), label="Ban threshold $\epsilon=-0.2$")
    # plt.plot(xD, np.squeeze(yD), label="Ban threshold $\epsilon=-0.3$")
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
    plt.plot(xA, np.squeeze(yA), '--', label="$CF_{th}=0.3$")
    plt.plot(xB, np.squeeze(yB), '-.', label="$CF_{th}=0.6$")
    plt.plot(xC, np.squeeze(yC), '-', label="$CF_{th}=0.9$")
    plt.ylabel("Accuracy")
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
