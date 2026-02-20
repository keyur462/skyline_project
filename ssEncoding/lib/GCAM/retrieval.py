import math, copy
import numpy as np
import matplotlib.pyplot as plt
from shapely import affinity
from sklearn.neighbors import NearestNeighbors

#
def affinity_geometry(geo, mx, my, x_rand, y_rand, scale_ratio=0.005):

    #
    df_visual = copy.deepcopy(geo)

    #
    geo_translate = affinity.translate(df_visual, mx+x_rand-df_visual.centroid.x, my+y_rand-df_visual.centroid.y)

    #
    scale_x = max(df_visual.exterior.xy[0]) - min(df_visual.exterior.xy[0])
    scale_y = max(df_visual.exterior.xy[1]) - min(df_visual.exterior.xy[1])
    geo_scale = affinity.scale(geo_translate, scale_ratio/scale_x, scale_ratio/scale_x, origin='centroid')

    return geo_scale

def compute_NN(X, Y):
    neigh = NearestNeighbors(n_neighbors = 1)
    neigh.fit(X)

    correct = 0
    total = X.shape[0]
    for i in range(total):
        nn_list = neigh.kneighbors(X[i].reshape(1,-1))[1][0][0]    #
        if Y[i] == Y[nn_list]:
            correct = correct + 1
    return round(correct*1.0/total, 3)

def compute_FT(X, Y, s, sY, k):
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)

    # print(neigh.kneighbors(s.reshape(1,-1)))
    knn_list = neigh.kneighbors(s.reshape(1,-1))[1][0]             #
    assert len(knn_list) == k

    # print('sY={}'.format(sY))
    correct = 0
    for i in knn_list:
        # print(Y[i])
        if Y[i] == sY:
            correct = correct+1
    return round(correct*1.0/k, 3), knn_list

def compute_ST(X, Y, s, sY, k):
    neigh = NearestNeighbors(n_neighbors = 2*k)
    neigh.fit(X)

    knn_list = neigh.kneighbors(s.reshape(1,-1))[1][0]             #
    assert len(knn_list) == 2*k

    correct = 0
    for i in knn_list:
        if Y[i] == sY:
            correct = correct+1
    return round(correct*1.0/k, 3)

def compute_DCG(X, Y, s, sY, k):
    l = X.shape[0]
    neigh = NearestNeighbors(n_neighbors = l)
    neigh.fit(X)

    ann_list = neigh.kneighbors(s.reshape(1, -1))[1][0]
    ann_ref, best_ref = [], []

    assert len(ann_list) == l

    for i in ann_list:
        if Y[i] == sY:
            ann_ref.append(1)
        else:
            ann_ref.append(0)
    for i in range(l):
        if i < k:
            best_ref.append(1)
        else:
            best_ref.append(0)
    print('sum1={}, sum2={}'.format(sum(ann_ref), sum(best_ref)))
    DCG, IDCG = 0, 0
    for i in range(l):
        DCG = DCG + (2**ann_ref[i]-1) / math.log(i+2, 2)             #Note:
        IDCG = IDCG + (2**best_ref[i]-1) / math.log(i+2, 2)          #Note:
    return round(DCG*1.0 / IDCG, 3)

def pecision_recall(X, Y, s, sY, k):
    l = X.shape[0]
    neigh = NearestNeighbors(n_neighbors = l)
    neigh.fit(X)

    ann_list = neigh.kneighbors(s.reshape(1, -1))[1][0]
    assert len(ann_list) == l

    #print('k = {}'.format(k))
    s_true = 0
    pecision, recall = [], []
    s_trues = []
    for i in range(l):
        if Y[ann_list[i]] == sY:
            s_true = s_true + 1
        #print('s_true = {}'.format(s_true))
        s_trues.append(s_true)
        pecision.append(round(s_true*1.0/(i+1), 3))
        recall.append(round(s_true*1.0/k, 3))
    #print('s_trues = {}'.format(s_trues))
    #print('pecision = {}, recall={}'.format(pecision, recall))
    return pecision, recall

def plot_pecision_recall(pecision, recall):
    plt.figure()
    plt.plot(recall, pecision, 'co-', linewidth=2.0, markersize=10.0)

def plot_embedding__(X, Y, df_geos, title=None):
    import random

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    
    for i in range(X.shape[0]):
        x_rand = random.uniform(0,10)/500.0
        y_rand = random.uniform(0,10)/500.0
        if i < 30:
            plt.text(X[i, 0]+x_rand, X[i, 1]+y_rand,'{}_{}'.format(i, Y[i]), ha='center',va='bottom')
        #plt.scatter(X[i, 0]+x_rand, X[i, 1]+y_rand, s=50, color = plt.get_cmap("Paired")(Y[i]))

        #
        geo = df_geos[i][0]
        affinity_geo = affinity_geometry(geo, X[i, 0], X[i, 1], x_rand, y_rand)

        #
        coords = affinity_geo.exterior.xy
        plt.fill(coords[0], coords[1], color = plt.get_cmap("Paired")(Y[i]))
    
    plt.gca().autoscale_view()
    plt.gca().set_aspect(1)
    if title is not None:
        plt.title(title)

def plot_embedding____(X, Y, df_geos, M, N, ej_geos, title=None):
    import random
    x_min, x_max = np.min(np.concatenate((X, M), axis=0), 0), np.max(np.concatenate((X, M), axis=0), 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    
    for i in range(X.shape[0]):
        x_rand = random.uniform(0,10)/500.0
        y_rand = random.uniform(0,10)/500.0
        if i < 30 and False:
            plt.text(X[i, 0]+x_rand, X[i, 1]+y_rand,'{}_{}'.format(i, Y[i]), ha='center',va='bottom')
        #plt.scatter(X[i, 0]+x_rand, X[i, 1]+y_rand, s=50, color = plt.get_cmap("Paired")(Y[i]))

        #
        geo = df_geos[i][0]
        affinity_geo = affinity_geometry(geo, X[i, 0], X[i, 1], x_rand, y_rand)

        #
        coords = affinity_geo.exterior.xy
        plt.fill(coords[0], coords[1], color = plt.get_cmap("Paired")(Y[i]))

    #m_min, m_max = np.min(M, 0), np.max(M, 0)
    M = (M - x_min) / (x_max - x_min)

    for i in range(M.shape[0]):
        x_rand = random.uniform(0,10)/500.0
        y_rand = random.uniform(0,10)/500.0
        #if i < 30:
        #    plt.text(M[i, 0]+x_rand, M[i, 1]+y_rand,'{}_{}'.format(i, N[i]), ha='center',va='bottom')
        #plt.scatter(M[i, 0]+x_rand, M[i, 1]+y_rand, s=50, color = plt.get_cmap("Paired")(N[i]))

        #
        geo = ej_geos[i][0]
        affinity_geo = affinity_geometry(geo, M[i, 0], M[i, 1], x_rand, y_rand, scale_ratio=0.025)

        #
        coords = affinity_geo.exterior.xy
        plt.fill(coords[0], coords[1], color = plt.get_cmap("Paired")(N[i]))
    
    plt.gca().autoscale_view()
    plt.gca().set_aspect(1)
    if title is not None:
        plt.title(title)

def plot_embedding(X, Y, df_geos, Query, Result, title=None):
    import random

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()

    print(X.shape)
    print(Y.shape)
    
    for i in range(X.shape[0]):
        x_rand = random.uniform(0,10)/500.0
        y_rand = random.uniform(0,10)/500.0
        plt.scatter(X[i, 0]+x_rand, X[i, 1]+y_rand, s=5, color = plt.get_cmap("Paired")(Y[i]))

        #
        geo = df_geos[i][0]
        affinity_geo = affinity_geometry(geo, X[i, 0], X[i, 1], x_rand, y_rand)

        #
        coords = affinity_geo.exterior.xy
        plt.fill(coords[0], coords[1], color = plt.get_cmap("Paired")(Y[i]))

    plt.scatter(X[Query, 0], X[Query, 1], s=200, color = plt.get_cmap("Paired")(Y[Query]))
    for i in Result:
        plt.scatter(X[i, 0], X[i, 1], s=60, color = plt.get_cmap("Paired")(Y[i]))
    
    plt.gca().autoscale_view()
    plt.gca().set_aspect(1)
    if title is not None:
        plt.title(title)