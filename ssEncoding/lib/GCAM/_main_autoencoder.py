#!/usr/bin/env python
# coding: utf-8

# In[26]:


from preprocess import *

import time
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import keras
from keras import models
from keras import layers
from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.models import load_model
import keras.backend  as K
from keras import callbacks
from scipy.spatial.distance import euclidean,cosine
from shapely import affinity
from shapely.geometry import Polygon
from keras.utils import plot_model
import shapely

import pickle
import tensorflow as tf

import math, collections, scipy.sparse
from matplotlib.collections import PolyCollection

import graph
import autoencoder_data
from sklearn import manifold, datasets, decomposition
#from transformer import spatial_transformer_network as transformer
import retrieval

helper_print_with_time('main begin')

np.random.seed(2019)


# In[27]:


hparams={
    # preprocessing params
    'if_simplify':True
    ,'tor_dist':0.1
    ,'tor_cos':0.99
    ,'if_scale_y':False
    
    ,'scale_type':'final'
    ,'seq_length':64            #
    ,'rotate_type':'equal'      #
    ,'rotate_length':1          #
    ,'norm_type':'minmax'       #

    #model params
    ,'GPU':True                 #
    ,'epochs':400               #
    ,'optimizer':'rmsprop'      #
    ,'z_size':128               #
    ,'rnn_type':'lstm'          #lstm or gru
}

k_list=[2, 4]
color_code3 = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
# In[5]:

print("----------------------------------------------------------")
print("---------------Deep Autoencoder methods-------------------")
print("----------------------------------------------------------")

def _weight_variable(shape, name):
    initial = tf.truncated_normal_initializer(0, 0.1)
    var = tf.get_variable(name, shape, tf.float32, initializer=initial)
    return var

def _bias_variable(shape, name):
    initial = tf.constant_initializer(0.1)
    var = tf.get_variable(name, shape, tf.float32, initializer=initial)
    return var

# def chebyshev5(self, x, L, Fout, K):
def chebyshev5(x, L, W, Fout, K):
    N, M, Fin = x.get_shape()
    N, M, Fin = int(N), int(M), int(Fin)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
    # print()
    L = scipy.sparse.csr_matrix(L)
    L = graph.rescale_L(L, lmax=2)
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    L = tf.SparseTensor(indices, L.data, L.shape)
    L = tf.sparse_reorder(L)
    # Transform to Chebyshev basis
    x0 = tf.transpose(x, perm=[1, 2, 0])                      # M x Fin x N
    x0 = tf.reshape(x0, [M, Fin*N])                           # M x Fin*N
    x = tf.expand_dims(x0, 0)                                 # 1 x M x Fin*N
    def concat(x, x_):
        x_ = tf.expand_dims(x_, 0)                            # 1 x M x Fin*N
        return tf.concat([x, x_], axis=0)                     # K x M x Fin*N
    if K > 1:
        x1 = tf.sparse_tensor_dense_matmul(L, x0)
        x = concat(x, x1)
    for k in range(2, K):
        x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0    # M x Fin*N
        x = concat(x, x2)
        x0, x1 = x1, x2
    x = tf.reshape(x, [K, M, Fin, N])                         # K x M x Fin x N
    x = tf.transpose(x, perm=[3,1,2,0])                       # N x M x Fin x K
    x = tf.reshape(x, [N*M, Fin*K])                           # N*M x Fin*K
    # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
    # W = self._weight_variable([Fin*K, Fout], regularization=False)
    x = tf.matmul(x, W)                                       # N*M x Fout
    return tf.reshape(x, [N, M, Fout])                        # N x M x Fout

def reluing(vertices, b):
    """Bias and ReLU. One bias per filter."""
    # N, M, F = vertices.get_shape()
    # b = self._bias_variable([1, 1, int(F)], regularization=False)
    return tf.nn.relu(vertices + b)      

def sigmoiding(vertices, b):
    """Bias and Sigmoid. One bias per filter."""
    # N, M, F = vertices.get_shape()
    # b = self._bias_variable([1, 1, int(F)], regularization=False)
    return tf.nn.sigmoid(vertices + b) 

def mpooling(x, p):
    """Max pooling of size p. Should be a power of 2."""
    if p > 1:
        x = tf.expand_dims(x, 3)  # N x M x F x 1
        x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='VALID')
        #tf.maximum
        return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x

def apooling(x, p):
    """Max pooling of size p. Should be a power of 2."""
    if p > 1:
        x = tf.expand_dims(x, 3)  # N x M x F x 1
        x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='VALID')
        #tf.average
        return tf.squeeze(x, [3])  # N x M/p x F
    else:
        return x

def upsampling(x, p):
    if p > 1:
        N, M, F = x.get_shape()
        N, M, F = int(N), int(M), int(F)
        x = tf.expand_dims(x, 2)  # N x M x 1 x F
        x = tf.image.resize_nearest_neighbor(x, [M*p,1])
        # tf.upsampling
        return tf.squeeze(x, [2])
    else:
        return x

#bu_shape=gpd.read_file('./data/final_simulate_without.shp',encode='utf-8')



#using the original building data...
bu_filename = 'test_5010'     #'test_5010r'

#using the rotating building data...
#bu_filename = 'test_5010r'



bu_shape=gpd.read_file('./data/' + bu_filename + '.shp', encode='utf-8')
bu_N, bu_M = bu_shape.shape
bu_y = np.array([color_code3[str(i)] for i in bu_shape['tangent'].values])
bu_t = np.array([i for i in bu_shape['isT'].values])
#bu_geos = np.array([geo for geo in bu_shape[['geometry']].values])

bu_tm = np.array([i for i in range(bu_N) if bu_t[i]==1])
bu_T = len(bu_tm)
print(bu_tm)
print(bu_N, bu_M, bu_T)             #5115, 6, 10


bu_use=copy.deepcopy(bu_shape)
bu_mbr,bu_use=get_shape_mbr(bu_use)
#
bu_use=get_shape_normalize_final(bu_use,hparams['if_scale_y'])
#
if hparams['if_simplify']:
    bu_use=get_shape_simplify(bu_use,hparams['tor_dist'],hparams['tor_cos'],simplify_type=0)
#
bu_use = reset_start_point(bu_use)
#
bu_geos = np.array([geo for geo in bu_use[['geometry']].values])
#
#
bu_node=get_node_features(bu_use)
#
bu_line=get_line_features_final(bu_node, hparams['seq_length'])
#
bu_detail=get_inter_features(bu_line)
bu_detail=get_neat_features(bu_detail, hparams['seq_length'], hparams['rotate_length'])
#
bu_features=get_multi_features_final(bu_detail, bu_use,k_list)
bu_features=get_overall_features_final(bu_features, bu_use)
#
bu_features=get_normalize_features_final(bu_features, hparams['norm_type'])
#
#bu_geos = np.array([geo for geo in bu_use[['geometry']].values])

#
cols=[
    'k1_l_bc','k1_s_abc','k1_s_obc','k1_c_obc','k1_r_obc','k1_rotate_bac','k1_rotate_boc'
    ,'k2_l_bc','k2_s_abc','k2_s_obc','k2_c_obc','k2_r_obc','k2_rotate_bac','k2_rotate_boc'
    ,'l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius'
]
cols=[
    c for c in bu_features.columns if 'k' in c
]+['Elongation','Circularity','Rectangularity','Convexity','l_oa','MeanRedius']
# +['l_oa','Area','Perimeter','Elongation','Circularity','MeanRedius']
bu_seq = get_train_sequence(bu_features,cols,hparams['rotate_type'])

print('OBJECTID Count:',bu_seq['OBJECTID'].nunique())
print('sequence Count:',bu_seq.__len__())
print('timestamps Count:',len([x for x in bu_seq.columns if 'f' in x]))
print('features Count:',len(bu_seq.loc[0,'f_0']))

featureList=[x for x in bu_seq.columns if 'f_' in x]
de_input_list,en_input_list,de_target_list=get_seq2seq_train_dataset(bu_seq)

# bu_seq=bu_seq.sample(frac=1).reset_index(drop=True)
dataset_en_input=np.array([[timestamps for timestamps in sample] for sample in bu_seq[en_input_list].values])
dataset_de_input=np.array([[timestamps for timestamps in sample] for sample in bu_seq[de_input_list].values])
dataset_de_output=np.array([[timestamps for timestamps in sample] for sample in bu_seq[de_target_list].values])

print(type(dataset_en_input))
print(dataset_en_input.shape)
print(dataset_en_input[0][0])

print(bu_seq.shape)
print(type(bu_seq))

###################################################################################################
# load_data_res = np.array(autoencoder_data.load_data("poly_c10.json")) #, [0.7, 0.3]
# polys, num_points = load_data_res[0], load_data_res[1]
# bu_x = np.array([polys[i] for i in range(len(polys))])
# bu_y = np.array([num_points[i][1] for i in range(len(num_points))])

bu_x = np.array([[timestamps for timestamps in sample] for sample in bu_seq[en_input_list].values])

#bu_temp = pd.merge(bu_seq[['OBJECTID','LID']],bu_shape[['OBJECTID','tangent']],how='left',on='OBJECTID')
#bu_y = np.array([color_code3[str(i)] for i in bu_temp['tangent'].values])

print(bu_y)
print(type(bu_y))

print(bu_x.shape)
print(bu_y.shape)
print(bu_t.shape)
print(bu_geos.shape)

print(bu_x[0][0], 6)
print(bu_x[0][1], 6)
print(bu_x[0][2], 6)
print(bu_x[0][3], 6)

#
if False:
    plt.figure()
    for i in range(100):
        import random
        from shapely import affinity
        x_rand = 0.#random.uniform(0,10)/500.0
        y_rand = 0.#random.uniform(0,10)/500.0
        
        index = int(random.uniform(0,5010))
        plt.text(int(i/10)+x_rand, int(i%10)+y_rand, '{}_'.format(bu_y[index]), ha='center',va='bottom')
        #plt.scatter(i+x_rand, y_rand, s=50, color = plt.get_cmap("Paired")(tm_y[i]))

        #
        geo = bu_geos[index][0]
        #affinity_geo = retrieval.affinity_geometry(geo, 1, 1, x_rand+i, y_rand)
        affinity_geo = affinity.translate(geo, int(i/10)+geo.centroid.x, int(i%10)+geo.centroid.y)
        #
        coords = affinity_geo.exterior.xy
        plt.fill(coords[0], coords[1], color = plt.get_cmap("Paired")(bu_y[index]))

    plt.gca().autoscale_view()
    plt.gca().set_aspect(1)

    plt.show()



if bu_filename == 'test_1499':
    k_tiers = [145, 149, 151, 150, 151, 144, 153, 152, 153, 151]
else:
    k_tiers = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
nn_average, ft_average, st_average, dcg_average = 0.0, 0.0, 0.0, 0.0
pecisions, recalls = np.zeros(bu_N-bu_T), np.zeros(bu_N-bu_T)

def distance_between_two_features(feature_1, feature_2):
    distan = 0.
    assert len(feature_1) == len(feature_2)
    for i in range(0, len(feature_1)):
        distan += pow((feature_1[i]-feature_2[i]), 2)
    return math.sqrt(distan)


#
#k_convos = [[1], [24, 2], [24, 24, 4], [24, 24, 24, 8], [24, 24, 24, 24, 16], [12, 12, 12, 12, 16]]
#k_convos = [[24, 24, 24, 24, 16], [16, 16, 16, 16, 16],[12, 12, 12, 12, 16],[12, 12, 12, 12, 16]]
k_convos = [[24, 24, 24, 8]]
#k_convos = [[12, 12, 12, 12, 16]]
for k_ in range(len(k_convos)):
    # tf.reset_default_graph()
    nn_average, ft_average, st_average, dcg_average = 0.0, 0.0, 0.0, 0.0
    pecisions, recalls = np.zeros(bu_N-bu_T), np.zeros(bu_N-bu_T)

    k_order = 3                 #
    k_convo = k_convos[k_]
    batch_size = 50             #15
    training_epochs = 500       #3
    print('-------------------------------------')
    print('------------------{}------------------'.format(k_convo))
    print('-------------------------------------')
    adjacencies_1 = np.zeros((64, 64), dtype=np.float32)
    for i in range(0, 64-1):
        adjacencies_1[i, i+1] = adjacencies_1[i+1, i] = 1
    adjacencies_1[0, 63] = adjacencies_1[63, 0] = 1

    adjacencies_2 = np.zeros((32, 32), dtype=np.float32)
    for i in range(0, 32-1):
        adjacencies_2[i, i+1] = adjacencies_2[i+1, i] = 1
    adjacencies_2[0, 31] = adjacencies_2[31, 0] = 1

    adjacencies_3 = np.zeros((16, 16), dtype=np.float32)
    for i in range(0, 16-1):
        adjacencies_3[i, i+1] = adjacencies_3[i+1, i] = 1
    adjacencies_3[0, 15] = adjacencies_3[15, 0] = 1

    adjacencies_4 = np.zeros((8, 8), dtype=np.float32)
    for i in range(0, 8-1):
        adjacencies_4[i, i+1] = adjacencies_4[i+1, i] = 1
    adjacencies_4[0, 7] = adjacencies_4[7, 0] = 1

    adjacencies_5 = np.zeros((4, 4), dtype=np.float32)
    for i in range(0, 4-1):
        adjacencies_5[i, i+1] = adjacencies_5[i+1, i] = 1
    adjacencies_5[0, 3] = adjacencies_5[3, 0] = 1

    adjacencies_6 = np.zeros((2, 2), dtype=np.float32)
    for i in range(0, 2-1):
        adjacencies_6[i, i+1] = adjacencies_6[i+1, i] = 1
    adjacencies_6[0, 1] = adjacencies_6[1, 0] = 1

    L_graph = []
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_1), normalized=True, rescaled=True))
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_2), normalized=True, rescaled=True))
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_3), normalized=True, rescaled=True))
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_4), normalized=True, rescaled=True))
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_5), normalized=True, rescaled=True))
    L_graph.append(graph.laplacian(scipy.sparse.csr_matrix(adjacencies_6), normalized=True, rescaled=True))

    [n_train, n_vertices, n_features] = bu_x.shape
    num_steps = int(training_epochs * n_train / batch_size)

    print('n_train    = {}'.format(n_train))
    print('n_vertices = {}'.format(n_vertices))
    print('n_features = {}'.format(n_features))
    print('num_steps  = {}'.format(num_steps))

    ph_vertices = tf.placeholder(tf.float32, [batch_size, n_vertices, n_features], 'vertices_')

    weights, biases = [], []
    for i in range(len(k_convo)):
        if i == 0:
            weights.append(_weight_variable([n_features*k_order, k_convo[i]], 'e_k{}'.format(i)))
        else:
            weights.append(_weight_variable([k_convo[i-1]*k_order, k_convo[i]], 'e_k{}'.format(i)))
        biases.append(_bias_variable([1, 1, k_convo[i]], 'e_b{}'.format(i)))
    for i in range(len(k_convo)):
        if i == 0:
            weights.append(_weight_variable([k_convo[len(k_convo)-1-i]*k_order, k_convo[len(k_convo)-1-i]], 'd_k{}'.format(i)))
        else:
            weights.append(_weight_variable([k_convo[len(k_convo)-i]*k_order, k_convo[len(k_convo)-1-i]], 'd_k{}'.format(i)))
        biases.append(_bias_variable([1, 1, k_convo[len(k_convo)-1-i]], 'd_b{}'.format(i)))
    weights.append(_weight_variable([k_convo[0]*k_order, n_features], 'w_out'))
    biases.append(_bias_variable([1, 1, n_features], 'b_out'))
    
    fc_loc1 = tf.tile(tf.expand_dims(_weight_variable((2, 3), 'trans_1'), 0), [batch_size, 1, 1])
    fc_loc2 = tf.tile(tf.expand_dims(_weight_variable((2, 3), 'trans_2'), 0), [batch_size, 1, 1])
    '''
    initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()

    W_fc2  = tf.Variable(tf.zeros([64*1, 6]), name='sp_weight_fc2')
    b_fc2  = tf.Variable(initial_value=initial, name='sp_biases_fc2')
    fc_loc2 = tf.add(tf.matmul(tf.zeros([batch_size, 64*1]), W_fc2), b_fc2)
    '''
    #ph_vertices1 = tf.reshape(ph_vertices, [batch_size, n_vertices, 1, n_features])
    #ph_vertices2 = transformer(ph_vertices1, fc_loc2)


    ph_vertices3 = tf.reshape(ph_vertices, [batch_size, n_vertices, n_features])
    for i in range(len(k_convo)):
        ph_vertices3 = mpooling(sigmoiding(chebyshev5(ph_vertices3, L_graph[i], weights[i], k_convo[i], k_order), biases[i]), 2)
    encoding_ = ph_vertices3
    for i in range(len(k_convo)):
        ph_vertices3 = upsampling(sigmoiding(chebyshev5(ph_vertices3, L_graph[len(k_convo)-i], weights[len(k_convo)+i], k_convo[len(k_convo)-1-i], k_order), biases[len(k_convo)+i]), 2)
    logits_ = sigmoiding(chebyshev5(ph_vertices3, L_graph[0], weights[-1], n_features, k_order), biases[-1])

    #logits_1 = tf.reshape(logits_, [batch_size, n_vertices, 1, n_features])
    #logits_2 = transformer(logits_1, fc_loc2)
    #logits_3 = tf.reshape(logits_2, [batch_size, n_vertices, n_features])

    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_vertices, logits=logits_)
    loss = tf.nn.l2_loss(ph_vertices - logits_)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)


    # # training

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    indices = collections.deque()
    for step in range(1, num_steps+1):
        # Be sure to have used all the samples before using one a second time.
        if len(indices) < batch_size:
            indices.extend(np.random.permutation(n_train))
        idx = [indices.popleft() for i in range(batch_size)]
        batch_vertices = [bu_x[dx] for dx in idx]
        batch_vertices = np.array(batch_vertices).reshape((-1, n_vertices, n_features))

        #print(batch_vertices.shape)
        # batch_cost, logits_, _ = sess.run([cost, logits_, optimizer], feed_dict={ph_vertices: batch_vertices, ph_L1: L1, ph_L2: L2, ph_L3: L3, ph_L4: L4})
        batch_cost, batch_logits, batch_encoding, _ = sess.run([cost, logits_, encoding_, optimizer], feed_dict={ph_vertices: batch_vertices})
        # print(logits_)
        # exit()
        if step % 2000 == 0:
            # print(batch_encoding)
            print("step:", '%04d' % step, "cost=", "{:.9f}".format(batch_cost))

    bu_encoding = []
    assert (hparams['rotate_length']+0)*bu_N == n_train
    for i in range(bu_N):
        #batch_vertices = bu_x[(hparams['rotate_length']+0)*i]
        #print(np.array(batch_vertices).shape)

        batch_vertices = bu_x[(hparams['rotate_length']+0)*i].tolist() * batch_size
        #print(np.array(batch_vertices).shape)
        batch_vertices = np.array(batch_vertices).reshape((batch_size, hparams['seq_length'], n_features))
        batch_encoding = sess.run([encoding_], feed_dict={ph_vertices: batch_vertices})
        #print(np.array(batch_encoding).shape)
        #print(np.array(batch_encoding[0][0]).shape)
        bu_encoding.append(batch_encoding[0][0])
    sess.close()


    bu_encoding = np.array(bu_encoding)
    [m_, p_, q_] = bu_encoding.shape
    print(bu_encoding.shape)
    bu_encoding = bu_encoding.reshape(m_, p_*q_)
    #bu_encoding = bu_encoding.reshape(m_*n_, p_*q_)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(bu_encoding)
    retrieval.plot_embedding____(X_tsne[0:5000, :], bu_y[0:5000], bu_geos[0:5000], X_tsne[5000:, :], bu_y[5000:], bu_geos[5000:])
    
    #NN, it is time-consuming!!!
    nn_correct = 0
    if False:
        for i in range(bu_N):
            shape_similarity = []
            for j in range(bu_N):
                if j == i: continue
                distance = distance_between_two_features(bu_encoding[i], bu_encoding[j])
                shape_similarity.append((round(distance, 4), j))
            shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])
            if bu_y[shape_similarity[0][1]] == bu_y[i]:
                nn_correct = nn_correct + 1
    nn_average = nn_correct*1.0/bu_N

    for i in range(bu_T):
        #if i != 1:  continue
        i_tmplate, k_tier = bu_tm[i], k_tiers[i]
        shape_similarity = []
        for j in range(bu_N):
            if j in bu_tm: continue
            distance = distance_between_two_features(bu_encoding[i_tmplate], bu_encoding[j])
            shape_similarity.append((round(distance, 4), j))
        shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])
        #print(shape_similarity)

        #NN
        #nn_correct = 0
        #if bu_y[shape_similarity[0][1]] == bu_y[i_tmplate]:
        #    nn_correct = nn_correct + 1
        #nn_average = nn_average + nn_correct #round(nn_correct*1.0/bu_N, 3)

        #FT
        ft_correct = 0
        for j in range(k_tier):
            if bu_y[shape_similarity[j][1]] == bu_y[i_tmplate]:
                ft_correct = ft_correct + 1
        ft_average = ft_average + round(ft_correct*1.0/k_tier, 3)

        #ST
        st_correct = 0
        for j in range(2*k_tier):
            if bu_y[shape_similarity[j][1]] == bu_y[i_tmplate]:
                st_correct = st_correct + 1
        st_average = st_average + round(st_correct*1.0/k_tier, 3)

        #DCG
        ann_ref, best_ref = [], []
        for j in range(bu_N-bu_T):
            if bu_y[shape_similarity[j][1]] == bu_y[i_tmplate]:
                ann_ref.append(1)
            else:
                ann_ref.append(0)
        for j in range(bu_N-bu_T):
            if j < k_tier:
                best_ref.append(1)
            else:
                best_ref.append(0)
        print('sum1={}, sum2={}'.format(sum(ann_ref), sum(best_ref)))
        #print(ann_ref)
        #print(best_ref)
        DCG, IDCG = 0.0, 0.0
        for j in range(bu_N-bu_T):
            DCG  = DCG  + (2**ann_ref[j]-1) / math.log(j+2, 2)             #Note:i from 0
            IDCG = IDCG + (2**best_ref[j]-1) / math.log(j+2, 2)            #Note:i from 0
        dcg_average = dcg_average + round(DCG*1.0/IDCG, 3)
        print(round(DCG*1.0/IDCG, 3))

        #pecision_recall
        s_true, pecision, recall = 0, [], []
        for j in range(bu_N-bu_T):
            if bu_y[shape_similarity[j][1]] == bu_y[i_tmplate]:
                s_true = s_true + 1
            pecision.append(round(s_true*1.0/(j+1), 3))
            recall.append(round(s_true*1.0/k_tier, 3))
        for j in range(bu_N-bu_T):
            pecisions[j] = pecisions[j] + pecision[j]
            recalls[j] = recalls[j] + recall[j]

        #
        if False:
            fig = plt.figure(figsize=(15, 15))
            for j in range(225):
                if j == 0:
                    x, y = bu_geos[i_tmplate][0].exterior.xy
                else:
                    x, y = bu_geos[shape_similarity[j-1][1]][0].exterior.xy
                plt.subplot(15, 15, 1+j)
                plt.plot(x, y)
                if j == 0:
                    plt.text(0., -1.,'{}'.format(bu_y[i_tmplate]), ha='center',va='bottom')
                else:
                    plt.text(0., -1.,'{}_{}'.format(bu_y[shape_similarity[j-1][1]], shape_similarity[j-1][0]), ha='center',va='bottom')

                plt.xlim(-1,1)
                plt.ylim(-1,1)
                plt.xticks([])
                plt.yticks([])
            #plt.show()
        #break

    print("NN  = {}".format(round(nn_average, 3)))
    print("FT  = {}".format(round(ft_average / bu_T, 3)))
    print("ST  = {}".format(round(st_average / bu_T, 3)))
    print("DCG = {}".format(round(dcg_average / bu_T, 3)))

    for j in range(bu_N-bu_T):
        pecisions[j] = pecisions[j] / bu_T
        recalls[j] = recalls[j] / bu_T
    
    retrieval.plot_pecision_recall(pecisions, recalls)
    #for i in range(bu_N-bu_T):
        #if i == 0 or i == bu_N-bu_T-1 or i % int((bu_N-bu_T)/20) == 0:
        #print(round(recalls[i], 3), round(pecisions[i], 3))
    plotslist = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
    j = 0
    for i in range(bu_N-bu_T):
        if i == 0 or i == bu_N-bu_T-1 :
            print(round(recalls[i], 3), round(pecisions[i], 3))
        else:
            if recalls[i] == plotslist[j]:
                print(round(recalls[i], 3), round(pecisions[i], 3))
                j = j + 1
            elif recalls[i] > plotslist[j]:
                print(round((recalls[i]+recalls[i-1])/2.0, 3), round((pecisions[i]+pecisions[i-1])/2., 3))
                j = j + 1

plt.show()
