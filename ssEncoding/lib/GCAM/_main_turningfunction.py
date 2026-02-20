import copy
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import turningfunction
import retrieval, preprocess

color_code1 = {'O':1, 'I':2, 'L':3, 'J':4, 'T':5, 'U':6, 'E':7, 'F':8, 'Z':9, 'Y':0, 'X':11}
color_code3 = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

print("----------------------------------------------------------")
print("---------------Turning function methods-------------------")
print("----------------------------------------------------------")

#
tm_filename = 'template_er'
tm_shape = gpd.read_file('./data/'+tm_filename+'.shp', encode='utf-8')
tm_N, tm_M = tm_shape.shape

#
tm_use = copy.deepcopy(tm_shape)
tm_mbr, tm_use = preprocess.get_shape_mbr(tm_use)
tm_use = preprocess.get_shape_normalize_final(tm_use, False)
#tm_use = preprocess.reset_start_point(tm_use)
tm_geos = np.array([geo for geo in tm_use[['geometry']].values])
#
tm_y = np.array([color_code3[str(i)] for i in tm_shape['tangent'].values])

tm_turning_functions_collection = []
for i in range(tm_N):
    #
    point_num = len(tm_geos[i][0].exterior.xy[0])
    #
    building_coords = np.array([tm_geos[i][0].exterior.xy[0][0:point_num-1], tm_geos[i][0].exterior.xy[1][0:point_num-1]]).T

    turning_functions = []
    for i in range(point_num-1):
        #
        reset_building_coords = np.concatenate((building_coords[i:point_num-1,:], building_coords[0:i,:]),axis=0)
        #print(reset_building_coords.shape)
        reset_building_coords = reset_building_coords.tolist()

        #
        [[CX,CY], area, peri] = turningfunction.get_basic_parametries_of_Poly(reset_building_coords)
        #
        turning_func = turningfunction.turning_function(reset_building_coords, peri)
        #
        if True:
            turning_func = np.array(turning_func)
            func_min, func_max = min(turning_func[:,2]), max(turning_func[:,2])
            for j in range(len(turning_func)):
                turning_func[j][2] = (turning_func[j][2] - func_min)/(func_max - func_min)
        #
        turning_functions.append(turning_func)
        
        #
        if False:
            for j in range(len(turning_func)):
                (turning_func[j][0], turning_func[j][1], turning_func[j][2])
            turningfunction.draw_turning_function(turning_func, reset_building_coords)

    tm_turning_functions_collection.append(turning_functions)


#using the original building data...
bu_filename = 'test_5000'     #'test_5000r'

#using the rotating building data...
#bu_filename = 'test_5000r'


bu_shape = gpd.read_file('./data/'+bu_filename+'.shp', encode='utf-8')
bu_N, bu_M = bu_shape.shape

#
bu_use = copy.deepcopy(bu_shape)
bu_mbr, bu_use = preprocess.get_shape_mbr(bu_use)
bu_use = preprocess.get_shape_normalize_final(bu_use, False)
#bu_use = preprocess.reset_start_point(bu_use)
bu_geos = np.array([geo for geo in bu_use[['geometry']].values])
#
if bu_filename == 'test_1499':
    bu_y = np.array([color_code3[str(i)] for i in bu_shape['tangent'].values])
else:
    bu_y = np.array([color_code3[str(i)] for i in bu_shape['tangent'].values])

bu_turning_functions_collection = []
for i in range(bu_N):
    #
    point_num = len(bu_geos[i][0].exterior.xy[0])
    #
    building_coords = np.array([bu_geos[i][0].exterior.xy[0][0:point_num-1], bu_geos[i][0].exterior.xy[1][0:point_num-1]]).T

    turning_functions = []
    for i in range(point_num-1):
        #
        reset_building_coords = np.concatenate((building_coords[i:point_num-1,:], building_coords[0:i,:]),axis=0)
        #print(reset_building_coords.shape)
        reset_building_coords = reset_building_coords.tolist()

        #
        [[CX,CY], area, peri] = turningfunction.get_basic_parametries_of_Poly(reset_building_coords)
        #
        turning_func = turningfunction.turning_function(reset_building_coords, peri)
        #
        if True:
            turning_func = np.array(turning_func)
            func_min, func_max = min(turning_func[:,2]), max(turning_func[:,2])
            for j in range(len(turning_func)):
                turning_func[j][2] = (turning_func[j][2] - func_min)/(func_max - func_min)
        #
        turning_functions.append(turning_func)

        #
        if False:
            for j in range(len(turning_func)):
                (turning_func[j][0], turning_func[j][1], turning_func[j][2])
            turningfunction.draw_turning_function(turning_func, reset_building_coords)

    bu_turning_functions_collection.append(turning_functions)



if bu_filename == 'test_1499':
    k_tiers = [145, 149, 151, 150, 151, 144, 153, 152, 153, 151]
else:
    k_tiers = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
nn_average, ft_average, st_average, dcg_average = 0.0, 0.0, 0.0, 0.0
pecisions, recalls = np.zeros(bu_N), np.zeros(bu_N)


#NN, it is time-consuming!
import time
nn_correct = 0
if False:
    for i in range(bu_N):
        shape_similarity = []
        turning_functions_1 = bu_turning_functions_collection[i]
        if i % 5 == 0:
            print(time.strftime("%H:%M:%S",time.localtime()), 'i = {}'.format(i))
        for j in range(bu_N):
            if j == i: continue
            min_distance = 999999.9
            turning_functions_2 = bu_turning_functions_collection[j]
            for k in range(len(turning_functions_1)):
                distance = turningfunction.distance_between_two_functions(turning_functions_1[k], turning_functions_2[0])
                if distance < min_distance:
                    min_distance = distance
            shape_similarity.append((round(min_distance, 4), j))
        shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])
        if bu_y[shape_similarity[0][1]] == bu_y[i]:
            nn_correct = nn_correct + 1
        if i > 100:
            break
nn_average = nn_correct*1.0/bu_N

for i in range(tm_N):
    #if i != 1: continue
    k_tier = k_tiers[i]
    #print(tm_y[i])
    shape_similarity = []
    turning_functions_1 = tm_turning_functions_collection[i]
    for j in range(bu_N):
        min_distance = 999999.9
        turning_functions_2 = bu_turning_functions_collection[j]
        for k in range(len(turning_functions_1)):
            distance = turningfunction.distance_between_two_functions(turning_functions_1[k], turning_functions_2[0])
            if distance < min_distance:
                min_distance = distance
        shape_similarity.append((round(min_distance, 4), j))
    shape_similarity = sorted(shape_similarity, key = lambda distance: distance[0])

    #NN
    #nn_correct = 0
    #if tm_y[i] == bu_y[shape_similarity[0][1]]:
    #    nn_correct = nn_correct + 1
    #nn_average = nn_average + round(nn_correct*1.0/bu_N, 3)

    #FT
    ft_correct = 0
    for j in range(k_tier):
        if tm_y[i] == bu_y[shape_similarity[j][1]]:
            ft_correct = ft_correct + 1
    ft_average = ft_average + round(ft_correct*1.0/k_tier, 3)

    #ST
    st_correct = 0
    for j in range(2*k_tier):
        if tm_y[i] == bu_y[shape_similarity[j][1]]:
            st_correct = st_correct + 1
    st_average = st_average + round(st_correct*1.0/k_tier, 3)

    #DCG
    ann_ref, best_ref = [], []
    for j in range(bu_N):
        if tm_y[i] == bu_y[shape_similarity[j][1]]:
            ann_ref.append(1)
        else:
            ann_ref.append(0)
    for j in range(bu_N):
        if j < k_tier:
            best_ref.append(1)
        else:
            best_ref.append(0)
    print('sum1={}, sum2={}'.format(sum(ann_ref), sum(best_ref)))
    #print(ann_ref)
    #print(best_ref)
    DCG, IDCG = 0.0, 0.0
    for j in range(bu_N):
        DCG  = DCG  + (2**ann_ref[j]-1) / math.log(j+2, 2)             #Note:i from 0
        IDCG = IDCG + (2**best_ref[j]-1) / math.log(j+2, 2)            #Note:i from 0
    dcg_average = dcg_average + round(DCG*1.0/IDCG, 3)
    print(round(DCG*1.0/IDCG, 3))

    #pecision_recall
    s_true, pecision, recall = 0, [], []
    for j in range(bu_N):
        if tm_y[i] == bu_y[shape_similarity[j][1]]:
            s_true = s_true + 1
        pecision.append(round(s_true*1.0/(j+1), 3))
        recall.append(round(s_true*1.0/k_tier, 3))
    for j in range(bu_N):
        pecisions[j] = pecisions[j] + pecision[j]
        recalls[j] = recalls[j] + recall[j]

print("NN  = {}".format(round(nn_average, 3)))
print("FT  = {}".format(round(ft_average / tm_N, 3)))
print("ST  = {}".format(round(st_average / tm_N, 3)))
print("DCG = {}".format(round(dcg_average / tm_N, 3)))

for j in range(bu_N):
    pecisions[j] = pecisions[j] / tm_N
    recalls[j] = recalls[j] / tm_N
retrieval.plot_pecision_recall(pecisions, recalls)

for i in range(bu_N):
    #if i == 0 or i == bu_N-1 or i % int(bu_N/20) == 0:
    print(round(recalls[i], 3), round(pecisions[i], 3))

plotslist = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
j = 0
for i in range(bu_N):
    if i == 0 or i == bu_N-1 :
        print(round(recalls[i], 3), round(pecisions[i], 3))
    else:
        if recalls[i] == plotslist[j]:
            print(round(recalls[i], 3), round(pecisions[i], 3))
            j = j + 1
        elif recalls[i] > plotslist[j]:
            print(round((recalls[i]+recalls[i-1])/2.0, 3), round((pecisions[i]+pecisions[i-1])/2., 3))
            j = j + 1
plt.show()
