import os, json, math, collections

import numpy as np
import tensorflow as tf

import geoutils

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


from time import time

from itertools import chain
from PIL import Image
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist

# ensure output dir exists
if not os.path.isdir('out'):
    os.mkdir('out')

def two_points_distance(from_pt, to_pt):
    dx, dy = from_pt[0] - to_pt[0], from_pt[1] - to_pt[1]
    return  math.sqrt(dx*dx + dy*dy)

def rotate_angle(dx1, dy1, dx2, dy2):
    # normalize
    dist = math.sqrt(dx1*dx1 + dy1*dy1)
    dx1 /= dist; dy1 /= dist
    dist = math.sqrt(dx2*dx2 + dy2*dy2)
    dx2 /= dist; dy2 /= dist

    # dot product
    dot = dx1*dx2 + dy1*dy2
    if abs(dot-1.0) <= 1.0e-16:
        angle = 0.0;
    elif abs(dot+1.0) <= 1.0e-16:
        angle = math.pi
    else:
        # to avoid going beyond our domain because of precison.
        if dot >= 1: dot = 1
        if dot <= -1: dot = -1
        angle = math.acos(dot)
        #cross product
        cross = dx1*dy2 - dx2*dy1
        # vector p2 is clockwise from vector p1
        # with respect to the origin (0.0)
        if cross < 0:
            angle = 2*math.pi - angle
    return angle

def three_points_vector(prev_pt, curr_pt, next_pt):
    dx1, dy1 = prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1]
    dx2, dy2 = next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]
    return [dx1+dx2, dy1+dy2]

def three_points_angle(prev_pt, curr_pt, next_pt):
    #IPoint pt1, IPoint pt, IPoint pt2
    ax, ay = prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1]
    bx, by = next_pt[1] - curr_pt[0], next_pt[1] - curr_pt[1]

    dx, dy = ax * by - ay * bx, ax * bx + ay + by
    angle = math.atan2(dy, dx)
    return angle

def three_points_angle_ex(prev_pt, curr_pt, next_pt):
    dx1, dy1 = prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1]
    dx2, dy2 = next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]
    c = math.sqrt(dx1*dx1 + dy1*dy1) * math.sqrt(dx2*dx2 + dy2*dy2)
    if c == 0: return -1
    return rotate_angle(dx1, dy1, dx2, dy2)

def point_on_segment_by_length(from_pt, to_pt, length):
    b_in = False
    dis = two_points_distance(from_pt, to_pt)
    if length <= dis:
        b_in = True
    if dis == 0.0:
        dis = 1.0
    dx, dy = from_pt[0]-to_pt[0], from_pt[1]-to_pt[1]
    ax, ay = length * dx / dis, length * dy / dis
    return [from_pt[0]+ax, from_pt[1]+ay, b_in]
        
def turning_function(coords, perimeter):
    n_coords = len(coords)
    ref_length = perimeter / (n_coords - 1)
    turning_function = []
    is_cw, is_normalize = False, True
    s = 0.0
    for i in range(1, n_coords):
        prev_pt, curr_pt, next_pt = coords[i-1], coords[i], coords[(i+2)%n_coords if i==n_coords-1 else i+1]
        if False:
            if i == 1:
                ref_pt = [curr_pt[0]+ref_length, curr_pt[1]]
            else:
                ref_pt = point_on_segment_by_length(prev_pt, curr_pt, ref_length)
        else:
            ref_pt = [curr_pt[0]+ref_length, curr_pt[1]]
        turning_angle = three_points_angle_ex(ref_pt, curr_pt, next_pt)
        if is_cw:
            turning_angle = 2*math.pi - turning_angle

        if i == 1:
            begin_turning = turning_angle
            totle_turning = begin_turning
        else:
            totle_turning += turning_angle

        if is_normalize:
            from_pos = s / perimeter
        else:
            from_pos = s
        s += two_points_distance(curr_pt, next_pt)
        if is_normalize:
            to_pos = s / perimeter
        else:
            to_pos = s
        turning_value = turning_angle #totle_turning - begin_turning
        turning_function.append([from_pos, to_pos, turning_value])
    return turning_function

def turning_function_value(turning_func, x):
    ans = 0.0;
    if x < .0 or x > 1.0:
        return ans;
    for i in range(len(turning_func)):
        if x >= turning_func[i][0] and x < turning_func[i][1]:
            ans = turning_func[i][2]
            return ans
    return ans
        
def draw_turning_function(turning_func, building_coords):
    fig = plt.figure(21)
    funct_plot = fig.add_subplot(1, 2, 1)
    coord_plot = fig.add_subplot(1, 2, 2)
    ax, ay = [], []
    for i in range(len(turning_func)):
        ax.append(turning_func[i][0])
        ax.append(turning_func[i][1])
        ay.append(turning_func[i][2])
        ay.append(turning_func[i][2])
    funct_plot.plot(ax, ay)

    building_coords = np.array(building_coords)
    coord_plot.plot(building_coords[:, 0], building_coords[:, 1])
    l = len(building_coords)
    for i in range(1, l):
        coord_plot.text(building_coords[i, 0], building_coords[i, 1], i)

    for i in range(len(turning_func)):
        if not i == -1:
            continue
        for j in range(3):
            p = turning_func[i][j+3]
            a, b, r = p[0], p[1], 0.25
            theta = np.arange(0, 2*np.pi, 0.05)# + [0.0]
            x, y= a + r * np.cos(theta), b + r * np.sin(theta)
            coord_plot.fill(x, y, 'r')

    funct_plot.set_aspect(1)
    coord_plot.set_aspect(1)
    plt.show()

def load_data(filename, split=None):
    #print("Parsing the {} data".format(filename))
    file = open(filename,'r',encoding='utf-8')
    data = json.load(file)
    feature_size = len(data['features'])
    # import random
    # random.shuffle(data['features'])
    examples, num_points = [], []
    for i in range(0, feature_size):
        # get the geometry objects.
        # if i >=100 :  continue
        if i % int(feature_size/10) == 0:
            print('processing: {:2d}%'.format(int(i*100/feature_size)))
        label = data['features'][i]['attributes']['tangent']
        geome_dict=data['features'][i]['geometry']
        geo_content=geome_dict.get('rings')
        if geo_content==None:
            geo_content=geome_dict.get('paths')
        if geo_content==None:
            print("Please Check the input data.")
            file.close()
            return [], []

        # get the coords.
        building_coords = []
        for j in range(0, len(geo_content[0])):
            building_coords.append([geo_content[0][j][0], geo_content[0][j][1]])

        # adjust the start point.
        # building_coords = FFT.adjust_StartAndDirection(building_coords)
        [[CX,CY], area, peri] = geoutils.get_basic_parametries_of_Poly(building_coords)

        if True:
            turning_func = turning_function(building_coords, peri)
            if True:
                turning_func = np.array(turning_func)
                func_min, func_max = min(turning_func[:,2]), max(turning_func[:,2])
                for j in range(len(turning_func)):
                    turning_func[j][2] = (turning_func[j][2] - func_min)/(func_max - func_min)

            turning_features, n_features = [], 64
            for j in range(n_features):
                turning_features.append(turning_function_value(turning_func, j*1.0/n_features))
            
            if (i == 12 and len(building_coords) < 64) and False:
                for j in range(len(turning_func)):
                    print(turning_func[j][0], turning_func[j][1], turning_func[j][2])
                draw_turning_function(turning_func, building_coords)
                exit()

        # coords = [[(coord[0]-x_min)/(x_max-x_min), (coord[1]-y_min)/(y_max-y_min)] for coord in building_coords]
        # coords_x = [(coord[0]-x_min)/(x_max-x_min) for coord in building_coords]
        # coords_y = [(coord[1]-y_min)/(y_max-y_min) for coord in building_coords]
        # coords = [[(coord[0]-x_min)/(x_max-x_min), (coord[1]-y_min)/(y_max-y_min)] for coord in building_coords]
        coords = [[(coord[0]-CX)/math.sqrt(area), (coord[1]-CY)/math.sqrt(area)] for coord in building_coords]
        [[CX,CY], area, peri] = geoutils.get_basic_parametries_of_Poly(coords)

        if False:
            fig = plt.figure(1, figsize=(14,14))
            ax  = fig.add_subplot(1,1,1)
            ax.fill(np.array(coords)[:, 0], np.array(coords)[:, 1], c='k')
            ax.set_axis_off()
            ax.set_aspect(1)

            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            fig.savefig("out/poly"+"%04d" % i+'.jpg', dpi=2)
            plt.close()
 
            img  = Image.open("out/poly"+"%04d" % i+'.jpg')
            img  = img.convert('L')
            img  = np.array(img)
            img  = 1.0 - img / 255.0

            pixel_features = list(chain(*img))
            #print(pixel_features)
            #exit()

        if False:
            inter_count = 20
            coords_features, track_ind, track_dis = [], 0, 0.0
            coords_features.append(math.sqrt(coords[0][0]*coords[0][0] + coords[0][1]*coords[0][1]))

            for j in range(1, inter_count):
                inter_dis = peri*j / inter_count
                while track_dis < inter_dis:
                    track_ind += 1
                    track_dis += two_points_distance(coords[track_ind-1], coords[track_ind])
                track_rad = 1.0 - (track_dis-inter_dis) / two_points_distance(coords[track_ind-1], coords[track_ind])

                inter_coord_x = coords[track_ind-1][0] + track_rad*(coords[track_ind][0]-coords[track_ind-1][0])
                inter_coord_y = coords[track_ind-1][1] + track_rad*(coords[track_ind][1]-coords[track_ind-1][1])

                coords_features.append(inter_coord_x*inter_coord_x + inter_coord_y*inter_coord_y)

        if False:
            # calculating the angle feature.
            angle_vectors = []
            count = len(coords)
            if area > 0:
                for j in range(1, count):
                    prev_pt, curr_pt, next_pt = coords[j-1], coords[j], coords[(j+2)%count if j==count-1 else j+1]
                    vector = three_points_vector(prev_pt, curr_pt, next_pt)
                    angle_vectors.append(vector)
            else:
                # need to be improved.
                print("area < 0")

            if False:
                # normalising
                if i == 1:
                    print(angle_vectors)
                angle_vectors = np.array(angle_vectors)
                x_min, x_max = np.min(angle_vectors[:, 0]), np.max(angle_vectors[:, 0])
                y_min, y_max = np.min(angle_vectors[:, 1]), np.max(angle_vectors[:, 1])
                angle_vectors = [[(vector[0]-x_min)/(x_max-x_min), (vector[1]-y_min)/(y_max-y_min)] for vector in angle_vectors]
                if i == 1:
                    print(angle_vectors)

        # collecting
        examples.append(turning_features)
        num_points.append([coords, label])
    file.close()

    # padding.
    maximum = max([len(examples[i]) for i in range(len(examples))])
    examples_padding = []
    for i in range(len(examples)):
        examples_padding.append(np.pad(examples[i],
                                       ((maximum-len(examples[i])), (0)),
                                       'constant', constant_values=(0)))
    if split == None:
        return [examples_padding, num_points]
    # splitting
    split.insert(0, 0)
    # print(len(examples_padding))
    split_n = [int(len(examples_padding)*sum(split[0:i+1])) for i in range(len(split))]
    return [[examples_padding[split_n[i-1]:split_n[i]], num_points[split_n[i-1]:split_n[i]]] for i in range(1, len(split_n))]

#print(X.shape)
#X = digits.data
#exit()
#load_data_res = np.array(load_data("data/MNIST_data/poly_c10.json")) #, [0.7, 0.3]
#polys, num_points = load_data_res[0], load_data_res[1]
#train_X = np.array([polys[i] for i in range(len(polys))])
#train_y = np.array([num_points[i][1] for i in range(len(num_points))])
