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
from tensorflow.examples.tutorials.mnist import input_data


# ensure output dir exists
if not os.path.isdir('out'):
    os.mkdir('out')

def det(point1,point2,point3):
    return (point2[0]-point1[0])*(point3[1]-point1[1])-(point2[1]-point1[1])*(point3[0]-point1[0])

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

def distance_between_two_functions(function_1, function_2):
    # ...
    step, s = 60, 0.0
    for i in range(step):
        x = i*1./step
        value_1 = turning_function_value(function_1, x)
        value_2 = turning_function_value(function_2, x)
        s = s + abs(value_1 - value_2)*math.pi*2 / step
    return math.sqrt(s)
        
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

# A basic function for the calcalation of basic parametries of a polygon
# Args:
#   A=[[x1,y1],[x2,y2],...,[xn,yn]]: The input polygon
# Output:
#  [CX,CY]: The center point of the polygon.
#     area: The area of the polygon.
#     peri: The perimeter of the polygon.
def get_basic_parametries_of_Poly(A):
    CX,CY,area,peri=0,0,0,0
    if len(A) < 1:
        raise Exception('ILLEGAL_ARGUMENT')
        return [[CX,CY], area, peri]
    # closure the polygon.
    if A[0][0] != A[len(A)-1][0] or A[0][1] != A[len(A)-1][1]:
        A.append(A[0])
    # calculate the center point [CX,CY] and perometry L.
    for i in range(0,len(A)-1):
        CX += A[i][0]
        CY += A[i][1]
        peri += math.sqrt(pow(A[i+1][0]-A[i][0],2) + pow(A[i+1][1]-A[i][1],2))

    CX = CX/(len(A)-1)
    CY = CY/(len(A)-1)
    #calculate the area.
    if len(A) < 3:
        raise Exception('ILLEGAL_ARGUMENT')
        return [[CX,CY], area, peri]
    indication_point = A[0]
    for i in range(1,len(A)-1):
        #vector_pp1=[A[i][0]-A[0][0],A[i][1]-A[0][1]]
        #vector_pp2=[A[i+1][0]-A[0][0],A[i+1][1]-A[0][1]]
        #vector_cross=vector_pp1[0]*vector_pp2[1]-vector_pp1[1]*vector_pp2[0]
        #sign=0;
        #if(vector_cross>0):
        #   sign=1
        #else:
        #   sign=-1
        area += det(indication_point,A[i],A[i+1])
    return [[CX,CY], abs(area)*0.5, abs(peri)]

'''
# test code...
# Input polygon:    building_coords=[[x1,y1],[x2,y2],...,[xn,yn]]

[[CX,CY], area, peri] = get_basic_parametries_of_Poly(building_coords)

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
'''