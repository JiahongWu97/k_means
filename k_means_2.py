# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:18:09 2018

@author: Administrator
"""
#导入数据:直接拖把、nonzero非零放行列、画图、不同文件交互：同一目录import、
import numpy as np
from center_method import *
from line_three import *
from distance_method import *

def get_data(file):
    data = []
    f = open(file)
    for i in f.readlines():
        modify = i.strip().split('\t')
        float_ = map(float, modify)
        data.append(float_)
    return data

#def naive_k_means(data , k, cen_method=plus_plus, dist_method = euclidean, final_value = 1e-6, max_iteration = 100, online = True):
def naive_k_means_original(data , k, center, dist_method = euclidean, final_value = 1e-6, max_iteration = 100, online = True):
    row = np.shape(data) [0]
    table = np.zeros((row, 2))
#    center = center.copy()
    cluster_changed = True
    plot_center(center, data, offset=0)
    while cluster_changed:
        cluster_changed = False
        for i in range(row):
            min_dist, min_index = dist_method(data[i, :], center[0, :]), 0
            for j in range(1, k):
                dist = dist_method(data[i, :], center[j, :], 0)
                if dist < min_dist:
                    min_dist, min_index = dist, j
            temp = table[i, 0]
            table[i, :] = min_index, min_dist
            if table[i, 0] != temp:
                cluster_changed = True            
        for i in range(k):
            temp = data[np.nonzero(table[:, 0] == i) [0]]
            if np.size(temp) != 0:
                center[i, :] = np.mean(temp, 0)
    my_plot(data, center, table[:, 0], table[:, 1], offset=0)
def hunt_all(data, center, d = euclidean):
    min_dist, min_index = d(data, center[0, :]), 0
    for j in range(1, np.shape(center) [0]):
        dist = d(data, center[j, :])
        if dist < min_dist:
            min_dist, min_index = dist, j
    return  min_index, min_dist
#    返回并没有错，但是array的dtype就统一为了 float，没得办法
if __name__ == '__main__':     
#    k_means(iris, 3)
    data = np.load('iris.npy')
#    a = mat([[1,2],[3,4]])
#    b=mat([[1],[2]])
    c = random_center(data, 3)

    naive_k_means(data , 3, c)
#    a, b = hunt_all(data[0, :], c)