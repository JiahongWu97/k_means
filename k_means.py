# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 09:18:09 2018

@author: Administrator
"""
#导入数据:直接拖把、nonzero非零放行列、画图、不同文件交互：同一目录import、
import numpy as np
# from center_method import *
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

# def naive_k_means(data , k, center, dist_method = euclidean, final_value = 1e-6, max_iteration = 100, online = True):
def naive_k_means(data , k, center, max_iteration = 200, dist_method = euclidean):
    row = np.shape(data) [0]
    table = np.zeros(row)
    index = np.zeros(row, dtype = int)
#    center = center.copy()
    cluster_changed = True
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    for i in range(row):
        index[i], table[i] = hunt_all(data[i, :], center)
        cen_num[index[i]] += 1 #用累加的变量记录簇
        cen_sum[index[i], :] += data[i, :]
    total = 1
#    plot_center(center, data, offset=0)
    while cluster_changed and total < max_iteration:
        for i in range(k):
            # if cen_num[i] == 0:
            #     print('*'*10+'fail'+'*'*10)
            #     return -1, -1
            # else:
            if cen_num[i] != 0:
                center[i, :] = cen_sum[i, :] / cen_num[i] #求mean
        cluster_changed = False
        total += 1
        for i in range(row):
            temp = index[i]
            index[i], table[i] = hunt_all(data[i, :], center)      #找最近中心点
            # see_equal(data[i, :], center) #相等应该确实没那么巧
            if index[i] != temp:
                cluster_changed = True
                cen_num[temp] -= 1
                cen_num[index[i]] += 1
                cen_sum[temp, :] -= data[i, :]  #对应簇的辅助变量更新
                cen_sum[index[i], :] += data[i, :]
    # print(total)
    return index, table
#    my_plot(data, center, index, table, offset=0)
def hunt_all(data, center):
    t = data - center[0, :]
    min_dist, min_index = np.dot(t,t), 0
    for j in range(1, np.shape(center)[0]):
        t = data - center[j, :]
        dist = np.dot(t,t)
        if dist < min_dist:
            min_dist, min_index = dist, j
    return  min_index, min_dist
def see_equal(data, center, d = euclidean):
    reg = d(data, center, 1)
    index = np.argsort(reg)
    t = 0
    while reg[index[t]] == reg[index[t+1]]:
        t += 1
    if t > 0:
        print(t+1,'个相等', index[0])

# from elkan_triangle import elkan_hyperbola_neighbor_2
if __name__ == '__main__':
   # k_means(iris, 3)
   #  data = np.load('iris.npy')
    data = np.random.rand(400,3)
   # a = mat([[1,2],[3,4]])
#    b=mat([[1],[2]])
    k = 5
    center = random_center(data, k)
    # center = plus_triangle(data, k)
    i, j = naive_k_means(data , k, center.copy())
    # i7, j7 = elkan_hyperbola_neighbor_2(data, k, center.copy())
    # print(sum(i7) - sum(i), sum(j) - sum(j7))

#    a, b = hunt_all(data[0, :], c)