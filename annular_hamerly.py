# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:35:42 2019

@author: Administrator
"""
import numpy as np
from center_method import *
from line_three import *
from distance_method import *

def get_cen_sqrt(data):
    k = np.shape(data) [0]
    cen = np.ones((k, k)) * np.inf
    for i in range(k):
        for j in range(i + 1, k):
            cen[i, j] = np.sqrt(euclidean(data[i, :], data[j, :])) / 2
    for i in range(1, k):
        for j in range(0, i):
            cen[i, j] = cen[j, i]
    return cen

def point_all_center_sqrt_2(x, c, d = euclidean):
    first, second, index1, index2 = d(x, c[0, :]), d(x, c[1, :]), 0, 1
    if first > second:
        first, second, index1, index2 = second, first, 1, 0
    for i in range(2, np.shape(c) [0]):
        temp = d(x, c[i, :])
        if temp < first:
            first, second, index1, index2 = temp, first, i, index1
        elif temp < second:
            second, index2 = temp, i
    return index1, index2, np.sqrt(first), np.sqrt(second)

def max_2_index(x):
    first, second, a, b = x[0], x[1], 0, 1
    if first < second:
        first, second, a, b = second, first, 1, 0
    for i in range(2, np.size(x)):
        if x[i] > first:
            first, second, a, b = x[i], first, i, a
        elif x[i] > second:
            second, b = x[i], i
    return a, b

def annular(data, k, center, dist_method = euclidean, max_iteration = 100):
    row = np.shape(data) [0]
    table = np.zeros((row, 2))#第一列放low，第二列high
    index = np.zeros((row, 2), int) #first, second
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i, 0], index[i, 1], table[i, 1], table[i, 0] = point_all_center_sqrt_2(data[i, :], center)
        cen_sum[index[i, 0], :] += data[i, :]
        cen_num[index[i, 0]] += 1
    p = np.zeros(k)
    for i in range(k):
        temp = center[i, :].copy()
        if cen_num[i] == 0:
            print('*'*10+'fail'+'*'*10)
            return -1, -1
        else:
            center[i, :] = cen_sum[i, :] / cen_num[i]
        p[i] = np.sqrt(dist_method(center[i, :], temp))
    first, second = max_2_index(p)
    for i in range(row):
        table[i, 1] += p[index[i, 0]]
        if first == index[i, 0]:
            table[i, 0] -= p[second]
        else:
            table[i, 0] -= p[first]
    cluster_changed = True 
    # center_norm = np.sqrt(np.power(center, 2).sum(1))
    center_norm = np.linalg.norm(center, axis = 1)
    # data_norm = np.sqrt(np.power(data, 2).sum(1))
    data_norm = np.linalg.norm(data, axis = 1)
    while cluster_changed and total < max_iteration:
        cluster_changed = False
        total += 1
        s = get_cen_sqrt(center).min(1)
        arg = np.argsort(center_norm)
        # arg = np.arange(k)
        for i in range(row):
            m = max(s[index[i, 0]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i, 0], :]))
                if table[i, 1] > m:
                    temp = index[i, 0]
#                    index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
                    second = np.sqrt(dist_method(data[i, :], center[index[i, 1], :]))
                    low = np.searchsorted(center_norm[arg], data_norm[i] - second) #[low,high)
                    high = np.searchsorted(center_norm[arg], data_norm[i] + second, 'right')
                    # print(center_norm[arg])
                    # arg=[j for j in range(k) if data_norm[i] - table[i, 1] <= center_norm[j] <=data_norm[i] + table[i, 1]]
                    # low, high = 0, len(arg)

                    # print(low, high, arg)
                    if high - low >= 2:
                        index[i, 0], index[i,1], table[i, 1], table[i, 0] = point_all_center_sqrt_2(data[i, :], center[arg[low : high], :])
                        # print('yuan', index[i, :])
                        index[i, 0], index[i,1] = arg[low : high][index[i, 0]], arg[low : high][index[i,1]]
#                        print('hou', index[i, :])
                    else:
                        # print('too tight')
                        index[i, 0], index[i, 1], table[i, 1], table[i, 0] = point_all_center_sqrt_2(data[i, :], center)
                    if temp != index[i, 0]:
                        cluster_changed = True
#                        print('asd')
                        cen_num[temp] -= 1
                        cen_num[index[i, 0]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i, 0], :] += data[i, :]
        if cluster_changed:
            for i in range(k):
                temp = center[i, :].copy()
                if cen_num[i] == 0:
                    print('*'*10+'fail'+'*'*10)
                    return -1, -1
                else:
                    center[i, :] = cen_sum[i, :] / cen_num[i]
                p[i] = np.sqrt(dist_method(center[i, :], temp))
            first, second = max_2_index(p)
            for i in range(row):
                table[i, 1] += p[index[i, 0]]
                if first == index[i, 0]:
                    table[i, 0] -= p[second]
                else:
                    table[i, 0] -= p[first]
            center_norm = np.linalg.norm(center, axis = 1)
            # print(center_norm)
            # center_norm = np.sqrt(np.power(center, 2).sum(1))

    table = dist_method(data, center[index[:, 0], :], 1)
    print(total)
#    my_plot(data, center, index[:, 0], table, 0, 'annular')
    return index, table
#from harmerly_triangle import * #mmp,重载了方法/
from harmerly_triangle import hamerly_sqrt
from center_method import plus_triangle    
import time
import matplotlib.pyplot as plt
if __name__ == '__main__':
#    a = array([[1,2,3]]).T
#    b = array([[-4,9,9]]).T
#    data = array([[1,2,3], [4,5,6]])
#    data  = np.load('iris.npy')

    for d in range(10):
        print((d + 1), '次')
        data = np.random.randn(3000,3) * 2
        dim = [2,4,8,16,32,64]
        k = 5
        t = []
        center=plus_triangle(data, k)
#    for d in dim:
#        data = np.random.randn(30000,d) * 2
#    #    k = 3
#        center=plus_triangle(data, k)
#        print(d)
#        a = time.process_time()
#        ii, jj = hamerly_sqrt(data , k, center.copy())
#        b = time.process_time()
#        t.append(b - a)
#        
#        a = time.process_time()
#        i, j = annular(data , k, center)
#        b = time.process_time()
#        t.append(b - a)
        ii, jj = hamerly_sqrt(data , k, center.copy())
        i, j = annular(data , k, center.copy())
        # my_plot(data, center, i[:, 0], j, 0, 'annular')

        print(np.sum(abs(i[:,0] - ii)), np.sum(j)-np.sum(jj))
#        
#    plt.plot(kk, t[1::2], label='annular', c='k',linewidth=2)
##    plt.plot(kk, c[2::3], '--',label='standard', c='b',linewidth=2)
#    plt.plot(kk, t[0::2], '-+',label='hamerly sqrt', c='r',linewidth=2)
#    
##    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
##    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
##    
#    plt.legend()
#    plt.grid()
#    plt.xlabel('dimensional')
##    plt.xlabel('number')
#    plt.ylabel('time')

#    [17.36401597200006,
# 13.541370495000024,
# 26.15816011100003,
# 28.94751338499998,
# 39.00431896899988,
# 46.38107525600003,
# 57.00992558500002,
# 68.50688099099989,
# 66.08529510799985,
# 79.24973440899998,
# 93.23470593100001,
# 111.43527967600016]