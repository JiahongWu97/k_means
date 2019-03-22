# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:35:42 2019

@author: Administrator
"""
import numpy as np
from center_method import *
from line_three import *
from distance_method import *

#def get_cen_sqrt(data):
#    k = np.shape(data) [0]
#    cen = np.ones((k, k)) * np.inf
#    for i in range(k):
#        for j in range(i + 1, k):
#            cen[i, j] = np.sqrt(euclidean(data[i, :], data[j, :]))
#    for i in range(1, k):
#        for j in range(0, i ):
#            cen[i, j] = cen[j, i]
#    return cen
class Heap:
    def __init__(self):
        self.data = []
    def __len__(self):
        return len(self.data)
    def get_min(self):
        return self.data[0][0]
    def left(self, j):
        return 2 * j + 1
    def right(self, j):
        return 2 *j + 2
    def parent(self, j):
        return (j - 1) // 2
    def has_left(self, j):
        return self.left(j) < len(self)
    def has_right(self, j):
        return self.right(j) < len(self)
    def upheap(self, j):
     parent = self. parent(j)
     if j > 0 and self. data[j][0] < self. data[parent][0]:
         self. swap(j, parent)
         self. upheap(parent) # recur at position of parent
    def add(self, key, value):
        self.data.append((key, value))
        self.upheap(len(self.data)-1)
    def swap(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]
    def down(self, j):
        if self.has_left(j):
            left = self.left(j)
            small = left
            if self.has_right(j):
                right = self.right(j)
                if self.data[right][0] < self.data[left][0]:
                    small = right
            if self.data[j][0] > self.data[small][0]:
                self.swap(j, small)
                self.down(small)
    def remove_min(self):
     self.swap(0, len(self. data)-1) # put minimum item at the end
     v = self.data.pop( )
     self.down(0) 
     return v
def point_all_center_sqrt(x, c, d = euclidean):
    first, second, index = d(x, c[0, :]), d(x, c[1, :]), 0
    if first > second:
        first, second, index = d(x, c[1, :]), d(x, c[0, :]), 1
    for i in range(2, np.shape(c) [0]):
        temp = d(x, c[i, :])
        if temp < first:
            first, second, index = temp, first, i
        elif temp < second:
            second = temp
    return index, np.sqrt(second) - np.sqrt(first)
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

def heap_pure(data, k, center, dist_method = euclidean, max_iteration = 100):
    heap_list = [Heap() for i in range(k)]
    row = np.shape(data) [0]
    z = np.zeros(k)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index, table = point_all_center_sqrt(data[i, :], center)
        heap_list[index].add(table, i)
        cen_sum[index, :] += data[i, :]
        cen_num[index] += 1
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
    for j in range(k):
        if first == j:
            z[j] += p[second] + p[j]
        else:
            z[j] += p[first] + p[j]
    cluster_changed = True 
    while cluster_changed and total < max_iteration:
        cluster_changed = False
        total += 1
#        s = get_cen_sqrt(center).min(1)
        for j in range(k):
#            m = max(s[index[i]] / 2, table[i, 0])
            while len(heap_list[j]) > 0 and heap_list[j].get_min() < z[j]:
#                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
#                if table[i, 1] > m:
                    i = heap_list[j].remove_min()[1] #值和data的索引
                    
                    index, table = point_all_center_sqrt(data[i, :], center)
                    table += z[index]
                    heap_list[index].add(table, i)
                    if j != index:
                        cluster_changed = True
#                        print('asd')
                        cen_num[j] -= 1
                        cen_num[index] += 1
                        cen_sum[j, :] -= data[i, :]
                        cen_sum[index, :] += data[i, :]
#        if cluster_changed:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*'*10+'fail'+'*'*10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        for j in range(k):
            if first == j:
                z[j] += p[second] + p[j]
            else:
                z[j] += p[first] + p[j]
        
    table = np.zeros((row))
#    index = [i[1] for j in range(k) for i in heap_list[j].data]
    index = np.zeros((row), int)
    for j in range(k):
        for i in heap_list[j].data:
            index[i[1]], table[i[1]] = j,  dist_method(data[i[1], :], center[j, :])           
    print(total)
#    my_plot(data, center, index, table, 0, 'heap_pure')

    return index, table
#def hamerly_sqrt(data, k, center_method=plus_plus, dist_method = euclidean, max_iteration = 100):
def heap_for(data, k, center, dist_method = euclidean, max_iteration = 100):
    row = np.shape(data) [0]
    table = np.zeros((row))#第一列放low，第二列high
    index = np.zeros((row), int)
    z = np.zeros(k)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
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
    for j in range(k):
        if first == j:
            z[j] += p[second] + p[j]
        else:
            z[j] += p[first] + p[j]
    cluster_changed = True 
    while cluster_changed and total < max_iteration:
        cluster_changed = False
        total += 1
#        s = get_cen_sqrt(center).min(1)
        for i in range(row):
#            m = max(s[index[i]] / 2, table[i, 0])
            if table[i] < z[index[i]]:
#                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
#                if table[i, 1] > m:
                    temp = index[i]
                    index[i], table[i] = point_all_center_sqrt(data[i, :], center)
                    table[i] += z[index[i]]
                    if temp != index[i]:
                        cluster_changed = True
#                        print('asd')
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
#        if cluster_changed:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*'*10+'fail'+'*'*10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        for j in range(k):
            if first == j:
                z[j] += p[second] + p[j]
            else:
                z[j] += p[first] + p[j]
        
    table = dist_method(data, center[index[:], :], 1)
    print(total)
#    my_plot(data, center, index, table, 0, 'heap_for')

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
    dim = [2,4,8,16,32,64]
    k = 8
    t = []
    for d in dim:
        data = np.random.randn(30000,d) * 2
    #    k = 3
        center=plus_triangle(data, k)
        print(d)
        a = time.process_time()
        ii, jj = heap_for(data , k, center.copy())
        b = time.process_time()
        t.append(b - a)
        
        a = time.process_time()
        i, j = heap_pure(data , k, center)
        b = time.process_time()
        t.append(b - a)

#        print(np.sum(i - ii), np.sum(j)-np.sum(jj))
        
    plt.plot(kk, t[1::2], label='heap pure', c='k',linewidth=2)
#    plt.plot(kk, c[2::3], '--',label='standard', c='b',linewidth=2)
    plt.plot(kk, t[0::2], '-+',label='heap for', c='r',linewidth=2)
    
#    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
#    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
#    
    plt.legend()
    plt.grid()
    plt.xlabel('dimensional')
#    plt.xlabel('number')
    plt.ylabel('time')
#
#[15.564208726000004,
# 20.56385205799998,
# 26.229412247,
# 35.44282548699999,
# 40.35982875000002,
# 55.97567111899997,
# 58.494563450999976,
# 77.52783965000003,
# 69.70725398000002,
# 96.608412111,
# 96.67335016099992,
# 130.54452192999997]