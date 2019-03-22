# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:56:27 2019

@author: Administrator
"""
import numpy as np
from distance_method import *
from line_three import *
from  center_method import *
class Wiki_Node:
    __slots__ = 'indices', 'mi', 'ma', 'w', 'count', 'left', 'right'
    def __init__(self, indices, mi, ma, w, count, left = None, right = None):
        self.indices = indices
        self.mi = mi
        self.ma = ma
        self.w = w
        self.count = count
        self.left = left
        self.right = right
    def in_order(self, p):
#        if p.left is not None:
        if not isinstance(p.left, Wiki_Leaf):
            for o in self.in_order(p.left):
                yield o
        yield p
        if not isinstance(p.right, Wiki_Leaf): 
            for o in self.in_order(p.right):
                yield o
    def __iter__(self):
        for p in self.in_order(self):
            yield p.w
    def __repr__(self):
        a=[]
        for i in self:
            a.append(i)
        return str(a)

class Wiki_Leaf:
    __slots__ = 'e', 'i'
#    __slots__ = 'e', 'left', 'right'
    def __init__(self, e, i):
        self.e = e
        self.i = i
#    def __init__(self, e, left = None, right = None):
#        self.e = e
#        self.left = left
#        self.right = right

class Tree:
    class _Node:
        __slots__ = 'mi', 'ma', 'w', 'count', 'left', 'right'
        def __init__(self, mi, ma, w, count, left = None, right = None):
            self.mi = mi
            self.ma = ma
            self.w = w
            self.count = count
            self.left = left
            self.right = right
    class _Leaf:
        __slots__ = 'e', 'left', 'right'
        def __init__(self, e, left = None, right = None):
            self.e = e
            self.left = left
            self.right = right
    def __init__(self):
        self.root = None
        self.size = 0
    def __len__(self):
        return self.size
    def add_root(self, mi, ma, w, count):
        self.size = 1
        self.root = self._Node(mi, ma, w, count)
        return self.root
    def add_left(self, p, mi, ma, w, count):
#        self.size += 1
        p.left = self._Node(mi, ma, w, count)
        return p.left
    def add_right(self, p, mi, ma, w, count):
#        self.size += 1
        p.right = self._Node(mi, ma, w, count)
        return p.right
    def add_left_leaf(self, p, e):
        p.left = self._Leaf(e)
    def add_right_leaf(self, p, e):
        p.right = self._Leaf(e)
    def in_order(self, p):
        if p.left is not None:
            for o in self.in_order(p.left):
                yield o
        yield p
        if p.right is not None:
            for o in self.in_order(p.right):
                yield o
    def __iter__(self):
        for p in self.in_order(self.root):
            yield p.w
    def __repr__(self):
        a=[]
        for i in self:
            a.append(i)
        return str(a)
    
def cell_kdtree(data, a, depth=0, root=None, left=True, bucketsize = 1):
    row, col = shape(data)
    axis = depth % col
    data = data[data[:,axis].argsort(0)]
    median = (row-1) // 2 # choose median
    try:
#        print(data[median, axis] ,data[median + 1, axis])
        while data[median, axis] == data[median + 1, axis]:
            median += 1
    except IndexError:
        pass
    if len(a) == 0:
        root = a.add_root(data.min(0), data.max(0), sum(data, 0), row)
    else:
        if left:
            if row > bucketsize:
                root = a.add_left(root, data.min(0), data.max(0), sum(data, 0), row)
            else:
                a.add_left_leaf(root, data)
                return 
        else:
            if row > bucketsize:
                root = a.add_right(root, data.min(0), data.max(0), sum(data, 0), row)
            else:
                a.add_right_leaf(root, data)
                return
    if median >= 0:
        cell_kdtree(data[:median+1, :], a, depth + 1,  root, True)
    if median+1 <= row - 1:
        cell_kdtree(data[median+1 :, :], a, depth + 1,  root, False)

def cell_kdtree_2(data, depth=0, bucket_size=3):
    row, col = shape(data)
    if row > bucket_size:
        axis = depth % col
        data = data[data[:,axis].argsort(0)]
        median = (row-1) // 2 # choose median
        try:
            while data[median, axis] == data[median + 1, axis]:
                median += 1
        except IndexError:
            pass

        return Wiki_Node(data.min(0), data.max(0), sum(data, 0), row,  #这里优化
                     cell_kdtree_2(data[:median+1, :], depth+1),
                     cell_kdtree_2(data[median+1 :, :], depth+1)
                     )
    else:
        return Wiki_Leaf(data)
#我操死你妈的，numoy的array 指针（变量名）作为行参，要copy一下。
def cell_kdtree_3(data, a, mi=None, ma=None, depth=0, root=None, left=True, bucketsize = 1):
    row, col = shape(data)
    axis = depth % col
    data = data[data[:,axis].argsort(0)]
    median = (row-1) // 2 # choose median
    try:
        while data[median, axis] == data[median + 1, axis]:
            median += 1
    except IndexError:
        pass
    if len(a) == 0:
        root = a.add_root(mi.copy(), ma.copy(), sum(data, 0), row)
    else:
        if left:
            if row > bucketsize:
#                print(mi, ma)
                root = a.add_left(root,mi.copy(), ma.copy(), sum(data, 0), row)
            else:
                a.add_left_leaf(root, data)
                return 
        else:
            if row > bucketsize:
#                print(mi, ma)
                root = a.add_right(root, mi.copy(), ma.copy(), sum(data, 0), row)
            else:
                a.add_right_leaf(root, data)
                return
    ori_mi = mi.copy()
#        print(ori_mi)
    ori_ma = ma.copy()
#        print(ma)
#        mi[axis], ma[axis] = data[median, axis], data[median, axis]
    mi[axis] = data[median, axis]
    ma[axis]=data[median, axis]
#        print(data[median, axis])
#    print(mi, ma)
    temp_mi = mi.copy()

    if median >= 0:
        cell_kdtree_3(data[:median+1, :], a, ori_mi, ma, depth + 1, root, True)
    if median+1 <= row - 1:
#        print(temp_mi, ori_ma)
        cell_kdtree_3(data[median+1 :, :], a, temp_mi, ori_ma, depth + 1, root, False)

def cell_kdtree_4(data, indices, mi, ma, sum_reg, depth=0, bucket_size=3):
#    row, col = shape(data)
    row = np.size(indices)
    if row > bucket_size:
        axis = depth % np.shape(data) [1]
#        data = data[data[:,axis].argsort(0)]
        indices_2 = np.argsort(data[indices, axis])
        median = (row-1) // 2 # choose median    
        try:
            while data[indices[indices_2[median]], axis] == data[indices[indices_2[median + 1]], axis]:
                median += 1
        except IndexError:
            pass
        ori_mi = mi.copy()
        ori_ma = ma.copy()
        mi[axis] = data[indices[indices_2[median]], axis]
        ma[axis] = data[indices[indices_2[median]], axis]
        temp_mi = mi.copy()
        reg = np.sum(data[indices[indices_2[:median+1]], :], 0)
#       数值就不需要copy，不怕递归过程中被修改。
        return Wiki_Node(indices, ori_mi.copy(), ori_ma.copy(), sum_reg, row,  #这里优化
                     cell_kdtree_4(data, indices[indices_2[:median+1]], ori_mi.copy(), ma.copy(), reg, depth+1, bucket_size),
                     cell_kdtree_4(data, indices[indices_2[median+1 :]], temp_mi.copy(), ori_ma.copy(), sum_reg - reg, depth+1, bucket_size)
                     )  #妈的，这种形式的递归，ori_mi回不来，不是一开始就附了值貌似;因为需要 list.copy(), array.copy*()啊 as hole
    else:
        return Wiki_Leaf(data[indices, :], indices)

def cell_kdtree_5(data, indices, depth=0, bucket_size=1):
    row = np.size(indices)
    if row > bucket_size:
        axis = depth % np.shape(data) [1]
#        data = data[data[:,axis].argsort(0)]
        indices_2 = np.argsort(data[indices, axis])
        median = (row-1) // 2 # choose median    
        try:
            while data[indices[indices_2[median]], axis] == data[indices[indices_2[median + 1]], axis]:
                median += 1
        except IndexError:
            pass
        return Wiki_Node(indices, data[indices, :].min(0), data[indices, :].max(0), data[indices, :].sum(0), row,  #这里优化
                     cell_kdtree_5(data, indices[indices_2[:median+1]], depth+1, bucket_size),
                     cell_kdtree_5(data, indices[indices_2[median+1 :]], depth+1, bucket_size)
                     )  #妈的，这种形式的递归，ori_mi回不来，不是一开始就附了值貌似;因为需要 list.copy(), array.copy*()啊 as hole
    else:
        return Wiki_Leaf(data[indices, :], indices)

class Candidate_Center:
    def __init__(self, data):
        self.element = data.copy()
        self.k, self.col = np.shape(data)
        self.w = np.zeros((self.k, self.col))
        self.count = np.zeros((self.k))
    def __getitem__(self, k):
        return self.element[k, :]
    def __setitem__(self, k, j):
        self.element[k, :] = j
    def update(self):
        f = False
        for i in range(self.k):
            if self.count[i] != 0: 
                temp = self.w[i, :] / self.count[i]
                if np.any(self[i] != temp) :
                   f = True
                   self[i] = temp
                   
            else:
                print('fail, bad center')
                return 
        self.w = np.zeros((self.k, self.col))
        self.count = np.zeros((self.k))
        return f
    def __repr__(self):
        return str(self.element)
    
def euclidean(a, b, axis = 0):
    return np.sum(np.power(a - b, 2), axis)

def brute_z(point, center, blacklist):
    best = np.inf
    index = -1
    for i in range(center.k):
        if i not in blacklist:
            temp = euclidean(point, center[i])
            if temp < best:
                best = temp
                index = i
#            best, index = (temp, i) if temp < best
    return index
def brute_leaf(point, center, blacklist, index, table):
    row = np.shape(point.e) [0]
    for j in range(row):
        best = np.inf
        inde = -1
        for i in range(center.k):
            if i not in blacklist:
                temp = euclidean(point.e[j, :], center[i])
                if temp < best:
                    best = temp
                    inde = i
        center.w[inde, :] += point.e[j, :]
        center.count[inde] += 1
        index[point.i[j]] = inde
        table[point.i[j]] = best

def farther(z, z_good, u, col, count):
    temp = z - z_good
    count[0] += 2
    v = np.zeros((col))
    for i in range(col):
        v[i] = u.mi[i] if temp[i] < 0 else u.ma[i]
    return euclidean(z, v) >= euclidean(z_good, v)
            
def efficient(u, center_class, blacklist, index, table, count):
    if isinstance(u, Wiki_Leaf):
        brute_leaf(u, center_class, blacklist, index, table)
        count[0] += (center_class.k - len(blacklist)) * np.shape(u.e)[0]
    else:
        z_good = brute_z((u.mi + u.ma)/2, center_class, blacklist)
        count[0] += center_class.k - len(blacklist)
        for i in range(center_class.k):
            if i != z_good and i not in blacklist and farther(center_class[i], center_class[z_good], u, center_class.col, count):
                blacklist.append(i)
        if len(blacklist) == center_class.k - 1:
            center_class.w[z_good, :] += u.w
            center_class.count[z_good] += u.count
            index[u.indices] = z_good
        else:
            temp = blacklist.copy()
            efficient(u.left, center_class, blacklist, index, table, count)
            efficient(u.right, center_class, temp, index, table, count)
def compare_k_means(data, initial, index, table):
    c = Candidate_Center(initial.copy()) #赋值给类的一定得小心，copy()!!!
#    也可以在类的初始化中，copy()，copy万岁。
    brute_leaf(data, c, [], index, table)
    c.update()
#    print(c.w, c.count)
#    print(c)
    return c.element
def hunt_once(point, center, index, table):
    row = np.shape(point) [0]
    for j in range(row):
        best = np.inf
        inde = -1
        for i in range(center.k):
#            if i not in blacklist:
                temp = euclidean(point[j, :], center[i])
                if temp < best:
                    best = temp
                    inde = i
        center.w[inde, :] += point[j, :]
        center.count[inde] += 1
        index[[j]] = inde
        table[[j]] = best

def kdtree_kmeans(data, k, initial, bucket_size = 5, max_iteration = 200):
    total = 0
    row = np.shape(data)[0]
    cluster_changed = True
    cen = Candidate_Center(initial)
    kdtree = cell_kdtree_5(data, np.arange(np.shape(data)[0]), 0, bucket_size)
    index = np.zeros(row, int)
    count = [0]
    while total < max_iteration and cluster_changed:
#        cluster_changed = False
        total += 1


        # print('这是kdtree的第', total, '次迭代')

#        print(total)
        table = np.zeros(row)
        efficient(kdtree, cen, [], index, table, count)
        cluster_changed = cen.update()
    for i in range(np.shape(data)[0]):
        if table[i] == 0:
            table[i] = euclidean(data[i, :], cen[index[i]])
            count[0] += 1
#    my_plot(data, cen.element, index, table, 0, 'kdtree_kmeans')
    print(total)
    return index, table, count[0]

import sys
from matplotlib import *
#from time import time
#import random
from numpy.random import rand, randn
if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    a=[1,2,3,4]
#    a.sort(key=itemgetter(axis))
#    a = mat([[1,2],[6,8]])
#    b = mat([[4,2],[7,-3]])
#    a=mat([[1,6],[7,0],[-6,9]])
    a = np.array([[-4,-6],[0,0],[0,0],[1,1],[4,4], [3,9]])
#    a = np.array([[-4,6],[0,0],[0,0],[1,1],[4,4], [3,9],[-1,-1],[2,2]])
    data = np.load('iris.npy')
    a = np.unique(a,axis=0)
#    initial = array([[-1,1],[3,9],[3,2]])
#    initial = a[[100, 40,53,22], :]
    a = rand(6,4)
    dim = 4
    row = 100
    k = 3
#    a = array([[random.gauss(0,1) for j in range(dim)] for i in range(row)])
#    initial = array([[random.gauss(0,1) for j in range(dim)] for i in range(k)])
    a = rand(row, dim)            
    initial = rand(k, dim)
    
    
#    c = compare_k_means(a, initial.copy())
#    print(c)
#    print('*'*20)
    
    
    
#    t = Tree()
#    c = cell_kdtree(a, t)
#    b = cell_kdtree_2(a)
#    tt = Tree()
#    d = cell_kdtree_3(a, tt, (a.min(0)), (a.max(0)))
    b = cell_kdtree_4(a, np.arange(np.shape(a)[0]), a.min(0), a.max(0), a.sum(0), 0, 1)
    c = cell_kdtree_5(a, np.arange(np.shape(a)[0]))
#    b = cell_kdtree_2(a)

#     cen = Candidate_Center(initial)
#     index = np.zeros(row, int)
#     table = np.zeros(row)
#     # efficient(b, cen, [], index, table)
#     cen.update()
#
#     cen_2 = Candidate_Center(initial)
#     index_2 = np.zeros(row, int)
#     table_2 = np.zeros(row)
#     hunt_once(a, cen_2, index_2, table_2)
#     cen_2.update()
#
    
    
#    cen.update()
#    print(cen.element - c)
#    print(cen_2.element-cen.element)
    ini = plus_plus(data, 3)
    i, j, t = kdtree_kmeans(data, k, ini)
    