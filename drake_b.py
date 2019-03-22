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
    k = np.shape(data)[0]
    cen = np.ones((k, k)) * np.inf
    for i in range(k):
        for j in range(i + 1, k):
            cen[i, j] = np.linalg.norm(data[i, :] - data[j, :]) / 2
    for i in range(1, k):
        for j in range(0, i ):
            cen[i, j] = cen[j, i]
    return cen

def point_all_center(x, c, b, d = euclidean):
    # temp = d(x, c, 1)
    temp = np.linalg.norm(x - c, axis = 1)
    arg = temp.argsort()
    index = arg[0]
    low_index = arg[1:b+1]
    table = temp[arg[0]]
    low = temp[arg[1:b+1]]
    return index, (table), (low), low_index
def point_partial_center(x, c, y, a, d = euclidean):
    comb = np.hstack((y, a))
    b = np.size(comb)
    temp = np.linalg.norm(x - c[comb, :], axis=1)
    # temp = np.sqrt(d(x, c[comb, :], 1))
    arg = temp.argsort()
    index = comb[arg[0]]
    low_index = comb[arg[1:b]]
    table = temp[arg[0]]
    low = temp[arg[1:b]]
    return index, table, low, low_index


def drake(data, k, center, max_iteration = 200):
    row = np.shape(data)[0]
    table = np.zeros((row))#high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    b = int(np.ceil(k/4))
    low = np.zeros((row, b))
    low_index = np.zeros((row, b), int)
    for i in range(row):
        index[i], table[i], low[i, :], low_index[i, :] = point_all_center(data[i, :], center, b)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
    count = row * k
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            # if cen_num[i] == 0:
            #     print('*' * 10 + 'fail' + '*' * 10)
            #     return -1, -1
            # else:
            if cen_num[i] != 0:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.linalg.norm(center[i, :] - temp)
        first = np.max(p)
        cluster_changed = False
        # print('这是Drake的第', total, '次迭代')
        total += 1
        s = get_cen_sqrt(center).min(1)
        m = -1
        for i in range(row):
            table[i] += p[index[i]]
            low[i, b - 1] -= first
            for z in range(b - 2, -1, -1):
                low[i, z] = np.min((low[i, z] - p[low_index[i, z]], low[i, z + 1])) #propagate,
            if table[i] > s[index[i]] :
                    temp = index[i]
                    flag = True
                    for z in range(b):
                        if table[i] <= low[i, z]:
                            flag = False    
                            if z > 0:
                                index[i], table[i], low[i, :z], low_index[i, :z] = point_partial_center(data[i, :], center, index[i], low_index[i, :z]) #为何我用z+1就有误差了？
                                count += z
                            break
                    if flag:
                        index[i], table[i], low[i, :b], low_index[i, :b] = point_all_center(data[i, :], center, b)
                        count += k
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
                    m = max(m, z+1)
        b = max(int(np.ceil(k/8)), m)
    table = np.linalg.norm(data - center[index[:], :], axis = 1) ** 2
    count += row
    print(total)
    return index, table, count

from harmerly_triangle import hamerly_sqrt
from k_means import naive_k_means
from elkan_triangle import elkan_sqrt
import time
from center_method import plus_triangle
from box_kdtree import kdtree_kmeans
if __name__ == '__main__':
#     data  = np.load('iris.npy')
#     data  = np.load('birch_data.npy')
    t =[]
    # dim = [2,4,8,16,32,64]
    # kk = [2,4,6, 8,12, 16,22, 32,64,128, 256, 512]
    # kk = [128, 148, 168, 188, 208, 238, 268, 298,356, 400]
    # kk = [12,17,22, 27,32,37]
    # kk = [2,3,4,5,6, 7,8,9,10]
    # kk = [4,8,16,32,64,128]
    # kk = [2,4]
    k=3
    data = np.random.rand(2000,3)
    center=plus_triangle(data, k)
    # # a = time.process_time()
    ii, jj, c1 = drake(data , k, center.copy())
    # # b = time.process_time()
    i2, j2 = naive_k_means(data, k, center.copy())
    print(np.sum(i2 - ii), np.sum(j2)-np.sum(jj))

    # kk=[160, 200, 256, 512]

#todo all triangle styles

#     count = []
#     for dim in kk:
#         data = np.random.rand(6800,dim)
# #        k = 13
#         print(dim,'维了')
#         center=plus_triangle(data, k)
#
#         a = time.process_time()
#         ii, jj, ch, original = hamerly_sqrt(data, k, center.copy())
#         b = time.process_time()
#         t.append(b - a)
#
#         a = time.process_time()
#         i1, j1 = naive_k_means(data, k, center.copy())
#         b = time.process_time()
#         t.append(b - a)
#         print(np.sum(i1 - ii), np.sum(j1) - np.sum(jj))
#
#         a = time.process_time()
#         i2, j2, ce = elkan_sqrt(data, k, center.copy())
#         b = time.process_time()
#         t.append(b - a)
#         print(np.sum(i2 - ii), np.sum(j2) - np.sum(jj))
#
#         a = time.process_time()
#         i3, j3, ctree = kdtree_kmeans(data, k, center.copy(), 50)
#         b = time.process_time()
#         t.append(b - a)
#         print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
#
#         a = time.process_time()
#         i4, j4, cd = drake(data , k, center.copy())
#         b = time.process_time()
#         t.append(b-a)
#         print(np.sum(i4 - ii), np.sum(j4) - np.sum(jj))
#
#         count.append(ch)
#         count.append(original)
#         count.append(ce)
#         count.append(ctree)
#         count.append(cd)
# # kk=dim
#     plt.plot(kk, t[2::5], label='Elkan', c='k',linewidth=2)
#     plt.plot(kk, t[1::5], '--',label='standard', c='b',linewidth=2)
#     plt.plot(kk, t[0::5], '-+',label='Hamerly', c='r',linewidth=2)
#     # plt.plot(kk, t[3::5], '-<',label='kdtree', c='g',linewidth=2)
#     plt.plot(kk, t[4::5], '->',label='Drake', c='y',linewidth=2)
#     #
# ##    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
# ##    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
# ##
#     plt.legend()
#     plt.grid()
#     plt.xlabel('dimensional')
#     ##     plt.xlabel('number')
#     plt.ylabel('distance calculations')
#     # plt.ylabel('time')
#     plt.show()

#        print(np.sum(i - ii), np.sum(j)-np.sum(jj))

    #

    #    i = drake(data , k, center)


#[4,8,16,32,64]簇，全，以下均为高斯。
# [10.359375,
#  32.859375,
#  65.59375,
#  2.890625,
#  11.5,
#  40.15625,
#  461.0625,
#  515.71875,
#  25.875,
#  95.625,
#  199.453125,
#  1763.96875,
#  786.765625,
#  87.984375,
#  402.609375,
#  702.640625,
#  7337.625,
#  1769.296875,
#  776.75,
#  940.953125]

# 2\4个簇，全
# [12.609375,
#  31.75,
#  54.125,
#  2.9375,
#  11.84375,
#  25.265625,
#  81.015625,
#  140.203125,
#  5.90625,
#  19.859375]

# [2,4,8,16,32,64,128]簇，少naive
#
# [12.609375,
#  31.75,
#  2.9375,
#  11.84375,
#  25.265625,
#  81.015625,
#  5.90625,
#  19.859375,
# 45.453125,
#  384.0,
#  24.46875,
#  87.890625,
#  92.96875,
#  589.71875,
#  51.25,
#  251.125,
#  285.796875,
#  1272.78125,
#  174.265625,
#  910.796875,
#  475.9375,
#  1794.984375,
#  518.609375,
#  989.671875,
#  927.25,
#  1905.0625,
#  1170.28125,
#  1423.6875]

    #
    # plt.plot(kk, t[1::4], label='Elkan', c='k',linewidth=2)
    # # plt.plot(kk, t[1::5], '--',label='standard', c='b',linewidth=2)
    # plt.plot(kk, t[0::4], '-+',label='Hamerly', c='r',linewidth=2)
    # plt.plot(kk, t[2::4], '-<',label='kdtree', c='g',linewidth=2)
    # plt.plot(kk, t[3::4], '->',label='Drake', c='y',linewidth=2)
    # plt.legend()
    # plt.grid()
    # plt.xlabel('k')
    # ##     plt.xlabel('number')
    # plt.ylabel('time')
    # plt.show()


