# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:35:42 2019

@author: Administrator
"""
import numpy as np
from center_method import *
from line_three import *
from distance_method import *

def get_cen(data):
    k = np.shape(data) [0]
    cen = np.ones((k, k)) * np.inf
    for i in range(k):
        for j in range(i + 1, k):
            cen[i, j] = euclidean(data[i, :], data[j, :])
    for i in range(1, k):
        for j in range(0, i ):
            cen[i, j] = cen[j, i]
    return cen
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

def elkan_square(data, k, center, max_iteration = 200, dist_method = euclidean):
#def elkan(data, k, center_method=plus_plus, dist_method = euclidean, max_iteration = 100):
    row = np.shape(data) [0]
    table = np.zeros((row))
    index = np.zeros((row), int)
#    center = center_method(data, k)
    low = np.zeros((row, k))
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 0
    for i in range(row):
        low[i, :] = dist_method(np.ones((k, 1)) * np.mat(data[i, :]), center, 1).T
        index[i] = low[i, :].argmin()
#        table[i] = low[np.arange(row), index]
        table[i] = low[i, index[i]]
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    new_center = np.zeros(np.shape(center))
    for i in range(k):
        if cen_num[i] == 0:
            print('*'*10+'fail'+'*'*10)
            return
        else:
            new_center[i, :] = cen_sum[i, :] / cen_num[i]
    reg = dist_method(new_center, center, 1) #matric的话就是3*1； 而array就自然得变成了1*3
#    return reg
    for i in range(row):
        table[i] += reg[index[i]] + 2 * np.sqrt(reg[index[i]] * table[i])
        for j in range(k):
            low[i, j] = low[i, j] + reg[j] -2 * np.sqrt(reg[j]*low[i,j]) if low[i,j] > reg[j] else 0
    center = new_center.copy()
    cluster_changed = True 
    while cluster_changed and total < max_iteration:
        cluster_changed = False;
        total += 1;
        cen_comp = get_cen(center)
        s = cen_comp.min(1)
        good = np.nonzero(table[:] <= s[index[:]] / 4) [0]
#        return good
        for i in range(row):
            temp = index[i]
            for j in range(k):
                if i not in good and j != index[i] and table[i] > low[i, j] and table[i] > cen_comp[index[i], j] / 4:
                    table[i] = dist_method(data[i, :], center[index[i], :])
                    low[i, index[i]] = table[i]
                    if table[i] > low[i, j] and table[i] > cen_comp[index[i], j] / 4:
                        low[i, j] = dist_method(data[i, :], center[j, :])
                        if low[i, j] < table[i]:
                            table[i] = low[i, j]
                            index[i] = j
#                            cluster_changed = True
            if temp != index[i]:
                cluster_changed = True
                cen_num[temp] -= 1
                cen_num[index[i]] += 1
                cen_sum[temp, :] -= data[i, :]
                cen_sum[index[i], :] += data[i, :]
                
        if cluster_changed:
            for i in range(k):
                if cen_num[i] == 0:
                    print('*'*10+'fail'+'*'*10)
                    return
                else:
                    new_center[i, :] = cen_sum[i, :] / cen_num[i]
            reg = dist_method(new_center, center, 1) #matric的话就是3*1； 而array就自然得变成了1*3
        #    return reg
            for i in range(row):
                table[i] += reg[index[i]] + 2 * np.sqrt(reg[index[i]] * table[i])
                for j in range(k):
                    low[i, j] = low[i, j] + reg[j] -2 * np.sqrt(reg[j]*low[i,j]) if low[i,j] > reg[j] else 0
            center = new_center.copy()
    table = dist_method(data, center[index[:], :], 1)
    print(total+1)
    return index, table
#    my_plot(data, center, index, table, 0, 'elkan_sqrt')
def elkan_sqrt(data, k, center, max_iteration=200):
    row = np.shape(data)[0]
    table = np.zeros((row))  # high
    index = np.zeros((row), int)
    low = np.zeros((row, k))
    cen_sum = np.zeros(np.shape(center))  # static information
    cen_num = np.zeros((k))
    total = 1  # iteration times
    for i in range(row):
        # low[i, :] = np.sqrt(dist_method(data[i, :], center, 1))  # TODO 貌似，变成了一行，默认array了
        low[i, :] = np.linalg.norm(data[i, :] -  center, axis = 1)
        # low[i, :] = np.sqrt(dist_method(np.ones((k, 1)) * np.mat(data[i, :]), center, 1)).T
        index[i] = low[i, :].argmin()
        #        table[i] = low[:, index] #todo 这样不得行，[]是对整体的索引，无法拆分到每一行吧
        table[i] = low[i, index[i]]
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    # new_center = np.zeros(np.shape(center))
    reg = np.zeros(k)
    cluster_changed = True
    count = row * k
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            # if cen_num[i] == 0:
            #     print('*' * 10 + 'fail' + '*' * 10)
            #     return -1, -1
            # else:
            if cen_num[i] != 0:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            reg[i] = np.linalg.norm(center[i, :] - temp)
        # reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        for i in range(row):
            table[i] += reg[index[i]]
            for j in range(k):
                # low[i, j] = max(low[i, j] - reg[j], 0)
                low[i, j] = low[i, j] - reg[j]
        # center = new_center.copy()  # todo copy牛逼

        cluster_changed = False
        total += 1
        # print('这是Elkan的第', total, '次迭代')
        cen_comp = get_cen_sqrt(center)
        s = cen_comp.min(1)
        good = np.nonzero(table[:] <= s[index[:]] )[0]
        for i in range(row):
            if i not in good:
                dist_calculate = True
                temp = index[i]
                for j in range(k):
                     if j != index[i] and table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                         if dist_calculate:
                            table[i], dist_calculate = np.linalg.norm(data[i, :] - center[index[i], :]), False
                            low[i, index[i]] = table[i]
                            count += 1
                         # low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                         low[i, j] = np.linalg.norm(data[i, :] - center[j, :])
                         count += 1
                        # if table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                        #     low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                         if low[i, j] < table[i]:
                            table[i], index[i] = low[i, j], j
                if temp != index[i]:
                    cluster_changed = True
                    cen_num[temp] -= 1
                    cen_num[index[i]] += 1
                    cen_sum[temp, :] -= data[i, :]
                    cen_sum[index[i], :] += data[i, :]
    table = np.linalg.norm(data - center[index[:], :], axis = 1) ** 2
    count += row
    print(total)
    return index, table, count

def elkan_hyperbola(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row))  # high
    index = np.zeros((row), int)
    low = np.zeros((row, k))
    cen_sum = np.zeros(np.shape(center))  # static information
    cen_num = np.zeros((k))
    total = 1  # iteration times
    for i in range(row):
        low[i, :] = np.sqrt(dist_method(data[i, :], center, 1))  # 貌似，变成了一行，默认array了
        # low[i, :] = np.sqrt(dist_method(np.ones((k, 1)) * np.mat(data[i, :]), center, 1)).T
        index[i] = low[i, :].argmin()
        #        table[i] = low[:, index] # 这样不得行，[]是对整体的索引，无法拆分到每一行吧
        table[i] = low[i, index[i]]
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    new_center = np.zeros(np.shape(center))
    #todo make hyperbola
    cen_low = np.zeros((k, k))
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                new_center[i, :] = cen_sum[i, :] / cen_num[i]
        reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        for i in range(k):
            # norm_i = np.linalg.norm(center[i, :])
            norm_i = sum(np.power(center[i, :], 2))
            m_i = max(table[index == i])
            for j in range(k):
                if reg[j] == 0:
                    cen_low[i, j] = 0
                    continue
                if j != i:
                    t = (center[i, :] - center[j, :]).dot(new_center[j, :] - center[j, :]) / reg[j] ** 2
                    xc = np.linalg.norm(center[j, :] + t * (new_center[j, :] - center[j, :]) - center[i, :]) * 2 / \
                         reg[j]
                    yc = 1 - 2 * t
                    r = m_i * 2 / reg[j]
                    if xc <= r:
                        cen_low[i, j] = max(0, min(2, 2 * (r - yc))) * reg[j] / 2
                        continue
                    if yc > r:
                        yc -= 1
                    if norm_i > r ** 2:
                        cen_low[i, j] = (xc * r - yc * np.sqrt(norm_i - r ** 2)) / norm_i * reg[j]
                    else:
                        # print('这里出事了')
                        # print((xc * r ) / norm_i )
                        cen_low[i, j] = reg[j]
        for i in range(row):
            table[i] += reg[index[i]]
            for j in range(k):
                if j != index[i]:
                    low[i, j] -= cen_low[index[i], j]
        center = new_center.copy()  # copy牛逼
        cluster_changed = False
        total += 1
        cen_comp = get_cen_sqrt(center)
        s = cen_comp.min(1)
        good = np.nonzero(table[:] <= s[index[:]])[0]
        for i in range(row):
            if i not in good:
                dist_calculate = True
                temp = index[i]
                for j in range(k):
                     if j != index[i] and table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                         if dist_calculate:
                            table[i], dist_calculate = np.sqrt(dist_method(data[i, :], center[index[i], :])), False
                            low[i, index[i]] = table[i]
                         low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                        # if table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                        #     low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                         if low[i, j] < table[i]:
                            table[i], index[i] = low[i, j], j
                if temp != index[i]:
                    cluster_changed = True
                    cen_num[temp] -= 1
                    cen_num[index[i]] += 1
                    cen_sum[temp, :] -= data[i, :]
                    cen_sum[index[i], :] += data[i, :]
        # if cluster_changed:
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table
def elkan_hyperbola_neighbor(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row))  # high
    index = np.zeros((row), int)
    low = np.zeros((row, k))
    cen_sum = np.zeros(np.shape(center))  # static information
    cen_num = np.zeros((k))
    total = 1  # iteration times
    for i in range(row):
        low[i, :] = np.sqrt(dist_method(data[i, :], center, 1))  # 貌似，变成了一行，默认array了
        index[i] = low[i, :].argmin()
        table[i] = low[i, index[i]]
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    new_center = np.zeros(np.shape(center))
    #todo make hyperbola
    cen_low = np.zeros((k, k))
    # todo make neighbor
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                new_center[i, :] = cen_sum[i, :] / cen_num[i]
        cen_comp = get_cen_sqrt(new_center)
        reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        neighbor = []
        for i in range(k):
            norm_i = sum(np.power(center[i, :], 2))
            m_i = max(table[index == i])
            neighbor.append(np.nonzero(m_i > cen_comp[i, :])[0])  #小于等于不用管了，大于的话就要管
            for j in range(k):
                if reg[j] == 0:
                    cen_low[i, j] = 0
                    continue
                if j != i:
                    t = (center[i, :] - center[j, :]).dot(new_center[j, :] - center[j, :]) / reg[j] ** 2
                    xc = np.linalg.norm(center[j, :] + t * (new_center[j, :] - center[j, :]) - center[i, :]) * 2 / \
                         reg[j]
                    yc = 1 - 2 * t
                    r = m_i * 2 / reg[j]
                    if xc <= r:
                        cen_low[i, j] = max(0, min(2, 2 * (r - yc))) * reg[j] / 2
                        continue
                    if yc > r:
                        yc -= 1
                    if norm_i > r ** 2:
                        cen_low[i, j] = (xc * r - yc * np.sqrt(norm_i - r ** 2)) / norm_i * reg[j]
                    else:
                        # print('这里出事了')
                        cen_low[i, j] = reg[j]
        for i in range(row):
            table[i] += reg[index[i]]
            for j in range(k):
                if j != index[i]:
                    low[i, j] -= cen_low[index[i], j]
        center = new_center.copy()  # copy牛逼
        cluster_changed = False
        total += 1
        # cen_comp = get_cen_sqrt(center)
        s = cen_comp.min(1)
        good = np.nonzero(table[:] <= s[index[:]])[0]
        for i in range(row):
            if i not in good:
                dist_calculate = True
                temp = index[i]
                for j in range(k):
                     if j != index[i] and table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                         if dist_calculate:
                            table[i], dist_calculate = np.sqrt(dist_method(data[i, :], center[index[i], :])), False
                            low[i, index[i]] = table[i] #每次计算就更新上吧，万一下次迭代就用上了
                         low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                        # if table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                        #     low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                         if low[i, j] < table[i]:
                            table[i], index[i] = low[i, j], j
                if temp != index[i]:
                    cluster_changed = True
                    cen_num[temp] -= 1
                    cen_num[index[i]] += 1
                    cen_sum[temp, :] -= data[i, :]
                    cen_sum[index[i], :] += data[i, :]
        # if cluster_changed:
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table
def elkan_hyperbola_neighbor_2(data, k, center, max_iteration=200, dist_method=euclidean):
    #修了good，我觉得还是这样好
    row = np.shape(data)[0]
    table = np.zeros((row))  # high
    index = np.zeros((row), int)
    low = np.zeros((row, k))
    cen_sum = np.zeros(np.shape(center))  # static information
    cen_num = np.zeros((k))
    total = 1  # iteration times
    for i in range(row):
        low[i, :] = np.sqrt(dist_method(data[i, :], center, 1))  # 貌似，变成了一行，默认array了
        index[i] = low[i, :].argmin()
        table[i] = low[i, index[i]]
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    new_center = np.zeros(np.shape(center))
    #todo make hyperbola
    cen_low = np.zeros((k, k))
    # todo make neighbor
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                new_center[i, :] = cen_sum[i, :] / cen_num[i]
        cen_comp = get_cen_sqrt(new_center)
        reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        # neighbor = []
        for i in range(k):
            norm_i = sum(np.power(center[i, :], 2))
            m_i = max(table[index == i])
            # neighbor.append(np.nonzero(m_i >= cen_comp[i, :] )[0])
            for j in range(k):
                if reg[j] == 0:
                    cen_low[i, j] = 0
                    continue
                if j != i:
                    t = (center[i, :] - center[j, :]).dot(new_center[j, :] - center[j, :]) / reg[j] ** 2
                    xc = np.linalg.norm(center[j, :] + t * (new_center[j, :] - center[j, :]) - center[i, :]) * 2 / \
                         reg[j]
                    yc = 1 - 2 * t
                    r = m_i * 2 / reg[j]
                    if xc <= r:
                        cen_low[i, j] = max(0, min(2, 2 * (r - yc))) * reg[j] / 2
                        continue
                    if yc > r:
                        yc -= 1
                    if norm_i > r ** 2:
                        cen_low[i, j] = (xc * r - yc * np.sqrt(norm_i - r ** 2)) / norm_i * reg[j]
                    else:
                        # print('这里出事了')
                        cen_low[i, j] = reg[j]
        # for i in range(row):
        center = new_center.copy()  # copy牛逼
        cluster_changed = False
        total += 1
        # cen_comp = get_cen_sqrt(center)
        # s = cen_comp.min(1)
        # good = np.nonzero(table[:] <= s[index[:]])[0]
        for i in range(row):
                table[i] += reg[index[i]]
                for j in range(k):
                    if j != index[i]:
                        low[i, j] -= cen_low[index[i], j]
                temp = index[i]
                neighbor = [j for j in range(k) if table[i] > cen_comp[index[i], j]] #这里同时把good 优化了
                dist_cal = True
                for j in neighbor:
                        # if  table[i] > low[i, j]  :
                            if table[i] > low[i, j] and table[i] > cen_comp[index[i], j] :
                                if dist_cal:
                                    table[i], dist_cal = np.sqrt(dist_method(data[i, :], center[index[i], :])), False
                                    low[i, index[i]] = table[i]
                                low[i, j] = np.sqrt(dist_method(data[i, :], center[j, :]))
                                if low[i, j] < table[i]:
                                    table[i], index[i] = low[i, j], j
                                    # index[i] = j
                if temp != index[i]:
                    cluster_changed = True
                    cen_num[temp] -= 1
                    cen_num[index[i]] += 1
                    cen_sum[temp, :] -= data[i, :]
                    cen_sum[index[i], :] += data[i, :]
        # if cluster_changed:
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table

#    my_plot(data, center, index, table, 0, 'elkan_sqrt')
from center_method import plus_triangle
from k_means import  naive_k_means
from box_kdtree import kdtree_kmeans
from drake_b import  drake
from harmerly_triangle import  hamerly_sqrt
import time
if __name__ == '__main__':
    # data  = np.load('iris.npy')
    data = np.random.rand(2000, 4)
    k = 5
    center=plus_triangle(data, k)
    # # elkan_square(data , k, center)
    i, j, c = elkan_sqrt(data , k, center.copy())
    # # i2, j2 = kdtree_kmeans(data, k, center.copy())
    i3, j3, c2, tt2 = hamerly_sqrt(data , k, center.copy())
    # i4, j4 = naive_k_means(data, k, center.copy())
    # print(sum(i4) - sum(i), sum(j) - sum(j4))
    # i5, j5 = elkan_hyperbola(data, k, center.copy())
    # print('plus neighbor'+'*'*20)
    # i6, j6, c3 = drake(data, k, center.copy())
    # i7, j7 = elkan_hyperbola_neighbor_2(data, k, center.copy())
    print(sum(i3) - sum(i), sum(j) - sum(j3))
    # print(sum(i6) - sum(i), sum(j) - sum(j6))
    # print(sum(i7) - sum(i), sum(j) - sum(j7))

    # data = np.load('birch_data.npy')
    # t = []
    # dim = [4, 8, 16, 32, 64]
    # kk = [2, 4, 8, 16, 32, 64, 128]
    # # kk = [8, 16, 32, 64, 128]
    # # kk = [2,4]
    # k = 30
    # # data = np.random.randn(2000,1)*2
    # # center=plus_triangle(data, k)
    # # # a = time.process_time()
    # # ii, jj = drake(data , k, center.copy())
    # # # b = time.process_time()
    # # i2, j2 = naive_k_means(data, k, center.copy())
    # # print(np.sum(i2 - ii), np.sum(j2)-np.sum(jj))
    # # kk=[9]
    # for k in kk:
    #     # data = np.random.randn(2000,1)*2
    #     #        k = 13
    #     print(k, '维了')
    #     center = plus_triangle(data, k)
    #
    #     a = time.process_time()
    #     ii, jj = elkan_sqrt(data, k, center.copy())
    #     b = time.process_time()
    #     t.append(b - a)
    #
    #     a = time.process_time()
    #     i2, j2 = elkan_hyperbola(data, k, center.copy())
    #     b = time.process_time()
    #     t.append(b - a)
    #     print(np.sum(i2 - ii), np.sum(j2) - np.sum(jj))
    #
    #     a = time.process_time()
    #     i3, j3 = elkan_hyperbola_neighbor(data, k, center.copy(), 50)
    #     b = time.process_time()
    #     t.append(b - a)
    #     print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
    #
    #     a = time.process_time()
    #     i4, j4 = elkan_hyperbola_neighbor_2(data, k, center.copy())
    #     b = time.process_time()
    #     t.append(b - a)
    #     print(np.sum(i4 - ii), np.sum(j4) - np.sum(jj))
    #
    # # kk=dim
    # plt.plot(kk, t[0::4], label='Elkan', c='k', linewidth=2)
    # plt.plot(kk, t[1::4], '--', label='Elkan_hyperbola', c='b', linewidth=2)
    # plt.plot(kk, t[2::4], '-+', label='Elkan_all', c='r', linewidth=2)
    # plt.plot(kk, t[3::4], '-<', label='Elkan_my_neighbor', c='g', linewidth=2)
    # #
    # ##    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
    # ##    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
    # ##
    # plt.legend()
    # plt.grid()
    # plt.xlabel('k')
    # ##     plt.xlabel('number')
    # plt.ylabel('time')
    # plt.show()
    #

