# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:35:42 2019

@author: Administrator
"""
import numpy as np
from center_method import *
# from line_three import *
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
def point_all_center_sqrt(x, c):
    # first, second, index = d(x, c[0, :]), d(x, c[1, :]), 0
    t = x - c[0, :]
    first, index = np.dot(t, t), 0
    t = x - c[1, :]
    second = np.dot(t, t)
    if first > second:
        first, second, index = second, first, 1
    for i in range(2, np.shape(c)[0]):
        t = x - c[i, :]
        temp = np.dot(t, t)
        if temp < first:
            first, second, index = temp, first, i
        elif temp < second:
            second = temp
    return index, np.sqrt(first), np.sqrt(second)
def point_not_all_center_sqrt(x, c, ind, ind_value, d = euclidean):
    li = list(range(np.shape(c)[0]))
    li.remove(ind)
    # first, second, index = ind_value ** 2, d(x, c[li[0], :]), ind  #哈哈哈，又是开根
    first, second, index = ind_value , np.linalg.norm(x - c[li[0], :]), ind  #哈哈哈，又是开根
    if first > second:
        first, second, index = second, first, li[0]
    for i in li[1:]:
        # temp = d(x, c[i, :])
        temp = np.linalg.norm(x - c[i, :])
        if temp < first:
            first, second, index = temp, first, i
        elif temp < second:
            second = temp
    # return index, np.sqrt(first), np.sqrt(second)
    return index, first, second

def point_all_center_sqrt_neighbor(x, c, j, value, d = euclidean):
    temp = c[j, :]
    first, second, index = value ** 2, d(x, temp[0, :]), -1 #2333，又是开根的锅
    if first > second:
        first, second, index = second, first, 0
    for i in range(1, np.shape(temp)[0] - 1):
        reg = d(x, temp[i, :])
        if reg < first:
            first, second, index = reg, first, i
        elif reg < second:
            second = reg
    return j[index], np.sqrt(first), np.sqrt(second)

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

def hamerly_sqrt(data, k, center, max_iteration=200):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    count = row * k
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
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
            # p[i] = np.sqrt(dist_method(center[i, :], temp))
            p[i] = np.linalg.norm(center[i, :] - temp)
        first, second = max_2_index(p)
        cluster_changed = False
        total += 1
        # print('这是Hamerly的第', total, '次迭代')
        s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i, 1] += p[index[i]]
            if first == index[i]:
                table[i, 0] -= p[second]
            else:
                table[i, 0] -= p[first] #更新bound，省下了一个for
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.linalg.norm(data[i, :] - center[index[i], :])
                count += 1
                if table[i, 1] > m:
                    temp = index[i]
                    index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
                    count += k
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
    # table = dist_method(data, center[index[:], :], 1)
    table = np.linalg.norm(data - center[index[:], :], axis = 1) ** 2
    count += row
    table_reg = sum(table)
    print(total, table_reg)
    return index, table_reg, table, count, total*row*k
def hamerly_sqrt_new(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    count = row * k
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        cluster_changed = False
        total += 1
        # print('这是Hamerly的第', total, '次迭代')
        s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i, 1] += p[index[i]]
            if first == index[i]:
                table[i, 0] -= p[second]
            else:
                table[i, 0] -= p[first] #更新bound，省下了一个for
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i, 1] > m:
                    temp = index[i]
                    index[i], table[i, 1], table[i, 0] = point_not_all_center_sqrt(data[i, :], center, temp, table[i, 1])
                    count += k-1
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
    table = dist_method(data, center[index[:], :], 1)
    count += row
    print(total)
    return index, table, count, total*row*k

def hamerly_hyperbola(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    # p = np.zeros(k)
    new_center = np.zeros(np.shape(center))
    cen_low = np.zeros((k, k))
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)  # handle fail
                return -1, -1
            else:
                new_center[i, :] = cen_sum[i, :] / cen_num[i]
        reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        # cen_low = np.zeros((k, k))
        for i in range(k):
            norm_i = sum(np.power(center[i, :], 2))
            m_i = max(table[index == i, 1])
            for j in range(k):
                if j != i:
                    if reg[j] == 0:
                        cen_low[i, j] = 0
                        # print('le cao asd')
                        continue
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
                        cen_low[i, j] = reg[j]
                    # if cen_low[i, j] < 0:
                        # print('cao le asdas ')
        cen_index = np.max(cen_low, 1)
        # for i in range(row):
        center = new_center.copy()
        cluster_changed = False
        total += 1
        s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i, 1] += reg[index[i]]
            table[i, 0] -= cen_index[index[i]]
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                if table[i, 1] > m:
                    temp = index[i]
                    index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
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

def hamerly_hyperbola_neighbor(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    new_center = np.zeros(np.shape(center))
    cen_low = np.zeros((k))
    neighbor = [None] * k
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)  # handle fail
                return -1, -1
            else:
                new_center[i, :] = cen_sum[i, :] / cen_num[i]
        reg = np.sqrt(dist_method(new_center, center, 1))  # matric的话就是3*1； 而array就自然得变成了1*3
        cen_comp = get_cen_sqrt(new_center)
        s = np.min(cen_comp, 1)
        for i in range(k):
            norm_i = sum(np.power(center[i, :], 2))
            m_i = max(table[index == i, 1])
            neighbor[i] = [j for j in range(k) if m_i + s[i] >= cen_comp[i, j]]
            cen_low[i] = - np.inf
            for j in neighbor[i]:
                # for j in range(k):
                #     if j != i :
                if reg[j] <= cen_low[i]:
                    continue
                if reg[j] == 0:
                    cen_low[i] = max(0, cen_low[i])
                    continue
                t = (center[i, :] - center[j, :]).dot(new_center[j, :] - center[j, :]) / reg[j] ** 2
                xc = np.linalg.norm(center[j, :] + t * (new_center[j, :] - center[j, :]) - center[i, :]) * 2 / reg[
                    j]
                yc = 1 - 2 * t
                r = m_i * 2 / reg[j]
                if xc <= r:
                    # cen_low[i, j] = max(0, min(2, 2*(r - yc))) * reg[j] / 2
                    cen_low[i] = max(cen_low[i], max(0, min(2, 2 * (r - yc))) * reg[j] / 2)
                    continue
                if yc > r:
                    yc -= 1
                if norm_i > r ** 2:
                    cen_low[i] = max(cen_low[i], (xc * r - yc * np.sqrt(norm_i - r ** 2)) / norm_i * reg[j])
                else:
                    cen_low[i] = max(cen_low[i], reg[j])
        # for i in range(row):
        center = new_center.copy()
        cluster_changed = False
        total += 1
        for i in range(k):
            neighbor[i].append(i) #todo 机智好吧
        for i in range(row):
            table[i, 1] += reg[index[i]]
            # todo 这里的max，或者说hamerly用的first，second不错的学问，逻辑
            table[i, 0] -= cen_low[index[i]]
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                if table[i, 1] > m:
                    temp = index[i]
                    # index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
                    index[i], table[i, 1], table[i, 0] = point_all_center_sqrt_neighbor(data[i, :], center, neighbor[index[i]])
                    #todo 和annular比比看，这样子是严谨的，但能提升多少？
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

def hamerly_neighbor(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
    cluster_changed = True
    count = row * k
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        cen_comp = get_cen_sqrt(center)
        s = np.min(cen_comp, 1)
        cluster_changed = False
        total += 1
        for i in range(row):
            table[i, 1] += p[index[i]]
            if first == index[i]:
                table[i, 0] -= p[second]
            else:
                table[i, 0] -= p[first]
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i, 1] > m:
                    # temp = index[i]
                    neighbor = [j for j in range(k) if table[i, 1] + s[index[i]] >= cen_comp[index[i], j]] #cen_comp对角线为无穷大
                    neighbor.append(index[i]) #neighbor[-1] 存了原先的index
                    index[i], table[i, 1], table[i, 0] = point_all_center_sqrt_neighbor(data[i, :], center, neighbor, table[i, 1])
                    count += len(neighbor) - 1
                    if neighbor[-1] != index[i]:
                        cluster_changed = True
                        cen_num[neighbor[-1]] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[neighbor[-1], :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
    table = dist_method(data, center[index[:], :], 1)
    count += row
    print(total)
    return index, table, count
def point_all_center_sqrt_localfilter(x, c, low, p, ind, inde_value, d = euclidean):
    li = list(range(np.shape(c)[0]))
    li.remove(ind)
    first, second, index = inde_value, np.linalg.norm(x - c[li[0], :]), ind
    count = 1
    if first > second:
        first, second, index = second, first, li[0]
    for i in li[1:]:
        # if i == ind:
        #     reg = inde_value
        #     # count += 1
        #     if reg < first:
        #         first, second, index = reg, first, i
        #     elif reg < second:
        #         second = reg
        #     # continue
        if second > low - p[i] : #它用来弄一维也合适
            count += 1
            reg = np.linalg.norm(x - c[i, :])
            if reg < first:
                first, second, index = reg, first, i
            elif reg < second:
                second = reg
        # elif i == ind:
            # count += 1

        # else:
        #     reg = np.linalg.norm(x - c[i, :])
        #     if reg < first:
        #         first, second, index = reg, first, i
        #         print('cao，错过了first', i, ind)
        #     elif reg < second:
        #         second = reg
        #         print('完蛋，错过了second', i, ind)
    return index, first, second, count

def hamerly_data_lowerbound(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
    count = row * k
    cluster_changed = True
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        cluster_changed = False
        total += 1
        # print('这是Hamerly的第', total, '次迭代')
        s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i, 1] += p[index[i]]
            table_reg = table[i, 0]
            if first == index[i]:
                table[i, 0] -= p[second]
            else:
                table[i, 0] -= p[first] #更新bound，省下了一个for
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i, 1] > m:
                    temp = index[i]
                    index[i], table[i, 1], table[i, 0], count_med = point_all_center_sqrt_localfilter(data[i, :], center, table_reg, p, temp, table[i, 1])
                    count += count_med
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
        # if cluster_changed:
    table = dist_method(data, center[index[:], :], 1)
    count += row
    print(total)
    return index, table, count
def point_all_center_sqrt_localfilter2(x, c, j, low, p, inde_value, d = euclidean):
    temp = c[j, :]
    first, second, index = inde_value, np.linalg.norm(x - temp[0, :]), -1
    count = 1
    if first > second:
        first, second, index = second, first, 0
    for i in range(1, np.shape(temp)[0] - 1):
        if second > low - p[i] : #它用来弄一维也合适
            count += 1
            reg = np.linalg.norm(x - temp[i, :])
            if reg < first:
                first, second, index = reg, first, i
            elif reg < second:
                second = reg
    return j[index], first, second, count

def hamerly_neighbor_2(data, k, center, max_iteration=200, dist_method=euclidean):
    row = np.shape(data)[0]#针对中心点需要开根了。
    table = np.zeros((row, 2))  # 第一列放low，第二列high
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    for i in range(row):
        index[i], table[i, 1], table[i, 0] = point_all_center_sqrt(data[i, :], center)
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
    p = np.zeros(k)
    cluster_changed = True
    count = row * k
    while cluster_changed and total < max_iteration:
        for i in range(k):
            temp = center[i, :].copy()
            if cen_num[i] == 0:
                print('*' * 10 + 'fail' + '*' * 10)
                return -1, -1
            else:
                center[i, :] = cen_sum[i, :] / cen_num[i]
            p[i] = np.sqrt(dist_method(center[i, :], temp))
        first, second = max_2_index(p)
        cen_comp = get_cen_sqrt(center)
        s = np.min(cen_comp, 1)
        cluster_changed = False
        total += 1
        for i in range(row):
            table[i, 1] += p[index[i]]
            table_reg = table[i, 0]
            if first == index[i]:
                table[i, 0] -= p[second]
            else:
                table[i, 0] -= p[first]
            m = max(s[index[i]], table[i, 0])
            if table[i, 1] > m:
                table[i, 1] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i, 1] > m:
                    # temp = index[i]
                    neighbor = [j for j in range(k) if table[i, 1] + s[index[i]] >= cen_comp[index[i], j]]
                    neighbor.append(index[i])
                    # print(len(neighbor), neighbor)
                    # index[i], table[i, 1], table[i, 0] = point_all_center_sqrt_neighbor(data[i, :], center, neighbor)
                    index[i], table[i, 1], table[i, 0], count_med = point_all_center_sqrt_localfilter2(data[i, :], center, neighbor, table_reg, p[neighbor], table[i, 1])
                    count += count_med
                    # count += len(neighbor)
                    if neighbor[-1] != index[i]:
                        cluster_changed = True
                        cen_num[neighbor[-1]] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[neighbor[-1], :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
    table = dist_method(data, center[index[:], :], 1)
    count += row
    print(total)
    return index, table, count


from center_method import plus_triangle
from k_means import  naive_k_means
from box_kdtree import kdtree_kmeans
# from elkan_triangle import elkan_sqrt, elkan_hyperbola
import time
#todo 我宣布，两不等式没啥用，norm这一内置函数到底有多牛逼?
if __name__ == '__main__':
    # data  = np.load('iris.npy')

    data = np.random.rand(200, 200)
    # data = np.load('birch_data.npy')
    k = 3
    center=plus_triangle(data, k)
    # # elkan_square(data , k, center)
    # i, j = elkan_sqrt(data , k, center.copy())
    # i2, j2, c5 = kdtree_kmeans(data, k, center.copy())
    i, j, c1, r = hamerly_sqrt(data , k, center.copy())
    i4, j4 = naive_k_means(data, k, center.copy())
    print(sum(i4) - sum(i), sum(j) - sum(j4))
    # i5, j5 = elkan_hyperbola(data, k, center.copy())
    # i5, j5, c2, r2 = hamerly_sqrt_new(data, k, center.copy())
    # i5, j5, c2 = hamerly_data_lowerbound(data, k, center.copy())
    # # print('plus neighbor'+'*'*20)
    # # i6, j6 = elkan_hyperbola_neighbor(data, k, center.copy())
    # # i7, j7 = elkan_hyperbola_neighbor_2(data, k, center.copy())
    # print(sum(i5) - sum(i), sum(j) - sum(j5))
    # i9, j9, c3 = hamerly_neighbor(data , k, center.copy())
    # i2, j2, c4 = hamerly_neighbor_2(data , k, center.copy())
    # print(sum(i) - sum(i9), sum(j) - sum(j9))
    # print(sum(i2) - sum(i9), sum(j2) - sum(j9))
    #发现 也减不到哪里去
    # print('计算比例:', c1/r)
    # print('hanerly',c1)
    # print('lb',c2)
    # print('center',c3)
    # print('low+center',c4)

    # # print(sum(i6) - sum(i), sum(j) - sum(j6))
    # # print(sum(i7) - sum(i), sum(j) - sum(j7))
    #
    # i6, j6 = hamerly_sqrt(data , k, center.copy())
    # print('*' * 20+'hamerly_hyperbola' + '*' * 20)
    # print(sum(i4) - sum(i6), sum(j6) - sum(j4))
    #
    # i7, j7 = hamerly_hyperbola(data , k, center.copy())
    # print(sum(i4) - sum(i7), sum(j7) - sum(j4))
    #
    # print('*' * 20+'hamerly_hyperbola_new_efficient' + '*' * 20)
    # i8, j8 = hamerly_hyperbola_neighbor(data , k, center.copy())
    # print(sum(i4) - sum(i8), sum(j8) - sum(j4))
    # print('*' * 20+'hamerly_neighbor' + '*' * 20)
    # # my_plot(data, center, i7, j7, 0, 'hamerly_hyperbola')

#todo hyperbola
    # data = np.load('birch_data.npy')
    # t = []
    # dim = [4, 8, 16, 32, 64]
    # kk = [2, 4, 8, 16, 32, 64, 128]
    # kk = [8, 16, 32, 64]
    # # kk = [2,4]
    # k = 30
    # data = np.random.randn(10000,30)*4
    # # center=plus_triangle(data, k)
    # # # a = time.time()
    # # ii, jj = drake(data , k, center.copy())
    # # # b = time.time()
    # # i2, j2 = naive_k_means(data, k, center.copy())
    # # print(np.sum(i2 - ii), np.sum(j2)-np.sum(jj))
    # # kk=[9]
    # for k in kk:
    #     # data = np.random.randn(2000,1)*2
    #     #        k = 13
    #     print(k, '维了')
    #     center = plus_triangle(data, k)
    #
    #     a = time.time()
    #     ii, jj = hamerly_sqrt(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #
    #     a = time.time()
    #     i2, j2 = hamerly_hyperbola(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(i2 - ii), np.sum(j2) - np.sum(jj))
    #
    #     a = time.time()
    #     i3, j3 = hamerly_hyperbola_neighbor(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
    #
    #     a = time.time()
    #     i4, j4 = hamerly_neighbor(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(i4 - ii), np.sum(j4) - np.sum(jj))
    #
    # # kk=dim
    # import matplotlib.pyplot as plt
    # plt.plot(kk, t[0::4], label='Hamerly', c='k', linewidth=2)
    # plt.plot(kk, t[1::4], '--', label='Hamerly_hyperbola', c='b', linewidth=2)
    # plt.plot(kk, t[2::4], '-+', label='Hamerly_efficient_all', c='r', linewidth=2)
    # plt.plot(kk, t[3::4], '-<', label='Hamerly_just_neighbor', c='g', linewidth=2)
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

#todo these two inequality
    # t = []
    # # dim = [2,4,8,16,32,64]
    # kk = [2, 4, 6, 8, 12, 16, 22, 32, 64, 128, 256, 512]
    # # kk = [32, 64, 128, 256]
    # # kk = [128, 148, 168, 188, 208, 238, 268, 298, 356, 400]
    # # kk = [12,17,22, 27,32,37]
    # # kk = [2,3,4,5,6, 7,8,9,10]
    # # kk = [4,8,16,32,64,128]
    # # kk = [2,4]
    # k = 16
    # # data = np.random.randn(2000,1)*2
    # # center=plus_triangle(data, k)
    # # # a = time.time()
    # # ii, jj = drake(data , k, center.copy())
    # # # b = time.time()
    # # i2, j2 = naive_k_means(data, k, center.copy())
    # # print(np.sum(i2 - ii), np.sum(j2)-np.sum(jj))
    #
    # # kk=[160, 200, 256, 512]
    # count = []
    # for dim in kk:
    #     data = np.random.rand(6800, dim)
    #     #        k = 13
    #     print(dim, '了')
    #     center = plus_triangle(data, k)
    #
    #     a = time.time()
    #     i2, j2, c_low = hamerly_data_lowerbound(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #
    #     a = time.time()
    #     ii, jj, ch, original = hamerly_sqrt_new(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(i2 - ii), np.sum(j2) - np.sum(jj))
    #
    #     a = time.time()
    #     i1, j1, c_center = hamerly_neighbor(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(i1 - ii), np.sum(j1) - np.sum(jj))
    #
    #
    #     a = time.time()
    #     i3, j3, c_2 = hamerly_neighbor_2(data, k, center.copy())
    #     b = time.time()
    #     t.append(b - a)
    #     print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
    #     #
    #     # a = time.time()
    #     # i3, j3, c_2, ori = hamerly_sqrt_new(data, k, center.copy())
    #     # b = time.time()
    #     # t.append(b - a)
    #     # print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
    #
    #
    #     count.append(ch)
    #     count.append(original)
    #     count.append(c_center)
    #     count.append(c_low)
    #     count.append(c_2)
    # # kk=dim
    # import matplotlib.pyplot as plt
    # # plt.plot(kk, t[2::5], label='center_inequality', c='k', linewidth=2)
    # # plt.plot(kk, t[1::5], '--', label='original', c='b', linewidth=2)
    # # plt.plot(kk, t[0::5], '-+', label='Hamerly', c='r', linewidth=2)
    # # plt.plot(kk, t[3::5], '-<',label='lb_inequality', c='g',linewidth=2)
    # # plt.plot(kk, t[4::5], '->', label='combine_inequality', c='y', linewidth=2)
    #
    # plt.plot(kk, t[2::4], label='center_inequality', c='k', linewidth=2)
    # plt.plot(kk, t[3::4], '--', label='combine_inequality', c='b', linewidth=2)
    # plt.plot(kk, t[1::4], '-+', label='Hamerly', c='r', linewidth=2)
    # plt.plot(kk, t[0::4], '-<',label='lb_inequality', c='g',linewidth=2)
    # # plt.plot(kk, t[4::5], '->', label='combine_inequality', c='y', linewidth=2)
    # #
    # ##    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
    # ##    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
    # ##
    # plt.legend()
    # plt.grid()
    # # plt.xlabel('k')
    # plt.xlabel('dimensional')
    # ##     plt.xlabel('number')
    # # plt.ylabel('distance calculations')
    # plt.ylabel('time')
    # plt.show()
