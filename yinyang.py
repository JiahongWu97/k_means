# Created by Jiahong Wu at 2019/3/12
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
        for j in range(0, i ):
            cen[i, j] = cen[j, i]
    return cen
def hunt_all_localfilter(x, c, jj, low, p, d = euclidean):
    temp = c[jj, :]
    # first, index = d(x, temp[0, :]), 0
    first, index = np.linalg.norm(x - temp[0, :]), 0
    for i in range(1, np.shape(temp)[0]):
        if first > low - p[i]: #todo 新的不等式
            reg = np.linalg.norm(x - temp[i, :])
            if reg < first:
                first, index = reg, i
        # else: #测试local filter
        #     reg = d(x, temp[i, :])
        #     if reg < first:
        #         print('原', low - p[i], first)
        #         print('后', reg, first) #为什么bound更新会失效
        #         first, index = reg, i
                # print('yeah')
    return jj[index], first
def hunt_all(data, center, jj, d = euclidean):
    temp = center[jj, :]
    min_dist, min_index = d(data, temp[0, :]), 0
    for j in range(1, np.shape(temp)[0]):
        reg = d(data, temp[j, :])
        if reg < min_dist:
            min_dist, min_index = reg, j
    return  jj[min_index], np.sqrt(min_dist)

def point_all_center_sqrt_localfilter(x, c, j, low, p, d = euclidean):
    temp = c[j, :]
    first, second, index = np.linalg.norm(x - temp[0, :]), np.linalg.norm(x - temp[1, :]), 0
    count_med = 2
    if first > second:
        first, second, index = second, first, 1
    for i in range(2, np.shape(temp)[0]):
        if second > low - p[i]: #它用来弄一维也合适
            count_med +=1
            reg = np.linalg.norm(x - temp[i, :])
            if reg < first:
                first, second, index = reg, first, i
            elif reg < second:
                second = reg
    return j[index], first, second, count_med

def point_all_center_sqrt_neighbor(x, c, j, d = euclidean):
    temp = c[j, :]
    first, second, index = d(x, temp[0, :]), d(x, temp[1, :]), 0
    if first > second:
        first, second, index = second, first, 1
    for i in range(2, np.shape(temp)[0]):
        reg = d(x, temp[i, :])
        if reg < first:
            first, second, index = reg, first, i
        elif reg < second:
            second = reg
    return j[index], np.sqrt(first), np.sqrt(second)
def yinyang_understand_second(data, k, center, max_iteration=200, dist_method=euclidean): #需要k//10 >= 2
    t = int(np.ceil(k / 10))
    row = np.shape(data)[0]
    table = np.ones((row)) * np.inf
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    g = [None] * t
    for i in range(t):
            g[i] = list(range(i*10, k)) if i == t - 1 else range(i*10, i*10+10) #生成group
    low = np.zeros((row, t))
    for i in range(row):
        for j in range(t):
            ind, low[i, j] = hunt_all(data[i, :], center, g[j])
            if low[i, j] < table[i]:
                table[i], index[i] = low[i, j], ind
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
        exclude = index[i] // 10
        temp = list(g[exclude])
        temp.remove(index[i])
        low[i, exclude] = hunt_all(data[i, :], center, temp)[1]  #修补 todo 问题出在修补这
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
        delta = [max(p[g[i]]) for i in range(t)]
        cluster_changed = False
        total += 1
        # print('这是yinyang的第', total, '次迭代')
        # s = get_cen_sqrt(center).min(1)
        # low -= delta
        for i in range(row):
            table[i] += p[index[i]]
            low[i, :] -= delta
            judge = min(low[i, :])
            # m = max(s[index[i]], table[i, 0])
            if table[i] > judge:
                table[i] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                if table[i] > judge:
                    temp = index[i]
                    exclude_pre = temp // 10
                    group = False
                    if table[i] <= low[i, exclude_pre]:
                        group = True
                        table_reg = table[i]
                    for j in range(t):
                        if table[i] > low[i, j]: #这里出了问题;大于或者大于等于没有啥影响
                            # print(j+1, t)
                            ind, low[i, j] = hunt_all_localfilter(data[i, :], center, g[j], low[i, j] + delta[j], p[g[j]])
                            if low[i, j] < table[i]:
                                table[i], index[i] = low[i, j], ind
                        # else:
                        #     t1, t2 = low[i, 2 * j], low[i, 2 * j + 1]
                        #     ind, low[i, j] = hunt_all_localfilter(data[i, :], center, g[j], low[i, j] + delta[j], p[g[j]])
                        #     if low[i, j] < table[i]:
                        #         table[i], index[i] = low[i, j], ind
                        #         print('yeah')
                        #     # print(low[i, 2*j], 'vs', t1,'+++++',t2,'vs', low[i, 2*j+1])#我特么居然看到更新后反而更小的。
                        #     # print(low[i, 2*j]<= t1,t2>= low[i, 2*j+1])#我特么居然看到更新后反而更小的。因为包含了table的那一组
                        #     if low[i, 2*j] < table[i]:
                        #         table[i], index[i] = low[i, 2*j], ind
                        #         print(i, 1+j, t)#这特么说明了，需要更新边界为最新值？也确实是可以避开的。大了嘛。想想我那拍脑袋想的更新，带来了啥

                        #     low[i, 2*j] -= max(delta)
                            # low[i, 2*j] = -np.inf
                            # low[i, 2*j+1] -= delta[j]
                            # low[i, 2*j+1] = -np.inf
                    exclude = index[i] // 10  # 修补
                    if group:
                        if exclude_pre != exclude:
                            # low[i, 2 * exclude_pre], low[i, 2 * exclude_pre + 1] = table[i], table[i]
                            low[i, exclude_pre] = table_reg
                    else:
                        tem = list(g[exclude])
                        tem.remove(index[i])
                        low[i, exclude] = hunt_all(data[i, :], center, tem)[1]  # 修补 todo 问题出在修补这

                    # low[i, 2 * exclude] = low[i, 2 * exclude + 1]
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
                # else:
                #     print('yeah')
            # else:
            #     print('yeah')
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table

def yinyang(data, k, center, max_iteration=200, dist_method=euclidean): #需要k//10 >= 2
    t = int(np.ceil(k / 10))
    row = np.shape(data)[0]
    table = np.ones((row)) * np.inf
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    g = [None] * t
    for i in range(t):
            g[i] = range(i*10, k) if i == t - 1 else range(i*10, i*10+10) #生成group
    low = np.zeros((row, 2 * t))#用空间复杂度牺牲时间复杂度，好吗？还好不是多太多吧
    for i in range(row):
        for j in range(t):
            ind, low[i, 2 *j], low[i, 2*j+1] = point_all_center_sqrt_neighbor(data[i, :], center, g[j])
            if low[i, 2*j] < table[i]:
                table[i], index[i] = low[i, 2*j], ind
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
        exclude = index[i] // 10 #修补 todo 问题出在修补这
        low[i, 2*exclude] = low[i, 2*exclude+1]
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
        delta = [max(p[g[i]]) for i in range(t)]
        cluster_changed = False
        total += 1
        # print('这是yinyang的第', total, '次迭代')
        # s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i] += p[index[i]]
            low[i, ::2] -= delta
            # low[i, 1::2] -= delta #貌似可以删
            judge = min(low[i, ::2])
            # m = max(s[index[i]], table[i, 0])
            if table[i] > judge:
                table[i] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                if table[i] > judge:
                    temp = index[i]
                    exclude_pre = temp // 10
                    if table[i] <= low[i, 2*exclude_pre]: #这样貌似省代码
                        low[i, 2*exclude_pre], low[i, 2*exclude_pre + 1] = table[i], table[i] #这一步也确保了无需low[i, 1::2] -= delta
                        # low[i, 2*exclude_pre] =  table[i]
                    for j in range(t):
                        if table[i] > low[i, 2*j]: #这里出了问题;大于或者大于等于没有啥影响
                            ind, low[i, 2*j], low[i, 2*j+1] = point_all_center_sqrt_localfilter(data[i, :], center, g[j], low[i, 2*j] + delta[j], p[g[j]])
                            if low[i, 2*j] < table[i]:
                                table[i], index[i] = low[i, 2*j], ind
                    exclude = index[i] // 10  # 修补
                    low[i, 2 * exclude] = low[i, 2 * exclude + 1] #贪图方便，所以之前要赋值2个
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table

def yinyang__modify_change(data, k, center, max_iteration=200, dist_method=euclidean): #需要k//10 >= 2
    t = int(np.ceil(k / 10))
    row = np.shape(data)[0]
    table = np.ones((row)) * np.inf
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    g = [None] * t
    for i in range(t):
            g[i] = range(i*10, k) if i == t - 1 else range(i*10, i*10+10) #生成group
    low = np.zeros((row, 2 * t))#用空间复杂度牺牲时间复杂度，好吗？还好不是多太多吧
    for i in range(row):
        for j in range(t):
            ind, low[i, 2 *j], low[i, 2*j+1] = point_all_center_sqrt_neighbor(data[i, :], center, g[j])
            if low[i, 2*j] < table[i]:
                table[i], index[i] = low[i, 2*j], ind
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
        exclude = index[i] // 10 #修补 todo 问题出在修补这
        low[i, 2*exclude] = low[i, 2*exclude+1]
    p = np.zeros(k)
    count = row*k
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
        delta = [max(p[g[i]]) for i in range(t)]
        cluster_changed = False
        total += 1
        # print('这是yinyang的第', total, '次迭代')
        s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i] += p[index[i]]
            low[i, ::2] -= delta
            # low[i, 1::2] -= delta #貌似可以删
            judge = min(low[i, ::2])
            m = max(s[index[i]], judge)
            if table[i] > m:
                table[i] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i] > m:
                    temp = index[i]
                    exclude_pre = temp // 10
                    group = False
                    if table[i] <= low[i, 2*exclude_pre]:
                        group = True
                        table_reg = table[i]
                    # if table[i] <= low[i, 2*exclude_pre]: #这样貌似省代码
                    #     low[i, 2*exclude_pre], low[i, 2*exclude_pre + 1] = table[i], table[i] #这一步也确保了无需low[i, 1::2] -= delta
                        # low[i, 2*exclude_pre] =  table[i]
                    for j in range(t):
                        if table[i] > low[i, 2*j]: #这里出了问题;大于或者大于等于没有啥影响
                            ind, low[i, 2*j], low[i, 2*j+1], count_med = point_all_center_sqrt_localfilter(data[i, :], center, g[j], low[i, 2*j] + delta[j], p[g[j]])
                            count += count_med
                            if low[i, 2*j] < table[i]:
                                table[i], index[i] = low[i, 2*j], ind
                    exclude = index[i] // 10  # 修补
                    # low[i, 2 * exclude] = low[i, 2 * exclude + 1]
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
                    if group:
                        if exclude_pre != exclude:
                            # low[i, 2 * exclude_pre], low[i, 2 * exclude_pre + 1] = table[i], table[i]
                            low[i, 2 * exclude_pre] = table_reg
                    else:
                        low[i, 2 * exclude] = low[i, 2 * exclude + 1]
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table, count, count/ total/ row/ k

def yinyang_equal_num(data, k, center, max_iteration=200, dist_method=euclidean): #需要k//10 >= 2
    t = int(np.ceil(k / 10))
    row = np.shape(data)[0]
    table = np.ones((row)) * np.inf
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    g = [None] * t
    for i in range(t):
            g[i] = range(i*10, k) if i == t - 1 else range(i*10, i*10+10) #生成group
    low = np.zeros((row, 2 * t))#用空间复杂度牺牲时间复杂度，好吗？还好不是多太多吧
    for i in range(row):
        for j in range(t):
            ind, low[i, 2 *j], low[i, 2*j+1] = point_all_center_sqrt_neighbor(data[i, :], center, g[j])
            if low[i, 2*j] < table[i]:
                table[i], index[i] = low[i, 2*j], ind
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
        exclude = index[i] // 10 #修补 todo 问题出在修补这
        low[i, 2*exclude] = low[i, 2*exclude+1]
    p = np.zeros(k)
    count = row*k
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
        delta = [max(p[g[i]]) for i in range(t)]
        cluster_changed = False
        total += 1
        # print('这是yinyang的第', total, '次迭代')
        # s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i] += p[index[i]]
            low[i, ::2] -= delta
            # low[i, 1::2] -= delta #貌似可以删
            judge = min(low[i, ::2])
            # m = max(s[index[i]], table[i, 0])
            if table[i] > judge:
                table[i] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i] > judge:
                    temp = index[i]
                    exclude_pre = temp // 10
                    group = False
                    if table[i] <= low[i, 2*exclude_pre]:
                        group = True
                        table_reg = table[i]
                    # if table[i] <= low[i, 2*exclude_pre]: #这样貌似省代码
                    #     low[i, 2*exclude_pre], low[i, 2*exclude_pre + 1] = table[i], table[i] #这一步也确保了无需low[i, 1::2] -= delta
                        # low[i, 2*exclude_pre] =  table[i]
                    for j in range(t):
                        if table[i] > low[i, 2*j]: #这里出了问题;大于或者大于等于没有啥影响
                            ind, low[i, 2*j], low[i, 2*j+1], count_med = point_all_center_sqrt_localfilter(data[i, :], center, g[j], low[i, 2*j] + delta[j], p[g[j]])
                            count += count_med
                            if low[i, 2*j] < table[i]:
                                table[i], index[i] = low[i, 2*j], ind
                    exclude = index[i] // 10  # 修补
                    # low[i, 2 * exclude] = low[i, 2 * exclude + 1]
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
                    if group:
                        if exclude_pre != exclude:
                            # low[i, 2 * exclude_pre], low[i, 2 * exclude_pre + 1] = table[i], table[i]
                            low[i, 2 * exclude_pre] = table_reg
                    else:
                        low[i, 2 * exclude] = low[i, 2 * exclude + 1]
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table, count, count/ total/ row/ k

def yinyang__initial(data, k, center, max_iteration=200, dist_method=euclidean): #需要k//10 >= 2
    t = int(np.ceil(k / 10))
    row = np.shape(data)[0]
    table = np.ones((row)) * np.inf
    index = np.zeros((row), int)
    cen_sum = np.zeros(np.shape(center))
    cen_num = np.zeros((k))
    total = 1
    # g = [None] * t
    center_initial = plus_triangle(center, t)
    i = hamerly_sqrt(center, t, center_initial, 5)[0]
    g = [np.nonzero(i == j)[0] for j in range(t)]     #生成group
    print(g)
    for j in range(t):
        if len(g[j]) < 2:
            print('fail')
            return -1, -1
    low = np.zeros((row, 2 * t))#用空间复杂度牺牲时间复杂度，好吗？还好不是多太多吧
    for i in range(row):
        for j in range(t):
            ind, low[i, 2 *j], low[i, 2*j+1] = point_all_center_sqrt_neighbor(data[i, :], center, g[j])
            if low[i, 2*j] < table[i]:
                table[i], index[i] = low[i, 2*j], ind
        cen_sum[index[i], :] += data[i, :]
        cen_num[index[i]] += 1
        exclude = index[i] // 10 #修补 todo 问题出在修补这
        low[i, 2*exclude] = low[i, 2*exclude+1]
    count = row*k
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
        delta = [max(p[g[i]]) for i in range(t)]
        cluster_changed = False
        total += 1
        # print('这是yinyang的第', total, '次迭代')
        # s = get_cen_sqrt(center).min(1)
        for i in range(row):
            table[i] += p[index[i]]
            low[i, ::2] -= delta
            # low[i, 1::2] -= delta #貌似可以删
            judge = min(low[i, ::2])
            # m = max(s[index[i]], table[i, 0])
            if table[i] > judge:
                table[i] = np.sqrt(dist_method(data[i, :], center[index[i], :]))
                count += 1
                if table[i] > judge:
                    temp = index[i]
                    for j in range(t):
                        if temp in g[j]:
                            exclude_pre = j
                    # exclude_pre = j for j in range(t) if temp in g[j]
                            break
                    group = False
                    if table[i] <= low[i, 2*exclude_pre]:
                        group = True
                        table_reg = table[i]
                    # if table[i] <= low[i, 2*exclude_pre]: #这样貌似省代码
                    #     low[i, 2*exclude_pre], low[i, 2*exclude_pre + 1] = table[i], table[i] #这一步也确保了无需low[i, 1::2] -= delta
                        # low[i, 2*exclude_pre] =  table[i]
                    for j in range(t):
                        if table[i] > low[i, 2*j]: #这里出了问题;大于或者大于等于没有啥影响
                            ind, low[i, 2*j], low[i, 2*j+1], count_med = point_all_center_sqrt_localfilter(data[i, :], center, g[j], low[i, 2*j] + delta[j], p[g[j]])
                            count += count_med
                            if low[i, 2*j] < table[i]:
                                table[i], index[i] = low[i, 2*j], ind
                    for j in range(t):
                        if index[i] in g[j]:
                            exclude = j  # 修补
                            break
                    # low[i, 2 * exclude] = low[i, 2 * exclude + 1]
                    if temp != index[i]:
                        cluster_changed = True
                        cen_num[temp] -= 1
                        cen_num[index[i]] += 1
                        cen_sum[temp, :] -= data[i, :]
                        cen_sum[index[i], :] += data[i, :]
                    if group:
                        if exclude_pre != exclude:
                            # low[i, 2 * exclude_pre], low[i, 2 * exclude_pre + 1] = table[i], table[i]
                            low[i, 2 * exclude_pre] = table_reg
                    else:
                        low[i, 2 * exclude] = low[i, 2 * exclude + 1]
    table = dist_method(data, center[index[:], :], 1)
    print(total)
    return index, table, count

from harmerly_triangle import hamerly_sqrt
from center_method import plus_triangle
from k_means import  naive_k_means
import time
from box_kdtree import kdtree_kmeans
from elkan_triangle import elkan_sqrt, elkan_hyperbola
if __name__ == '__main__':
    # data  = np.load('iris.npy')
    data = np.random.rand(4000, 2) * 4
    # data = np.load('birch_data.npy')
    # data = data[0:2000,:]
    # data = np.unique(data, 1)
    k = 18
    center=plus_triangle(data, k)
    # elkan_square(data , k, center)
    # i, j = elkan_sqrt(data , k, center.copy())

    # i2, j2 = kdtree_kmeans(data, k, center.copy())

    i2, j2, ch, rh = hamerly_sqrt(data , k, center.copy())

    # i2, j2 = naive_k_means(data, k, center.copy())

    # print(sum(i4) - sum(i), sum(j) - sum(j4))
    # i5, j5 = elkan_hyperbola(data, k, center.copy())
    # print('plus neighbor'+'*'*20)
    # i6, j6 = elkan_hyperbola_neighbor(data, k, center.copy())
    # i7, j7 = elkan_hyperbola_neighbor_2(data, k, center.copy())
    # print(sum(i5) - sum(i), sum(j) - sum(j5))
    # print(sum(i6) - sum(i), sum(j) - sum(j6))
    # print(sum(i7) - sum(i), sum(j) - sum(j7))

    # print(sum(i4) - sum(i2), sum(j2) - sum(j4))
    # i6, j6 = yinyang_understand_second(data , k, center.copy())
    # i6, j6 = yinyang(data , k, center.copy())
    i22, j22, c1, r = yinyang__modify_change(data , k, center.copy())
    i6, j6, c2 = yinyang__initial(data , k, center.copy())

    # print('*' * 20+'hamerly_hyperbola' + '*' * 20)
    # i7, j7 = hamerly_hyperbola(data , k, center.copy())
    # print('*' * 20+'hamerly_hyperbola_new_efficient' + '*' * 20)
    # i8, j8 = hamerly_hyperbola_neighbor(data , k, center.copy())
    # my_plot(data, center, i7, j7, 0, 'hamerly_hyperbola')
    print(sum(i2) - sum(i22), sum(j22) - sum(j2))
    print(sum(i2) - sum(i6), sum(j6) - sum(j2))

    # print(sum(i4) - sum(i7), sum(j7) - sum(j4))
    # print(sum(i4) - sum(i8), sum(j8) - sum(j4))

    # data = np.random.randn(10000, 3) * 5
    # t = []
    # # dim = [4, 8, 16, 32, 64]
    # # kk = [ 4, 8, 16, 25, 32, 46, 58, 64, 78, 82, 94]
    # kk = [ 58, 64, 78, 85, 94]
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
    #     a = time.process_time()
    #     ii, jj = yinyang__modify_change(data, k, center.copy())
    #     b = time.process_time()
    #     t.append(b - a)
    #     # print(np.sum(i2 - ii), np.sum(j2) - np.sum(jj))
    #
    #     a = time.process_time()
    #     i3, j3 = yinyang_understand_second(data, k, center.copy())
    #     b = time.process_time()
    #     t.append(b - a)
    #     print(np.sum(np.abs(i3 - ii)), np.sum(j3) - np.sum(jj))
    #
    # plt.plot(kk, t[0::2], label='yinyang_totalcase_reg', c='k', linewidth=2)
    # plt.plot(kk, t[1::2], '--', label='yinyang_compuate', c='b', linewidth=2)
    # # plt.plot(kk, t[2::3], '-+', label='yinyang_compuate', c='r', linewidth=2)
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

# [33.03125,
#  33.546875,
#  32.78125,
#  33.65625,
#  33.5625,
#  34.59375,
#  46.125,
#  45.203125,