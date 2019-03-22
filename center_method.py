# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 10:53:07 2018

@author: Administrator
"""
#from numpy import *
import numpy as np
from distance_method import *
from numpy.random import rand, randint
from random import sample
def k_means_for_sample(data , k, center, max_iteration = 200):
    row = np.shape(data) [0]
    table = np.zeros(row)
    index = np.zeros(row, dtype = int)
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
            if index[i] != temp:
                cluster_changed = True
                cen_num[temp] -= 1
                cen_num[index[i]] += 1
                cen_sum[temp, :] -= data[i, :]  #对应簇的辅助变量更新
                cen_sum[index[i], :] += data[i, :]
    # print(total)
    return center, table

def random_field(data, k):
    col = np.shape(data)[1]
    center = np.zeros((k, col))
    for i in range(col):
        min_i = np.min(data[:, i])
#        print(min_i, i)
        center[:, i] = min_i + rand(k) * (np.max(data[:, i]) - min_i)
    return center

def var_partition(data, k):

    dim = np.var(data, 0).argmax()


def forgy(data, k):
    row, col = np.shape(data)
    c_sum = np.zeros((k, col))
    c_num = np.zeros(k)
    for i in range(row):
        t = np.random.randint(k)
        c_sum[t, :] += data[i, :]
        c_num[t] += 1
    for i in range(k):
        if c_num[i] == 0:
            print('fail')
            return None
        c_sum[i, :] = c_sum[i, :] / c_num[i]
    return c_sum

def random_centroid(data, k):
    # row = np.shape(data)[0]
    # center_index = sample(range(row), k)
    # center_index = np.unique(randint(0, row, k))
    # while np.size(center_index) != k:
    # center_index = np.unique(randint(0, row, k))
    return data[sample(range(np.shape(data)[0]), k), :]
    
def plus_plus(data, k):
    row = np.shape(data)[0]
    table = np.zeros(row)
    # index = np.zeros(row, int)
    count = 0
    center = data[randint(row), :]
    ind = -1
    for i in range(row):
        t = data[i, :] - center
        table[i] = np.dot(t, t)
    while count < k - 1:
        temp = 0
        judge = rand() * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:  # 可以修成小于
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        count += 1
        for i in range(row):
                t = data[i, :] - center[-1, :]
                temp = np.dot(t, t)
                if temp < table[i]:
                    table[i] = temp
    return center

def plus_triangle(data, k, c = 0):
    row = np.shape(data)[0]
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[randint(row), :]
#    data_norm = np.power(data, 2).sum(1)
#     data_norm = np.sqrt(np.power(data, 2).sum(1))
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
#         table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
        t = data[i, :] - center
        table[i] = np.dot(t, t)
    while count < k - 1:
        temp = 0
        judge = rand() * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]: #可以修成小于
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        # center_norm = np.sqrt(np.power(center[-1, :], 2).sum(0))
        count += 1
        # reg = d(center[: count, :], center[-1, :], 1)#妈的，这么简单的，又给忘了
        reg = np.linalg.norm(center[: count, :] - center[-1, :], axis = 1) ** 2
#        print(reg)
        for i in range(row):
            # if reg[index[i]] / 4 < table[i] and data_norm[i] - np.sqrt(table[i]) < center_norm < data_norm[i] + np.sqrt(table[i]):
            if reg[index[i]] / 4 < table[i]:
#             if reg[index[i]] / 4 < table[i]:
                c += 1
                t = data[i, :] - center[-1, :]
                temp = np.dot(t, t)
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
#    print(row*(k), c)
#    print((1 - c/row/(k))*100)
    return center

def greedy_plus_plus(data, k):
    row = np.shape(data)[0]
    table = np.zeros(row)
    num_try = int(np.ceil(np.log2(k))) #greedy次数
    # index = np.zeros(row, int)
    count = 0
    ind = -1
    table_sum_reg = np.inf
    for i in range(num_try):
        c_index = randint(row)
        center = data[c_index, :]
        new_table_sum_reg = 0
        for i in range(row):
            t = data[i, :] - center
            table[i] = np.dot(t, t)
            new_table_sum_reg += table[i]
        if new_table_sum_reg <= table_sum_reg: #啊哈，希望用等号propagate下去，利于判断是否需重新
            table_sum_reg, ind = new_table_sum_reg, c_index
    if ind != c_index:#希望最后一次就成，这样就不需要重新算table了
        center = data[ind, :]
        for i in range(row):
            t = data[i, :] - center
            table[i] = np.dot(t, t)
    while count < k - 1:
        for_table_sum_reg = table_sum_reg #上一次table合，存下来用来找概率区间
        for i in range(num_try):
            table_reg = table.copy()
            temp = 0
            judge = rand() * for_table_sum_reg
            for i in range(row):
                if judge >= temp and judge <= temp + table_reg[i]:  # 可以修成小于
                    ind = i
                    break
                temp += table_reg[i]
            center_reg = np.row_stack((center, data[ind, :])) #临时
            new_table_sum_reg = 0
            for i in range(row):
                t = data[i, :] - center_reg[-1, :]
                temp = np.dot(t, t)
                if temp < table_reg[i]:
                    table_reg[i] = temp
                new_table_sum_reg += table_reg[i]
            if new_table_sum_reg <= table_sum_reg:
                table_sum_reg, c_index = new_table_sum_reg, ind
        if ind != c_index:
            center = np.row_stack((center, data[c_index, :]))
            for i in range(row):
                t = data[i, :] - center[-1, :]
                temp = np.dot(t, t)
                if temp < table[i]:
                    table[i] = temp
        else:
            center = center_reg.copy()
            table = table_reg.copy()
        count += 1
    return center

def just_maximum(data, k):
    row = np.shape(data)[0]
    table = np.zeros(row)
    count = 0
    center = data[randint(row), :]
    ind = -1
    for i in range(row):
        t = data[i, :] - center
        table[i] = np.dot(t, t)
    while count < k - 1:
        temp = 0
        ind = np.argmax(table)
        center = np.row_stack((center, data[ind, :]))
        count += 1
        for i in range(row):
                t = data[i, :] - center[-1, :]
                temp = np.dot(t, t)
                if temp < table[i]:
                    table[i] = temp
    return center

# from k_means import naive_k_means
def bradley_sample(data, k, j = 10, rate = .1):
    sp = random_centroid(data, k) #可以换其他
    # a1 = random_field(data, k)
    # a3 = plus_plus(data, k)
    # a4 = plus_triangle(data, k)
    # a5 = greedy_plus_plus(data, k)

    row, col = np.shape(data)
    fm = np.zeros((k, col))
    cm = np.zeros((k * j, col))
    for i in range(j):
        s = data[sample(range(row), int(rate*row)), :]
        cm[k * i: k * (i + 1), :] = k_means_for_sample(s, k, sp)[0]
    # cm = np.unique(cm, axis = 0)
    reg = np.inf
    for i in range(j):
        fm, table = k_means_for_sample(cm, k, cm[k * i: k * (i + 1), :])
        table_sum = sum(table)
        if table_sum < reg:
            reg = table_sum
            fms = fm.copy()
    return fms

from line_three import *
from try_some_functionality import generate_gaussian_clusters
# import  matplotlib.pyplot as plt
if __name__ == '__main__':
     # a = np.array([[1],[2],[1]])
     
#     print(np.unique(a, axis=1))
#      a = np.reshape(np.arange(16), (4,4))
#      data = np.load('iris.npy')
     # data = np.load('birch_data.npy')
#     c=    random_center(data, 3)
#     c=    plus_plus(data, 3)
#      data  = np.random.rand(300, 2)
#      k = 3
     # center=plus_triange(data, k)
     # print(center)
     # print(np.random.choice(3,2)) #这个搞不定，会重复
     k=5
     data, real_c, z = generate_gaussian_clusters(10000, 2, k, .08)

     a1 =  random_field(data, k)
     a2 =  random_centroid(data, k)
     a3 =  plus_plus(data, k)
     # a4 =  plus_triangle(data, k)
     a5 =  greedy_plus_plus(data, k)
     a6 =  bradley_sample(data, k)
     a7 = just_maximum(data, k)
     a8 = forgy(data, k)
     plot_center(a1, data,'random field')
     plot_center(a8, data,'forgy')
     # plot_center(a2, data,'random centroid')
     # plot_center(a3, data,'plus_plus')
     # plot_center(a5, data,'greedy ++')
     # plot_center(a6, data,'bradley_sample')
     # plot_center(a7, data,'maximum')
