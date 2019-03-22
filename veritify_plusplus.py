# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 10:53:07 2018

@author: Administrator
"""
#from numpy import *
import numpy as np
from distance_method import *
from numpy.random import rand, randint
#from k_means import *

def plus_old(data, k, ran, d = euclidean):
    row = np.shape(data) [0]
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[row//2, :]
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
        table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
#         table[i] = np.min(d(data[i, :], center, 1))
    # c = row
    while count < k - 1:
        temp = 0
        judge = ran[count] * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        count += 1
#        print(count)
#        reg = d(center[: count, :], center[-1, :], 1) #记录d（c,c'）
#        return (reg)
        for i in range(row):
#            if reg[index[i]] / 4 < table[i]:
#                 c += 1
                temp = d(data[i, :], center[-1, :])
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
    # print(row*(k), c)
    # print((1 - c/row/(k))*100)
    return center, row* k


def plus_just_new(data, k, ran, d = euclidean):
    row = np.shape(data) [0]
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[row//2, :]
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
        table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
#         table[i] = np.min(d(data[i, :], center, 1))
    # c = row
    while count < k - 1:
        temp = 0
        judge = ran[count] * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        count += 1
#        print(count)
#        reg = d(center[: count, :], center[-1, :], 1) #记录d（c,c'）
#        return (reg)
        for i in range(row):
#            if reg[index[i]] / 4 < table[i]:
#                 c += 1
                temp = d(data[i, :], center[-1, :])
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
    # print(row*(k), c)
    # print((1 - c/row/(k))*100)
    return center, row* k

def plus_triange(data, k, ran, d = euclidean):
    row = np.shape(data) [0]
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[row//2, :]
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
        table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
        # table[i] = np.min(d(data[i, :], center, 1))
    c = row
    while count < k - 1:
        temp = 0
        judge = ran[count] * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        count += 1
#        print(count)
        reg = d(center[: count, :], center[-1, :], 1) #记录d（c,c'）
#        return (reg)
        for i in range(row):
            if reg[index[i]] / 4 < table[i]:
                c += 1
                temp = d(data[i, :], center[-1, :])
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
    # print(row*(k), c)
    # print((1 - c/row/(k))*100)
    return center, c

def plus_norm(data, k, ran, d = euclidean):
    row = np.shape(data) [0]
    c = row
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[row//2, :]
#    data_norm = np.power(data, 2).sum(1)
#     data_norm = np.sqrt(np.power(data, 2).sum(1))
    data_norm = np.linalg.norm(data, axis=1)
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
        table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
    while count < k - 1:
        temp = 0
        judge = ran[count] * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        # center_norm = np.sqrt(np.power(center[-1, :], 2).sum(0))
        center_norm = data_norm[ind]
#        print(center_norm)
        count += 1
        # reg = d(center[: count, :], center[-1, :], 1)#妈的，这么简单的，又给忘了
#        print(reg)
        for i in range(row):
            table_reg = np.sqrt(table[i])
            if  data_norm[i] - table_reg < center_norm < data_norm[i] + table_reg:
#             if reg[index[i]] / 4 < table[i]:
                c += 1
                temp = d(data[i, :], center[-1, :])
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
    return center,c

def plus_triange_norm(data, k, ran, d = euclidean):
    row = np.shape(data) [0]
    c = row
    table = np.zeros(row)
    index = np.zeros(row, int)
    count = 0
    center = data[row//2, :]
#    data_norm = np.power(data, 2).sum(1)
#     data_norm = np.sqrt(np.power(data, 2).sum(1))
    data_norm = np.linalg.norm(data, axis=1)
    for i in range(row):
#            table[i] = np.min(sum(np.power(np.ones((count, 1)) * np.mat(data[i, :]) - center, 2),1))
        table[i] = np.min(d(np.ones((1, 1)) * np.mat(data[i, :]), center, 1))
    while count < k - 1:
        temp = 0
        judge = ran[count] * np.sum(table)
        for i in range(row):
            if judge >= temp and judge <= temp + table[i]:
                ind = i
                break
            temp += table[i]
        center = np.row_stack((center, data[ind, :]))
        # center_norm = np.sqrt(np.power(center[-1, :], 2).sum(0))
        center_norm = data_norm[ind]
#        print(center_norm)
        count += 1
        reg = d(center[: count, :], center[-1, :], 1)#妈的，这么简单的，又给忘了
#        print(reg)
        for i in range(row):
            table_reg = np.sqrt(table[i])
            if reg[index[i]] / 4 < table[i] and data_norm[i] - table_reg < center_norm < data_norm[i] + table_reg:
#             if reg[index[i]] / 4 < table[i]: 
                c += 1
                temp = d(data[i, :], center[-1, :])
#                print(temp)
                if temp < table[i]:
                    table[i], index[i] = temp, count
    return center,c

def sample(data, k):
    
    return
import time
import matplotlib.pyplot as plt
if __name__ == '__main__':
     a = np.array([[1],[2],[1]])
     
#     print(np.unique(a, axis=1))
#      a = np.reshape(np.arange(16), (4,4))
#      kk = [3, 6, 9, 12, 15, 18, 21, 24]
#      dim = [2, 4, 8, 16, 32]
     n = 30000
     dim = 30
     c = []
#
#      data  = np.random.rand(n, dim)
#      k = 6
#      ran = np.random.rand(k-1)
#      i, j = plus_just_new(data, k, ran)
#      i1, j1 = plus_triange(data, k, ran)
#      i2, j2 = plus_norm(data, k, ran)
#      i3, j3 = plus_triange_norm(data, k, ran)
#      print(np.sum(i-i1))
#      print(np.sum(i-i1))
#      print(np.sum(i-i2))
#      print('original: ',j)
#      print('triangle: ',j1)
#      print('norm: ',j2)
#      print('triangle+norm: ',j3)

     count = []
     kk = [2, 4, 6, 8, 12, 16, 22, 32, 64, 128, 256]
     kk = [2, 3,4, 5,6, 7,8, 9,10,11,12, 16]

     # kk = [2, 3,4, 5,6, 7,8, 10,12, 16, 26, 36]
     for dim in kk:
         data  = np.random.rand(n, dim)
         k = 10

         ran = np.random.rand(k-1)
         print(k)
         a = time.process_time()
         i1, j1 =plus_just_new(data, k, ran)
         b = time.process_time()
         c.append(b - a)

         a = time.process_time()
         i2, j2 =plus_triange(data, k, ran)
         b = time.process_time()
         c.append(b - a)
    #     print(center)
         a = time.process_time()
         i3, j3 =plus_norm(data, k, ran)
         b = time.process_time()
         c.append(b - a)

         a = time.process_time()
         i4, j4 =plus_triange_norm(data, k, ran)
         b = time.process_time()
         c.append(b - a)
         print(np.sum(i1 - i2))
         print(np.sum(i1 - i3))
         print(np.sum(i1 - i4))
         count.append(j1)
         count.append(j2)
         count.append(j3)
         count.append(j4)

     #         print(c)
     plt.plot(kk, c[0::4], '-+',label='just_plus', c='r',linewidth=2)
     plt.plot(kk, c[1::4], label='just_triangle', c='k',linewidth=2)
     plt.plot(kk, c[2::4], '--',label='just_norm', c='b',linewidth=2)
     plt.plot(kk, c[3::4], '->',label='norm_triangle', c='y',linewidth=2)

#    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
#    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
#
     plt.legend()
     plt.grid()
     plt.xlabel('dimensional')
#      plt.xlabel('k')
#      plt.ylabel('time')
     plt.ylabel('distance_calculation')
