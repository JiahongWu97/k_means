# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:29:28 2019

@author: Administrator
"""
import numpy as np
def a45(b):
    b[0, :] = 0
    return 3
def aa(b):
    b[0]=4
def aaa(b):
    b=np.arange(2)
    return b
def re():
    return 1,2,3,4
def normalization(da):
    data = da.copy()
    mi = data.min(0)
    ma = data.max(0)
    for i in range(np.shape(data)[1]):
        data[:, i] = (data[:, i] - mi[i]) / (ma[i] - mi[i])
    return data
def generate_gaussian_clusters(n, dim, k, var = .05):
    data = np.random.randn(n, dim) * var
    center = np.random.sample((k, dim))
    c = 0
    for i in range(n):
        # z =
        c += np.dot(data[i, :], data[i, :])
        data[i, :] += center[np.random.randint(k), :]
    return data, center, c
from sklearn.cluster import KMeans
if __name__ == '__main__':
    #todo 函数返回较多，取其中
    a = re()[0:2] #tuple
    np.random.seed()
    #todo 试试归一化
    # a = np.reshape(np.arange(16, dtype= 'float'), (4,4))
    # a = np.reshape(np.arange(16), (4,4))
    # b = normalization(a)
    a, b = generate_gaussian_clusters(60000,2,4)
    import matplotlib.pyplot as plt
    plt.scatter(a[:, 0], a[:, 1], alpha=.1,s=1)
    plt.scatter(b[:, 0], b[:, 1], marker='d',s=23,c='k')
    # c = np.random.rand(2,4)
    # print(c)
    # b=[0]
    # aa(b)
#    print(c)
#    d = a(c)
#    print(c)
#    c = np.row_stack((c, c[1,:]))
##    c.tolist()
##    d = [i*j for j in range(4) for i in range(3)]
##    b = [5]
##    aa(b)
#    c[0,:] = np.ones(2)
#    a = np.random.rand(2)
#    b = np.random.rand(1)
#    c = np.hstack((4, 3))
    
#    c[0, :] = aaa(c[0, :])
#     c[0,1:3] = np.arange(2)
#     print(c)

# #todo 测试下在循环中变量的改变在条件中带来的影响
#     t = 1
#     for i in range(5):
#         if t >= 0:
#             print('ok', t)
#         t -= 1

    # i, j =2,3
    # a, b = 0, 1 if i < j#todo 必须要有else +值 ；且两个不得行

    #todo for循环有变量i， 在list生成语句中用i是否有影响
    # for i in range(5):
    #     print(i)
    #     a = [i for i in range(2)] #看来是没啥影响了
    #     print(a)

    #todo 大循环小循环的时间问题,好像真的外层循环数小点能够快点。
    # import time
    # a = time.process_time()
    # c = []
    # for i in range(1000000):
    #     for j in range(10):
    #       c.append(np.sqrt(i*j))
    #       # c = max(c)
    # b = time.process_time()
    # print(b - a, '外层大')
    #
    # a1 = time.process_time()
    # d = []
    # for i1 in range(10):
    #     for j1 in range(1000000):
    #       d.append(np.sqrt(i1*j1))
    #       # d = max(d)
    # b1 = time.process_time()
    # print(b1 - a1, '内层大')
#todo 看看随机采样整数，且不重复，是否好办
    # import random
    # a = random.sample(range(3), 3))
    # print(random.sample(range(6), 3)) #这个可以
    # a=np.random.rand(2)
    # b = np.row_stack((a,a))  #貌似生成了新的，不影响了
    # a = b