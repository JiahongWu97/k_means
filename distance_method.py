# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:23:15 2019

@author: Administrator
"""
import numpy as np
def euclidean(data, center, axis = 0):
    return np.sum(np.power(data - center, 2), axis)
def hunt_all(data, center, d = euclidean):
    min_dist, min_index = np.linalg.norm(data - center[0, :]), 0
    # min_dist, min_index = d(data, center[0, :]), 0
    for j in range(1, np.shape(center)[0]):
        dist = np.linalg.norm(data - center[j, :])
        # dist = d(data, center[j, :])
        if dist < min_dist:
            min_dist, min_index = dist, j
    return  min_index, min_dist
def hunt_all_2(data, center, d = euclidean):
    min_dist, min_index = np.dot(data - center[0, :],data - center[0, :]), 0
    # min_dist, min_index = d(data, center[0, :]), 0
    for j in range(1, np.shape(center)[0]):
        dist = np.dot(data - center[j, :],data - center[j, :])
        # dist = d(data, center[j, :])
        if dist < min_dist:
            min_dist, min_index = dist, j
    return  min_index, min_dist
def hunt_all_3(data, center, d = euclidean):
    t = data - center[0, :]
    min_dist, min_index = np.dot(t,t), 0
    # min_dist, min_index = d(data, center[0, :]), 0
    for j in range(1, np.shape(center)[0]):
        t = data - center[j, :]
        dist = np.dot(t,t)
        # dist = d(data, center[j, :])
        if dist < min_dist:
            min_dist, min_index = dist, j
    return  min_index, min_dist
if __name__ == '__main__':
     a = np.reshape(np.arange(16), (4,4))
     # data = a[0,:]
     # center=a[1,:]
     # b = np.power(data - center, 2)
     # c = np.sum(np.power(a-a, 2), 1)
#     axis, 0\1更适用于mat，array就有点奇怪了（应该是因为array会自动变成1 * n）
     import time
     b = np.random.rand(100000, 200)

     a1 = time.process_time()
     # c = np.linalg.norm(b, axis=1)
     # c = np.power(np.linalg.norm(b, axis=1), 2)
     # c = np.dot(b,b)
     hunt_all_3(b[0, :], b)
     b1 = time.process_time()

     a2 = time.process_time()
     # cc = np.sqrt(np.sum(np.power(b, 2), 1))
     # cc = (np.sum(np.power(b, 2), 1))
     # c = np.linalg.norm(b, axis=1)
     hunt_all_2(b[0, :], b)
     b2 = time.process_time()

     print(b1-a1, b2-a2)
#todo 算欧式距离，norm是真的快
#todo np.dot牛逼，但是我看不到源码，是因为更底层的语言？