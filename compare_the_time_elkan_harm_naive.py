# -*- coding: utf-8 -*-
"""f
Created on Fri Mar  1 20:53:00 2019

@author: Administrator
"""
from k_means import naive_k_means
from harmerly_triangle import hamerly_sqrt
from elkan_triangle import elkan_sqrt, elkan_square
from distance_method import *
from center_method import *
from box_kdtree import kdtree_kmeans
from line_three import *
#from time import time
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, rand
#array([ 8.8125  ,  2.671875,  0.875   , 46.953125,  1.84375 ,  0.875   ,
#       48.46875 ,  1.671875,  0.90625 , 83.8125  ,  1.65625 ,  1.109375,
#       38.25    ,  1.640625,  1.328125, 39.796875,  1.703125,  1.328125,
#       49.21875 ,  1.75    ,  1.765625, 78.125   ,  1.6875  ,  1.578125,
#       44.828125,  1.875   ,  1.671875])
if __name__ == '__main__':
    nn = [200,500,1000,5000,10000,100000,300000]
    s = []
    nn = [200,500,1000,5000,8000,10000,30000]
    n=2000
#    dim = 3
    kk = [2,3,4,5,6,8,10,12,14,16,20]
    k = 10
#    kk=[3];
    dim = 30
#    for dim in kk:    
    for n in nn:
        data = np.array([[(rand()*2-1)+randn()*0.05*j for j in range(1, dim+1)] for i in range(n)])
        center = plus_plus(data, k)
    
#        print('dimensional',dim)
        print('number', n)
        a = time.process_time()
        index, table =  hamerly_sqrt(data, k, center.copy())  #函数内部涉及到直接赋值，强行改变了，得copy
#        index, table = elkan_sqrt(data, k, center)
        b = time.process_time()
        s.append(b-a)
#        my_plot(data, center, index, table, 0, 'hamerly_sqrt')


        a = time.process_time()
        index, table = naive_k_means(data, k, center.copy())
        b = time.process_time()
        s.append(b-a)
#        my_plot(data, center, index, table, 0, 'naive')

        a = time.process_time()
        index, table = elkan_sqrt(data, k, center.copy())
        b = time.process_time()
#        my_plot(data, center, index, table, 0, 'elkan_square')
        s.append(b-a)
        
#        a = time.process_time()
#        index, table = kdtree_kmeans(data, k, center.copy())
#        b = time.process_time()
#        s.append(b-a)
#        my_plot(data, center, index, table, 0, 'kdtree_kmeans')

        
    kk = nn
#    plt.plot(kk, s[1::4], label='naive', c='k',linewidth=2)
#    plt.plot(kk, s[2::4], '--',label='Elkan', c='b',linewidth=2)
#    plt.plot(kk, s[0::4], '-+',label='Hamerly', c='r',linewidth=2)
#    plt.plot(kk, s[3::4], '--o',label='kdtree', c='m',linewidth=2)

    plt.plot(kk, s[1::3], label='naive', c='k',linewidth=2)
    plt.plot(kk, s[2::3], '--',label='Elkan', c='b',linewidth=2)
    plt.plot(kk, s[0::3], '-+',label='Hamerly', c='r',linewidth=2)
    
#    plt.plot(kk, s[0::2], label='square', c='k',linewidth=2)
#    plt.plot(kk, s[1::2], '*',label='sqrt', c='r',linewidth=2)
#    
    plt.legend()
    plt.grid()
#    plt.xlabel('dimensional')
    plt.xlabel('number')
    plt.ylabel('time')
#    
    