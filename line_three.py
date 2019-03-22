# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:37:01 2018

@author: Administrator
"""

#from matplotlib import *
#from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# color = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k']
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 'k','aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
mar = ['<', '*', '+', 'v', '.', 'o', ',', '^', '1', '2', '3', '4', '8','s', 'p', 'P', 'h', 'H', 'x', 'X', 'D', 'd', '|', '_', '<', '*','+', '<', '*', '+', 'v', '.', 'o', ',', '^', '1', '2', '3', '4','8', 's', 'p', 'P', 'h', 'H', 'x', 'X', 'D', 'd', '|', '_', '<','*', '+', '<', '*', '+', 'v', '.', 'o', ',', '^', '1', '2', '3','4', '8', 's', 'p', 'P', 'h', 'H', 'x', 'X', 'D', 'd', '|', '_','<', '*', '+', '<', '*', '+', 'v', '.', 'o', ',', '^', '1', '2','3', '4', '8', 's', 'p', 'P', 'h', 'H', 'x', 'X', 'D', 'd', '|','_', '<', '*', '+']


#ax.plot([0,1],[0,2],[4,5])
def my_plot(data, center, index, tab, string = 'naive', offset=0):
    # color = ['b', 'y', 'r', 'g', 'm', 'c', 'k'];
    # mar = ['+','o','v',',','>','*','.'];
#    '-':    '_draw_solid',
#        '--':   '_draw_dashed',
#        '-.':   '_draw_dash_dot',
#        ':':    '_draw_dotted',
    fig = plt.figure()
    ax = fig.gca( projection='3d')
    k = np.shape(center)[0]
    for i in range(k):
        row = np.nonzero(index==i)[0] #b[a==3,:]即可
        x = data[row][:,0+offset]
        y = data[row][:,1+offset]
        z = data[row][:,2+offset]
        ax.scatter(x,y,z,c=color[i],marker=mar[i],s=75)
        for j in range(len(row)):
#            连center和data
            ax.plot([x[j],center[i,0+offset]],[y[j],center[i,1+offset]],[z[j],center[i,2+offset]],c=color[i],ls='-',lw=.2,alpha = 0.5)
    
    plt.title('SSE: '+str(sum(tab))+' '+string)
    plt.show()



def plot_center(center, data, char, offset=0):
    fig = plt.figure()
    k = np.shape(center)[0]
    if np.shape(center)[1] >= 3:
        ax = fig.gca(projection='3d')
        for i in range(k):
            x = center[i,0+offset]
            y = center[i,1+offset]
            z = center[i,2+offset]
            ax.scatter(x,y,z,s=75,c=color[i],marker=mar[i])
        x = data[:,0+offset]
        y = data[:,1+offset]
        z = data[:,2+offset]
        ax.scatter(x,y,z,c='k',s=1, alpha = .007)
    else:
        for i in range(k):
            x = center[i,0+offset]
            y = center[i,1+offset]
            plt.scatter(x,y,c=color[i],marker=mar[i],s=75, alpha = 1)
        x = data[:,0+offset]
        y = data[:,1+offset]
        plt.scatter(x,y,c='k',s=1, alpha = .02)
    plt.title('initial centers:'+ ' ' +char)
    plt.show()
def plot_final(data, k, index, j, char, offset=0):
    fig = plt.figure()
    col = np.shape(data)[1]
    center = np.zeros((k, col))
    for i in range(k):
        center[i, :] = np.average(data[index == i, :], 0)
    if col >= 3:
        ax = fig.gca(projection='3d')
        for i in range(k):
            x = center[i,0+offset]
            y = center[i,1+offset]
            z = center[i,2+offset]
            ax.scatter(x,y,z,s=55,c=color[i],marker=mar[i])
            x = data[index == i,0+offset]
            y = data[index == i,1+offset]
            z = data[index == i,2+offset]
            ax.scatter(x,y,z,c=color[i],s=5, alpha = .1)
    else:
        for i in range(k):
            x = center[i,0+offset]
            y = center[i,1+offset]
            plt.scatter(x,y,s=55,c='k',marker=mar[i])
            x = data[index == i,0+offset]
            y = data[index == i,1+offset]
            plt.scatter(x,y,c=color[i],s=5, alpha = .1)
    plt.title('final:'+ ' ' +char+' '+str(j))
    plt.show()