# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:20:13 2018

@author: Administrator
"""

from matplotlib import *
#fig = pyplot.figure()
#fig.suptitle('Wawawa')
#fig, ax = pyplot.subplots(2,2)

#fig.show()
x = np.linspace(0,2,100)
pyplot.plot(x, x, label='linear')
pyplot.plot(x, x**2,label='quadratic')
pyplot.plot(x, x**3, label='cubic')
pyplot.title(2)
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.legend()
pyplot.show()


x = arange(0, 10, 0.2)
y = sin(x)
fig, ax = pyplot.subplots()
ax.plot(x, y)
pyplot.show()

pyplot.xlim((0,1))
pyplot.ylim((0,3))
pyplot.xticks(linspace(-1,1,5))

import numpy as np
import matplotlib.pyplot as plt

# 数据个数
n = 1024
# 均值为0, 方差为1的随机数
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)

# 计算颜色值
color = np.arctan2(y, x)
# 绘制散点图
plt.scatter(x, y, s = 75, c = color, alpha = 0.5)
# 设置坐标轴范围
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))

# 不显示坐标轴的值
plt.xticks(())
plt.yticks(())

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# 定义figure
fig = plt.figure()
# 将figure变为3d
ax = Axes3D(fig)

# 数据数目
n = 256
# 定义x, y
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)

# 生成网格数据
X, Y = np.meshgrid(x, y)

# 计算每个点对的长度
R = np.sqrt(X ** 2 + Y ** 2)
# 计算Z轴的高度
Z = np.sin(R)

# 绘制3D曲面
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
# 绘制从3D曲面到底部的投影
ax.contour(X, Y, Z, zdim = 'z', offset = -2, cmap = 'rainbow')

# 设置z轴的维度
ax.set_zlim(-2, 2)





pyplot.subplot(1,2,1)
ax.plot([0,1],[0,2],[4,5])
pyplot.subplot(1,2,2)
pyplot.plot([3,1],[4,2])


plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

import numpy as np

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.scatter([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

