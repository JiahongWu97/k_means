# Created by Jiahong Wu at 2019/3/20
from center_method import *
from harmerly_triangle import hamerly_sqrt
from k_means import naive_k_means
import time
import matplotlib.pyplot as plt
from try_some_functionality import generate_gaussian_clusters
from line_three import *
if __name__ == '__main__':
    kk = [2, 4, 8, 16, 32, 64, 100]
    # kk = [ 5, 25, 50, 100]
    # kk = [5]
    v = []
    # data = np.load('iris.npy')
    # data = np.load('birch_data.npy')
    # plt.scatter(data[:,0], data[:, 1], linewidths=.01, alpha=.1)
    t = []
    a, b, z = 0, 0, 0
    num_try = 20
    for k in kk:
        data, real_c, z = generate_gaussian_clusters(10000, 6, k, .2)
        print(k, '维了', z)
        for j in range(num_try):
            # data = np.random.rand(10000, 8)
    #        k = 13
            # plot_center(real_c, data, 'really')
            print(j+1, '次了')

            a = time.process_time()
            c1 = random_field(data, k)
            # plot_center(c1, data, 'random field')
            i1, j1 = hamerly_sqrt(data, k, c1)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j1))
            # plot_final(data, k, i1, j1, 'random field')

            a = time.process_time()
            c2 = random_centroid(data, k)
            # plot_center(c2, data, 'random centroid')
            i2, j2 = hamerly_sqrt(data, k, c2)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j2))
            # plot_final(data, k, i2, j2, 'random centroid')

            a = time.process_time()
            c3 = plus_plus(data, k)
            # plot_center(c3, data, 'plus_plus')
            i3, j3 = hamerly_sqrt(data, k, c3)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j3))
            # plot_final(data, k, i3, j3, 'plus_plus')

            a = time.process_time()
            c4 = greedy_plus_plus(data, k)
            # plot_center(c4, data, 'greedy ++')
            i4, j4 = hamerly_sqrt(data, k, c4)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j4))
            # plot_final(data, k, i4, j4, 'greedy ++')

            a = time.process_time()
            c5 = just_maximum(data, k)
            # plot_center(c5, data, 'maximum')
            i5, j5 = hamerly_sqrt(data, k, c5)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j5))
            # plot_final(data, k, i5, j5, 'maximum')

            a = time.process_time()
            c6 = bradley_sample(data, k)
            # plot_center(c6, data, 'sample')
            i6, j6 = hamerly_sqrt(data, k, c6)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j6))
            # plot_final(data, k, i6, j6, 'sample')

            a = time.process_time()
            c7 = forgy(data, k)
            # plot_center(c7, data, 'forgy')
            i7, j7 = hamerly_sqrt(data, k, c7)[0:2]
            b = time.process_time()
            t.append(b - a)
            v.append((j7))
            # plot_final(data, k, i7, j7, 'forgy')

#结果为啥都差不多,震惊，c123456除了顺序，其他都一致，为啥？因为在函数内部改变了呢，又是那个问题copy。还是看看常规蒙特卡罗该咋弄。
    # np.savetxt('t.txt', t)
    # np.savetxt('v.txt', v)
    # t = np.loadtxt('t.txt', )
    t = np.array(t).reshape(len(kk), num_try * 7)
    v = np.array(v).reshape(len(kk), num_try * 7)

    plt.figure()
    plt.plot(kk, np.average(t[:, 0 : num_try], 1), label='plus_plus', c='k',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try : num_try * 2], 1), '-',label='random_centroid', c='b',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try * 2 : num_try * 3], 1), '-',label='random_field', c='r',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try * 3 : num_try * 4], 1), '-',label='greedy_++', c='g',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try * 4 : num_try * 5], 1), '-',label='maximum', c='y',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try * 5 : num_try * 6], 1), '-',label='sample', c='c',linewidth=2)
    plt.plot(kk, np.average(t[:, num_try * 6 : num_try * 7], 1), '-',label='forgy', c='aqua',linewidth=2)
    plt.legend()
    plt.grid()
    # plt.xlabel('dimensional')
    plt.xlabel('k')
    # plt.ylabel('SSE value')
    plt.ylabel('time')
    plt.show()

    plt.figure()
    plt.plot(kk, np.average(v[:, 0 : num_try], 1), label='plus_plus', c='k',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try : num_try * 2], 1), '--',label='random_centroid', c='b',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 2 : num_try * 3], 1), '-+',label='random_field', c='r',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 3 : num_try * 4], 1), '-<',label='greedy_++', c='g',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 4 : num_try * 5], 1), '->',label='maximum', c='y',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 5 : num_try * 6], 1), '-^',label='sample', c='c',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 6 : num_try * 7], 1), '-^',label='forgy', c='aqua',linewidth=2)
    plt.legend()
    plt.grid()
    # plt.xlabel('dimensional')
    plt.xlabel('k')
    plt.ylabel('SSE value')
    # plt.ylabel('time')
    plt.show()

    plt.figure()
    plt.plot(kk, np.average(v[:, 0 : num_try], 1), label='plus_plus', c='k',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try : num_try * 2], 1), '--',label='random_centroid', c='b',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 2 : num_try * 3], 1), '-+',label='random_field', c='r',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 3 : num_try * 4], 1), '-<',label='greedy_++', c='g',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 4 : num_try * 5], 1), '->',label='maximum', c='y',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 5 : num_try * 6], 1), '-^',label='sample', c='c',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 6 : num_try * 7], 1), '-^',label='forgy', c='aqua',linewidth=2)
    plt.legend()
    plt.grid()
    # plt.xlabel('dimensional')
    plt.xlabel('k')
    plt.ylabel('SSE value min')
    # plt.ylabel('time')
    plt.show()

    plt.figure()
    plt.plot(kk, np.average(v[:, 0 : num_try], 1), label='plus_plus', c='k',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try : num_try * 2], 1), '--',label='random_centroid', c='b',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 2 : num_try * 3], 1), '-+',label='random_field', c='r',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 3 : num_try * 4], 1), '-<',label='greedy_++', c='g',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 4 : num_try * 5], 1), '->',label='maximum', c='y',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 5 : num_try * 6], 1), '-^',label='sample', c='c',linewidth=2)
    plt.plot(kk, np.average(v[:, num_try * 6 : num_try * 7], 1), '-^',label='forgy', c='aqua',linewidth=2)
    plt.legend()
    plt.grid()
    # plt.xlabel('dimensional')
    plt.xlabel('k')
    plt.ylabel('SSE value max')
    # plt.ylabel('time')
    plt.show()
