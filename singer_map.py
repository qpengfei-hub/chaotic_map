import random

import matplotlib
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.cluster import KMeans



def singer_map(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    p = 1.07
    num = 500
    dimension = 2
    clusters = 5
    singer_map = numpy.zeros(shape=(num, dimension))

    x, y, xn = [], [], x1
    for j in range(dimension):  # singer map
        for i in range(num):
            xn = p * (7.86 * xn - 23.31 * xn ** 2 + 28.75 * xn ** 3 - 13.302875 * xn ** 4)
            singer_map[i, j] = xn

    clf = KMeans(n_clusters=clusters)
    clf.fit(singer_map)  # 分组
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    for i in range(len(labels)):
        if labels[i] == 0:
            pyplot.scatter(singer_map[i][0], singer_map[i][1], c=('r'))
        elif labels[i] == 1:
            pyplot.scatter(singer_map[i][0], singer_map[i][1], c=('green'))
        elif labels[i] == 2:
            pyplot.scatter(singer_map[i][0], singer_map[i][1], c=('black'))
        elif labels[i] == 3:
            pyplot.scatter(singer_map[i][0], singer_map[i][1], c=('orange'))
        else:
            pyplot.scatter(singer_map[i][0], singer_map[i][1], c=('purple'))

    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)
    plt.ioff()
    plt.show()
    return


def singer_map_plt(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    iters = 100
    start = 1
    end = 1.073
    xn = x1
    x, y, xn = [], [], x1
    stable_x, stable_y = [], []
    u = np.arange(start, end , 0.001)
    for p in range(int(start * 1000), int(end * 1000)):
        p /= 1000
        for i in range(iters):
            xn = p * (7.86 * xn - 23.31 * xn ** 2 + 28.75 * xn ** 3 - 13.302875 * xn ** 4)
            y.append(xn)
            x.append(i)
            if i % 1 == 0:
                stable_y.append(xn)
                stable_x.append(p)
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("tent_map,xn-n图像")
        plt.grid(True)
        plt.xlabel("n")
        plt.xlim(-1, iters)
        plt.ylabel("Xn")
        plt.ylim(-0.01, 1.01)
        plt.scatter(x, y, marker=".", linewidths=1)
        # Stable-r
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("tent_map,stable-r图像")
        plt.grid(True)
        plt.xlabel("r")
        plt.xlim(start-0.05, end+0.05)
        plt.ylabel("Stable")
        plt.ylim(-0.01, 1.01)
        plt.scatter(stable_x, stable_y, marker=".", linewidths=1)
        plt.pause(0.1)
    # 关闭交互模式
    plt.ioff()
    plt.show()
    return


def singer_map10(x0, a, b, iters):
    # x0=0.2, a = 0.9, b = 1.08, iters = 500
    x = x0
    u = np.arange(a, b, 0.001)
    for i in range(iters):
        x = u * (7.86 * x - 23.31 * x ** 2 + 28.75 * x ** 3 - 13.302875 * x ** 4)
        if i >= iters - 100:
            plt.plot(u, x, ',k')
    plt.xlabel('u')
    plt.ylabel('x')
    plt.title('Singer map')
    plt.show()

if __name__ == "__main__":
    x1 = random.random()
    singer_map(x1)
    pass