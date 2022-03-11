import random
import math
import matplotlib
import numpy
import numpy as geek  # geek.mod()
from operator import mod  # mod()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.cluster import KMeans


def circle_map(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    num = random.randint(1000,5000)
    dimension = 2
    clusters = 5
    circle_map = numpy.zeros(shape=(num, dimension))

    x, y = [], []
    for j in range(dimension):  # circle map
        xn = random.random()
        for i in range(num):
            xn = mod(xn + 0.2 - 0.5 / (2 * math.pi) * math.sin(2 * math.pi * xn), 1)
            circle_map[i, j] = xn

    clf = KMeans(n_clusters=clusters)
    clf.fit(circle_map)  # 分组
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    for i in range(len(labels)):
        if labels[i] == 0:
            pyplot.scatter(circle_map[i][0], circle_map[i][1], c=('r'))
        elif labels[i] == 1:
            pyplot.scatter(circle_map[i][0], circle_map[i][1], c=('green'))
        elif labels[i] == 2:
            pyplot.scatter(circle_map[i][0], circle_map[i][1], c=('black'))
        elif labels[i] == 3:
            pyplot.scatter(circle_map[i][0], circle_map[i][1], c=('orange'))
        else:
            pyplot.scatter(circle_map[i][0], circle_map[i][1], c=('purple'))

    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=500)
    plt.ioff()
    plt.show()
    return


def circle_map_plt(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    stable_x, stable_y = [], []
    x, y, xn = [], [], x1
    for n in range(500):
        xn = mod(xn + 0.2 - 0.5 / (2 * math.pi) * math.sin(2 * math.pi * xn), 1)
        y.append(xn)
        x.append(n)
        if n % 10 == 0:
            stable_y.append(xn)
            stable_x.append(n / 5)
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("circle_map,xn-n图像")
        plt.grid(True)
        plt.xlabel("n")
        plt.xlim(-1, 500)
        plt.ylabel("Xn")
        plt.ylim(-0.01, 1.01)
        plt.scatter(x, y, marker=".", linewidths=1)
        # Stable-r
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("circle_map,stable-r图像")
        plt.grid(True)
        plt.xlabel("r")
        plt.xlim(-0.01, 100)
        plt.ylabel("Stable")
        plt.ylim(-0.01, 1.01)
        plt.scatter(stable_x, stable_y, marker=".", linewidths=1)
        plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()
    plt.show()

    return

if __name__ == "__main__":
    x1 = random.random()
    circle_map_plt(x1)
    pass