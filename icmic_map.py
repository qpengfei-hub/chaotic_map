import math
import random

import matplotlib
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.cluster import KMeans



def icmic_map(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    p = 70   # 1.7 - 3.7; 4.8 - 无穷
    num = 800
    dimension = 2
    clusters = 5
    icmic_map = numpy.zeros(shape=(num, dimension))

    x, y, xn = [], [], x1
    for j in range(dimension):  # icmic map:无限折叠迭代混沌映射
        for i in range(num):
            xn = math.sin(p / xn)
            # -1 <= xn <= 1
            icmic_map[i, j] = xn

    clf = KMeans(n_clusters=clusters)
    clf.fit(icmic_map)  # 分组
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    for i in range(len(labels)):
        if labels[i] == 0:
            pyplot.scatter(icmic_map[i][0], icmic_map[i][1], c=('r'))
        elif labels[i] == 1:
            pyplot.scatter(icmic_map[i][0], icmic_map[i][1], c=('green'))
        elif labels[i] == 2:
            pyplot.scatter(icmic_map[i][0], icmic_map[i][1], c=('black'))
        elif labels[i] == 3:
            pyplot.scatter(icmic_map[i][0], icmic_map[i][1], c=('orange'))
        else:
            pyplot.scatter(icmic_map[i][0], icmic_map[i][1], c=('purple'))

    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)
    plt.ioff()
    plt.show()
    return


def icmic_map_plt(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    stable_x, stable_y = [], []
    for r in range(1, 1600):
        r /= 200
        x, y, xn = [], [], x1
        for n in range(80):
            xn = math.sin(r / xn)
            # -1 <= xn <= 1
            y.append(xn)
            x.append(n)
            if n >= 70:
                stable_y.append(xn)
                stable_x.append(r)

        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("icmic_map,xn-n图像")
        plt.grid(True)
        plt.xlabel("n")
        plt.xlim(-1, 80)
        plt.ylabel("Xn")
        plt.ylim(-1.01, 1.01)
        plt.scatter(x, y, marker=".", linewidths=1)
        # Stable-r
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("icmic_map,stable-r图像")
        plt.grid(True)
        plt.xlabel("r")
        plt.xlim(-0.01, 8.20)
        plt.ylabel("Stable")
        plt.ylim(-1.01, 1.01)
        plt.scatter(stable_x, stable_y, marker=".", linewidths=1)
        plt.pause(0.1)
    # 关闭交互模式
    plt.ioff()
    plt.show()
    return

if __name__ == "__main__":
    x1 = random.random()
    # -1 <= xn <= 1
    icmic_map_plt(x1)
    pass