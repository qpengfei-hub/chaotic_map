import matplotlib
import numpy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.cluster import KMeans



def pwlcm(x1):
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    p = 0.7
    num = 800
    dimension = 2
    clusters = 5
    chaotic_map = numpy.zeros(shape=(num, dimension))
    x, y, xn = [], [], x1
    for j in range(dimension):  # chaotic map
        for i in range(num):
            if xn < p and xn > 0:
                xn = xn / p
            else:
                xn = (1 - xn) / (1 - p)
            chaotic_map[i, j] = xn

    clf = KMeans(n_clusters=clusters)
    clf.fit(chaotic_map)  # 分组
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    for i in range(len(labels)):
        if labels[i] == 0:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('r'))
        elif labels[i] == 1:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('green'))
        elif labels[i] == 2:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('black'))
        elif labels[i] == 3:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('orange'))
        else:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('purple'))

    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=500)
    plt.ioff()
    plt.show()
    return


if __name__ == "__main__":
    x1 = random.random()
    # x1 = 0.002
    pwlcm(x1)
    pass