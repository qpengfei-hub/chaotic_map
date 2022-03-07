import matplotlib
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.cluster import KMeans
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D


def chaos(x1):
    """
    X(n+1) = r * Xn(1-Xn)一个极为简洁的递推公式,产生出惊人的变化。
    混沌现象图示,递归公式为X(n+1) = r * Xn(1-Xn),展示随着r从0增大的过程中Xn稳态值
    的变化的过程。x1为初始值,实际上稳态的曲线只跟r相关,与x1的值无关,当r超过3.5之后形态
    发生大的变化,产生伪随机数。对混沌现象有兴趣的朋友可以深入研究下~~
    :return: 图像
    """
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    stable_x, stable_y = [], []
    for r in range(1, 1000):
        r /= 200
        x, y, xn = [], [], x1
        for n in range(80):
            xn = r * xn * (1 - xn)
            if xn > 1:
                xn = 0.999999
            if xn < 0:
                xn = 0.000001
            y.append(xn)
            x.append(n)
            if n >= 70:
                stable_y.append(xn)
                stable_x.append(r)
        # print(r, y[0:20])
        # Xn-n
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("Chaos,xn-n图像")
        plt.grid(True)
        plt.xlabel("n")
        plt.xlim(-1, 80)
        plt.ylabel("Xn")
        plt.ylim(-0.01, 1.01)
        plt.scatter(x, y, marker=".", linewidths=1)
        # Stable-r
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Chaos,stable-r图像")
        plt.grid(True)
        plt.xlabel("r")
        plt.xlim(0, 5.5)
        plt.ylabel("Stable")
        plt.ylim(-0.01, 1.01)
        plt.scatter(stable_x, stable_y, marker=".", linewidths=1)
        plt.pause(0.1)
    # 关闭交互模式
    plt.ioff()
    plt.show()
    return


def logistic_map_kmeans_plt():
    num = 800
    dimension = 2
    clusters = 5
    r = 3.98
    xn = 0.02
    chaotic_map = numpy.zeros(shape=(num, dimension))
    matplotlib.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.figure(figsize=(12, 9), dpi=80)
    # 开启交互模式
    plt.ion()
    for n in range(3):
        for j in range(dimension):  # chaotic map
            for i in range(num):
                xn = r * xn * (1 - xn)
                if xn > 1:
                    xn = 0.99999
                if xn < 0:
                    xn = 0.00001
                chaotic_map[i, j] = xn

        clf = KMeans(n_clusters=clusters)
        clf.fit(chaotic_map)  # 分组

        centers = clf.cluster_centers_  # 两组数据点的中心点
        labels = clf.labels_  # 每个数据点所属分组

    print('centers:', centers)
    for i in range(len(labels)):
        if  labels[i] == 0:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('r'))
        elif labels[i] == 1:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('green'))
        elif labels[i] == 2:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('black'))
        elif labels[i] == 3:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('orange'))
        else:
            pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], c=('purple'))

    #  pyplot.scatter(chaotic_map[i][0], chaotic_map[i][1], cmap='rainbow')
    pyplot.scatter(centers[:, 0], centers[:, 1], marker='*', s=100)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    logistic_map_kmeans_plt()
    # chaos(0.1)
    pass