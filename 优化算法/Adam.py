# -*- encoding: utf-8 -*-
"""
@File    : Adam.py
@Time    : 2019/12/31 15:56
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.arange(0., 10., 0.2)
m = len(x)
x0 = np.full(m, 1.0)
train_data = np.vstack([x0, x]).T
y = 4 * x + 1 + np.random.randn(m)

beat1 = 0.9
beat2 = 0.999
# 衰减力度,可以用来调节,该值越大那么之前的梯度对现在方向的影响也越大
eps = 0.9


def Adam(alpha, loops, epsilon):
    """

    :param alpha:
    :param loops:
    :param epsilon:
    :return:
    """
    count = 0  # loop的次数
    theta = np.random.randn(2)  # 随机theta向量初始的值,也就是随机起点位置
    k = np.random.randn(2)
    l = np.random.randn(2)
    err = np.zeros(2)  # 上次thata的值，初始为0向量

    while count < loops:
        count += 1

        for i in range(m):
            # np.dot()两个数组的点积
            cost = (np.dot(theta, train_data[i]) - y[i]) * train_data[i]
            k = beat1 * k + (1 - beat1) * cost
            l = beat2 * l + (1 - beat2) * (cost ** 2)
            kb = k / (1 - beat1 ** count)
            lb = l / (1 - beat2 ** count)

            theta = theta - alpha * kb / (np.sqrt(lb) + epsilon)

        if np.linalg.norm(theta - err) < epsilon:  # 判断是否收敛
            break
        else:
            err = theta  # 没有则将当前thata向量赋值给err,作为下次判断收敛的参数之一
    print(u'Sgd结果:\tloop_counts: %d\ttheta[%f, %f]' % (count, theta[0], theta[1]))
    return theta


if __name__ == '__main__':
    theta = Adam(alpha=0.001, loops=10000, epsilon=1e-8)

    # 将训练数据导入stats的线性回归算法，以作验证
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    print(u'Stats结果:\tintercept(截距):%s\tslope(斜率):%s' % (intercept, slope))

    plt.plot(x, y, 'k+')
    plt.plot(x, theta[1] * x + theta[0], 'r')
    plt.show()