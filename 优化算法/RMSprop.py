# -*- encoding: utf-8 -*-
"""
@File    : RMSprop.py
@Time    : 2019/12/31 15:48
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 构造训练数据h(x) = thata0 * x0 + thata1 * x1
x = np.arange(0., 20., 0.2)
m = len(x)

x0 = np.full(m, 0.1)
# 通过矩阵变化得到测试集[x0 x1]
train_data = np.vstack([x0, x]).T
# 通过随机增减一定值构造标准答案h(x)=4*x+1
# randn函数返回一个或一组样本，具有标准正态分布
y = 4 * x + 1 + np.random.randn(m)
decay_rate = 0.99


def RMSprop(alpha, loops, epsilon):
    """
    AdaDelta的一种形式
    :param alpha:
    :param loops:
    :param epsilon:
    :return:
    """
    count = 0
    # 随机theta向量初始的值,也就是随机起点位置
    theta = np.random.randn(2)
    err = np.zeros(2)

    while count < loops:
        count += 1
        cache = np.zeros(2)
        for i in range(m):
            # 求导结果
            cost = (np.dot(theta, train_data[i]) - y[i]) * train_data[i]

            cache = decay_rate * cache + (1 - decay_rate) * cost ** 2
            theta = theta - alpha * cost / (np.sqrt(cache) + epsilon)
        if np.linalg.norm(theta - err) < epsilon:
            break
        else:
            # 如果没有则将当前theta向量赋值给err，作为下次判断收敛的参数
            err = theta
    print('Bgd结果:\t loop_counts: %d \t theta[%f, %f]' % (count, theta[0], theta[1]))
    return theta


if __name__ == '__main__':
    theta = RMSprop(alpha=0.0001, loops=10000, epsilon=1e-5)

    # 将训练数据导入stats的线性回归算法，以作验证
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    print(u'Stats结果:\tintercept(截距):%s\tslope(斜率):%s' % (intercept, slope))

    plt.plot(x, y, 'k+')
    plt.plot(x, theta[1] * x + theta[0], 'r')
    plt.show()