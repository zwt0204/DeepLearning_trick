# -*- encoding: utf-8 -*-
"""
@File    : Adamax.py
@Time    : 2019/12/31 16:05
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
# 通过矩阵变化得到测试集 [x0 x1]
train_data = np.vstack([x0, x]).T
# 通过随机增减一定值构造'标准'答案 h(x)=4*x+1
y = 4 * x + 1 + np.random.randn(m)

alpha = 0.9
beat = 0.999

# 随机生成[1, m]的噪声数据
# print(np.random.randn(m))


def adamax(lr, loops, epsilon):
    """
    [增量梯度下降]
    alpha:步长,
    loops:循环次数,
    epsilon:收敛精度
    """
    count = 0  # loop的次数
    thata = np.random.randn(2)  # 随机thata向量初始的值,也就是随机起点位置
    err = np.zeros(2)  # 上次thata的值，初始为0向量
    s = np.random.randn(2)
    r = np.random.randn(2)

    while count < loops:
        count += 1

        for i in range(m):
            # np.dot()两个数组的点积
            g = (np.dot(thata, train_data[i]) - y[i]) * train_data[i]

            s = s * alpha + (1 - alpha) * g
            r = np.maximum(r * beat, np.abs(g))
            s_i = s / (1 - alpha**count)
            thata = thata - lr * s_i / (r+epsilon)

        if np.linalg.norm(thata - err) < epsilon:  # 判断是否收敛
            break
        else:
            err = thata  # 没有则将当前thata向量赋值给err,作为下次判断收敛的参数之一
    print(u'Sgd结果:\tloop_counts: %d\tthata[%f, %f]' % (count, thata[0], thata[1]))
    return thata


if __name__ == '__main__':
    thata = adamax(lr=0.001, loops=10000, epsilon=1e-5)

    # 将训练数据导入stats的线性回归算法，以作验证
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    print(u'Stats结果:\tintercept(截距):%s\tslope(斜率):%s' % (intercept, slope))

    plt.plot(x, y, 'k+')
    plt.plot(x, thata[1] * x + thata[0], 'r')
    plt.show()

