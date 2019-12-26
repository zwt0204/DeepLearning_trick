# -*- encoding: utf-8 -*-
"""
@File    : 分段常数衰减.py
@Time    : 2019/12/26 17:03
@Author  : zwt
@git   : 
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import tensorflow as tf

num_epoch = tf.Variable(0, name='global_step', trainable=False)
boundaries = [10, 20, 30]
learing_rates = [0.1, 0.07, 0.025, 0.0125]

y = []
N = 40

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for num_epoch in range(N):
        learing_rate = tf.train.piecewise_constant(num_epoch, boundaries=boundaries, values=learing_rates)
        # num_epoch训练参数
        # boundaries 学习率参数应用区间列表
        # learing_rates 学习率列表
        lr = sess.run([learing_rate])
        y.append(lr)

x = range(N)
plt.plot(x, y, 'r-', linewidth=2)
plt.title('piecewise_constant')
plt.show()