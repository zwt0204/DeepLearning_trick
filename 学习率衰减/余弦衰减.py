# -*- encoding: utf-8 -*-
"""
@File    : 余弦衰减.py
@Time    : 2019/12/26 17:44
@Author  : zwt
@git   : 
@Software: PyCharm
"""
# coding:utf-8
import matplotlib.pyplot as plt
import tensorflow as tf

y = []
z = []
w = []
N = 200
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        # 余弦衰减
        learing_rate1 = tf.train.cosine_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50)

        # 线性余弦衰减
        learing_rate2 = tf.train.linear_cosine_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50,
            num_periods=0.2, alpha=0.5, beta=0.2)

        # 噪声线性余弦衰减
        learing_rate3 = tf.train.noisy_linear_cosine_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50,
            initial_variance=0.01, variance_decay=0.1, num_periods=0.2, alpha=0.5, beta=0.2)

        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        lr3 = sess.run([learing_rate3])
        y.append(lr1)
        z.append(lr2)
        w.append(lr3)

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, z, 'b-', linewidth=2)
plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, w, 'g-', linewidth=2)
plt.title('cosine_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()