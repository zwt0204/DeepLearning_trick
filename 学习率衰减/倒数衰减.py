# -*- encoding: utf-8 -*-
"""
@File    : 倒数衰减.py
@Time    : 2019/12/26 17:44
@Author  : zwt
@git   : 
@Software: PyCharm
"""

import matplotlib.pyplot as plt
import tensorflow as tf

y = []
z = []
N = 200
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(N):
        # 阶梯型衰减
        learing_rate1 = tf.train.inverse_time_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=20,
            decay_rate=0.2, staircase=True)

        # 连续型衰减
        learing_rate2 = tf.train.inverse_time_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=20,
            decay_rate=0.2, staircase=False)

        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])

        y.append(lr1)
        z.append(lr2)

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, z, 'r-', linewidth=2)
plt.plot(x, y, 'g-', linewidth=2)
plt.title('inverse_time_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()