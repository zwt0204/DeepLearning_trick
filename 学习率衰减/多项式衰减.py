# -*- encoding: utf-8 -*-
"""
@File    : 多项式衰减.py
@Time    : 2019/12/26 17:42
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
        # cycle=False
        learing_rate1 = tf.train.polynomial_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50,
            end_learning_rate=0.01, power=0.5, cycle=False)
        # cycle=True
        learing_rate2 = tf.train.polynomial_decay(
            learning_rate=0.1, global_step=global_step, decay_steps=50,
            end_learning_rate=0.01, power=0.5, cycle=True)

        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        y.append(lr1)
        z.append(lr2)

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, z, 'g-', linewidth=2)
plt.plot(x, y, 'r--', linewidth=2)
plt.title('polynomial_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()