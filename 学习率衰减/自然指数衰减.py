# -*- encoding: utf-8 -*-
"""
@File    : 自然指数衰减.py
@Time    : 2019/12/26 17:40
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import matplotlib.pyplot as plt
import tensorflow as tf

num_epoch = tf.Variable(0, name='global_step', trainable=False)

y = []
z = []
w = []
m = []
N = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for num_epoch in range(N):
        # 阶梯型衰减
        learing_rate1 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)

        # 标准指数型衰减
        learing_rate2 = tf.train.natural_exp_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)

        # 阶梯型指数衰减
        learing_rate3 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)

        # 标准指数衰减
        learing_rate4 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)

        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        lr3 = sess.run([learing_rate3])
        lr4 = sess.run([learing_rate4])

        y.append(lr1)
        z.append(lr2)
        w.append(lr3)
        m.append(lr4)

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])

plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.plot(x, w, 'r-', linewidth=2)
plt.plot(x, m, 'g-', linewidth=2)

plt.title('natural_exp_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()