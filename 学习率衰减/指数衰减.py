# -*- encoding: utf-8 -*-
"""
@File    : 指数衰减.py
@Time    : 2019/12/26 17:36
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import matplotlib.pyplot as plt
import tensorflow as tf

num_epoch = tf.Variable(0, name='global_step', trainable=False)

y = []
z = []
N = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for num_epoch in range(N):
        """
        learning_rate: 初始学习率
        global_step: 当前训练轮次，epoch
        decay_step: 定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内保持学习率不变
        decay_rate，衰减率系数
        staircase： 定义是否是阶梯型衰减，还是连续衰减，默认是False，即连续衰减（标准的指数型衰减）
        """
        # 阶梯型衰减
        learing_rate1 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)
        # 标准指数型衰减
        learing_rate2 = tf.train.exponential_decay(
            learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)
        lr1 = sess.run([learing_rate1])
        lr2 = sess.run([learing_rate2])
        y.append(lr1)
        z.append(lr2)

x = range(N)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([0, 0.55])

plt.plot(x, y, 'r-', linewidth=2)
plt.plot(x, z, 'g-', linewidth=2)
plt.title('exponential_decay')
ax.set_xlabel('step')
ax.set_ylabel('learing rate')
plt.show()