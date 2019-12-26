# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2019/12/25 18:34
@Author  : zwt
@git   : 
@Software: PyCharm
"""
import numpy as np
from clr import cyclic_learning_rate


# def cyclical_learning_rate(batch_step,
#                            step_size,
#                            base_lr=0.001,
#                            max_lr=0.006,
#                            mode='triangular',
#                            gamma=0.999995):
#     cycle = np.floor(1 + batch_step / (2. * step_size))
#     x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)
#
#     lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))
#
#     if mode == 'triangular':
#         pass
#     elif mode == 'triangular2':
#         lr_delta = lr_delta * 1 / (2. ** (cycle - 1))
#     elif mode == 'exp_range':
#         lr_delta = lr_delta * (gamma ** (batch_step))
#     else:
#         raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')
#
#     lr = base_lr + lr_delta
#
#     return lr


import matplotlib.pyplot as plt

num_epochs = 50
num_train = 50000
batch_size = 100
iter_per_ep = num_train // batch_size

batch_step = -1
collect_lr = []
for e in range(num_epochs):
    for i in range(iter_per_ep):
        batch_step += 1
        cur_lr = cyclic_learning_rate(global_step=batch_step,
                                        step_size=iter_per_ep * 5)

        collect_lr.append(cur_lr)

plt.scatter(range(len(collect_lr)), collect_lr)
plt.ylim([0.0, 0.01])
plt.xlim([0, num_epochs * iter_per_ep + 5000])
plt.show()