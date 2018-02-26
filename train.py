#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from scipy.io  import loadmat as load
import matplotlib.pyplot as plt
import time

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.001))

def model(X, w_layer_1, w_layer_2, w_layer_3, p_keep_input, p_keep_hidden):

    hidden_1 = tf.nn.relu(tf.matmul(X, w_layer_1))

    hidden_1 = tf.nn.dropout(hidden_1, p_keep_hidden)
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w_layer_2))

    hidden_2 = tf.nn.dropout(hidden_2, p_keep_hidden)


    return tf.matmul(hidden_2, w_layer_3)

# 导入数据
dict = load('./data1.mat')
trX=dict['xyl_train']
trY=dict['rgb_train']
teX=dict['xyl_test']
teY=dict['rgb_test']

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

# 在该模型中我们一共有4层，一个输入层，两个隐藏层，一个输出层
# 定义输入层到第一个隐藏层之间的连接矩阵
w_layer_1 = init_weights([3, 20])

# 定义第一个隐藏层到第二个隐藏层之间的连接矩阵
w_layer_2 = init_weights([20, 20])


# 定义第一个隐藏层到第二个隐藏层之间的连接矩阵
w_layer_3 = init_weights([20, 3])
# dropout 系数
# 定义有多少有效的神经元将作为输入神经元，比如 p_keep_intput = 0.8，那么只有80%的神经元将作为输入
p_keep_input = tf.placeholder("float")

# 定义有多少的有效神经元将在隐藏层被激活
p_keep_hidden = tf.placeholder("float")

# 构建模型
py_x = model(X, w_layer_1, w_layer_2, w_layer_3, p_keep_input, p_keep_hidden)
tf.summary.histogram('predict',py_x)

global_step=tf.Variable(0,trainable=False)
counter = global_step.assign_add(1)
lr=tf.train.exponential_decay(0.1,global_step,200000,0.5,staircase=True)
tf.summary.scalar('lr',lr)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
cost = tf.reduce_mean(tf.reduce_sum(tf.square(Y - py_x),
                     reduction_indices=[1]))

tf.summary.scalar('cost',cost)

train_op = optimizer.minimize(cost)
predict_op = tf.argmax(py_x, 1)
predict_cost = tf.argmax(cost, 1)

start_time = time.time()
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logs/t_11', sess.graph)
    
    for i in range(10000000):

        sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input:1.0, p_keep_hidden:1.0})
        step = sess.run(counter)
        
        if i % 100 == 0:
            summary_str = sess.run(merged_summary_op,feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0})
            summary_writer.add_summary(summary_str, i)
            c = sess.run(cost, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0})
            print("Iterate [%d]: time: %4.4f, cost: %.8f" % (i, time.time() - start_time,c))
            # print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, Y: teY, p_keep_input: 1.0,
            #                                                           p_keep_hidden: 1.0})))
            # plt.suptitle("learning rate=%f, training epochs=%i, with tf.truncated_normal()" % ( sess.run(cost, feed_dict={X: teX, Y: teY, p_keep_input: 1.0,
            #                                                          p_keep_hidden: 1.0}), i), size=14)
            # plt.savefig('AC8.png', dpi=300)
            # plt.show()
