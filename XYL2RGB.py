#!/usr/bin/python
# -*- coding: UTF-8 -*

from config import Config as conf
from model import XYL2RGB
from scipy.io import loadmat as load
from data_pro import decode_from_tfrecords,get_batch
from utils import conv2d, deconv2d, linear, batch_norm, lrelu
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import sys
import os


def train():
    dict = load(r'F:\WH\Test\center_max.mat')

    xyl_train = dict["input_train"]
    rgb_train = dict["output_train"]
    xyl_test = dict["input_test"]
    rgb_test = dict["output_test"]

    print(xyl_train.shape)
    print(rgb_train.shape)
    print(xyl_test.shape)
    print(rgb_test.shape)
#    xyl_train = np.reshape(xyl_train,[train_shape,1,3,1])
#    rgb_train =np.reshape(rgb_train,[train_shape,1,3,1])
#    
#    xyl_test = np.reshape(xyl_test,[test_shape,1,3,1])
#    rgb_test =np.reshape(rgb_test,[test_shape,1,3,1])
    
    global_step=tf.Variable(0,trainable=False)
    counter = global_step.assign_add(1)
    lr=tf.train.exponential_decay(0.1,global_step,100000,0.1,staircase=True)
    tf.summary.scalar('lr',lr)

    model = XYL2RGB()

    # xyls,rgbs=get_batch(xyl_train,rgb_train,28)

    opt = tf.train.GradientDescentOptimizer(lr).minimize(model.loss)

    saver = tf.train.Saver()
    
    start_time = time.time()
    if not os.path.exists(conf.data_path + "/checkpoint"):
        os.makedirs(conf.data_path + "/checkpoint")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs/t_56', sess.graph)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess, coord)
        for i in range(100000):
            # xyl_batch_v, rgb_batch_v = sess.run([xyls, rgbs])
            # _, m = sess.run([opt, model.loss], feed_dict={model.xyl:xyl_batch_v, model.rgb:rgb_batch_v})
            _, m = sess.run([opt, model.loss], feed_dict={model.xyl: xyl_train, model.rgb: rgb_train})
            step = sess.run(counter)
            if (i+1) % 100  == 0:

                print("Iterate [%d]: time: %4.4f, train_loss: %.8f" % (i+1, time.time() - start_time, m))
                m_test = sess.run(model.loss, feed_dict={model.xyl:xyl_test, model.rgb:rgb_test})

                print("Iterate [%d]: time: %4.4f, test_loss: %.8f" % (i + 1, time.time() - start_time, m_test))

                summary_str = sess.run(merged_summary_op,feed_dict={model.xyl:xyl_test, model.rgb:rgb_test})
                summary_writer.add_summary(summary_str, i+1)
            if (i+1) % 1000  == 0:
                save_path = saver.save(sess, conf.data_path + "/checkpoint56/" + "model_%d.ckpt" % (i+1))
        # coord.request_stop()
        # coord.join(threads)

def train_shuffle_batch():
    dict = load(r'F:\WH\Test\single_8_14_x.mat')

    xyl = dict["input_train"]
    rgb = dict["output_train"]
    xyl_test = dict["input_test"]
    rgb_test = dict["output_test"]

    xyls,rgbs=get_batch(xyl,rgb,conf.batch_size)#batch
    
    predict = predictor(xyls)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(rgb - predict),1))
    tf.summary.scalar('loss',loss)
    
    opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    saver = tf.train.Saver()
    
    #counter = 0
    start_time = time.time()
    if not os.path.exists(conf.data_path + "/checkpoint"):
        os.makedirs(conf.data_path + "/checkpoint")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if conf.model_path == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./logs/t_1', sess.graph)
        
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        #启动QueueRunner, 此时文件名队列已经进队。
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)  
        
        for epoch in xrange(2000000):
            _, m = sess.run([opt, loss])
            
            if (epoch + 1) % 100  == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, epoch+1)
                print("Iterate [%d]: time: %4.4f, loss: %.8f" % (epoch+1, time.time() - start_time, m))
            if (epoch + 1) % 100000  == 0:
                save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))   
        
        coord.request_stop()
        coord.join(threads)          

                
                
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    train()
    # train_shuffle_batch()

