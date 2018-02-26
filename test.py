#!/usr/bin/python
# -*- coding: UTF-8 -*

from config import Config as conf
from model import XYL2RGB
from data_pro import decode_from_tfrecords,get_batch
from utils import conv2d, deconv2d, linear, batch_norm, lrelu
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import sys
import os



#def predictor(xyl):
#    batch_size = xyl.get_shape().as_list()[0]
#    with tf.variable_scope("pred"):
#        l0 = lrelu(linear(tf.reshape(xyl, [batch_size ,-1]), 8, "l0"))
#        l1 = lrelu(batch_norm(linear(l0,16, "l1"), "l1"))
#        l2 = lrelu(batch_norm(linear(l1,16, "l2"), "l2"))
#        l3 = lrelu(batch_norm(linear(l2,8, "l3"), "l3"))
#        l4 = linear(l3,3, "l4")
#        tf.summary.histogram('l4',l4)
#    return tf.nn.tanh(l4)
        
def predictor(xyl):
    batch_size = xyl.get_shape().as_list()[0]
    with tf.variable_scope("pred"):
        l0 = lrelu(linear(tf.reshape(xyl, [batch_size ,-1]), 16, "l0"))
        l1 = lrelu(linear(l0,16, "l1"))
        l2 = linear(l1,3, "l2")
        tf.summary.histogram('l2',l2)
    return tf.nn.tanh(l2)

def train_shuffle_batch():
    xyl, rgb=decode_from_tfrecords('./train.tfrecords')
    xyl = tf.cast(xyl,tf.float32)
    rgb = tf.cast(rgb,tf.float32)
    
    predict = predictor(xyl)
    loss = tf.reduce_mean(tf.square(rgb - predict))

    saver = tf.train.Saver()
    
    #counter = 0
    start_time = time.time()
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, conf.model_path)
       
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程

        threads=tf.train.start_queue_runners(sess=sess,coord=coord)  
        
        for epoch in xrange(conf.max_epoch):
            _, m = sess.run(loss)
            print "Iterate [%d]: time: %4.4f, loss: %.8f" % (epoch, time.time() - start_time, m)

        coord.request_stop()
        coord.join(threads)          

                
                
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    #train()
    train_shuffle_batch()

