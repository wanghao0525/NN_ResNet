import tensorflow as tf
from scipy.io import loadmat as load
from scipy.io import savemat as save
from model import XYL2RGB
from config import Config as conf
from data_pro import decode_from_tfrecords,get_batch
from utils import conv2d, deconv2d, linear, batch_norm, lrelu
import numpy as np
import time
import sys
import os

dict = load(r'F:\WH\Test\center_max.mat')

xyl_train = dict["input_train"]
rgb_train = dict["output_train"]
xyl_test = dict["input_test"]
rgb_test = dict["output_test"]

model = XYL2RGB()
# 使用和保存模型代码中一样的方式来声明变量
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    saver.restore(sess,r'C:\Users\Administrator\Desktop\XYL2RGB\checkpoint56\model_100000.ckpt') # 即将固化到硬盘中的Session从保存路径再读取出来
    p=sess.run(model.predict,feed_dict={model.xyl:xyl_test,model.rgb:rgb_test})# 打印v1、v2的值和之前的进行对比
    save('./test.mat', {'rgb_test': rgb_test, 'y_test':p})