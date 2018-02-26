from config import Config as conf
from utils import conv2d, deconv2d, linear, batch_norm, lrelu
import tensorflow as tf
import numpy as np
class XYL2RGB(object):

    def __init__(self):
        self.xyl = tf.placeholder("float", shape=(None,3))
        self.rgb = tf.placeholder("float", shape=(None,3))

        self.predict = self.predictor(self.xyl)
        tf.summary.histogram('predict',self.predict)
        
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.rgb - self.predict),1))
        tf.summary.scalar('loss',self.loss)
        
    def res_block(self,x,block_name):
        with tf.variable_scope(block_name):
            l1 = linear(tf.sigmoid(x),64, "l") + x
        return l1
    
    #log33  
    def predictor(self, xyl):
        with tf.variable_scope("pred"):
            l = linear(xyl, 64, "l0")
            tf.summary.histogram('l',l)
            for i in range(56): #34,56,110
                l = self.res_block(l,'b%d'%(i+1))
            output = linear(tf.sigmoid(l),3, "output")

            return tf.sigmoid(output)