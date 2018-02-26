#!/usr/bin/python
# -*- coding: UTF-8 -*

#tensorflow高效数据读取训练
import tensorflow as tf
from config import Config as conf
from utils import imread
import scipy.misc
import os
import scipy.io as sio


#奖数据打包，转换成tfrecords格式，以便后续高效读取
def encode_to_tfrecords(path,data_name='data', resize=conf.adjust_size):
    writer=tf.python_io.TFRecordWriter(path+ '/' + data_name + '.tfrecords')
    num_example=0
    
    data_fn = './data1.mat'
    data = sio.loadmat(data_fn)
    xyl_train = data["xyl_train"]
    rgb_train = data["rgb_train"]
    xyl_test = data["xyl_test"]
    rgb_test = data["rgb_test"]
    print(xyl_test.dtype)
    l1 = xyl_train.shape[0]
    l2 = xyl_test.shape[0]
    for i in range(l2):
        xyl = xyl_test[i,:].reshape((1,3))
        rgb = rgb_test[i,:].reshape((1,3))
        
        example=tf.train.Example(features=tf.train.Features(feature={
            'xyl':tf.train.Feature(bytes_list=tf.train.BytesList(value=[xyl.tobytes()])),
            'rgb':tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb.tobytes()])),
        }))
        serialized=example.SerializeToString()
        writer.write(serialized)
        num_example+=1
    print("样本数据量：",num_example)
    writer.close()
    
#读取tfrecords文件
def decode_from_tfrecords(filename,size=conf.adjust_size,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    example=tf.parse_single_example(serialized,features={
        'xyl':tf.FixedLenFeature([],tf.string),
        'rgb':tf.FixedLenFeature([],tf.string),
    })
    xyl=tf.decode_raw(example['xyl'],tf.float64)
    xyl=tf.reshape(xyl,[1,3])
    rgb=tf.decode_raw(example['rgb'],tf.float64)
    rgb=tf.reshape(rgb,[1,3])
    return xyl,rgb
    
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充
def get_batch(xyl, rgb,batch_size=1):
    xyl = tf.cast(xyl, tf.float32)
    rgb = tf.cast(rgb, tf.float32)
    # input_queue = tf.train.slice_input_producer([xyl, rgb], shuffle=False)
    # xyls, rgbs = tf.train.shuffle_batch([xyl,rgb],batch_size=batch_size,
    #                                              num_threads=1,capacity=50000,min_after_dequeue=20000)
    xyls,rgbs= tf.train.batch([xyl, rgb],batch_size=batch_size,num_threads=1,capacity=70,enqueue_many=True)

    # 调试显示
    #tf.image_summary('images', images)
    return xyls, rgbs
#这个是用于测试阶段，使用的get_batch函数
def get_test_batch(xyl, rgb):
    xyls, rgbs=tf.train.batch([xyl, rgb],num_threads=1,batch_size=1)
    return xyls, rgbs
    

    
def test():
    xyl, rgb=decode_from_tfrecords('./train.tfrecords')
    xyls,rgbs=get_batch(xyl,rgb,1)#batch 生成测试
    
#   test_xyl, test_rgb=decode_from_tfrecords(path + '/test.tfrecords',size=conf.train_size)
#   test_xyls, test_rgbs=get_test_batch(test_xyl, test_rgb)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        #启动QueueRunner, 此时文件名队列已经进队。
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)  
        for i in range(10):
            xyl, rgb = sess.run([xyls, rgbs])  
            print(xyl,rgb)
        coord.request_stop()
        coord.join(threads)
    




if __name__ == '__main__':
    path = '.'
    encode_to_tfrecords(path,'test')
#    test()

