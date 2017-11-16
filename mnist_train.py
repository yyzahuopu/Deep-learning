# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:00:14 2017

@author: SKJ
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

#配置神经网络参数
BATCH_SIZE = 100    #一个训练batch中的训练数据个数，数字越小，训练
                    #过程越接近随机梯度下降，数字越大越接近梯度下降
LEARNING_RATE_BASE = 0.8      #基础学习率
LEARNING_RATE_DECAY = 0.99    #学习率的衰减率
REGULARIZATION_RATE = 0.0001  #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000        #训练轮数
MOVING_AVERAGE_DECAY = 0.99   #滑动平均衰减率
#模型保存的路径和文件名
MODEL_SAVE_PATH = "C:/Users/SKJ/Documents/GitHub/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入输出的placeholder
    x = tf.placeholder(tf.float32,
            [None,mnist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,
            [None,mnist_inference.OUTPUT_NODE],name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    y = mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    
    #定义损失函数、学习率、滑动平均操作及训练过程
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,global_step)    
    #在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(
            tf.trainable_variables())    
    #计算交叉熵作为损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(y_,1),logits=y)
    #计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,   #基础学习率，随着迭代进行，更新变量时
                              #使用的学习率在这个基础上衰减
        global_step,          #当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  #过完所有训练数据需要
                                                #的迭代次数
        LEARNING_RATE_DECAY)  #学习率衰减速度

    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                   .minimize(loss,global_step=global_step)
    
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    #初始化TensorFLow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        #训练过程中不测试模型在验证数据上的表现，验证和测试过程由一个独立程序完成
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run(
            [train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            #每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s),loss on training "
                      "batch is %g." % (step,loss_value))
                
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                           global_step=global_step)

#主程序入口
def main(argv=None):#argv=1
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets(
            r"C:\Users\SKJ\Documents\GitHub\MNIST_data",one_hot=True)
    train(mnist)
  
#TF提供的一个主程序入口，会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
    

  

















