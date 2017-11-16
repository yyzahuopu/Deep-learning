# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:53:51 2017

@author: SKJ
"""

import tensorflow as tf

#MNIST数据集相关的常数
INPUT_NODE = 784    #输入层的节点数，这里等于图片像素
OUTPUT_NODE = 10    #输出层节点数，这里需要区分0~9这10个数字

LAYER1_NODE = 500   #隐藏层节点数，这里使用只有一个隐藏层的网络结构

def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,
       initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

#定义前向传播
def inference(input_tensor,regularizer,reuse=False):
    with tf.variable_scope('layer1',reuse=reuse):
        weights = get_weight_variable(
                [INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],
                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
        
    with tf.variable_scope('layer2',reuse=reuse):
        weights = get_weight_variable(
                [LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases
    return layer2
 




























