# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:49:22 2017

@author: SKJ
"""

import tensorflow as tf
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#在shape的一个维度上使用None可以方便使用不同的batch大小
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#前向传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(
        y_ *tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#定义规则来给出样本的标签，这里所有x1+x2<1的样例都被认为是正样本，其它为负
Y= [[int(x1 + x2) < 1] for (x1,x2) in X]

#创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    #初始化变量
    sess.run(init_op)
    
    print(sess.run(w1))
    print(sess.run(w2))
    
    """
    先运行得到在训练之前神经网络参数的值：
    w1 = [[-0.81131822  1.48459876  0.06532937]
          [-2.4427042   0.0992484   0.59122431]]
    w2 = [[-0.81131822],[ 1.48459876],[ 0.06532937]]
    """
    
    #设定训练轮数
    SETPS = 5000
    for i in range(SETPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            #每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on \
                   all data is %g" %(i,total_cross_entropy))
            """
            输出
            After 0 training step(s),cross entropy on all data is 0.0674925
            After 1000 training step(s),cross entropy on all data is 0.0163385
            After 2000 training step(s),cross entropy on all data is 0.00907547
            After 3000 training step(s),cross entropy on all data is 0.00714436
            After 4000 training step(s),cross entropy on all data is 0.00578471
            """     
    print(sess.run(w1))
    print(sess.run(w2))
    """
    训练后
    w1 = [[-1.96182752  2.58235407  1.68203771]
          [-3.46817183  1.06982315  2.11788988]]
    w2 = [[-1.82471502],[2.68546653],[1.41819501]]    
    """









