# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:52:48 2019

@author: yuany
"""
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.reset_default_graph()

#定义一下超参数
n_features = 1 # 特征维度
n_hidden = [128, 64] #每个RNN的隐藏单元个数
learning_rate = 0.001 #学习率

# load data for training the baseline model
def load_data(data, seq_len, predict_num):
    """ 载入数据
    seq：整条历史序列
    seq_len：编码器端输入的序列长度，69
    predict_num：解码器端输出的序列长度，23
    """
    sequence_length = seq_len + predict_num  # 序列长度（编码器输入+解码器输出）
    result = []
    seq_num = len(data) - sequence_length  # 可组成的序列数
    for index in range(seq_num):
        seq = data[index: index + sequence_length]
        result.append(seq)
        
    result = np.array(result)
#    print(result.shape)
    
	# 训练数据集
    x_train = result[:-1, : seq_len]
    y_train = result[:-1, seq_len :]
    
	# 测试数据集
    x_test = result[-1, : seq_len]
    y_test = result[-1, seq_len :]

	# [样本数、序列长度、特征维度]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
    y_test = np.reshape(y_test, (1, y_test.shape[0], 1))

    print('Data loaded...')
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    return [x_train, y_train, x_test, y_test]


def get_batch(x_train,y_train):
    input_batch,output_batch,target_batch = [],[],[]
    # seq2seq要求decoder的输入值前有一位，对decoder的输入进行padding
    y_output = np.insert(y_train, 0, 0, axis=1)
    # 对decoder的目标输出进行padding
    y_target = np.insert(y_train, y_train.shape[1], 0, axis=1)
    
    print('Converted to output and target...')
#    print(y_output.shape)
#    print(y_output[0,:,0])
#    print(y_target.shape)
#    print(y_target[0,:,:])
    
    for i in range(x_train.shape[0]): # 遍历所有样本序列，x_train, y_train结构均为[样本数、序列长度]
        input_batch.append(x_train[i,:,:])  #对encoder的输入进行编码
        output_batch.append(y_output[i,:,:]) #对decoder的输入进行编码
        target_batch.append(y_target[i,:,:]) # model的期望输出，不需要编码为one-hot向量 

    return input_batch,output_batch,target_batch

#定义model
encoder_input = tf.placeholder(tf.float32,[None,None,n_features])#(batch_size,max_len,num_class)
decoder_input = tf.placeholder(tf.float32,[None,None,n_features])#(batch_size,max_len+1,num_class) max_len+1 because add 'S'
target = tf.placeholder(tf.float32,[None,None,n_features])#(batch_size,max_len+1)max_len+1 because'E'

#encoder
with tf.variable_scope('encoder'):
    cell = cells = [tf.contrib.rnn.BasicRNNCell(num_units=n) for n in n_hidden] #根据每个RNN的隐藏单元个数 创建RNN cell
    encoder = tf.contrib.rnn.MultiRNNCell(cell)
    encoder = tf.contrib.rnn.DropoutWrapper(encoder,output_keep_prob=0.5) #dropout 随机失活
    _,encoder_output = tf.nn.dynamic_rnn(encoder,encoder_input,dtype=tf.float32)

#decoder
with tf.variable_scope('decoder'):
    cell = cells = [tf.contrib.rnn.BasicRNNCell(num_units=n) for n in n_hidden]#根据每个RNN的隐藏单元个数 创建RNN cell
    decoder = tf.contrib.rnn.MultiRNNCell(cell)
    decoder = tf.contrib.rnn.DropoutWrapper(decoder,output_keep_prob=0.5)#dropout 随机失活
    decoder_output,_ = tf.nn.dynamic_rnn(decoder,decoder_input,initial_state=encoder_output,dtype=tf.float32)

#还需要一个全连接层，把得到的值映射到29个类别
model = tf.layers.dense(decoder_output,n_features, activation = tf.nn.relu)

#计算loss 以及优化
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=target))
#loss = tf.losses.mean_squared_error(target, model)
loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, model))))
#optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#定义测试函数 测试某一条序列
def test(x_test):
    y_test = np.zeros((1,predict_num,1))
    input_batch,output_batch,_ = get_batch(x_test, y_test)
    predict = sess.run(model,feed_dict={encoder_input:input_batch,decoder_input:output_batch}) # model : [batch_size, max_len+1, n_features]
    
    return predict

if __name__ == '__main__':
    global_start_time = time.time()
    
    # 参数定义
    seq_len = 69
    predict_num = 23

    # 读取时序数据
    input_name = "D:\\黑龙江火灾\\Kong\\b1.csv"
    out_name = 'predict.csv'
    f = open(input_name, 'rb').read()
    data = f.decode().split('\n')
    data = data[:-1]
    data = list(map(float,data)) 
    x_train, y_train, x_test, y_test = load_data(data, seq_len, predict_num)
    
    # 建立一个graph
    sess = tf.Session()
    
    #初始化
    sess.run(tf.global_variables_initializer())
    
    #准备batch
    input_batch,output_batch,target_batch = get_batch(x_train,y_train)
    for i in range(2000):#跑5000轮
        _,loss_ = sess.run([optimizer,loss],
                            feed_dict={encoder_input:input_batch,decoder_input:output_batch,target:target_batch})
        if (i+1)%500 == 0:
            print('Epoch %04d:' % (i+1))
            print('cost:%.6f' % loss_)
            
    print('algorithm test...')
    # 绘图显示
    ax = plt.subplot(111)
    ax.plot(y_test[0,:,0],color = 'red',marker = '*', label='real time series')
    predict = test(x_test)
    ax.plot(predict[0,:-1,0],color = 'green',marker = 'o', label='predict by custom LSTM')
    ax.legend()
    plt.show()
    