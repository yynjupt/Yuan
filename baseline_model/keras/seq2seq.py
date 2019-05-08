# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:53:11 2019

@author: yuany
"""

import os
import warnings
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Masking
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM, TimeDistributed, Dropout
from keras.models import Sequential

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
#warnings.filterwarnings('ignore')  # Hide messy Numpy warnings
#np.random.seed(777)  # 固定种子保证结果能够重现
#
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.05
#session = tf.Session(config=config)
#KTF.set_session(session)

# define RMSE loss function
def root_mean_squared_error(y_true, y_pred):
    return KTF.sqrt(KTF.mean(KTF.square(y_pred - y_true))) 
    
# load data for training the baseline model
def load_data(data, seq_len, feature_num, predict_num, percent=0.8):
    """ 载入数据
    seq：整条历史序列
    seq_len：编码器端输入的序列长度，69
    predict_num：解码器端输出的序列长度，23
    percent = 0.8  # 80% 数据用来训练
    """
    sequence_length = seq_len + predict_num  # 序列长度为样本长度加 1 个预测值
    result = []
    seq_num = len(data) - sequence_length  # 可组成的序列数
    train_num = int(seq_num * percent)  # 用来训练的序列数
    for index in range(seq_num):
        seq = data[index: index + sequence_length]
        result.append(seq)
        
    result = np.array(result)  # [样本数、序列长度、特征维度]
    print(result.shape)
    
    # 训练数据集
    x_train = result[:train_num, :seq_len, :feature_num]
    y_train = result[:train_num, seq_len:, :feature_num]
    if len(x_train.shape) != 3:
        x_train = np.reshape(x_train, (seq_num-train_num, seq_len, feature_num))
        y_train = np.reshape(y_train, (seq_num-train_num, predict_num, feature_num))

    # 测试数据集
    x_test = result[train_num:, :seq_len, :feature_num]
    y_test = result[train_num:, seq_len:, :feature_num]
    if len(x_test.shape) != 3:
        x_test = np.reshape(x_test, (seq_num-train_num, seq_len, feature_num))
        y_test = np.reshape(y_test, (seq_num-train_num, predict_num, feature_num))

    return [x_train, y_train, x_test, y_test]


def build_baseline_model(n_timesteps_in, n_timesteps_out, n_features):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(n_timesteps_in, n_features)))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(RepeatVector(n_timesteps_out))
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features,activation='tanh')))
    
    start = time.time()
    # model.compile(optimizer = "rmsprop", loss = root_mean_squared_error)
    # model.compile(optimizer='Adam', loss=root_mean_squared_error)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    print('> Compilation Time : ', time.time() - start)
    return model
