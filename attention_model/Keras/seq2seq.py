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
from keras.models import Sequential
from keras.layers import LSTM
from attention_decoder import AttentionDecoder


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
    sequence_length = seq_len + predict_num  # 序列长度为样本长度加预测长度
    padding_length = seq_len*2 - sequence_length # 补0后序列长度，注意：输入输出序列长度必须相等
    result = []
    seq_num = len(data) - sequence_length  # 可组成的序列数
    train_num = int(seq_num * percent)  # 用来训练的序列数
    for index in range(seq_num):
        seq = data[index: index + sequence_length] + [0 for _ in range(padding_length)]
        result.append(seq)
        
    result = np.array(result)  # [样本数、序列长度、特征维度]
    print(result.shape)

    # 训练数据集
    if feature_num == 1:
        x_train = result[:train_num, :seq_len]
        y_train = result[:train_num, seq_len:]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    else:
        x_train = result[:train_num, :seq_len, :feature_num]
        y_train = result[:train_num, seq_len:, :feature_num]

    # 测试数据集
    if feature_num == 1:
        x_test = result[train_num:, :seq_len]
        y_test = result[train_num:, seq_len:]
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    else:
        x_test = result[train_num:, :seq_len, :feature_num]
        y_test = result[train_num:, seq_len:, :feature_num]

    return [x_train, y_train, x_test, y_test]


def build_attention_model(n_timesteps_in, n_features):
    model = Sequential()
    model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    model.add(AttentionDecoder(150, n_features))

    start = time.time()
    # model.compile(optimizer = "rmsprop", loss = root_mean_squared_error)
    # model.compile(optimizer='Adam', loss=root_mean_squared_error)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    print('> Compilation Time : ', time.time() - start)
    return model
