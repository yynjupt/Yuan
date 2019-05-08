# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:12:27 2019

@author: yuany

description: standard encoder-decoder model for remote sensing time series prediction
"""

import seq2seq
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

seed = 7
np.random.seed(seed)


# def save_csv(filename, data):
#     dataframe = pd.DataFrame({})
#
#     print('save prediction result!')


# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()
    seq_len = 69
    predict_num = 23
    feature_num = 6
    percent = 0.8  # 80% 数据用来训练

    input_name = r"D:\黑龙江火灾\TestData\test_deltaT_1.csv"
    out_name = 'predict.csv'

    print("(1)loading data......")
    f = open(input_name, 'rb').read()
    data = f.decode().split('\n')
    data = data[:-1]
    sequence = []
    for str in data:
        sequence.append(list(map(float, str.split(','))))
    x_train, y_train, x_test, y_test = seq2seq.load_data(sequence, seq_len, feature_num, predict_num, percent)

    ################################ 不包含注意力机制的编码器-解码器模型 #####################################
    # 建立模型
    print('(2)loading model...')
    model = load_model('baseline_model.h5')  # 加载神经网络

    # 时序预测
    print('(3)model prediction...')
    print('number of sequence for test:',x_test.shape[0])
    predition = model.predict(x_test, verbose=0)
    # save_csv(out_name, predition)

    # 绘图显示
    for i in range(4):
        ax = plt.subplot(221+i)
        j = i*10
        ax.plot(np.hstack((x_test[j,:,5],y_test[j, :, 5])), color='blue', marker='*', ls=':', label='predict by custom LSTM')
        ax.plot(np.hstack((x_test[j,:,5],predition[j, :, 5])), color='green', marker='o', ls=':', label='predict by custom LSTM')
        ax.legend()
    plt.show()

    print('>>>>>> Complete! <<<<<<<')
