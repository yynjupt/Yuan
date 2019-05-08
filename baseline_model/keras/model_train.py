# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:12:27 2019

@author: yuany
 
description: standard encoder-decoder model for remote sensing time series prediction
"""

import seq2seq
import time
import numpy as np
import matplotlib.pyplot as plt


seed = 7
np.random.seed(seed)

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
    x_train, y_train, x_test, y_test = seq2seq.load_data(sequence, seq_len, feature_num, predict_num,percent)

    ################################ 不包含注意力机制的编码器-解码器模型 #####################################
    # 建立模型
    print('(2)building model...')
    model = seq2seq.build_baseline_model(seq_len, predict_num, feature_num)  # 建立神经网络

    # 训练模型
    history = model.fit(x_train, y_train, epochs=100, batch_size=16, shuffle=True, verbose=2)
    model.save('baseline_model.h5')
    print('> Training duration (s): ', time.time() - global_start_time)

    # summarize history for loss
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('>>>>>> Complete! <<<<<<<')
