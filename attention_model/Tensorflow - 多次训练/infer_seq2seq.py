import tensorflow as tf
import numpy as np
import random
import time
# from model_seq2seq import Seq2seq
from model_seq2seq_contrib import Seq2seq
from train_seq2seq import Config, input_name
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tf.reset_default_graph()
tf.set_random_seed(0)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

model_path = "checkpoint/model.ckpt"

def save_csv(filename, data):
    f = open(filename, 'w')
    for index in range(data.shape[0]):
        f.write(np.str(data[index]))
        f.write('\n')
    f.close()


def plot_attention_matrix(src, tgt, matrix, name="attention_matrix.png"):
    plt.figure()
    src = [str(item) for item in src]
    tgt = [str(item) for item in tgt]
    df = pd.DataFrame(matrix, index=src, columns=tgt)
    ax = sns.heatmap(df, linewidths=.5, robust=True)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title("Attention heatmap")
    plt.show()
    plt.savefig(name, bbox_inches='tight')
    # plt.gcf().clear()
    return matrix

# load data for testing
def load_test_data(data, input_len, predict_len, percent):
    """载入数据
    data: 整条序列
    input_len：编码器输入序列长度
    predict_len：解码器预测序列长度
    percent：训练数据占比
    """
    sequence_length = input_len + predict_len  # 序列长度（编码器输入+解码器输出）
    result = []
    seq_num = len(data) - sequence_length  # 可组成的序列数
    train_nums = int(seq_num*percent)  # 用来训练的序列数
    for index in range(train_nums,seq_num+1):
        seq = data[index: index + sequence_length]
        result.append(seq)

    result = np.array(result)  # shape: [test_num,seq_length]
    #    print(result.shape)

    # 训练数据集
    x_test = result[:, :input_len]
    y_test = result[:, input_len:]

    # [样本数、序列长度、特征维度]
    if len(x_test.shape) != 3:
        # 特征维度
        num_features = result.shape[2]

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], num_features))

    print('Data loaded...')
    print(x_test.shape)
    print(y_test.shape)

    return [x_test, y_test]

def get_batch(x_test, y_test, batch_size):
    input_batch = x_test[-batch_size:, :, :]
    target_batch = y_test[-batch_size:, :, :-1]
    target_time_batch = y_test[-batch_size:, :, -1]

    input_lens = [x_test.shape[1] for _ in range(batch_size)]
    target_lens = [y_test.shape[1] for _ in range(batch_size)]

    return input_batch, input_lens, target_batch, target_time_batch, target_lens

if __name__ == "__main__":
    percent = 0.8  # 20% 数据用来测试

    # input_name = r"D:\黑龙江火灾\TestData\ForTest0.csv"
    out_name = r'D:\黑龙江火灾\out_Seq2Seq_with_attention.csv'

    config = Config()

    print("(1)load data......")
    f = open(input_name, 'rb').read()
    data = f.decode().split('\n')
    data = data[:-1]
    sequence = []
    for str1 in data:
        sequence.append(list(map(float, str1.split(','))))
    x_test, y_test = load_test_data(sequence, config.input_length, config.predict_length, percent)


    print("(2) build model......")
    model = Seq2seq(config=config, useTeacherForcing=False, useAttention=True)

    print("(3) run model......")
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        source_batch, source_lens, target_batch, target_time, target_lens = get_batch(x_test, y_test, config.batch_size)

        feed_dict = {
            model.encoder_input: source_batch,
            model.encoder_input_length: source_lens,
            model.decoder_target: target_batch,
            model.decoder_target_time: target_time,
            model.decoder_target_length: target_lens
        }

        print("show samples...\n")

        predict_batch, losses = sess.run([model.out, model.loss], feed_dict)

        # predict_batch, losses, att_mat = sess.run([model.out, model.loss, model.attention_matrices], feed_dict)

        print(losses)

        # 拟合结果
        for i in range(4):
            ax = plt.subplot(221+i)
            ax.scatter(target_time[i, :], target_batch[i, :, 0], color='blue', marker='*',
                       label='real time series')  # NDVI
            ax.scatter(target_time[i, :], predict_batch[i, :, 0], color='red', marker='o',
                       label='predicted time series')
            ax.legend()

        # # 画出注意力矩阵
        # x_ = source_batch[-1]  # [input_length, num_features]
        # y_ = target_time[-1]  # [predict_length]
        # mask_x = [i for i in range(x_.shape[0]) if x_[i,-1] == 0.]
        # x_valid = np.delete(x_, mask_x, axis=0)
        # # x_valid = x_
        # mask_y = [i for i in range(y_.shape[0]) if y_[i] == 0.]
        # y_valid = np.delete(y_, mask_y, axis=0)
        #
        # matrix = att_mat[:,-1,:].T
        # np.savetxt('attn_matrix.csv', matrix, delimiter=',')
        # matrix = np.delete(matrix, mask_x, axis=0)
        # matrix = np.delete(matrix, mask_y, axis=1)
        # plot_attention_matrix(src=x_valid[:,-1], tgt=y_valid, matrix=matrix)

        # save_csv(out_name, target_batch[-1, :, 6])

        plt.show()
