import tensorflow as tf
import numpy as np
import random
import time
# from model_seq2seq import Seq2seq
from model_seq2seq_contrib import Seq2seq
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(0)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

input_name = r"D:\黑龙江火灾\TestData\ForTest0.csv"

class Config(object):
    input_length = 50
    predict_length = 20
    hidden_dim = 50
    batch_size = 10
    num_features = 1  # dimension of the predict sequence
    learning_rate = 0.005
    lr_decay = 0.9  # Simulated annealing.
    momentum = 0.5  # Momentum technique in weights update
    lambda_l2_reg = 0.0  # L2 regularization of weights - avoids overfitting
    dropout_keep_prob = 0.5  # dropout rate
    mask_value = 0
    set_mask = False

# load data for training
def load_train_data(data, input_len, predict_len, percent):
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
    for index in range(train_nums):
        seq = data[index: index + sequence_length]
        result.append(seq)

    result = np.array(result)
    #    print(result.shape)

    # 训练数据集
    x_train = result[:, : input_len]
    y_train = result[:, input_len :]

    # [样本数、序列长度、特征维度]
    if len(x_train.shape) != 3:
        # 特征维度
        num_features = result.shape[2]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], num_features))

    print('Data loaded...')
    print(x_train.shape)
    print(y_train.shape)

    return [x_train, y_train]


def get_batch(x_train, y_train, batch_size):
    permutation = np.random.permutation(x_train.shape[0])
    input_batch = x_train[permutation[:batch_size], :, :]
    target_batch = y_train[permutation[:batch_size],:,:-1]
    target_time_batch = y_train[permutation[:batch_size],:,-1]

    input_lens = [x_train.shape[1] for i in range(batch_size)]
    target_lens = [y_train.shape[1] for i in range(batch_size)]

    return input_batch, input_lens, target_batch, target_time_batch, target_lens

if __name__ == "__main__":
    global_start_time = time.time()
    percent = 0.8  # 80% 数据用来训练

    config = Config()

    print("(1)load data......")
    f = open(input_name, 'rb').read()
    data = f.decode().split('\n')
    data = data[:-1]
    sequence = []
    for str in data:
        sequence.append(list(map(float, str.split(','))))
    x_train, y_train = load_train_data(sequence, config.input_length, config.predict_length, percent)

    print("(2) build model......")
    model = Seq2seq(config=config, useTeacherForcing=True, useAttention=True)


    print("(3) run model......")
    batches = 3000
    print_every = 500

    with tf.Session(config=tf_config) as sess:
        tf.summary.FileWriter('graph', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        losses = []
        total_loss = 0
        for batch in range(batches):
            source_batch, source_lens, target_batch, target_time, target_lens = get_batch(x_train, y_train, config.batch_size)

            feed_dict = {
                model.encoder_input: source_batch,
                model.encoder_input_length: source_lens,
                model.decoder_target: target_batch,
                model.decoder_target_time: target_time,
                model.decoder_target_length: target_lens
            }

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            total_loss += loss

            if batch % print_every == 0:
                print_loss = total_loss if batch == 0 else total_loss / print_every
                losses.append(print_loss)
                total_loss = 0
                print("-----------------------------")
                print("batch:",batch,"/",batches)
                print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                print("loss:",print_loss)

                # print("show samples...\n")
                # predict_batch = sess.run(model.out, feed_dict)
                # ax = plt.subplot(111)
                # # ax.plot(target_batch[0, :, 6],target_batch[0,:,4],color='blue', ls=':',marker='*',
                # #         label='real time series')  # NDVI
                # # ax.plot(target_batch[0, :, 6], predict_batch[0, :, 4], color='red', ls=':',marker='o',
                # #         label='predicted time series')
                # ax.scatter(target_batch[0, :, 6],target_batch[0,:,4],color='blue',marker='*',label='real time series') #NDVI
                # ax.scatter(target_batch[0, :, 6], predict_batch[0, :, 4], color='red', marker='o',
                #            label='predicted time series')
                # ax.legend()
                # plt.show()

        print(losses)

        print(saver.save(sess, "checkpoint/model.ckpt"))




