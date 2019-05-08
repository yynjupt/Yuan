import tensorflow as tf
from tensorflow.python.util import nest

class Seq2seq(object):

    def build_inputs(self, config):
        self.encoder_input = tf.placeholder(shape=(config.batch_size, None, config.num_features+1), dtype=tf.float32, name='encoder_input')
        self.encoder_input_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_target = tf.placeholder(shape=(config.batch_size, None, config.num_features), dtype=tf.float32, name='decoder_target')
        self.decoder_target_time = tf.placeholder(shape=(config.batch_size, None), dtype=tf.float32, name='decoder_target_time')
        self.decoder_target_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='decoder_target_length')

    def build_loss(self, set_mask, mask_value):
        '''
        自定义损失函数（只考虑在特征维上的预测损失，不考虑时间维预测损失）
        :param mask_zeros: 是否需要将序列中的补0值去掉
        :return: 总损失
        '''
        if set_mask is True:
            # mask = tf.sign(tf.reduce_max(tf.abs(self.decoder_target), 2)) # [batch_size*seq_length]
            # mask = tf.tile(tf.expand_dims(mask,dim=-1),[1,1,self.decoder_target.shape[2]])  # [batch_size*seq_length*1]
            mask = tf.sign(tf.abs(tf.subtract(self.decoder_target, mask_value)))  # Shape: batch_size*seq_length*num_features
            self.out *= mask  # mask off padding zeros
            losses = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.decoder_target, self.out))) / tf.reduce_sum(mask)) # RMSE
        else:
            losses = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.decoder_target, self.out))))

        return losses

    def __init__(self, config, useTeacherForcing, useAttention):

        self.build_inputs(config)

        with tf.variable_scope("encoder"):

            encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(config.hidden_dim),
                                                         output_keep_prob = config.dropout_keep_prob)

            # single-directional RNN
            # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.encoder_input,
            #                                                    sequence_length=self.encoder_input_length,
            #                                                    dtype=tf.float32, time_major=False)

            # bi-directional RNN
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, inputs=self.encoder_input,
                                                sequence_length=self.encoder_input_length, dtype=tf.float32)
            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], axis=2)   # shape = [B,T,2D]
            encoder_state = tf.concat([encoder_fw_final_state, encoder_bw_final_state], axis=1)  # shape = [B,2D], D = hidden_dims


        with tf.variable_scope("decoder"):
            # single-directional RNN
            # decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)

            # bi-directional RNN
            decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim*2)

            tokens_eos = tf.constant(config.mask_value, tf.float32, shape=[config.batch_size, config.num_features+1],
                                     name='token_End')  # 终止编码

            # 定义传入CustomHelper的三个函数
            def initial_fn():  # 第一轮循环的返回函数，传入前一时刻的观测值
                initial_elements_finished = (0 >= self.decoder_target_length)  # all False at the initial step
                initial_inputs = tf.concat([self.encoder_input[:, -1, :-1],
                                            tf.expand_dims(self.decoder_target_time[:, 0], axis=1)],axis=1)
                return initial_elements_finished, initial_inputs

            def next_inputs_fn1(time, outputs, state, sample_ids):  # 将上一个时刻的期望预测值作为下一个时刻的输入
                del sample_ids  # unused in next_input_fn
                next_time = time + 1
                elements_finished = (
                        next_time >= self.decoder_target_length)  # this operation produces boolean tensor of [batch_size]
                all_finished = tf.reduce_all(elements_finished)  # elements_finished -> boolean scalar
                next_inputs = tf.cond(all_finished, lambda: tokens_eos,
                                      lambda: tf.concat([tf.reshape(self.decoder_target[:, time, :],
                                                                    shape=outputs.shape),
                                                         tf.reshape(self.decoder_target_time[:, next_time],
                                                                    shape=[outputs.shape[0], 1])], axis=1))
                return elements_finished, next_inputs, state

            def next_inputs_fn2(time, outputs, state, sample_ids):  # 将上一个时刻的输出作为下一个时刻的输入
                del sample_ids  # unused in next_input_fn
                next_time = time + 1
                elements_finished = (
                        next_time >= self.decoder_target_length)  # this operation produces boolean tensor of [batch_size]
                all_finished = tf.reduce_all(elements_finished)  # elements_finished -> boolean scalar
                next_inputs = tf.cond(all_finished, lambda: tokens_eos,
                                      lambda: tf.concat([outputs, tf.reshape(self.decoder_target_time[:, next_time],
                                                                             shape=[outputs.shape[0], 1])], axis=1))
                return elements_finished, next_inputs, state

            def sample_fn(time, outputs, state):
                return outputs

            if useTeacherForcing:
                helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn1,
                                                     sample_ids_dtype=tf.float32,
                                                     sample_ids_shape=config.num_features)
            else:
                helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn2,
                                                         sample_ids_dtype=tf.float32,
                                                         sample_ids_shape=config.num_features)

            if useAttention:
                # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim*2,
                #                                                            memory=encoder_outputs,
                #                                                            normalize=True)
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim*2,
                                                                           memory=encoder_outputs,
                                                                           scale=True)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                alignment_history=True, output_attention=True,
                                                                name='AttentionWrapper')
                decoder_initial_state = attn_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, config.num_features)
                decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, helper, initial_state=decoder_initial_state)
            else:
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state=encoder_state,
                                                          output_layer=tf.layers.Dense(config.num_features))

            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

            if useAttention:
                self.attention_matrices = final_state.alignment_history.stack(name = 'predict_attention_matrix')

        self.out = final_outputs.rnn_output

        self.loss = self.build_loss(config.set_mask, config.mask_value)

        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
