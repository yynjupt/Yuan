import tensorflow as tf

# 自行编写的Seq2seq with Attention
class Seq2seq(object):

    def build_inputs(self, config):
        '''
        导入输入数据
        encoder_input:输入序列数据,shape=[batch_size,Seq_length,num_features]
        encoder_input_length：记录每条输入序列长度,shape=[batch_size,1]
        decoder_input:预测序列数据,shape=[batch_size,Seq_length,num_features]
        decoder_input_length：记录每条预测序列长度,shape=[batch_size,1]
        '''
        self.encoder_input = tf.placeholder(shape=(config.batch_size, None, config.num_features), dtype=tf.float32, name='encoder_input')
        self.encoder_input_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_target = tf.placeholder(shape=(config.batch_size, None, config.num_features), dtype=tf.float32, name='decoder_target')
        self.decoder_target_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='decoder_target_length')

    def build_loss(self, lambda_l2_reg = 0.005, mask_zeros=False):
        '''
        自定义损失函数（只考虑在特征维上的预测损失，不考虑时间维预测损失）
        :param lambda_l2_reg: 正则化参数
        :param mask_zeros: 是否需要将序列中的补0值去掉
        :return: 总损失
        '''
        if mask_zeros is True:
            mask = tf.sign(tf.reduce_max(tf.abs(self.decoder_target), 2)) # [batch_size*seq_length]
            mask = tf.tile(tf.expand_dims(mask,dim=-1),[1,1,self.decoder_target.shape[2]])  # [batch_size*seq_length*1]
            self.out *= mask

        losses = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.decoder_target[:,:,:-1], self.out[:,:,:-1])))) # RMSE

        # L2 regularization (to avoid overfitting)
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                # reg_loss += tf.contrib.layers.l2_regularizer(lambda_l2_reg)(tf_var)

        return losses + lambda_l2_reg*reg_loss

    def build_optim(self, loss, lr, lr_decay, momentum):
        '''
        优化方法：
        :param lr: learning_rate
        :param lr_decay: learning rate下降参数
        '''
        # return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        return tf.train.RMSPropOptimizer(learning_rate=lr, decay=lr_decay, momentum=momentum).minimize(loss)

    def attn(self, hidden, encoder_outputs):
        '''
        自定义注意力模块，用来计算上下文向量
        :param hidden: encoder的隐藏状态,shape=[batch_size,num_features]
        :param encoder_outputs:decoder输出,shape=[batch_size,seq_length,num_features]
        :return: context vector,shape=[batch_size,num_features]
        '''
        # hidden: B * D
        # encoder_outputs: B * S * D
        attn_weights = tf.matmul(encoder_outputs, tf.expand_dims(hidden, 2))
        # attn_weights: B * S * 1
        context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs, [0, 2, 1]), attn_weights))
        # context: B * D
        return context

    def __init__(self, config, useTeacherForcing=True, useAttention=True):

        self.build_inputs(config)

        with tf.variable_scope("encoder"):
            with tf.variable_scope("gru_cell"):
                encoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(config.hidden_dim), config.dropout_keep_prob)

            # sinle-layer bidirectional RNN
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, inputs=self.encoder_input, sequence_length=self.encoder_input_length, dtype=tf.float32, time_major=False)
            encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)  # hidden states of encoder
            encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)  # outputs of encoder

        with tf.variable_scope("decoder"):

            with tf.variable_scope("gru_cell"):
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(config.hidden_dim), config.dropout_keep_prob)
                decoder_initial_state = encoder_state

            # if useTeacherForcing and not useAttention:
            # decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
            # decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
            # decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded, initial_state=decoder_initial_state, sequence_length=self.seq_targets_length, dtype=tf.float32, time_major=False)

            tokens_eos = tf.constant(-1, tf.float32, shape=[config.batch_size, config.num_features], name='token_End') # 终止编码为-1
            tokens_go = tf.constant(-1, tf.float32, shape=[config.batch_size, config.num_features], name='token_Start') # 起始编码为-1

            W = tf.Variable(tf.random_uniform([config.hidden_dim, config.num_features]), dtype=tf.float32, name='decoder_out_W')
            b = tf.Variable(tf.constant(0.1, shape=[config.num_features]), dtype=tf.float32, name="decoder_out_b")

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:    # time step == 0
                    initial_elements_finished = (0 >= self.decoder_target_length)  # all False at the initial step
                    initial_state = decoder_initial_state # last time steps cell state
                    initial_input = tokens_go # last time steps cell state
                    if useAttention:
                        initial_input = tf.concat([initial_input, self.attn(initial_state, encoder_outputs)], 1)
                        initial_input = tf.nn.l2_normalize(initial_input,dim=0,epsilon=1e-12)
                    initial_output = None # none
                    initial_loop_state = None  # we don't need to pass any additional information
                    return (initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state)
                else:
                    def get_next_input():   # time step == 1,2,3,...
                        if useTeacherForcing:  # for model training
                            prediction = self.decoder_target[:,time-1,:]  # [batch_size, 1, n_features]
                        else:    # useTeacherForcing == False for model prediction
                            prediction = tf.add(tf.matmul(previous_output, W), b) # add a dense layer
                            input_time = tf.reshape(self.decoder_target[:,time-1,-1],shape=[prediction.shape[0],1])
                            prediction = tf.concat([prediction[:,:-1], input_time],axis=1)
                        next_input = prediction
                        return next_input

                    elements_finished = (time >= self.decoder_target_length)
                    finished = tf.reduce_all(elements_finished) #Computes the "logical and"
                    input = tf.cond(finished, lambda: tokens_eos, get_next_input)
                    if useAttention:
                        input = tf.concat([input, self.attn(previous_state, encoder_outputs)], 1)
                        input = tf.nn.l2_normalize(input, dim=0, epsilon=1e-12)
                    state = previous_state
                    output = previous_output
                    loop_state = None

                    return (elements_finished, input, state, output, loop_state)

            decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()
            decoder_outputs = tf.transpose(decoder_outputs, perm=[1,0,2]) # Seq_length*Batch_size*Dimension -> Batch_size*Seq_length*Dimension

            decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim))
            decoder_prediction = tf.add(tf.matmul(decoder_outputs_flat, W), b)
            decoder_prediction = tf.reshape(decoder_prediction, (decoder_batch_size, decoder_max_steps, config.num_features))

        self.out = decoder_prediction

        # self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.decoder_target, self.out)))) # RMSE
        self.loss = self.build_loss(config.lambda_l2_reg,config.mask_zeros)

        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        # self.train_op = self.build_optim(self.loss,lr=config.learning_rate,lr_decay=config.lr_decay,momentum=config.momentum)