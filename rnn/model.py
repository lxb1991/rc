import numpy as np
import tensorflow as tf


label_nums = 19
padding_word = '<pad>'
word_dim = 50

epochs = 1000
learning_rate = 0.001

use_lstm = True
lstm_hidden = 300

attention = False


class Batch:

    shuffle = True

    def __init__(self, samples):

        self.samples = samples
        self.labels = []
        if attention:
            self.batch_size = 10
        else:
            self.batch_size = 100

    def generate(self):

        if self.shuffle:
            shuffled_data = np.random.permutation(list(zip(self.samples, self.labels)))
        else:
            shuffled_data = list(zip(self.samples, self.labels))
        batch_nums = int(len(self.samples)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.samples))]


class Rnn:

    def __init__(self, embeddings, sentence_len, batch_size):

        self.samples = tf.placeholder(tf.int32, [None, sentence_len], 'samples')
        self.labels = tf.placeholder(tf.int32, [None, label_nums], 'labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                       initializer=tf.convert_to_tensor(embeddings, tf.float32))

        samples_embeddings = tf.nn.embedding_lookup(lookup_table, self.samples)
        with tf.name_scope('recurrent'):

            with tf.variable_scope('recurrent', reuse=tf.AUTO_REUSE):

                rnn_sample_dim = lstm_hidden

                if use_lstm:
                    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        rnn_sample_dim, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)
                    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        rnn_sample_dim, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)

                    fw_init_state = fw_lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                    bw_init_state = bw_lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                    rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, samples_embeddings,
                                                                         initial_state_fw=fw_init_state,
                                                                         initial_state_bw=bw_init_state)
                else:
                    fw_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_sample_dim)
                    bw_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_sample_dim)

                    fw_init_state = fw_rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                    bw_init_state = bw_rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                    rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, samples_embeddings,
                                                                         initial_state_fw=fw_init_state,
                                                                         initial_state_bw=bw_init_state)
                bilstm_out = tf.concat(rnn_out, 2)

                if attention:
                    weight = tf.get_variable('att_weight', shape=[lstm_hidden * 2, 1],
                                             initializer=tf.truncated_normal_initializer(0.0, 0.1))
                    alpha = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(x, weight), tf.tanh(bilstm_out)), axis=1)
                    att_out = tf.tanh(tf.matmul(alpha, bilstm_out, transpose_a=True))
                else:
                    rnn_out = tf.expand_dims(bilstm_out, axis=-1)
                    pooled = tf.nn.max_pool(rnn_out, ksize=[1, sentence_len, 1, 1], strides=[1, sentence_len, 1, 1],
                                            padding="SAME", name='max_pooling')
                    att_out = tf.reshape(pooled, [-1, rnn_sample_dim * 2])

        with tf.name_scope('fully_conn'):
            with tf.variable_scope('fully_conn', reuse=tf.AUTO_REUSE):

                fully_dim = rnn_sample_dim * 2
                fully_input = tf.nn.dropout(tf.squeeze(att_out), self.dropout)

                weight = tf.get_variable(name="weight", shape=[fully_dim, label_nums],
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                biases = tf.get_variable(name='biases', shape=[label_nums], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                l2_loss = tf.nn.l2_loss(weight)

                scores = tf.nn.xw_plus_b(fully_input, weight, biases, name='score')

        with tf.name_scope('accuracy'):

            self.predict_labels = tf.argmax(scores, axis=1, name='predict')
            self.correct = tf.equal(self.predict_labels, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='calculate_accuracy')

        with tf.name_scope('loss'):

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(None, self.labels, scores)
            self.reg_losses = tf.reduce_mean(losses) + 0.01 * l2_loss

        with tf.name_scope('optimizer'):

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt = optimizer.minimize(self.reg_losses)
