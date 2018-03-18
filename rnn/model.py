import numpy as np
import tensorflow as tf


class Batch:

    batch_size = 10

    def __init__(self, samples, labels):

        self.samples = samples
        self.labels = labels

    def generate(self, shuffle):

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(self.samples, self.labels)))
        else:
            shuffled_data = list(zip(self.samples, self.labels))

        batch_nums = int(len(self.samples)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.samples))]


class Rnn:

    MODEL_NAME = 'origin_rnn'

    def __init__(self, flags, embeddings, sentence_len):

        self.FLAGS = flags
        self.samples = tf.placeholder(tf.int32, [None, sentence_len], 'samples')
        self.labels = tf.placeholder(tf.int32, [None, self.FLAGS.labels_num], 'labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        self.l2_loss, self.scores = self.build_graph(embeddings)

        self.predict, self.correct, self.accuracy = self.proc_accuracy()
        self.losses = self.loss()
        self.opt = self.optimize()

    def build_graph(self, embeddings):
        with tf.variable_scope('embedding_layer'):
            lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                           initializer=tf.convert_to_tensor(embeddings, tf.float32))

            samples_embedding = tf.nn.embedding_lookup(lookup_table, self.samples)

        with tf.variable_scope('recurrent_layer', reuse=tf.AUTO_REUSE):

                if not self.FLAGS.rnn_cell:
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        self.FLAGS.rnn_hidden, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        self.FLAGS.rnn_hidden, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)
                else:
                    fw_cell = tf.nn.rnn_cell.BasicRNNCell(self.FLAGS.rnn_hidden)
                    bw_cell = tf.nn.rnn_cell.BasicRNNCell(self.FLAGS.rnn_hidden)

                fw_init_state = fw_cell.zero_state(batch_size=tf.to_int32(self.batch_size), dtype=tf.float32)
                bw_init_state = bw_cell.zero_state(batch_size=tf.to_int32(self.batch_size), dtype=tf.float32)

                rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, samples_embedding,
                                                                     initial_state_fw=fw_init_state,
                                                                     initial_state_bw=bw_init_state)
                cell_out = tf.concat(rnn_out, axis=2)

                if self.FLAGS.attention:
                    weight = tf.get_variable('att_weight', shape=[self.FLAGS.rnn_hidden * 2, 1],
                                             initializer=tf.truncated_normal_initializer(0.0, 0.1))
                    alpha = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(x, weight), tf.tanh(cell_out)), axis=1)
                    att_out = tf.tanh(tf.matmul(alpha, cell_out, transpose_a=True))
                else:
                    pooled = tf.reduce_max(cell_out, axis=1)
                    att_out = tf.reshape(pooled, [-1, self.FLAGS.rnn_hidden * 2])

        with tf.variable_scope('fully_conn_layer', reuse=tf.AUTO_REUSE):

            fully_dim = self.FLAGS.rnn_hidden * 2
            fully_input = tf.nn.dropout(tf.squeeze(att_out), self.dropout)

            weight = tf.get_variable(name="weight", shape=[fully_dim, self.FLAGS.labels_num],
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            biases = tf.get_variable(name='biases', shape=[self.FLAGS.labels_num], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            l2_loss = tf.nn.l2_loss(weight)

            scores = tf.nn.xw_plus_b(fully_input, weight, biases, name='score')

            return l2_loss, scores

    def proc_accuracy(self):
        with tf.name_scope('accuracy_layer'):

            predict = tf.argmax(self.scores, axis=1, name='predict')
            correct = tf.equal(predict, tf.argmax(self.labels, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='calculate_accuracy')
        return predict, correct, accuracy

    def loss(self):
        with tf.name_scope('loss'):

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(None, self.labels, self.scores)
            return tf.reduce_mean(losses) + 0.01 * self.l2_loss

    def optimize(self):
        with tf.name_scope('optimizer'):

            optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
            return optimizer.minimize(self.losses)
