import numpy as np
import tensorflow as tf


class Batch:

    batch_size = 10

    def __init__(self, sdp_l, sdp_r, relation_l, relation_r, pos_l, pos_r, channel_len):

        self.sdp_l = sdp_l
        self.sdp_r = sdp_r
        self.relation_l = relation_l
        self.relation_r = relation_r
        self.pos_l = pos_l
        self.pos_r = pos_r
        self.channel_len = channel_len
        self.labels = []

    def generate(self, shuffle):

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(self.sdp_l, self.sdp_r, self.relation_l, self.relation_r,
                                                           self.pos_l, self.pos_r, self.labels, self.channel_len)))
        else:
            shuffled_data = list(zip(self.sdp_l, self.sdp_r, self.relation_l, self.relation_r, self.pos_l, self.pos_r,
                                     self.labels, self.channel_len))
        batch_nums = int(len(self.sdp_l)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.sdp_l))]


class SDPNet:

    MODEL_NAME = 'sdp_lstm'

    def __init__(self, flags, embeddings, rel_len, pos_len, left_max, right_max):

        self.FLAGS = flags

        self.sdp_left = tf.placeholder(tf.int32, [None, left_max], 'sdp_left')
        self.sdp_right = tf.placeholder(tf.int32, [None, right_max], 'sdp_right')

        self.relation_left = tf.placeholder(tf.int32, [None, left_max], 'relation_left')
        self.relation_right = tf.placeholder(tf.int32, [None, right_max], 'relation_right')

        self.pos_left = tf.placeholder(tf.int32, [None, left_max], 'pos_left')
        self.pos_right = tf.placeholder(tf.int32, [None, right_max], 'pos_right')

        self.channel_len = tf.placeholder(tf.int32, [None, 6], 'channel_len')
        self.labels = tf.placeholder(tf.int32, [None, self.FLAGS.labels_num], 'labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        self.l2_loss, self.scores = self.build_graph(embeddings, rel_len, pos_len)

        self.predict, self.correct, self.accuracy = self.proc_accuracy()
        self.losses = self.loss()
        self.opt = self.optimize()

    def build_graph(self, embeddings, rel_len, pos_len):

        with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):

            lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                           initializer=tf.convert_to_tensor(embeddings, tf.float32))

            relation_emb = tf.get_variable('relation_embedding', dtype=tf.float32, shape=[rel_len, self.FLAGS.rel_dim],
                                           initializer=tf.truncated_normal_initializer(0.0, 0.1))

            pos_emb = tf.get_variable('pos_embedding', dtype=tf.float32, shape=[pos_len, self.FLAGS.pos_dim],
                                      initializer=tf.truncated_normal_initializer(0.0, 0.1))

            sdp_left_emb = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.sdp_left), self.dropout)
            sdp_right_emb = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.sdp_right), self.dropout)

            rel_left_emb = tf.nn.dropout(tf.nn.embedding_lookup(relation_emb, self.relation_left), self.dropout)
            rel_right_emb = tf.nn.dropout(tf.nn.embedding_lookup(relation_emb, self.relation_right), self.dropout)

            pos_left_emb = tf.nn.dropout(tf.nn.embedding_lookup(pos_emb, self.pos_left), self.dropout)
            pos_right_emb = tf.nn.dropout(tf.nn.embedding_lookup(pos_emb, self.pos_right), self.dropout)

        with tf.variable_scope('multi_channel_layer'):

            sdp_out = self.bi_lstm("sdp", self.FLAGS.word_hidden, sdp_left_emb, sdp_right_emb,
                                   tf.squeeze(tf.slice(self.channel_len, [0, 0], [-1, 1])),
                                   tf.squeeze(tf.slice(self.channel_len, [0, 1], [-1, 1])))
            rel_out = self.bi_lstm("rel", self.FLAGS.other_hidden, rel_left_emb, rel_right_emb,
                                   tf.squeeze(tf.slice(self.channel_len, [0, 2], [-1, 1])),
                                   tf.squeeze(tf.slice(self.channel_len, [0, 3], [-1, 1])))
            pos_out = self.bi_lstm("pos", self.FLAGS.other_hidden, pos_left_emb, pos_right_emb,
                                   tf.squeeze(tf.slice(self.channel_len, [0, 4], [-1, 1])),
                                   tf.squeeze(tf.slice(self.channel_len, [0, 5], [-1, 1])))

        with tf.variable_scope('fully_conn_layer', reuse=tf.AUTO_REUSE):

            fully_dim = self.FLAGS.word_hidden * 2
            feature_out = sdp_out

            if self.FLAGS.rel_channel:
                fully_dim += self.FLAGS.other_hidden * 2
                feature_out = tf.concat([feature_out, rel_out], axis=1)

            if self.FLAGS.pos_channel:
                fully_dim += self.FLAGS.other_hidden * 2
                feature_out = tf.concat([feature_out, pos_out], axis=1)

            fully_input = tf.nn.dropout(tf.squeeze(feature_out), self.dropout)

            weight = tf.get_variable(name="Weight", shape=[fully_dim, self.FLAGS.labels_num],
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            biases = tf.get_variable(name='Biases', shape=[self.FLAGS.labels_num], dtype=tf.float32,
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

    def bi_lstm(self, model_name, hidden_dim, left_input, right_input, left_len, right_len):

        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):

            left_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                hidden_dim, forget_bias=1.0, state_is_tuple=True))
            right_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                hidden_dim, forget_bias=1.0, state_is_tuple=True))

            left_init_state = left_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            right_init_state = right_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            left_out, left_state = tf.nn.dynamic_rnn(left_cell, left_input,
                                                     initial_state=left_init_state)
            right_out, right_state = tf.nn.dynamic_rnn(right_cell, right_input,
                                                       initial_state=right_init_state)

            left_pooled = tf.reduce_max(left_out, axis=1)
            right_pooled = tf.reduce_max(right_out, axis=1)

            return tf.concat([tf.reshape(left_pooled, [-1, hidden_dim]), tf.reshape(right_pooled, [-1, hidden_dim])], 1)
