import numpy as np
import tensorflow as tf


class Batch:

    batch_size = 100

    def __init__(self, sdp, relation, labels):

        self.sdp = sdp
        self.relation = relation
        self.labels = labels

    def generate(self, shuffle):

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(self.sdp, self.relation, self.labels)))
        else:
            shuffled_data = list(zip(self.sdp, self.relation, self.labels))
        batch_nums = int(len(self.sdp)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.sdp))]


class SDPCNN:

    MODEL_NAME = 'sdp_cnn'

    def __init__(self, flags, embeddings, relation_len, max_sdp_len, max_rel_len):

        self.FLAGS = flags
        self.sdp = tf.placeholder(tf.int32, [None, max_sdp_len], 'sdp')
        self.relation = tf.placeholder(tf.int32, [None, max_rel_len], 'relation')

        self.labels = tf.placeholder(tf.int32, [None, self.FLAGS.labels_num], 'labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self.l2_loss, self.scores = self.build_graph(embeddings, relation_len, max_sdp_len)

        self.predict, self.correct, self.accuracy = self.proc_accuracy()
        self.losses = self.loss()
        self.opt = self.optimize()

    def build_graph(self, embeddings, relation_len, sdp_len):

        with tf.variable_scope('embedding_layer'):

            lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                           initializer=tf.convert_to_tensor(embeddings, tf.float32))

            relation_emb = tf.get_variable('relation_embedding', dtype=tf.float32,
                                           shape=[relation_len, self.FLAGS.rel_dim],
                                           initializer=tf.truncated_normal_initializer(0.0, 0.1))

            sdp_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.sdp), self.dropout)
            rel_embedding = tf.nn.dropout(tf.nn.embedding_lookup(relation_emb, self.relation), self.dropout)

        if self.FLAGS.order:
            conv = []
            for i in range(Batch.batch_size):
                conv.append(self.concat2d(sdp_embeddings[i], rel_embedding[i], sdp_len))
            conv_input = tf.concat(conv, axis=0)
            convolve_sample_dim = self.FLAGS.embedding_dim * 2 + self.FLAGS.rel_dim
            kernel_window = 1
            pad = 'VALID'
        else:
            conv_input = tf.concat([sdp_embeddings, rel_embedding], axis=1)
            convolve_sample_dim = self.FLAGS.embedding_dim
            kernel_window = 3
            pad = 'SAME'

        with tf.name_scope('convolution'):
            with tf.variable_scope('convolution', reuse=tf.AUTO_REUSE):

                convolve_sample = tf.expand_dims(conv_input, axis=-1)

                convolve_kernel = [kernel_window, convolve_sample_dim, 1, self.FLAGS.kernels_num]

                weight = tf.get_variable(name='weight', shape=convolve_kernel, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                biases = tf.get_variable(name='biases', shape=[self.FLAGS.kernels_num], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                convolve_vec = tf.nn.conv2d(convolve_sample, weight, strides=[1, 1, convolve_sample_dim, 1],
                                            padding=pad, name='convolve_feature')
                feature_map = tf.nn.relu(tf.nn.bias_add(convolve_vec, biases), name='activate_feature')

                print(feature_map)
                pooled = tf.nn.max_pool(feature_map, ksize=[1, conv_input.shape[1], 1, 1],
                                        strides=[1, conv_input.shape[1], 1, 1], padding=pad, name='max_pooling')
                pooled_flat = tf.reshape(pooled, [-1, self.FLAGS.kernels_num])

        with tf.name_scope('fully_conn'):

            with tf.variable_scope('fully_conn', reuse=tf.AUTO_REUSE):

                fully_input = tf.nn.dropout(pooled_flat, self.dropout)

                weight = tf.get_variable(name="weight", shape=[self.FLAGS.kernels_num, self.FLAGS.labels_num],
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

    def concat2d(self, sdp_entity, rel, max_len):
        out = []
        for index in range(int(max_len)-1):
            c = tf.reshape(tf.concat([sdp_entity[index], sdp_entity[index+1], rel[index]], axis=0),
                           [-1, self.FLAGS.embedding_dim * 2 + self.FLAGS.rel_dim])
            out.append(c)
        return tf.reshape(tf.concat(out, axis=1), [1, -1, self.FLAGS.embedding_dim * 2 + self.FLAGS.rel_dim])
