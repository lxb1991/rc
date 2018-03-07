import numpy as np
import tensorflow as tf


label_nums = 19
padding_word = '<pad>'
word_dim = 50

epochs = 1000
conv_kernel_window = 3
conv_kernel_nums = 150
learning_rate = 0.001

pos_feature = True
pos_num = 123
pos_dim = 5
lexical_feature = True

attention_1 = False
attention_2 = True
attention_self = False


class Batch:

    batch_size = 100
    shuffle = True

    def __init__(self, samples, rel_pos1, rel_pos2, nearby_words1, nearby_words2):

        self.samples = samples
        self.labels = []
        self.rel_pos1 = rel_pos1
        self.rel_pos2 = rel_pos2
        self.nearby_words1 = nearby_words1
        self.nearby_words2 = nearby_words2

    def generate(self):

        if self.shuffle:
            shuffled_data = np.random.permutation(list(zip(self.samples, self.labels, self.rel_pos1, self.rel_pos2,
                                                           self.nearby_words1, self.nearby_words2)))
        else:
            shuffled_data = list(zip(self.samples, self.labels, self.rel_pos1, self.rel_pos2, self.nearby_words1,
                                     self.nearby_words2))
        batch_nums = int(len(self.samples)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.samples))]


class Cnn:

    def __init__(self, embeddings, sentence_len):

        self.samples = tf.placeholder(tf.int32, [None, sentence_len], 'samples')
        self.labels = tf.placeholder(tf.int32, [None, label_nums], 'labels')
        self.rel_pos1 = tf.placeholder(tf.int32, [None, sentence_len], "relative_entity1_pos")
        self.rel_pos2 = tf.placeholder(tf.int32, [None, sentence_len], "relative_entity2_pos")
        self.nearby_words1 = tf.placeholder(tf.int32, [None, 3], 'nearby_words1')
        self.nearby_words2 = tf.placeholder(tf.int32, [None, 3], 'nearby_words2')
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                       initializer=tf.convert_to_tensor(embeddings, tf.float32))
        pos1_lookup_table = tf.get_variable('pos1_embeddings', [pos_num, pos_dim])
        pos2_lookup_table = tf.get_variable('pos2_embeddings', [pos_num, pos_dim])

        word_embeddings = tf.nn.embedding_lookup(lookup_table, self.samples)

        # if lexical_feature:
        nearby_word1_emb = tf.nn.embedding_lookup(lookup_table, self.nearby_words1)
        nearby_word2_emb = tf.nn.embedding_lookup(lookup_table, self.nearby_words2)

        if attention_1:
            # score function is matrix matmul: w M q
            self.att_relation1 = tf.get_variable('att_rel1', shape=[word_dim, word_dim], dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(0.0, 0.1))
            self.att_relation2 = tf.get_variable('att_rel2', shape=[word_dim, word_dim], dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(0.0, 0.1))
            att_word = tf.reshape(word_embeddings, [-1, word_dim])
            att_entity1 = tf.slice(nearby_word1_emb, [0, 0, 0], [-1, 1, -1])
            att_entity2 = tf.slice(nearby_word1_emb, [0, 0, 0], [-1, 1, -1])
            att_score1 = tf.matmul(tf.reshape(tf.matmul(att_word, self.att_relation1, transpose_b=True),
                                              [-1, sentence_len, word_dim]), att_entity1, transpose_b=True)
            att_score2 = tf.matmul(tf.reshape(tf.matmul(att_word, self.att_relation2, transpose_b=True),
                                              [-1, sentence_len, word_dim]), att_entity2, transpose_b=True)
            att_weight = tf.nn.softmax(tf.div(att_score1 + att_score2, 2), axis=1)

            word_embeddings = tf.multiply(word_embeddings, att_weight)
        elif attention_2:
            att_v = tf.get_variable('att_V', shape=[1, 100], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.1))
            att_w = tf.get_variable('att_W', shape=[100, word_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.1))
            att_u = tf.get_variable('att_U', shape=[100, word_dim], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.1))

            att_entity1 = tf.slice(nearby_word1_emb, [0, 0, 0], [-1, 1, -1])
            att_entity2 = tf.slice(nearby_word2_emb, [0, 0, 0], [-1, 1, -1])
            att_score1 = tf.map_fn(lambda x: tf.matmul(att_v, x),
                                   tf.tanh(tf.add(tf.map_fn(lambda x: tf.matmul(att_w, x, transpose_b=True),
                                                            word_embeddings),
                                                  tf.map_fn(lambda x: tf.matmul(att_u, x, transpose_b=True),
                                                            att_entity1))))
            att_score2 = tf.map_fn(lambda x: tf.matmul(att_v, x),
                                   tf.tanh(tf.add(tf.map_fn(lambda x: tf.matmul(att_w, x, transpose_b=True),
                                                            word_embeddings),
                                                  tf.map_fn(lambda x: tf.matmul(att_u, x, transpose_b=True),
                                                            att_entity2))))
            att_weight = tf.reshape(tf.nn.softmax(tf.div(tf.add(att_score1, att_score2), 2)), [-1, sentence_len, 1])
            word_embeddings = tf.multiply(word_embeddings, att_weight)
        elif attention_self:

            att_w2 = tf.get_variable('att_w2', shape=[100, word_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            att_w1 = tf.get_variable('att_w1', shape=[1, 100], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            att_score = tf.map_fn(lambda x: tf.matmul(att_w1, x),
                                  tf.tanh(tf.map_fn(lambda y: tf.matmul(att_w2, y, transpose_b=True), word_embeddings)))
            att_weight = tf.reshape(tf.nn.softmax(att_score), [-1, sentence_len, 1])
            word_embeddings = tf.multiply(word_embeddings, att_weight)

        if pos_feature:
            samples_embeddings = tf.concat([word_embeddings,
                                            tf.nn.embedding_lookup(pos1_lookup_table, self.rel_pos1),
                                            tf.nn.embedding_lookup(pos2_lookup_table, self.rel_pos2)], axis=2)
        else:
            samples_embeddings = word_embeddings

        samples_embeddings = tf.nn.dropout(samples_embeddings, self.dropout)

        with tf.name_scope('convolution'):

            with tf.variable_scope('convolution', reuse=tf.AUTO_REUSE):

                convolve_sample = tf.expand_dims(samples_embeddings, axis=-1)
                if pos_feature:
                    convolve_sample_dim = word_dim + pos_dim * 2
                else:
                    convolve_sample_dim = word_dim
                convolve_kernel = [conv_kernel_window, convolve_sample_dim, 1, conv_kernel_nums]
                weight = tf.get_variable(name='weight', shape=convolve_kernel, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                biases = tf.get_variable(name='biases', shape=[conv_kernel_nums], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))

                convolve_vec = tf.nn.conv2d(convolve_sample, weight, strides=[1, 1, convolve_sample_dim, 1],
                                            padding='SAME', name='convolve_feature')

                feature_map = tf.nn.relu(tf.nn.bias_add(convolve_vec, biases), name='activate_feature')

                pooled = tf.nn.max_pool(feature_map, ksize=[1, sentence_len, 1, 1], strides=[1, sentence_len, 1, 1],
                                        padding="SAME", name='max_pooling')

                pooled_flat = tf.reshape(pooled, [-1, conv_kernel_nums])

        with tf.name_scope('fully_conn'):

            with tf.variable_scope('fully_conn', reuse=tf.AUTO_REUSE):

                if lexical_feature:
                    nearby_word = tf.concat([nearby_word1_emb, nearby_word2_emb], axis=2)
                    nearby_word_flat = tf.reshape(nearby_word, [-1, word_dim * 6])
                    fully_concat = tf.concat([pooled_flat, nearby_word_flat], axis=1)
                    fully_dim = conv_kernel_nums + word_dim * 6
                    fully_input = tf.reshape(fully_concat, [-1, fully_dim])
                else:
                    fully_input = pooled_flat
                    fully_dim = conv_kernel_nums

                fully_input = tf.nn.dropout(fully_input, self.dropout)

                weight = tf.get_variable(name="weight", shape=[fully_dim, label_nums],
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                biases = tf.get_variable(name='biases', shape=[label_nums], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                l2_loss = tf.nn.l2_loss(weight)

                scores = tf.nn.xw_plus_b(fully_input, weight, biases, name='score')

        with tf.name_scope('accuracy'):

            self.predict_labels = tf.argmax(scores, axis=1, name='predict')
            correct = tf.equal(self.predict_labels, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='calculate_accuracy')

        with tf.name_scope('loss'):

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(None, self.labels, scores)
            self.reg_losses = tf.reduce_mean(losses) + 0.01 * l2_loss

        with tf.name_scope('optimizer'):

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.opt = optimizer.minimize(self.reg_losses)
