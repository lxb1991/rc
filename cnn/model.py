import numpy as np
import tensorflow as tf


class Batch:

    batch_size = 100

    def __init__(self, samples, labels, rel_pos1, rel_pos2, entity1, entity2, nearby1, nearby2):

        self.samples = samples
        self.labels = labels
        self.rel_pos1 = rel_pos1
        self.rel_pos2 = rel_pos2
        self.entity1 = entity1
        self.entity2 = entity2
        self.nearby_words1 = nearby1
        self.nearby_words2 = nearby2

    def generate(self, shuffle):

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(self.samples, self.labels, self.rel_pos1, self.rel_pos2,
                                                  self.entity1, self.entity2, self.nearby_words1, self.nearby_words2)))
        else:
            shuffled_data = list(zip(self.samples, self.labels, self.rel_pos1, self.rel_pos2, self.entity1,
                                     self.entity2, self.nearby_words1, self.nearby_words2))

        batch_nums = int(len(self.samples)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.samples))]


class Cnn:

    MODEL_NAME = 'origin_cnn'

    def __init__(self, flags, embeddings, sentence_len):

        self.FLAGS = flags
        self.samples = tf.placeholder(tf.int32, [None, sentence_len], 'samples')
        self.labels = tf.placeholder(tf.int32, [None, flags.labels_num], 'labels')

        self.rel_pos1 = tf.placeholder(tf.int32, [None, sentence_len], "relative_entity1_pos")
        self.rel_pos2 = tf.placeholder(tf.int32, [None, sentence_len], "relative_entity2_pos")

        self.entity1 = tf.placeholder(tf.int32, [None], "entity1_pos")
        self.entity2 = tf.placeholder(tf.int32, [None], "entity2_pos")
        self.nearby_words1 = tf.placeholder(tf.int32, [None, 2], 'nearby_words1')
        self.nearby_words2 = tf.placeholder(tf.int32, [None, 2], 'nearby_words2')

        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.batch_size = tf.placeholder(tf.float32, name='batch_size')

        self.l2_loss, self.scores = self.build_graph(embeddings, sentence_len)

        self.predict, self.correct, self.accuracy = self.proc_accuracy()
        self.losses = self.loss()
        self.opt = self.optimize()

    def build_graph(self, embeddings, sentence_len):

        with tf.variable_scope("embedding_layer"):

            lookup_table = tf.get_variable('word_embeddings', dtype=tf.float32,
                                           initializer=tf.convert_to_tensor(embeddings, tf.float32))
            pos1_lookup_table = tf.get_variable('pos1_embeddings', [self.FLAGS.pos_num, self.FLAGS.pos_dim])
            pos2_lookup_table = tf.get_variable('pos2_embeddings', [self.FLAGS.pos_num, self.FLAGS.pos_dim])

            sentence_embedding = tf.nn.embedding_lookup(lookup_table, self.samples)
            entity1_embedding = tf.nn.embedding_lookup(lookup_table, self.entity1)
            entity2_embedding = tf.nn.embedding_lookup(lookup_table, self.entity2)

            if self.FLAGS.lexical:
                nearby_word1_emb = tf.nn.embedding_lookup(lookup_table, self.nearby_words1)
                nearby_word2_emb = tf.nn.embedding_lookup(lookup_table, self.nearby_words2)

            if self.FLAGS.pos:
                samples_embedding = tf.concat([sentence_embedding,
                                               tf.nn.embedding_lookup(pos1_lookup_table, self.rel_pos1),
                                               tf.nn.embedding_lookup(pos2_lookup_table, self.rel_pos2)], axis=2)
            else:
                samples_embedding = sentence_embedding

        samples_embedding = tf.nn.dropout(samples_embedding, self.dropout)

        with tf.variable_scope('convolution_layer', reuse=tf.AUTO_REUSE):

            samples = tf.expand_dims(samples_embedding, axis=-1)
            sample_dim = self.FLAGS.embedding_dim
            if self.FLAGS.pos:
                sample_dim += self.FLAGS.pos_dim * 2

            convolve_kernel = [self.FLAGS.kernel_window, sample_dim, 1, self.FLAGS.kernel_nums]
            weight = tf.get_variable(name='weight', shape=convolve_kernel, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            biases = tf.get_variable(name='biases', shape=[self.FLAGS.kernel_nums], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))

            convolve_vec = tf.nn.conv2d(samples, weight, strides=[1, 1, sample_dim, 1],
                                        padding='SAME', name='convolve_feature')

            feature_map = tf.nn.relu(tf.nn.bias_add(convolve_vec, biases), name='activate_feature')

            pooled = tf.nn.max_pool(feature_map, ksize=[1, sentence_len, 1, 1], strides=[1, sentence_len, 1, 1],
                                    padding="SAME", name='max_pooling')

            if self.FLAGS.attention:
                att_embedding = self.attention_emb()
                pooled_flat = tf.concat([tf.reshape(pooled, [-1, self.FLAGS.kernel_nums]), att_embedding], axis=1)
            else:
                pooled_flat = tf.reshape(pooled, [-1, self.FLAGS.kernel_nums])

        with tf.variable_scope('fully_conn_layer', reuse=tf.AUTO_REUSE):

            sample_dim = self.FLAGS.kernel_nums

            if self.FLAGS.attention:
                sample_dim += self.FLAGS.embedding_dim

            if self.FLAGS.lexical:
                nearby_word = tf.concat([tf.expand_dims(entity1_embedding, axis=1),
                                         tf.expand_dims(entity2_embedding, axis=1),
                                         nearby_word1_emb, nearby_word2_emb], axis=1)
                nearby_word_flat = tf.reshape(nearby_word, [-1, self.FLAGS.embedding_dim * 6])
                fully_concat = tf.concat([pooled_flat, nearby_word_flat], axis=1)

                sample_dim += self.FLAGS.embedding_dim * 6
                samples = tf.reshape(fully_concat, [-1, sample_dim])
            else:
                samples = pooled_flat

            samples = tf.nn.dropout(samples, self.dropout)

            weight = tf.get_variable(name="weight", shape=[sample_dim, self.FLAGS.labels_num],
                                     initializer=tf.truncated_normal_initializer(0.0, 0.1))
            biases = tf.get_variable(name='biases', shape=[self.FLAGS.labels_num], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            l2_loss = tf.nn.l2_loss(weight)

            scores = tf.nn.xw_plus_b(samples, weight, biases, name='score')

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

    def sentence_attention(self, att_w, att_b, sentence, entity1, entity2):

        weight = tf.map_fn(lambda i: tf.nn.tanh(
            tf.nn.xw_plus_b(tf.concat([tf.reshape(sentence[i], [-1, self.FLAGS.embedding_dim]), entity1, entity2], axis=1),
                            att_w, att_b)), tf.range(sentence.shape[0]), dtype=tf.float32)

        return tf.nn.softmax(tf.squeeze(weight))

    def attention_emb(self, self_att, word_embedding, sentence_len, entity1=None, entity2=None):

        with tf.variable_scope('attention_layer'):
            if not self_att:
                att_w = tf.get_variable('att_W', shape=[self.FLAGS.embedding_dim * 3, 1], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(0.0, 0.1))
                att_b = tf.get_variable('att_B', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
                emb = word_embedding
                for hop in range(1):
                    out = tf.map_fn(lambda i: self.sentence_attention(att_w, att_b, emb[i], entity1[i],
                                    entity2[i]), tf.range(tf.to_int32(self.batch_size)), dtype=tf.float32)
                    emb = tf.multiply(word_embedding, tf.expand_dims(tf.cast(out, tf.float32), axis=-1))
                    att_embeddings = tf.reduce_sum(emb, axis=1)
            else:

                att_w2 = tf.get_variable('att_w2', shape=[100, word_dim], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                att_w1 = tf.get_variable('att_w1', shape=[1, 100], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                att_score = tf.map_fn(lambda x: tf.matmul(att_w1, x),
                                      tf.tanh(tf.map_fn(lambda y: tf.matmul(att_w2, y, transpose_b=True),
                                                        word_embedding)))
                att_weight = tf.reshape(tf.nn.softmax(att_score), [-1, sentence_len, 1])

                att_embeddings = tf.reduce_sum(tf.multiply(word_embedding, att_weight), axis=1)

        return att_embeddings
