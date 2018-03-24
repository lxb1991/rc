import numpy as np
import tensorflow as tf


class Batch:

    batch_size = 100

    def __init__(self, samples, samples_len, entity, sdp, sdp_lens, relation, labels):

        self.samples = samples
        self.samples_len = samples_len
        self.entity = entity
        self.sdp = sdp
        self.sdp_len = sdp_lens
        self.relation = relation
        self.labels = labels

    def generate(self, shuffle):

        if shuffle:
            shuffled_data = np.random.permutation(list(zip(self.samples, self.samples_len, self.entity, self.sdp,
                                                       self.sdp_len, self.relation, self.labels)))
        else:
            shuffled_data = list(zip(self.samples, self.samples_len, self.entity, self.sdp, self.sdp_len, self.relation,
                                     self.labels))
        batch_nums = int(len(self.sdp)/self.batch_size)
        for start in range(batch_nums):
            yield shuffled_data[start*self.batch_size:min((start+1)*self.batch_size, len(self.sdp))]


class AttNN:

    MODEL_NAME = 'att_cnn'

    def __init__(self, flags, embeddings, rel_vocab, max_sdp_len, max_rel_len, max_sent_len):

        self.FLAGS = flags

        self.sentence = tf.placeholder(tf.int32, [None, max_sent_len], 'sentence')
        self.sentence_lens = tf.placeholder(tf.int32, [None], 'sentence_lens')
        self.entity = tf.placeholder(tf.int32, [None, 2], 'entity')

        self.sdp = tf.placeholder(tf.int32, [None, max_sdp_len], 'sdp')
        self.sdp_lens = tf.placeholder(tf.int32, [None], 'sdp_lens')
        self.relation = tf.placeholder(tf.int32, [None, max_rel_len], 'relation')

        self.labels = tf.placeholder(tf.int32, [None, self.FLAGS.labels_num], 'labels')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        self.l2_loss, self.scores = self.build_graph(embeddings, rel_vocab, max_sdp_len)

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

            sentence_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.sentence), self.dropout)
            entity_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.entity), self.dropout)
            sdp_embeddings = tf.nn.dropout(tf.nn.embedding_lookup(lookup_table, self.sdp), self.dropout)
            # rel_embedding = tf.nn.dropout(tf.nn.embedding_lookup(relation_emb, self.relation), self.dropout)

        with tf.name_scope('representation'):
            sentence_rep = self.lstm_cell('sentence', sentence_embeddings, self.FLAGS.rnn_hidden, self.sentence_lens)
            sdp_rep = self.lstm_cell('sdp', sdp_embeddings, self.FLAGS.rnn_hidden, self.sdp_lens)
            entity_rep = self.lstm_cell('entity', entity_embeddings, self.FLAGS.rnn_hidden)

        with tf.name_scope('fuse'):
            sdp_att = self.attention_emb(sentence_rep, sdp_rep)
            sdp_hidden = self.lstm_cell('sdp_att', sdp_att, self.FLAGS.rnn_hidden)

            entity_att = self.attention_emb(sentence_rep, entity_rep)
            entity_hidden = self.lstm_cell('entity_att', entity_att, self.FLAGS.rnn_hidden)

            fuse_att = self.attention_emb(sdp_hidden, entity_rep)
            fuse_hidden = self.lstm_cell('fuse_att', fuse_att, self.FLAGS.rnn_hidden)

            pooled = tf.reduce_max(tf.concat([sentence_rep, sdp_hidden, entity_hidden, fuse_hidden], 2), axis=1)

        with tf.name_scope('fully_conn'):

            with tf.variable_scope('fully_conn', reuse=tf.AUTO_REUSE):
                fully_dim = self.FLAGS.rnn_hidden * 8
                fully_input = tf.nn.dropout(pooled, self.dropout)

                weight = tf.get_variable(name="weight", shape=[fully_dim, self.FLAGS.labels_num],
                                         initializer=tf.truncated_normal_initializer(0.0, 0.1))
                biases = tf.get_variable(name='biases', shape=[self.FLAGS.labels_num], dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                l2_loss = tf.nn.l2_loss(weight)

                scores = tf.nn.xw_plus_b(fully_input, weight, biases, name='score')

                return l2_loss, scores

    def lstm_cell(self, layer_name, samples_embedding, hidden_size, lens=None):

            with tf.variable_scope(layer_name+'recurrent_layer', reuse=tf.AUTO_REUSE):

                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        hidden_size, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(
                        hidden_size, forget_bias=1.0, state_is_tuple=True), input_keep_prob=self.dropout)

                    fw_init_state = fw_cell.zero_state(batch_size=tf.to_int32(self.batch_size), dtype=tf.float32)
                    bw_init_state = bw_cell.zero_state(batch_size=tf.to_int32(self.batch_size), dtype=tf.float32)
                    if lens is not None:
                        rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, samples_embedding,
                                                                             sequence_length=lens,
                                                                             initial_state_fw=fw_init_state,
                                                                             initial_state_bw=bw_init_state)
                    else:
                        rnn_out, rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, samples_embedding,
                                                                             initial_state_fw=fw_init_state,
                                                                             initial_state_bw=bw_init_state)
                    cell_out = tf.concat(rnn_out, axis=2)
                    return cell_out

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

    @staticmethod
    def attention_emb(word_embedding, entity):

        sim_matrix = tf.matmul(word_embedding, entity, transpose_b=True)
        attn = tf.matmul(tf.nn.softmax(sim_matrix, 0), entity)
        att_embeddings = tf.concat([word_embedding, attn], -1)

        return att_embeddings
