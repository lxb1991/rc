from loader import semeval, embeddings as emb
from sdplstm import model
import tensorflow as tf
from syntactic import sdpchain
from textutil import util
from textutil import f1measure


JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']
pos_tag2id = {'OTH': 0, 'JJ': 1, 'NN': 2, 'RB': 3, 'PRP': 4, "VB": 5, 'CC': 6, 'CD': 7, 'DT': 8, 'IN': 9}


def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 0


def train(train_batch, test_batch, embeddings, relation_vocab, pos_vocab):

    measure = f1measure.F1Measure(test_batch.labels)

    with tf.Graph().as_default():
        with tf.variable_scope(model.SDPNet.MODEL_NAME, reuse=tf.AUTO_REUSE):
            sdp = model.SDPNet(FLAGS, embeddings, len(relation_vocab), len(pos_vocab), 8, 10)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(FLAGS.epoch_num):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate(True):
                    sdp_left, sdp_right, rel_left, rel_right, pos_left, pos_right, labels, channel_len = zip(*batch)
                    feed_dict = {sdp.sdp_left: sdp_left, sdp.sdp_right: sdp_right, sdp.relation_left: rel_left,
                                 sdp.relation_right: rel_right, sdp.pos_left: pos_left, sdp.pos_right: pos_right,
                                 sdp.labels: labels, sdp.batch_size: len(sdp_left), sdp.channel_len: channel_len,
                                 sdp.dropout: 0.5}
                    _, losses, accuracy = sess.run([sdp.opt, sdp.losses, sdp.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.sdp_l)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                test_loss = 0
                test_accuracy = 0
                predict = []
                for batch in test_batch.generate(False):
                    sdp_left, sdp_right, rel_left, rel_right, pos_left, pos_right, labels, channel_len = zip(*batch)
                    feed_dict = {sdp.sdp_left: sdp_left, sdp.sdp_right: sdp_right, sdp.relation_left: rel_left,
                                 sdp.relation_right: rel_right, sdp.pos_left: pos_left, sdp.pos_right: pos_right,
                                 sdp.labels: labels, sdp.batch_size: len(sdp_left), sdp.channel_len: channel_len,
                                 sdp.dropout: 1.0}
                    losses, p_labels, accuracy = sess.run([sdp.losses, sdp.predict, sdp.accuracy],
                                                          feed_dict=feed_dict)

                    test_loss += losses
                    test_accuracy += accuracy
                    predict.extend(p_labels)

                f1 = measure.f1_score(predict)
                batch_nums = len(test_batch.sdp_l) / test_batch.batch_size
                print("test=> loss: {0} accuracy: {1} f1_score: {2}".format(test_loss/batch_nums,
                                                                            test_accuracy/batch_nums, f1))

                if max_accuracy < (test_accuracy/batch_nums):
                    max_accuracy = (test_accuracy/batch_nums)
                if max_f1score < f1:
                    max_f1score = f1
                print('max accuracy: {0} max f1: {1}'.format(max_accuracy, max_f1score))


def update_vocab(vocab1, vocab2):

    for key in vocab2:
        if key not in vocab1:
            vocab1[key] = len(vocab1)
    # 添加 padding 的索引
    vocab1[util.padding_word] = len(vocab1)
    return vocab1


def build_batch(sdp_container, vocab, relation_vocab, pos_vocab, embedding):

    left_sdp, left_relation, left_pos, right_sdp, right_relation, right_pos, channel_len = \
        sdpchain.create_sdp_direct(sdp_container)

    util.pad_sentence(left_sdp, 8)
    util.pad_sentence(right_sdp, 10)
    util.pad_sentence(left_relation, 8)
    util.pad_sentence(right_relation, 10)
    util.pad_sentence(left_pos, 8)
    util.pad_sentence(right_pos, 10)

    left_sdp_ids, left_oov_vocab = util.word2index(left_sdp, vocab)
    embedding = emb.pad_embedding(vocab, embedding, left_oov_vocab, FLAGS.embedding_dim)
    right_sdp_ids, right_oov_vocab = util.word2index(right_sdp, vocab)
    embedding = emb.pad_embedding(vocab, embedding, right_oov_vocab, FLAGS.embedding_dim)

    left_relation_ids = util.other2index(left_relation, relation_vocab)
    right_relation_ids = util.other2index(right_relation, relation_vocab)

    left_pos_ids = util.other2index(left_pos, pos_vocab)
    right_pos_ids = util.other2index(right_pos, pos_vocab)

    batch = model.Batch(left_sdp_ids, right_sdp_ids, left_relation_ids, right_relation_ids, left_pos_ids,
                        right_pos_ids, channel_len)

    return batch, embedding


def main(unused_argv):

    train_labels = semeval.load_labels(FLAGS.train_data_path, FLAGS.train_pkl_path)
    test_labels = semeval.load_labels(FLAGS.test_data_path, FLAGS.test_pkl_path)

    train_sdp, train_relation, train_pos = semeval.load_sdp(FLAGS.train_sdp_path, FLAGS.train_sdp_pkl_path, True)
    test_sdp, test_relation, test_pos = semeval.load_sdp(FLAGS.test_sdp_path, FLAGS.test_sdp_pkl_path, False)

    relation_vocab = update_vocab(train_relation, test_relation)
    pos_vocab = update_vocab(train_pos, test_pos)

    vocab, embedding = emb.load_embedding(FLAGS.embedding_dim, emb.words_path, emb.em_path)
    print("origin_vocab len : {0}".format(len(vocab)))
    embedding = emb.padding_word(embedding, vocab, FLAGS.embedding_dim, util.padding_word)

    train_batch, embedding = build_batch(train_sdp, vocab, relation_vocab, pos_vocab, embedding)
    train_batch.labels = train_labels
    test_batch, embedding = build_batch(test_sdp, vocab, relation_vocab, pos_vocab, embedding)
    test_batch.labels = test_labels

    train(train_batch, test_batch, embedding, relation_vocab, pos_vocab)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_data_path", "../semeval08/cnn/train.txt", "training data dir")
tf.app.flags.DEFINE_string("test_data_path", "../semeval08/cnn/test.txt", "test data dir")
tf.app.flags.DEFINE_string("train_pkl_path", "../semeval08/sdp/pkl/train.pkl", "training pkl dir")
tf.app.flags.DEFINE_string("test_pkl_path", "../semeval08/sdp/pkl/test.pkl", "test pkl dir")
tf.app.flags.DEFINE_string("train_sdp_path", "../semeval08/sdp/train_sdp.txt", "training data dir")
tf.app.flags.DEFINE_string("test_sdp_path", "../semeval08/sdp/test_sdp.txt", "test data dir")
tf.app.flags.DEFINE_string("train_sdp_pkl_path", "../semeval08/sdp/pkl/train_sdp.pkl", "training pkl dir")
tf.app.flags.DEFINE_string("test_sdp_pkl_path", "../semeval08/sdp/pkl/test_sdp.pkl", "test pkl dir")
tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("labels_num", 19, "max num of labels")
tf.app.flags.DEFINE_integer("embedding_dim", 50, "embedding dim")
tf.app.flags.DEFINE_integer("rel_dim", 50, "sdp relation dim")
tf.app.flags.DEFINE_integer("pos_dim", 50, "sdp pos dim")
tf.app.flags.DEFINE_integer("epoch_num", 50, "epoch num")
tf.app.flags.DEFINE_integer("word_hidden", 200, "multi channel: word channel bi lstm hidden size")
tf.app.flags.DEFINE_integer("other_hidden", 100, "multi channel: other channel bi lstm hidden size")
tf.app.flags.DEFINE_boolean("rel_channel", True, "use relation channel")
tf.app.flags.DEFINE_boolean("pos_channel", True, "use rnn pos channel")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")


if '__main__' == __name__:

    tf.app.run()
