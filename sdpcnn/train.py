from loader import semeval, embeddings as emb
from sdpcnn import model
import tensorflow as tf
from syntactic import sdpchain
from textutil import util
from textutil import f1measure


def train(train_batch, test_batch, embeddings, relation_vocab, sdp_max, rel_max):

    measure = f1measure.F1Measure(test_batch.labels)

    with tf.Graph().as_default():
        with tf.variable_scope(model.SDPCNN.MODEL_NAME, reuse=tf.AUTO_REUSE):
            sdp_cnn = model.SDPCNN(FLAGS, embeddings, len(relation_vocab), sdp_max, rel_max)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(FLAGS.epoch_num):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate(True):
                    sdp_chain, rel, labels = zip(*batch)
                    feed_dict = {sdp_cnn.sdp: sdp_chain, sdp_cnn.relation: rel, sdp_cnn.labels: labels,
                                 sdp_cnn.dropout: 0.5}
                    _, losses, accuracy = sess.run([sdp_cnn.opt, sdp_cnn.losses, sdp_cnn.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.sdp)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                test_loss = 0
                test_accuracy = 0
                predict = []
                for batch in test_batch.generate(False):
                    sdp_chain, rel, labels = zip(*batch)
                    feed_dict = {sdp_cnn.sdp: sdp_chain, sdp_cnn.relation: rel, sdp_cnn.labels: labels,
                                 sdp_cnn.dropout: 1.0}
                    losses, p_labels, accuracy = sess.run([sdp_cnn.losses, sdp_cnn.predict, sdp_cnn.accuracy],
                                                          feed_dict=feed_dict)

                    test_loss += losses
                    test_accuracy += accuracy
                    predict.extend(p_labels)

                f1 = measure.f1_score(predict)
                batch_nums = len(test_batch.sdp) / test_batch.batch_size
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


def build_batch(sdp, relation, labels, vocab, relation_vocab, embedding, sdp_len, rel_len):

    util.pad_sentence(sdp, sdp_len)
    util.pad_sentence(relation, rel_len)

    sdp_ids, oov_vocab = util.word2index(sdp, vocab)
    embedding = emb.pad_embedding(vocab, embedding, oov_vocab, FLAGS.embedding_dim)

    relation_id = util.other2index(relation, relation_vocab)

    batch = model.Batch(sdp_ids, relation_id, labels)

    return batch, embedding


def main(unused_argv):

    train_labels = semeval.load_labels(FLAGS.train_data_path, FLAGS.train_pkl_path)
    test_labels = semeval.load_labels(FLAGS.test_data_path, FLAGS.test_pkl_path)

    train_sdp_container, train_relation, train_pos = semeval.load_sdp(FLAGS.train_sdp_path, FLAGS.train_sdp_pkl_path, True)
    test_sdp_container, test_relation, test_pos = semeval.load_sdp(FLAGS.test_sdp_path, FLAGS.test_sdp_pkl_path, False)

    relation_vocab = update_vocab(train_relation, test_relation)

    vocab, embedding = emb.load_embedding(FLAGS.embedding_dim, emb.words_path, emb.em_path)
    print("origin_vocab len : {0}".format(len(vocab)))
    embedding = emb.padding_word(embedding, vocab, FLAGS.embedding_dim, util.padding_word)

    train_sdp, train_rel, train_sdp_max, train_rel_max = sdpchain.create_sdp(train_sdp_container)
    test_sdp, test_rel, test_sdp_max, test_rel_max = sdpchain.create_sdp(test_sdp_container)

    sdp_max = max(train_sdp_max, test_sdp_max)
    rel_max = max(test_sdp_max, test_rel_max)

    print('sdp max len:{0}, relation max len:{1}'.format(sdp_max, rel_max))

    train_batch, embedding = build_batch(train_sdp, train_rel, train_labels, vocab, relation_vocab,
                                         embedding, sdp_max, rel_max)
    test_batch, embedding = build_batch(test_sdp, test_rel, test_labels, vocab, relation_vocab,
                                        embedding, sdp_max, rel_max)

    train(train_batch, test_batch, embedding, relation_vocab, sdp_max, rel_max)


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
tf.app.flags.DEFINE_integer("epoch_num", 300, "epoch num")
tf.app.flags.DEFINE_integer("kernels_num", 200, "cnn kernel size")
tf.app.flags.DEFINE_boolean("order", True, "sdp order, if order is true must require embedding_dim = rel_dim")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")


if '__main__' == __name__:

    tf.app.run()
