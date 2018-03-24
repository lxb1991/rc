from loader import semeval, embeddings as emb
from attnn import model
import tensorflow as tf
from syntactic import sdpchain
from textutil import util
from textutil import f1measure


def train(train_batch, test_batch, embeddings, relation_vocab, sdp_max, rel_max, sent_max):

    measure = f1measure.F1Measure(test_batch.labels)

    with tf.Graph().as_default():
        with tf.variable_scope(model.AttNN.MODEL_NAME, reuse=tf.AUTO_REUSE):
            att_cnn = model.AttNN(FLAGS, embeddings, len(relation_vocab), sdp_max, rel_max, sent_max)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(FLAGS.epoch_num):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate(True):
                    samples, samples_lens, entity, sdp_chain, sdp_lens, rel, labels = zip(*batch)
                    feed_dict = {att_cnn.sdp: sdp_chain, att_cnn.sdp_lens: sdp_lens,
                                 att_cnn.sentence: samples, att_cnn.sentence_lens: samples_lens,
                                 att_cnn.entity: entity, att_cnn.relation: rel, att_cnn.labels: labels,
                                 att_cnn.batch_size: len(samples), att_cnn.dropout: 0.5}
                    _, losses, accuracy = sess.run([att_cnn.opt, att_cnn.losses, att_cnn.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.sdp)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                test_loss = 0
                test_accuracy = 0
                predict = []
                for batch in test_batch.generate(False):
                    samples, samples_lens, entity, sdp_chain, sdp_lens, rel, labels = zip(*batch)
                    feed_dict = {att_cnn.sdp: sdp_chain, att_cnn.sdp_lens: sdp_lens,
                                 att_cnn.sentence: samples, att_cnn.sentence_lens: samples_lens,
                                 att_cnn.entity: entity, att_cnn.relation: rel, att_cnn.labels: labels,
                                 att_cnn.batch_size: len(samples), att_cnn.dropout: 1.0}
                    losses, p_labels, accuracy = sess.run([att_cnn.losses, att_cnn.predict, att_cnn.accuracy],
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


def count_len(sentences):
    lens = []
    for sent in sentences:
        lens.append(len(sent))
    return lens


def update_vocab(vocab1, vocab2):

    for key in vocab2:
        if key not in vocab1:
            vocab1[key] = len(vocab1)
    # 添加 padding 的索引
    vocab1[util.padding_word] = len(vocab1)
    return vocab1


def build_batch(samples, entities, sdp, relation, labels, vocab, relation_vocab, embedding, sdp_len, rel_len, sent_len):

    samples_lens = count_len(samples)
    sdp_lens = count_len(sdp)

    util.pad_sentence(samples, sent_len)
    util.pad_sentence(sdp, sdp_len)
    util.pad_sentence(relation, rel_len)

    sdp_ids, oov_vocab = util.word2index(sdp, vocab)
    embedding = emb.pad_embedding(vocab, embedding, oov_vocab, FLAGS.embedding_dim)
    samples_ids, oov_vocab = util.word2index(samples, vocab)
    embedding = emb.pad_embedding(vocab, embedding, oov_vocab, FLAGS.embedding_dim)

    relation_id = util.other2index(relation, relation_vocab)
    e1, e2, nearby_e1, nearby_e2 = util.nearby_entities(samples, vocab, entities)
    entity = []
    for e_1, e_2 in zip(e1, e2):
        e_temp = list()
        e_temp.append(e_1)
        e_temp.append(e_2)
        entity.append(e_temp)

    batch = model.Batch(samples_ids, samples_lens, entity, sdp_ids, sdp_lens, relation_id, labels)

    return batch, embedding


def main(unused_argv):

    train_samples, train_labels, train_entities, train_max_lens = semeval.load_data(FLAGS.train_data_path,
                                                                                    FLAGS.labels_num, True)

    test_samples, test_labels, test_entities, test_max_lens = semeval.load_data(FLAGS.test_data_path,
                                                                                FLAGS.labels_num, True)

    sentence_max_len = max(train_max_lens, test_max_lens)

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

    train_batch, embedding = build_batch(train_samples, train_entities, train_sdp, train_rel, train_labels, vocab,
                                         relation_vocab, embedding, sdp_max, rel_max, sentence_max_len)
    test_batch, embedding = build_batch(test_samples, test_entities, test_sdp, test_rel, test_labels, vocab,
                                        relation_vocab, embedding, sdp_max, rel_max, sentence_max_len)

    train(train_batch, test_batch, embedding, relation_vocab, sdp_max, rel_max, sentence_max_len)


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
tf.app.flags.DEFINE_integer("rnn_hidden", 100, "rnn_hidden dim")
tf.app.flags.DEFINE_integer("rel_dim", 50, "sdp relation dim")
tf.app.flags.DEFINE_integer("epoch_num", 300, "epoch num")
tf.app.flags.DEFINE_integer("kernels_num", 200, "cnn kernel size")
tf.app.flags.DEFINE_boolean("order", True, "sdp order, if order is true must require embedding_dim = rel_dim")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")


if '__main__' == __name__:

    tf.app.run()
