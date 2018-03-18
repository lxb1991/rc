from loader import semeval, embeddings as emb
from cnn import model
import tensorflow as tf
from textutil import util
from textutil import f1measure


def train(train_batch, test_batch, embeddings, max_len):

    measure = f1measure.F1Measure(test_batch.labels)

    with tf.Graph().as_default():

        with tf.variable_scope(model.Cnn.MODEL_NAME, reuse=tf.AUTO_REUSE):
            cnn = model.Cnn(FLAGS, embeddings, max_len)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(FLAGS.epoch_num):

                train_accuracy = train_runner(epoch, sess, train_batch, cnn)
                accuracy, labels = test_runner(sess, test_batch, cnn)

                if train_accuracy > 0.7:
                    f1 = measure.f1_score(labels)
                    print('epoch:{0} test f1 score: {1}'.format(epoch, f1))
                    if max_accuracy < accuracy:
                        max_accuracy = accuracy
                    if max_f1score < f1:
                        max_f1score = f1
                    print('max accuracy: {0} max f1: {1}\n'.format(max_accuracy, max_f1score))


def train_runner(epoch, sess, train_batch, cnn):

    train_loss = 0
    train_accuracy = 0

    for batch in train_batch.generate(True):
        samples, labels, rel_pos1, rel_pos2, e1, e2, nearby_w1, nearby_w2 = zip(*batch)

        feed_dict = {cnn.samples: samples, cnn.labels: labels, cnn.rel_pos1: rel_pos1,
                     cnn.rel_pos2: rel_pos2, cnn.entity1: e1, cnn.entity2: e2, cnn.nearby_words1: nearby_w1,
                     cnn.nearby_words2: nearby_w2, cnn.dropout: 0.5}

        _, losses, accuracy = sess.run([cnn.opt, cnn.losses, cnn.accuracy], feed_dict=feed_dict)

        train_loss += losses
        train_accuracy += accuracy
    batch_nums = len(train_batch.samples) / train_batch.batch_size
    print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss / batch_nums,
                                                              train_accuracy / batch_nums))
    return train_accuracy / batch_nums


def test_runner(sess, test_batch, cnn):
    test_loss = 0
    test_accuracy = 0
    p_labels = []
    for batch in test_batch.generate(False):
        samples, labels, rel_pos1, rel_pos2, e1, e2, nearby_w1, nearby_w2 = zip(*batch)

        feed_dict = {cnn.samples: samples, cnn.labels: labels, cnn.rel_pos1: rel_pos1,
                     cnn.rel_pos2: rel_pos2, cnn.entity1: e1, cnn.entity2: e2, cnn.nearby_words1: nearby_w1,
                     cnn.nearby_words2: nearby_w2, cnn.dropout: 1.0}

        losses, p_label, accuracy, correct = sess.run([cnn.losses, cnn.predict, cnn.accuracy, cnn.correct],
                                                      feed_dict=feed_dict)

        test_loss += losses
        test_accuracy += accuracy
        p_labels.extend(p_label.tolist())

    batch_nums = len(test_batch.samples) / test_batch.batch_size
    print("test=> loss: {0} accuracy: {1}".format(test_loss / batch_nums, test_accuracy / batch_nums))
    return test_accuracy / batch_nums, p_labels


def build_batch(samples, labels, entities, vocab, max_len):

    util.pad_sentence(samples, max_len)

    samples_id, oov_vocab = util.word2index(samples, vocab)
    rel_e1_pos, rel_e2_pos = util.entities_pos(samples, entities)
    e1, e2, nearby_e1, nearby_e2 = util.nearby_entities(samples, vocab, entities)

    batch = model.Batch(samples_id, labels, rel_e1_pos, rel_e2_pos, e1, e2, nearby_e1, nearby_e2)
    return batch, oov_vocab


def main(unused_argv):

    train_samples, train_labels, train_entities, train_max_lens = semeval.load_data(FLAGS.train_data_path,
                                                                                    FLAGS.labels_num, True)

    test_samples, test_labels, test_entities, test_max_lens = semeval.load_data(FLAGS.test_data_path,
                                                                                FLAGS.labels_num, True)

    max_len = max(train_max_lens, test_max_lens)

    print("load data complete, sentence max len: {0}".format(max_len))

    vocab, embeddings = emb.load_embedding(FLAGS.embedding_dim, emb.words_path, emb.em_path)
    print("origin_vocab len : {0}".format(len(vocab)))
    embeddings = emb.padding_word(embeddings, vocab, FLAGS.embedding_dim, util.padding_word)

    train_batch, train_oov = build_batch(train_samples, train_labels, train_entities, vocab, max_len)
    embeddings = emb.pad_embedding(vocab, embeddings, train_oov, FLAGS.embedding_dim)

    test_batch, test_oov = build_batch(test_samples, test_labels, test_entities, vocab, max_len)
    embeddings = emb.pad_embedding(vocab, embeddings, test_oov, FLAGS.embedding_dim)

    train(train_batch, test_batch, embeddings, max_len)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_data_path", "../semeval08/cnn/train.txt", "training data dir")
tf.app.flags.DEFINE_string("test_data_path", "../semeval08/cnn/test.txt", "test data dir")
tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("labels_num", 19, "max num of labels")
tf.app.flags.DEFINE_integer("embedding_dim", 50, "embedding dim")
tf.app.flags.DEFINE_integer("pos_num", 123, "position feature num")
tf.app.flags.DEFINE_integer("pos_dim", 5, "position feature dim")
tf.app.flags.DEFINE_integer("epoch_num", 200, "epoch num")
tf.app.flags.DEFINE_integer("kernel_window", 3, "convolution kernel window size")
tf.app.flags.DEFINE_integer("kernel_nums", 200, "convolution kernel nums")
tf.app.flags.DEFINE_boolean("lexical", True, "select lexical feature")
tf.app.flags.DEFINE_boolean("pos", True, "select position feature")
tf.app.flags.DEFINE_boolean("attention", False, "select attention mechanism")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

if '__main__' == __name__:

    tf.app.run()
