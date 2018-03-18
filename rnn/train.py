from loader import semeval, embeddings as emb
from rnn import model
import tensorflow as tf
from textutil import util
from textutil import f1measure


def train(train_batch, test_batch, embeddings, max_len):

    measure = f1measure.F1Measure(test_batch.labels)

    with tf.Graph().as_default():
        with tf.variable_scope(model.Rnn.MODEL_NAME, reuse=tf.AUTO_REUSE):
            rnn = model.Rnn(FLAGS, embeddings, max_len)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(FLAGS.epoch_num):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate(True):
                    samples, labels = zip(*batch)
                    feed_dict = {rnn.samples: samples, rnn.labels: labels, rnn.batch_size: len(samples),
                                 rnn.dropout: 0.5}

                    _, losses, accuracy = sess.run([rnn.opt, rnn.losses, rnn.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.samples)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                test_loss = 0
                test_accuracy = 0
                predict = []
                for batch in test_batch.generate(False):
                    samples, labels = zip(*batch)
                    feed_dict = {rnn.samples: samples, rnn.labels: labels, rnn.batch_size: len(samples),
                                 rnn.dropout: 1.0}

                    losses, p_labels, accuracy, correct = sess.run([rnn.losses, rnn.predict, rnn.accuracy,
                                                                    rnn.correct], feed_dict=feed_dict)
                    test_loss += losses
                    test_accuracy += accuracy
                    predict.extend(p_labels)

                batch_nums = len(test_batch.samples)/test_batch.batch_size
                f1 = measure.f1_score(predict)
                print("test=> loss: {0} accuracy: {1} f1_score: {2}".format(test_loss/batch_nums,
                                                                            test_accuracy/batch_nums, f1))

                if max_accuracy < (test_accuracy/batch_nums):
                    max_accuracy = (test_accuracy/batch_nums)
                if max_f1score < f1:
                    max_f1score = f1
                print('max accuracy: {0} max f1: {1}'.format(max_accuracy, max_f1score))


def build_batch(samples, labels, vocab, max_len):

    util.pad_sentence(samples, max_len)

    samples_id, oov_vocab = util.word2index(samples, vocab)

    batch = model.Batch(samples_id, labels)

    return batch, oov_vocab


def main(unused_argv):

    train_samples, train_labels, train_max_lens = semeval.load_data(FLAGS.train_data_path, FLAGS.labels_num)

    test_samples, test_labels, test_max_lens = semeval.load_data(FLAGS.test_data_path, FLAGS.labels_num)

    max_len = max(train_max_lens, test_max_lens)

    print("load data complete, sentence max len: {0}".format(max_len))

    vocab, embeddings = emb.load_embedding(FLAGS.embedding_dim, emb.words_path, emb.em_path)
    print("origin_vocab len : {0}".format(len(vocab)))
    embeddings = emb.padding_word(embeddings, vocab, FLAGS.embedding_dim, util.padding_word)

    train_batch, train_oov = build_batch(train_samples, train_labels, vocab, max_len)
    embeddings = emb.pad_embedding(vocab, embeddings, train_oov, FLAGS.embedding_dim)

    test_batch, test_oov = build_batch(test_samples, test_labels, vocab, max_len)
    embeddings = emb.pad_embedding(vocab, embeddings, test_oov, FLAGS.embedding_dim)

    train(train_batch, test_batch, embeddings, max_len)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_data_path", "../semeval08/rnn/train.txt", "training data dir")
tf.app.flags.DEFINE_string("test_data_path", "../semeval08/rnn/test.txt", "test data dir")
tf.app.flags.DEFINE_string("log_dir", "./logs", " the log dir")
tf.app.flags.DEFINE_integer("labels_num", 19, "max num of labels")
tf.app.flags.DEFINE_integer("embedding_dim", 50, "embedding dim")
tf.app.flags.DEFINE_integer("epoch_num", 30, "epoch num")
tf.app.flags.DEFINE_integer("rnn_hidden", 200, "rnn kernel nums")
tf.app.flags.DEFINE_boolean("rnn_cell", True, "use rnn or long short term memory")
tf.app.flags.DEFINE_boolean("attention", False, "select attention mechanism")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

if '__main__' == __name__:

    tf.app.run()
