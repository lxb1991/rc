from loader import semeval, embeddings as emb
from rnn import model
import numpy as np
import tensorflow as tf
import subprocess
from analyzer import statistic
train_file = "../semeval08/raw/train.txt"
test_file = "../semeval08/raw/test.txt"


def train():

    train_samples, train_labels, train_max_lens = semeval.load_data_raw(train_file, model.label_nums)

    test_samples, test_labels, test_max_lens = semeval.load_data_raw(test_file, model.label_nums)

    id2relations = semeval.load_relation_type('../semeval08/f1/relation_types.txt')

    max_len = max(train_max_lens, test_max_lens)

    print("load data complete, sentence max len: {0}".format(max_len))

    train_batch, test_batch, embeddings = process_data(train_samples, train_labels, test_samples, test_labels, max_len)

    train_runner(train_batch, test_batch, embeddings, id2relations, max_len)


def train_runner(train_batch, test_batch, embeddings, id2relations, max_len):

    with tf.Graph().as_default():
        with tf.variable_scope('rnn', reuse=False):
            cnn = model.Rnn(embeddings, max_len, train_batch.batch_size)
        with tf.variable_scope('rnn', reuse=True):
            test = model.Rnn(embeddings, max_len, len(test_batch.samples))
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(model.epochs):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate():
                    samples, labels = zip(*batch)
                    feed_dict = {cnn.samples: samples, cnn.labels: labels, cnn.dropout: 0.5}

                    _, losses, accuracy = sess.run([cnn.opt, cnn.reg_losses, cnn.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.samples)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                feed_dict = {test.samples: test_batch.samples, test.labels: test_batch.labels, test.dropout: 1.0}

                losses, predict_labels, accuracy, correct = sess.run([test.reg_losses, test.predict_labels,
                                                                      test.accuracy, test.correct], feed_dict=feed_dict)

                f1 = f1_score(predict_labels.tolist(), np.argmax(test_batch.labels, axis=1).tolist(), id2relations)
                print("test=> loss: {0} accuracy: {1} f1_score: {2}".format(losses, accuracy, f1))

                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                if max_f1score < f1:
                    max_f1score = f1
                print('max accuracy: {0} max f1: {1}'.format(max_accuracy, max_f1score))

                if train_accuracy/batch_nums > 0.98:
                    statistic.error_distribute(correct, 'rnn')
                    return


def f1_score(predict_labels, test_labels, id2relations):
    prediction_result_file = open('../semeval08/f1/prediction_result.txt', 'w')
    real_result_file = open('../semeval08/f1/real_result.txt', 'w')

    for index in range(len(predict_labels)):
        real_result_file.write(str(index) + '\t' + id2relations[test_labels[index]])
        prediction_result_file.write(str(index) + '\t' + id2relations[predict_labels[index]])

    prediction_result_file.close()
    real_result_file.close()

    output = subprocess.getoutput('perl ../semeval08/f1/semeval2010_task8_scorer-v1.2.pl ../semeval08/f1/prediction_result.txt ../semeval08/f1/real_result.txt')
    f1 = float(output[-10:-5])
    return f1


def process_data(train_samples, train_labels, test_samples, test_labels, max_len):

    pad_sentence(train_samples, max_len)
    pad_sentence(test_samples, max_len)

    vocab, embeddings = emb.load_embedding(model.word_dim, emb.words_path, emb.em_path)
    origin_vocab = len(vocab)
    print("origin_vocab len : {0}".format(origin_vocab))

    pad_id = len(vocab)
    vocab[model.padding_word] = pad_id
    embeddings = np.asarray(embeddings)
    embeddings = np.vstack((embeddings, np.zeros([model.word_dim], dtype=float)))

    train_batch = word2index(train_samples, vocab)
    train_batch.labels = train_labels
    test_batch = word2index(test_samples, vocab)
    test_batch.labels = test_labels

    oov_nums = len(vocab) - origin_vocab - 1
    oov_embeddings = np.random.normal(0, 0.1, [oov_nums, model.word_dim])
    embeddings = np.vstack((embeddings, oov_embeddings))
    print("load complete vocab len : {0}, embeddings len: {1}".format(len(vocab), len(embeddings)))

    return train_batch, test_batch, embeddings


def word2index(sentences, vocab):
    oov_nums = 0
    input_ids = []

    for i, sentence in enumerate(sentences):
        vocab_wid = []
        for index, word in enumerate(sentence):
            if word not in vocab:
                oov_nums += 1
                vocab[word] = len(vocab)

            vocab_wid.append(vocab[word])
        input_ids.append(vocab_wid)

    print("out of vocab word count ={}".format(oov_nums))
    return model.Batch(input_ids)


def pad_sentence(sentences, max_len):
    for sen in sentences:
        padding_num = max_len - len(sen)
        if padding_num > 0:
            sen.extend(padding_num * [model.padding_word])
    return sentences


if '__main__' == __name__:
    train()
