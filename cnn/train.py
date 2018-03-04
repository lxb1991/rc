from loader import semeval, embeddings as emb
from cnn import model
import numpy as np
import tensorflow as tf
import subprocess


def train():

    train_samples, train_labels, train_entities_locations, train_max_lens = \
        semeval.load_data('../semeval08/train.txt', model.label_nums)

    test_samples, test_labels, test_entities_locations, test_max_lens = \
        semeval.load_data('../semeval08/test.txt', model.label_nums)

    id2relations = semeval.load_relation_type('../semeval08/f1/relation_types.txt')

    max_len = max(train_max_lens, test_max_lens)

    print("load data complete, sentence max len: {0}".format(max_len))

    train_batch, test_batch, embeddings = process_data(train_samples, train_labels, train_entities_locations,
                                                       test_samples, test_labels, test_entities_locations, max_len)

    train_runner(train_batch, test_batch, embeddings, id2relations, max_len)


def train_runner(train_batch, test_batch, embeddings, id2relations, max_len):

    with tf.Graph().as_default():
        with tf.variable_scope('cnn', reuse=False):
            cnn = model.Cnn(embeddings, max_len)
        with tf.variable_scope('cnn', reuse=True):
            test = model.Cnn(embeddings, max_len)
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            max_accuracy = 0
            max_f1score = 0

            for epoch in range(model.epochs):

                train_loss = 0
                train_accuracy = 0

                for batch in train_batch.generate():
                    samples, labels, rel_pos1, rel_pos2, nearby_w1, nearby_w2 = zip(*batch)

                    feed_dict = {cnn.samples: samples, cnn.labels: labels, cnn.rel_pos1: rel_pos1,
                                 cnn.rel_pos2: rel_pos2, cnn.nearby_words1: nearby_w1, cnn.nearby_words2: nearby_w2,
                                 cnn.dropout: 0.5}

                    _, losses, accuracy = sess.run([cnn.opt, cnn.reg_losses, cnn.accuracy], feed_dict=feed_dict)

                    train_loss += losses
                    train_accuracy += accuracy
                batch_nums = len(train_batch.samples)/train_batch.batch_size
                print("train=> epoch: {0} loss: {1} accuracy: {2}".format(epoch, train_loss/batch_nums,
                                                                          train_accuracy/batch_nums))

                feed_dict = {test.samples: test_batch.samples, test.labels: test_batch.labels,
                             test.rel_pos1: test_batch.rel_pos1, test.rel_pos2: test_batch.rel_pos2,
                             test.nearby_words1: test_batch.nearby_words1, test.nearby_words2: test_batch.nearby_words2,
                             test.dropout: 1.0}

                losses, predict_labels, accuracy = sess.run([test.reg_losses, test.predict_labels, test.accuracy],
                                                            feed_dict=feed_dict)
                f1 = f1_score(predict_labels.tolist(), np.argmax(test_batch.labels, axis=1).tolist(), id2relations)
                print("test=> loss: {0} accuracy: {1} f1_score: {2}".format(losses, accuracy, f1))

                if max_accuracy < accuracy:
                    max_accuracy = accuracy
                if max_f1score < f1:
                    max_f1score = f1
                print('max accuracy: {0} max f1: {1}'.format(max_accuracy, max_f1score))

                if train_accuracy/batch_nums > 0.98:
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


def process_data(train_samples, train_labels, train_entities_loc, test_samples, test_labels, test_entities_loc,
                 max_len):

    pad_sentence(train_samples, max_len)
    pad_sentence(test_samples, max_len)

    vocab, embeddings = emb.load_embedding(model.word_dim, emb.words_path, emb.em_path)
    origin_vocab = len(vocab)
    print("origin_vocab len : {0}".format(origin_vocab))

    pad_id = len(vocab)
    vocab[model.padding_word] = pad_id
    embeddings = np.asarray(embeddings)
    embeddings = np.vstack((embeddings, np.zeros([model.word_dim], dtype=float)))

    train_batch = word2index(train_samples, vocab, train_entities_loc)
    train_batch.labels = train_labels
    test_batch = word2index(test_samples, vocab, test_entities_loc)
    test_batch.labels = test_labels

    oov_nums = len(vocab) - origin_vocab - 1
    oov_embeddings = np.random.normal(0, 0.1, [oov_nums, model.word_dim])
    embeddings = np.vstack((embeddings, oov_embeddings))
    print("load complete vocab len : {0}, embeddings len: {1}".format(len(vocab), len(embeddings)))

    return train_batch, test_batch, embeddings


def word2index(sentences, vocab, entities_locations):
    oov_nums = 0
    input_ids = []
    entities_rel_pos1 = []
    entities_rel_pos2 = []
    nearby_words1 = []
    nearby_words2 = []

    for i, sentence in enumerate(sentences):
        vocab_wid = []
        rel2en1_pos = []
        rel2en2_pos = []

        en1_pos = int(entities_locations[i][0])
        en2_pos = int(entities_locations[i][2])
        for index, word in enumerate(sentence):
            if word not in vocab:
                oov_nums += 1
                vocab[word] = len(vocab)

            vocab_wid.append(vocab[word])
            rel2en1_pos.append(convert2positive(index - en1_pos))
            rel2en2_pos.append(convert2positive(index - en2_pos))

        e1_temp = []
        e2_temp = []
        e1_temp.append(vocab[sentence[en1_pos]])
        e2_temp.append(vocab[sentence[en2_pos]])

        if en1_pos >= 1:
            e1_temp.append(vocab[sentence[en1_pos-1]])
        else:
            e1_temp.append(vocab[sentence[0]])

        if en1_pos < len(sentence)-1:
            e1_temp.append(vocab[sentence[en1_pos+1]])
        else:
            e1_temp.append(vocab[sentence[en1_pos]])

        if en2_pos >= 1:
            e2_temp.append(vocab[sentence[en2_pos-1]])
        else:
            e2_temp.append(vocab[sentence[0]])

        if en2_pos < len(sentence) - 1:
            e2_temp.append(vocab[sentence[en2_pos+1]])
        else:
            e2_temp.append(vocab[sentence[en2_pos]])

        input_ids.append(vocab_wid)
        entities_rel_pos1.append(rel2en1_pos)
        entities_rel_pos2.append(rel2en2_pos)
        nearby_words1.append(e1_temp)
        nearby_words2.append(e2_temp)

    print("out of vocab word count ={}".format(oov_nums))
    return model.Batch(input_ids, entities_rel_pos1, entities_rel_pos2, nearby_words1, nearby_words2)


def convert2positive(pos):

    if pos < -60:
        pos = 0
    elif -60 <= pos <= 60:
        pos += 61
    else:
        pos = 122
    return pos


def pad_sentence(sentences, max_len):
    for sen in sentences:
        padding_num = max_len - len(sen)
        if padding_num > 0:
            sen.extend(padding_num * [model.padding_word])
    return sentences


if '__main__' == __name__:
    train()
