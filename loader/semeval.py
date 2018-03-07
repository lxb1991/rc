import numpy as np


def load_data(file_path, label_nums):

    lines = list(open(file_path, "r").readlines())
    sentences = [s.split(' ') for s in clean_sentences(lines)]

    samples = []
    labels = []

    locations = []
    max_len = 0

    for sentence in sentences:
        samples.append(sentence[5:])
        oh_label = np.zeros([label_nums], dtype=np.int32)
        oh_label[int(sentence[0])] = 1
        labels.append(oh_label)
        locations.append(list(map(eval, sentence[1:5])))
        if len(sentence[5:]) > max_len:
            max_len = len(sentence[5:])

    return samples, labels, locations, max_len


def load_data_raw(file_path, label_nums):

    lines = list(open(file_path, "r").readlines())
    sentences = [s.split(' ') for s in clean_sentences(lines)]

    samples = []
    labels = []

    max_len = 0

    for sentence in sentences:
        samples.append(sentence[1:])
        oh_label = np.zeros([label_nums], dtype=np.int32)
        oh_label[int(sentence[0])] = 1
        labels.append(oh_label)
        if len(sentence[5:]) > max_len:
            max_len = len(sentence[1:])

    return samples, labels, max_len


def load_relation_type(file_path):
    relations = {}
    lines = open(file_path, 'r').readlines()
    for index, line in enumerate(lines):
        relations[index] = line
    return relations


def clean_sentences(sentences):

    return [sentence.strip().lower() for sentence in sentences]

