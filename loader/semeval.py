import numpy as np
import pickle
import os
from syntactic import sdpchain


def load_data(file_path, label_nums, contains_pos=False):

    if contains_pos:
        return load_contains_pos(file_path, label_nums)
    else:
        return load_single_label(file_path, label_nums)


def load_contains_pos(file_path, label_nums):
    """
        加载数据 类似'18 2 2 6 6 a misty ridge uprises from the surge .'
    """
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


def load_single_label(file_path, label_nums):
    """
        加载只有标签的数据 '18 a misty <e1> ridge </e1> uprises from the <e2> surge </e2> .'
    """
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


def load_sdp(file_path, pkl_path, is_save):
    """
        返回 最短依存路径 sdp 集合
    """
    relation_type = {}
    pos_type = {}
    if not os.path.exists(pkl_path):
        lines = list(open(file_path, "r").readlines())
        sentences = [s.split(' ') for s in clean_sentences(lines)]

        line_tag = 0
        sdp_container = []

        sdp = []
        l_relation = []
        r_relation = []
        pos = []
        for sentence in sentences:

            if line_tag == 0:
                sdp.extend(sentence)
            elif line_tag == 1:
                for word in sentence:
                    if word not in relation_type:
                        relation_type[word] = len(relation_type)
            elif line_tag == 2:
                for word in sentence:
                    if word.startswith('l'):
                        l_relation.append(word[1:])
                    else:
                        r_relation.append(word[1:])
            elif line_tag == 3:
                for word in sentence:
                    if word not in pos_type:
                        pos_type[word] = len(pos_type)
                pos.extend(sentence)
                sdp_container.append(sdpchain.SdpChain(sdp, l_relation, r_relation, pos))
                line_tag = -1
                sdp = []
                l_relation = []
                r_relation = []
                pos = []
            line_tag += 1
        with open(pkl_path, 'wb') as out:
            pickle.dump(sdp_container, out, protocol=True)

        if is_save:
            with open('../semeval08/sdp/pkl/relation_type.pkl', 'wb') as out:
                pickle.dump(relation_type, out, protocol=True)
            with open('../semeval08/sdp/pkl/pos_type.pkl', 'wb') as out:
                pickle.dump(pos_type, out, protocol=True)
    else:
        print('load pkl from path:{0}'.format(pkl_path))
        sdp_container = pickle.load(open(pkl_path, 'rb'))
        relation_type = pickle.load(open('../semeval08/sdp/pkl/relation_type.pkl', 'rb'))
        pos_type = pickle.load(open('../semeval08/sdp/pkl/pos_type.pkl', 'rb'))
    return sdp_container, relation_type, pos_type


def load_labels(file_path, pkl_path):
    """
        只加载 labels 的信息
        :param file_path 代表原始数据
        :param pkl_path 代表 pkl 文件
    """

    if os.path.exists(pkl_path):
        labels = pickle.load(open(pkl_path, 'rb'))
        print('load pkl from path: {0}'.format(pkl_path))
    else:
        with open(file_path, 'r') as raw_file:
            lines = raw_file.readlines()
            sentences = [sent.split(' ') for sent in lines]
            labels = []
            for sentence in sentences:
                one_hot = np.zeros(19)
                one_hot[eval(sentence[0])] = 1
                labels.append(one_hot)
        with open(pkl_path, 'wb') as out:
            pickle.dump(labels, out, protocol=True)
    return labels


def load_file_lines(file_path):
    lines = open(file_path, 'r').readlines()
    elements = []
    elements2id = {}
    for ele_id, line in enumerate(lines):
        elements.append(line.strip())
        elements2id[line.strip()] = ele_id
    return elements, elements2id


def split_sentence(sentence, sp):
    mid = sentence.index(sp)
    l = []
    r = []
    for wid in range(len(sentence)):
        if wid < mid:
            l.append(sentence[wid])
        elif wid > mid:
            r.append(sentence[wid])
    return l, r


def load_sentence_len(file_path):

    lines = list(open(file_path, "r").readlines())
    sentences = [s.split(' ') for s in clean_sentences(lines)]
    return [len(x) for x in sentences]


def load_real_result(file_path):
    lines = list(open(file_path, "r").readlines())
    sentences = [s.split(' ') for s in clean_sentences(lines)]
    real = []
    for sentence in sentences:
        real.append(sentence[0])
    return real


def load_relation_type(file_path):
    """
        加载 labels 数字标签所对应的19种关系类型， 用于 F1 score 的计算
        结果中的line需要带 '\n' 为了写入文件方便换行
    """
    relations = {}
    lines = open(file_path, 'r').readlines()
    for index, line in enumerate(lines):
        relations[index] = line
    return relations


def clean_sentences(sentences):
    return [sentence.strip().lower() for sentence in sentences]
