

padding_word = '<pad>'


def pad_sentence(sentences, max_len):
    """
        按照max_len长度，用 padding_word 补齐句子 sentences
    """
    for sen in sentences:
        padding_num = max_len - len(sen)
        if padding_num > 0:
            sen.extend(padding_num * [padding_word])
    return sentences


def word2index(sentences, vocab):
    """
        将 sentences 中 单词 解析为 vocab 中的索引
    """
    oov_vocab = {}
    input_ids = []

    for sentence in sentences:
        vocab_wid = []
        for word in sentence:
            if word not in vocab:
                oov_vocab[word] = len(vocab)
                vocab[word] = oov_vocab[word]
            vocab_wid.append(vocab[word])
        input_ids.append(vocab_wid)

    print("out of vocab word count ={}".format(len(oov_vocab)))
    return input_ids, oov_vocab


def other2index(sentences, vocab):
    """
        将 sentences 中 单词 解析为 vocab 中的索引
    """
    input_ids = []

    for sentence in sentences:
        vocab_wid = []
        for word in sentence:
            vocab_wid.append(vocab[word])
        input_ids.append(vocab_wid)

    return input_ids


def entities_pos(sentences, entities):
    """
        找到单词相对与实体的相对位置
        :param: entities 为实体位置
    """
    entities_rel_pos1 = []
    entities_rel_pos2 = []

    for i, sentence in enumerate(sentences):
        en1_pos = int(entities[i][0])
        en2_pos = int(entities[i][2])
        rel2en1_pos = []
        rel2en2_pos = []
        for index, word in enumerate(sentence):
            rel2en1_pos.append(convert2positive(index - en1_pos))
            rel2en2_pos.append(convert2positive(index - en2_pos))

        entities_rel_pos1.append(rel2en1_pos)
        entities_rel_pos2.append(rel2en2_pos)

    return entities_rel_pos1, entities_rel_pos2


def nearby_entities(sentences, vocab, entities):
    """
       找到实体的临近词
    """
    entities1 = []
    entities2 = []
    nearby_entity1 = []
    nearby_entity2 = []

    for i, sentence in enumerate(sentences):
        en1_pos = int(entities[i][0])
        en2_pos = int(entities[i][2])

        entities1.append(vocab[sentence[en1_pos]])
        entities2.append(vocab[sentence[en2_pos]])

        nearby_e1 = []
        nearby_e2 = []
        # 实体e1的临近词
        if en1_pos >= 1:
            nearby_e1.append(vocab[sentence[en1_pos - 1]])
        else:
            nearby_e1.append(vocab[sentence[0]])

        if en1_pos < len(sentence) - 1:
            nearby_e1.append(vocab[sentence[en1_pos + 1]])
        else:
            nearby_e1.append(vocab[sentence[en1_pos]])
        # 实体e2的临近词
        if en2_pos >= 1:
            nearby_e2.append(vocab[sentence[en2_pos - 1]])
        else:
            nearby_e2.append(vocab[sentence[0]])

        if en2_pos < len(sentence) - 1:
            nearby_e2.append(vocab[sentence[en2_pos + 1]])
        else:
            nearby_e2.append(vocab[sentence[en2_pos]])

        nearby_entity1.append(nearby_e1)
        nearby_entity2.append(nearby_e2)

    return entities1, entities2, nearby_entity1, nearby_entity2


def convert2positive(pos):
    """
        tensor flow 中需要的是索引值， 所以转化为正数
    """
    if pos < -60:
        pos = 0
    elif -60 <= pos <= 60:
        pos += 61
    else:
        pos = 122
    return pos


def max_list(sentences):
    """
        返回列表的最长长度
    """
    max_len = 0
    for sent in sentences:
        if len(sent) > max_len:
            max_len = len(sent)
    return max_len
