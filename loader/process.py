import re

raw_train = "../semeval08/raw/TRAIN_FILE.TXT"
raw_test = "../semeval08/raw/TEST_FILE.TXT"
raw_relations = "../semeval08/f1/relation_types.txt"
train_file = "../semeval08/rnn/train.txt"
test_file = "../semeval08/rnn/test.txt"


def process_raw():
    """
        生成rnn下的文件
    """
    relations = {}
    process_relation(relations)

    process_raw_file(relations, raw_train, train_file)
    process_raw_file(relations, raw_test, test_file)


def process_relation(relations):

    lines = open(raw_relations, 'r').readlines()
    for index, line in enumerate(lines):
        relations[line.strip()] = index
    return relations


def process_raw_file(relations, input_file, out_file):

    with open(input_file, 'r') as raw_file:
        line = raw_file.readline()
        line_num = 0
        train_sentences = []
        while line:
            if line_num == 0:
                sentence = clean_str(line[line.index('\"')+1:-2])
            elif line_num == 1:
                relation = relations[line.strip()]
            elif line_num == 3:
                sentence = str(relation) + ' ' + sentence + '\n'
                train_sentences.append(sentence)
                line_num = -1
            line_num += 1
            line = raw_file.readline()

    with open(out_file, 'w') as train:
        train.writelines(train_sentences)


def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"<e1>", " <e1> ", string)
    string = re.sub(r"</e1>", " </e1> ", string)
    string = re.sub(r"<e2>", " <e2> ", string)
    string = re.sub(r"</e2>", " </e2> ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    # process_raw()
    pass
