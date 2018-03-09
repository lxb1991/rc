import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from loader import semeval

root_path = '../semeval08/analysis/'


def error_distribute(labels, model_name):
    line_num = 0
    error_line = []
    for label in labels:
        if label:
            error_line.append(str(line_num))
        line_num += 1
    pickle.dump(error_line, open(root_path + model_name + '.pkl', 'wb'), protocol=True)


def draw_distribute(model, comp_model):

    if not os.path.exists(root_path + model + '.pkl') or not os.path.exists(root_path + comp_model + '.pkl'):
        raise FileNotFoundError('model need run the method of error_distribute')
    model_right = pickle.load(open(root_path + model + '.pkl', 'rb'))
    com_model_right = pickle.load(open(root_path + comp_model + '.pkl', 'rb'))

    sent_len = semeval.load_sentence_len('../semeval08/test.txt')

    model_wrong = sorted(list(set(range(2717)) - set([int(x) for x in model_right])))
    comp_model_wrong = sorted(list(set(range(2717)) - set([int(x) for x in com_model_right])))

    error = sorted(list(set(range(2717)) - set([int(x) for x in model_right]) - set([int(x) for x in com_model_right])))
    print(len(error))
    print(sorted([sent_len[x] for x in error]))

    # model_len = [sent_len[x] for x in model_wrong]
    # com_model_len = [sent_len[x] for x in comp_model_wrong]
    # print(sorted(model_len))
    # print(sorted(com_model_len))
    # len_statistic = np.zeros(40)
    # for r in range(40):
    #     threshold = (r+1) * 2
    #     for l in model_len:
    #         if threshold - 2 < l < threshold:
    #             len_statistic[r] += 1
    # com_len_statistic = np.zeros(40)
    # for r in range(40):
    #     threshold = (r+1) * 2
    #     for l in com_model_len:
    #         if threshold - 2 < l < threshold:
    #             com_len_statistic[r] += 1

    # model_x = range(102)
    # model_y = np.zeros(102)
    # com_model_y = np.zeros(102)
    # batch_size = 27
    # batch = 0
    # for r in model_wrong:
    #     if r <= batch * batch_size:
    #         model_y[batch] += 1
    #     if r > batch * batch_size:
    #         batch += 1
    # batch = 0
    # for r in comp_model_wrong:
    #     if r <= batch * batch_size:
    #         com_model_y[batch] += 1
    #     if r > batch * batch_size:
    #         batch += 1

    plt.bar(range(len(sorted([sent_len[x] for x in error]))), sorted([sent_len[x] for x in error]))
    plt.show()


def class_distribute(model):
    if not os.path.exists(root_path + model + '.pkl'):
        raise FileNotFoundError('model need run the method of error_distribute')
    model_line = pickle.load(open(root_path + model + '.pkl', 'rb'))
    real = semeval.load_real_result('../semeval08/test.txt')
    model_wrong = list(set(range(2717)) - set([int(x) for x in model_line]))
    classes = np.zeros(19)
    all_classes = np.zeros(19)
    for r in real:
        all_classes[eval(r)] += 1
    for line in model_wrong:
        cls = eval(real[line])
        classes[cls] += 1

    plt.plot(range(len(classes)), classes)
    plt.bar(range(len(classes)), classes)
    plt.plot(range(len(all_classes)), all_classes)
    plt.show()


if '__main__' == __name__:
    # draw_distribute("rnn", "cnn_temp")
    class_distribute('cnn')
