import numpy as np
import os
import pickle


def load_embedding(dim, words_file, em_file):

    embeddings = []
    vocab = {}

    if os.path.exists(words_pkl_path):
        print("load pkl ...")
        vocab = pickle.load(open(words_pkl_path, "rb"))
        embeddings = pickle.load(open(em_pkl_path, "rb"))
    else:
        words = open(words_file, "r").readlines()
        all_embedding = open(em_file, "r").readlines()

        for wid in range(len(words)):
            word = words[wid].strip()
            embedding = list(map(eval, all_embedding[wid].strip().split()))
            if len(embedding) != dim:
                print("Error ,parse the " + wid + "id embedding error")
            else:
                embeddings.append(embedding)
                vocab[word] = wid
        pickle.dump(vocab, open(words_pkl_path, "wb"), protocol=True)
        pickle.dump(embeddings, open(em_pkl_path, "wb"), protocol=True)

    return vocab, embeddings


def padding_word(embeddings, vocab, dim, pad_word):

    pad_id = len(vocab)
    vocab[pad_word] = pad_id
    embeddings = np.asarray(embeddings)
    return np.vstack((embeddings, np.zeros([dim], dtype=float)))


def pad_embedding(vocab, embeddings, oov_vocab, dim):

    oov_nums = len(oov_vocab)
    oov_embeddings = np.random.normal(0, 0.1, [oov_nums, dim])
    embeddings = np.vstack((embeddings, oov_embeddings))
    print("padding complete vocab len : {0}, embeddings len: {1}".format(len(vocab), len(embeddings)))

    return embeddings


words_path = "../semeval08/senna/words.lst"
em_path = "../semeval08/senna/embeddings.txt"
words_pkl_path = "../semeval08/senna/pkl/words.pkl"
em_pkl_path = "../semeval08/senna/pkl/embeddings.pkl"
