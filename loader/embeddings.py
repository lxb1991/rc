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


words_path = "../semeval08/senna/words.lst"
em_path = "../semeval08/senna/embeddings.txt"
words_pkl_path = "../semeval08/senna/words.pkl"
em_pkl_path = "../semeval08/senna/embeddings.pkl"
