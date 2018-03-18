from nltk.parse.stanford import StanfordDependencyParser
import os


# source from >> https://github.com/Sshanu/Relation-Classification
def lca(tree, entity1_index, entity2_index):
    #  find ancestor
    node = entity1_index
    entity1_path = list()
    entity2_path = list()
    entity1_path.append(entity1_index)
    entity2_path.append(entity2_index)
    while node != tree.root:
        node = tree.nodes[node['head']]
        entity1_path.append(node)
    node = entity2_index
    while node != tree.root:
        node = tree.nodes[node['head']]
        entity2_path.append(node)
    for l1, l2 in zip(entity1_path[::-1], entity2_path[::-1]):
        if l1 == l2:
            ancestor = l1
    return ancestor


def path_lca(tree, node, lca_node):
    path = list()
    path.append(node)
    while node != lca_node:
        node = tree.nodes[node['head']]
        path.append(node)
    return path


def parse(sentences, out_file):

    i = 0
    with open(out_file, 'w') as out:
        for sentence in sentences:
            batch = []
            parse_tree = dep_parser.parse(sentence[5:])
            for tree in parse_tree:
                print(i)
                node1 = tree.nodes[int(sentence[1]) + 1]
                node2 = tree.nodes[int(sentence[3]) + 1]
                if node1['address'] and node2['address']:
                    lca_node = lca(tree, node1, node2)
                    path1 = path_lca(tree, node1, lca_node)
                    lca_node_index = len(path1) - 1
                    path2 = path_lca(tree, node2, lca_node)[:-1]
                    path1.extend(path2[::-1])

                    word_path = [p["word"] for p in path1]
                    rel_path = [p["rel"] for p in path1 if p != lca_node]
                    rel_path_dir = ['l' + p["rel"] if i < lca_node_index else 'r' + p["rel"] for i, p in enumerate(path1)
                                    if p != lca_node]
                    pos_path = [p["tag"] for p in path1]
                    batch.append(' '.join(word_path) + '\n')
                    batch.append(' '.join(rel_path) + '\n')
                    batch.append(' '.join(rel_path_dir) + '\n')
                    batch.append(' '.join(pos_path) + '\n')
                else:
                    print(i, node1["address"], node2["address"])
                i += 1
            out.writelines(batch)


def parse_train(file):
    lines = list(open('../semeval08/cnn/train.txt', 'r').readlines())
    train_sentences = [s.strip().lower().split(' ') for s in lines]
    parse(train_sentences, file)


def parse_test(file):
    lines = list(open('../semeval08/cnn/test.txt', 'r').readlines())
    test_sentences = [s.strip().lower().split(' ') for s in lines]
    parse(test_sentences, file)


if '__main__' == __name__:
    os.environ['JAVAHOME'] = '/home/lxb/tool/jdk1.8.0_161'
    os.environ['STANFORD_PARSER'] = "/home/lxb/work/mygithub/rc/semeval08/stanford/stanford-parser.jar"
    os.environ['STANFORD_MODELS'] = "/home/lxb/work/mygithub/rc/semeval08/stanford/stanford-parser-3.8.0-models.jar"
    mod_path = "/home/lxb/work/mygithub/rc/semeval08/stanford/englishPCFG.ser.gz"
    dep_parser = StanfordDependencyParser(model_path=mod_path,
                                          corenlp_options=['-originalDependencies',
                                                           '-outputFormatOptions', 'basicDependencies'])
    #train_out = '../semeval08/sdp/train_sdp.txt'
    #parse_train(train_out)
    test_out = '../semeval08/sdp/test_sdp.txt'
    parse_test(test_out)
