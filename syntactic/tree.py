from nltk.parse.stanford import StanfordDependencyParser
from nltk.tree import Tree
import os
import traceback


class DepTreeHelper:

    parser_jar = "../semeval08/stanford/stanford-parser.jar"
    model_jar = "../semeval08/stanford/stanford-parser-3.8.0-models.jar"

    def __init__(self):
        # fix the bug of "Use software specific configuration paramaters or set the JAVAHOME environment variable."
        os.environ['JAVAHOME'] = "/home/lxb/tool/jdk1.8.0_161"

        self.dep_parser = StanfordDependencyParser(path_to_jar=self.parser_jar, path_to_models_jar=self.model_jar)

    def dep_graph(self, sentence):
        parse_tree = list(self.dep_parser.parse(sentence))
        return parse_tree[0]

    def dep_tree(self, graph, sentence, entity1_pos, entity2_pos):
        """
        调用获取 sentence 对应的 dependency map
        map中以 位置为索引key TreeNode为value
        """
        dp_tree = graph.tree()
        root = TreeNode(None, "root", -1)
        word_map = dict()
        word_map[-1] = root
        self.convert_tree(dp_tree, root, dict(), sentence, word_map)
        for index, w in enumerate(sentence):
            if index in word_map.keys():
                tree_node = word_map[index]

                tree_node.pos_entity1 = self.relative_pos(tree_node, word_map[entity1_pos])
                tree_node.pos_entity2 = self.relative_pos(tree_node, word_map[entity2_pos])
            else:
                node = TreeNode(root, w, index)
                word_map[index] = node
        return word_map

    @staticmethod
    def relation(graph, child_chain):
        root = graph.nodes
        rel = []
        for index in child_chain:
            rel.append(root[index+1]['rel'])
        return rel

    def sdp(self, entity1, entity2):
        shared_ancestor = self.ancestor(entity1, entity2)
        left = []
        right = []
        while entity1.index != shared_ancestor.index:
            left.append(entity1)
            entity1 = entity1.parent
        left.append(shared_ancestor)
        while entity2.index != shared_ancestor.index:
            right.append(entity2)
            entity2 = entity2.parent
        right.append(shared_ancestor)
        return left, right

    def traversal(self, root):

        if len(root.children) == 0:
            print(root.text, root.index)
            return
        print(root.text, root.index)
        for sub in root.children:
            self.traversal(sub)

    def convert_tree(self, raw_tree, parent, count_dict, sentences, word_map):

        if type(raw_tree) is not Tree:
            if raw_tree in count_dict:
                count_dict[raw_tree] += 1
            else:
                count_dict[raw_tree] = 0
            sentence_index = self.find_index(sentences, raw_tree, count_dict[raw_tree])
            tree_node = TreeNode(parent, raw_tree, sentence_index)
            parent.add_child(tree_node)
            word_map[sentence_index] = tree_node
            return

        if raw_tree.label() in count_dict:
            count_dict[raw_tree.label()] += 1
        else:
            count_dict[raw_tree.label()] = 0
        sentence_index = self.find_index(sentences, raw_tree.label(), count_dict[raw_tree.label()])
        tree_node = TreeNode(parent, raw_tree.label(), sentence_index)
        parent.add_child(tree_node)
        word_map[sentence_index] = tree_node

        for sub_tree in raw_tree:
            self.convert_tree(sub_tree, tree_node, count_dict, sentences, word_map)

    @staticmethod
    def find_index(sentence, entry, count):
        i = 0
        for index, word in enumerate(sentence):
            if word == entry:
                if count == i:
                    return index
                i += 1
        print(sentence, entry, count)
        raise Exception("error index")

    def ancestor(self, word, entry):
        if self.is_parent(word, entry.index):
            return entry
        if self.is_parent(entry, word.index):
            return word

        word_parent = word
        while word.index != -1:
            entry_parent = entry
            while entry_parent.index != -1:
                if word_parent.index == entry_parent.index:
                    return word_parent
                entry_parent = entry_parent.parent
            word_parent = word_parent.parent

        return word_parent

    @staticmethod
    def is_parent(word, parent_index):
        temp_parent = word
        while temp_parent.index != -1:
            if temp_parent.index == parent_index:
                return True
            temp_parent = temp_parent.parent
        return False

    def relative_pos(self, word, entry):
        parent = self.ancestor(word, entry)
        relative_word = self.distance(parent, word)
        relative_entry = self.distance(parent, entry)
        relative = relative_word + relative_entry
        if self.is_parent(word, entry.index):
            relative = -relative

        return relative

    @staticmethod
    def distance(parent, child):
        dis = 0
        child_parent = child
        while parent.index != child_parent.index:
            child_parent = child_parent.parent
            dis += 1
        return dis


class TreeNode:

    def __init__(self, parent, text, index):

        self.index = index
        self.text = text
        self.parent = parent
        self.children = list()
        self.pos_entity1 = 99
        self.pos_entity2 = 99

    def add_child(self, child):
        self.children.append(child)


def process():

    lines = list(open('../semeval08/train.txt', 'r').readlines())
    sentences = [s.split(' ') for s in [l.strip().lower() for l in lines]]
    helper = DepTreeHelper()
    with open('./dp.txt', 'w') as file:
        line_num = 0
        for sentence in sentences:
            try:
                line_num += 1
                batch = list()
                batch.append(' '.join(sentence) + '\n')

                graph = helper.dep_graph(sentence[5:])
                locations = list(map(eval, sentence[1:5]))
                sent_map = helper.dep_tree(graph, sentence[5:], locations[0], locations[2])
                l_sdp, r_sdp = helper.sdp(sent_map[locations[0]], sent_map[locations[2]])

                l_word = []
                r_word = []
                l_chain = []
                r_chain = []
                for l in l_sdp:
                    l_word.append(l.text)
                    l_chain.append(l.index)
                for r in r_sdp:
                    r_word.append(r.text)
                    r_chain.append(r.index)
                batch.append(' '.join(l_word) + ' # ' + ' '.join(r_word) + '\n')

                sdp_str = []
                if len(l_chain):
                    sdp_str.extend(helper.relation(graph, l_chain))
                sdp_str.append('#')
                if len(r_chain):
                    sdp_str.extend(helper.relation(graph, r_chain))
                batch.append(' '.join(sdp_str) + '\n')

                file.writelines(batch)
                print(line_num)
            except Exception as e:
                traceback.print_exc()
                print('error line no: {0}'.format(line_num))
                print('error sentence: {0}'.format(sentence))
                file.writelines(sentence)


if '__main__' == __name__:

    sent = "either we saw our mothers show fear from bugs / spiders and learned to fear them irrationally .".split(' ')
    helper = DepTreeHelper()
    graph = helper.dep_graph(sent)
    sent_map = helper.dep_tree(graph, sent, 6, 8)
    l_sdp, r_sdp = helper.sdp(sent_map[6], sent_map[8])

    l_word = []
    r_word = []
    l_chain = []
    r_chain = []
    for l in l_sdp:
        l_word.append(l.text)
        l_chain.append(l.index)
    for r in r_sdp:
        r_word.append(r.text)
        r_chain.append(r.index)
    print(' '.join(l_word) + ' # ' + ' '.join(r_word) + '\n')

    sdp_str = []
    if len(l_chain):
        sdp_str.extend(helper.relation(graph, l_chain))
    sdp_str.append('#')
    if len(r_chain):
        sdp_str.extend(helper.relation(graph, r_chain))
    print(' '.join(sdp_str) + '\n')
