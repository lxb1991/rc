

class SdpChain:

    def __init__(self, sdp, l_relation, r_relation, pos):
        self.sdp = sdp
        self.l_relation = l_relation
        self.r_relation = r_relation
        self.pos = pos

    def create_sdp_chain(self):
        temp = []
        temp.extend(self.l_relation)
        temp.extend(self.r_relation)
        return self.sdp, temp

    def sdp_direct(self):

        left_sdp_chain = []
        left_relation_chain = []
        left_pos_chain = []

        right_sdp_chain = []
        right_relation_chain = []
        right_pos_chain = []

        word_index = 0

        if self.l_relation:
            for l in self.l_relation:
                left_relation_chain.append(l)
                left_sdp_chain.append(self.sdp[word_index])
                left_pos_chain.append(self.pos[word_index])
                word_index += 1
        else:
            left_sdp_chain.append(self.sdp[word_index])
            left_pos_chain.append(self.pos[word_index])

        if self.r_relation:
            for r in self.r_relation:
                right_relation_chain.append(r)
                right_sdp_chain.append(self.sdp[word_index])
                right_pos_chain.append(self.pos[word_index])
                word_index += 1
        else:
            right_sdp_chain.append(self.sdp[word_index])
            right_pos_chain.append(self.pos[word_index])

        return (left_sdp_chain, left_relation_chain, left_pos_chain, right_sdp_chain, right_relation_chain,
                right_pos_chain)


def create_sdp(container):
    sdp_container = []
    relation_container = []
    sdp_max = 0
    rel_max = 0
    for chain in container:
        sdp_chain, relation_chain = chain.create_sdp_chain()
        sdp_container.append(sdp_chain)
        relation_container.append(relation_chain)
        if len(sdp_chain) > sdp_max:
            sdp_max = len(sdp_chain)
        if len(relation_chain) > rel_max:
            rel_max = len(relation_chain)
    return sdp_container, relation_container, sdp_max, rel_max


def create_sdp_direct(container):
    left_sdp = []
    right_sdp = []

    left_relation = []
    right_relation = []

    left_pos = []
    right_pos = []

    channel_len = []

    for chain in container:
        l_sdp, l_relation, l_pos, r_sdp, r_relation, r_pos = chain.sdp_direct()
        len_set = []
        left_sdp.append(l_sdp)
        len_set.append(len(l_sdp))
        left_relation.append(l_relation)
        len_set.append(len(l_relation))
        left_pos.append(l_pos)
        len_set.append(len(l_pos))

        right_sdp.append(r_sdp)
        len_set.append(len(r_sdp))
        right_relation.append(r_relation)
        len_set.append(len(r_relation))
        right_pos.append(r_pos)
        len_set.append(len(r_pos))
        channel_len.append(len_set)

    return left_sdp,  left_relation, left_pos, right_sdp, right_relation, right_pos, channel_len
