import math

class LinearInterpolator:

    def __init__(self, unigram, bigram, trigram, ):
        self.unigram = unigram
        self.bigram = bigram
        self.trigram = trigram

    def get_probability(self, sentence_components, l1, l2, l3):
        log_prob = 0
        for i in range(1, len(sentence_components)):
            u_prob = self.unigram.get_indexed_probability(sentence_components, i)
            b_prob = self.bigram.get_indexed_probability(sentence_components, i)
            t_prob = self.trigram.get_indexed_probability(sentence_components, i)

            # if t_prob == 0:
            #     l2 += l1 / 2
            #     l3 += l1 / 2
            #     l1 = 0
            #
            # if b_prob == 0:
            #     l3 = 1
            #     l1 = 0
            #     l2 = 0

            total_prob = (l1 * t_prob) + (l2 * b_prob) + (l3 * u_prob)
            log_prob += math.log(total_prob, 2)

        return log_prob