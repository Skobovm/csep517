from tags import START
from bigram_hmm import BigramHMM
import math
from collections import OrderedDict
import copy

UNK_THRESHOLD = 5
LAPLACE_SMOOTHING = False

class TrigramHMM:

    def __init__(self):
        self.bigram = BigramHMM()

        # Tracks the transition counts, but for tuples
        self.transition_map = {}

        # Tracks the MLE of any given tag, over whole set
        self.tag_probabilities = {}


    def add_sentence(self, sentence):
        self.bigram.add_sentence(sentence)

        sentence_copy = copy.deepcopy(sentence)

        # Making the sentences trigram friendly after adding in bigram
        sentence_copy.insert(0, [START, START])

        # 1 = first seen, 2 = 2nd seen, etc...
        previous_tag1 = None
        previous_tag2 = None

        # iterate over every word
        for word_tuple in sentence_copy:
            tag = word_tuple[1]

            # Want to be careful with START tag - don't count as a word
            if previous_tag1 and previous_tag2:
                condition = (previous_tag1, previous_tag2)
                if condition not in self.transition_map:
                    self.transition_map[condition] = {'__TOTAL__': 0}
                self.transition_map[condition]['__TOTAL__'] += 1

                # Update the count of the transition
                if tag not in self.transition_map[condition]:
                    self.transition_map[condition][tag] = {'count': 0}
                self.transition_map[condition][tag]['count'] += 1

            # Current tag is now the previous tag
            previous_tag1, previous_tag2 = previous_tag2, tag

    def _calculate_transition_mle(self):
        for condition in self.transition_map:
            total_count = self.transition_map[condition]['__TOTAL__']

            for transition in self.transition_map[condition]:
                if transition == '__TOTAL__':
                    continue

                self.transition_map[condition][transition]['probability'] = self.transition_map[condition][transition]['count'] / total_count
                self.transition_map[condition][transition]['log_probability'] = math.log(self.transition_map[condition][transition]['probability'], 2)

    def finalize(self):
        # Finalized the emissions and bigram transitions
        self.bigram.finalize()

        # Calculate the transition probabilities
        self._calculate_transition_mle()

    # The emissions would be the same for both models
    def get_emission_probabilities(self, word):
        return self.bigram.get_emission_probabilities(word)

    def get_emission_probability(self, tag, word):
        return self.bigram.get_emission_probability(tag, word)

    def get_transition_probability(self, tag1, tag2, next_tag):
        if LAPLACE_SMOOTHING:
            k_constant = 1
            condition = (tag1, tag2)
            condition_count = 0
            tag_count = 0
            if condition in self.transition_map:
                condition_count = self.transition_map[condition]['__TOTAL__']

                next_tag_probabilities = self.transition_map[condition]

                # If it doesn't exist, we can't calculate this
                if next_tag in next_tag_probabilities:
                    tag_count = next_tag_probabilities[next_tag]['count']

            probability = (tag_count + k_constant) / (condition_count + (k_constant * len(self.transition_map)))
            return math.log(probability, 2)
        else:
            condition = (tag1, tag2)
            if condition not in self.transition_map:
                # print('tag not in transition map!')
                return float('-inf')

            next_tag_probabilities = self.transition_map[condition]

            # If it doesn't exist, we can't calculate this
            if next_tag not in next_tag_probabilities:
                return float('-inf')

            return next_tag_probabilities[next_tag]['log_probability']

    def get_bigram_transition_probability(self, tag, next_tag):
        return self.bigram.get_transition_probability(tag, next_tag)

    def get_tag_probability(self, tag):
        return self.bigram.get_tag_probability(tag)


