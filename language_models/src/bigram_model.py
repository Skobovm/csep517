import unigram_model as uni
import special_chars
import math

ASSERT_SUM = True

class BigramModel:
    def __init__(self, unigram_model):
        # We will use this model for removing unknowns
        self.unigram_model = unigram_model

        # Uses tuples as keys and contains the counts
        self.bigrams = {}
        self.total_bigrams = 0

    def add_sentence(self, sentence_components):
        for i in range(1, len(sentence_components)):
            first = sentence_components[i - 1]
            second = sentence_components[i]

            # START isn't in unigram
            if first not in self.unigram_model.word_map and first != special_chars.START:
                first = special_chars.UNK

            if second not in self.unigram_model.word_map:
                second = special_chars.UNK

            condition = first
            if condition not in self.bigrams:
                self.bigrams[condition] = {
                    '__TOTAL__': 0
                }

            if second not in self.bigrams[condition]:
                self.bigrams[condition][second] = {
                    'count': 0,
                    'probability': 0,
                    'numeric_probability': 0
                }
            self.bigrams[condition][second]['count'] += 1
            self.bigrams[condition]['__TOTAL__'] += 1

    def calculate_probabilities(self):
        for condition in self.bigrams.keys():
            total = self.bigrams[condition]['__TOTAL__']

            for key in self.bigrams[condition].keys():
                if key == '__TOTAL__':
                    continue
                numeric_probability = self.bigrams[condition][key]['count'] / total
                self.bigrams[condition][key]['probability'] = math.log(numeric_probability, 2)
                self.bigrams[condition][key]['numeric_probability'] = numeric_probability

    def get_probability(self, sentence_components):
        log_probability = 0
        for i in range(1, len(sentence_components)):
            first = sentence_components[i - 1]
            second = sentence_components[i]

            # START isn't in unigram
            if first not in self.unigram_model.word_map and first != special_chars.START:
                first = special_chars.UNK

            if second not in self.unigram_model.word_map:
                second = special_chars.UNK

            condition = first
            if condition not in self.bigrams:
                # Doesn't exist, so probability is 0
                return -float('inf')
            if second not in self.bigrams[condition]:
                # Doesn't exist, so probability is 0
                return -float('inf')

            log_probability += self.bigrams[condition][second]['probability']

        return log_probability