import unigram_model as uni
import special_chars
import math

ASSERT_SUM = True
K_SMOOTHING_HW_DEF = True

class TrigramModel:
    def __init__(self, unigram_model):
        # We will use this model for removing unknowns
        self.unigram_model = unigram_model

        # Uses tuples as keys and contains the counts
        self.trigrams = {}
        self.total_trigrams = 0

    def add_sentence(self, sentence_components):
        for i in range(1, len(sentence_components)):
            first = sentence_components[i - 2] if i > 1 else special_chars.START
            second = sentence_components[i - 1]
            third = sentence_components[i]

            # START isn't in unigram
            if first not in self.unigram_model.word_map and first != special_chars.START:
                first = special_chars.UNK

            if second not in self.unigram_model.word_map and second != special_chars.START:
                second = special_chars.UNK

            if third not in self.unigram_model.word_map:
                third = special_chars.UNK

            condition = (first, second)
            if condition not in self.trigrams:
                self.trigrams[condition] = {
                    '__TOTAL__': 0
                }

            if third not in self.trigrams[condition]:
                self.trigrams[condition][third] = {
                    'count': 0,
                    'probability': 0,
                    'numeric_probability': 0
                }
            self.trigrams[condition][third]['count'] += 1
            self.trigrams[condition]['__TOTAL__'] += 1

    def calculate_probabilities(self):
        for condition in self.trigrams.keys():
            total = self.trigrams[condition]['__TOTAL__']

            for key in self.trigrams[condition].keys():
                if key == '__TOTAL__':
                    continue
                numeric_probability = self.trigrams[condition][key]['count'] / total
                self.trigrams[condition][key]['probability'] = math.log(numeric_probability, 2)
                self.trigrams[condition][key]['numeric_probability'] = numeric_probability

    # Used for LIRP
    def get_indexed_probability(self, sentence_components, i):
        first = sentence_components[i - 2] if i > 1 else special_chars.START
        second = sentence_components[i - 1]
        third = sentence_components[i]

        # START isn't in unigram
        if first not in self.unigram_model.word_map and first != special_chars.START:
            first = special_chars.UNK

        if second not in self.unigram_model.word_map and second != special_chars.START:
            second = special_chars.UNK

        if third not in self.unigram_model.word_map:
            third = special_chars.UNK

        condition = (first, second)
        if condition not in self.trigrams:
            # Doesn't exist, so probability is 0
            return 0
        if third not in self.trigrams[condition]:
            # Doesn't exist, so probability is 0
            return 0

        return self.trigrams[condition][third]['numeric_probability']

    def get_probability(self, sentence_components, k_num=None):
        log_probability = 0

        if k_num != None:
            # K-smoothing

            # This is the size of the set of words + STOP
            V_star = len(self.unigram_model.word_map) + 1

            for i in range(1, len(sentence_components)):
                first = sentence_components[i - 2] if i > 1 else special_chars.START
                second = sentence_components[i - 1]
                third = sentence_components[i]

                # START isn't in unigram
                if first not in self.unigram_model.word_map and first != special_chars.START:
                    first = special_chars.UNK

                if second not in self.unigram_model.word_map and second != special_chars.START:
                    second = special_chars.UNK

                if third not in self.unigram_model.word_map:
                    third = special_chars.UNK

                condition = (first, second)
                no_condition_count = 0
                if condition not in self.trigrams:
                    # This is the case where denominator/conditional doesn't exist - use 1/|V*|
                    vocabulary_size = len(self.unigram_model.word_map)
                    probability = 1 / vocabulary_size
                    log_probability += math.log(probability, 2)
                else:
                    if third not in self.trigrams[condition]:
                        # Doesn't exist, so probability is 0
                        count = 0
                    else:
                        count = self.trigrams[condition][third]['count']

                    denominator_count = self.trigrams[condition]['__TOTAL__'] if condition in self.trigrams else 0
                    probability = (k_num + count) / ((V_star * k_num) + denominator_count)

                    log_probability += math.log(probability, 2)
        else:
            # Non-smoothed probability
            for i in range(1, len(sentence_components)):
                first = sentence_components[i - 2] if i > 1 else special_chars.START
                second = sentence_components[i - 1]
                third = sentence_components[i]

                # START isn't in unigram
                if first not in self.unigram_model.word_map and first != special_chars.START:
                    first = special_chars.UNK

                if second not in self.unigram_model.word_map and second != special_chars.START:
                    second = special_chars.UNK

                if third not in self.unigram_model.word_map:
                    third = special_chars.UNK

                condition = (first, second)
                if condition not in self.trigrams:
                    # Doesn't exist, so probability is 0
                    return -float('inf')
                if third not in self.trigrams[condition]:
                    # Doesn't exist, so probability is 0
                    return -float('inf')

                log_probability += self.trigrams[condition][third]['probability']

        return log_probability
