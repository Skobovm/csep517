import special_chars
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

DRAW_DISTRIBUTION = False
ASSERT_SUM = True
UNK_CUTOFF = 10

class UnigramModel:

    def __init__(self):
        self.word_map = {}
        self.unique_words = 0
        self.probabilities = {}
        self.total_words = 0
        self.numeric_probabilities = {}

    def add_sentence(self, sentence_components):
        for component in sentence_components:
            if component == special_chars.START:
                # Don't want this as part of unigram model (per slides)
                continue
            if component in self.word_map:
                self.word_map[component] += 1
            else:
                self.unique_words += 1
                self.word_map[component] = 1
            self.total_words += 1

    def get_unk_keys(self):
        keys_to_unk = []
        for key in self.word_map:
            times_seen = self.word_map[key]

            if times_seen < UNK_CUTOFF:
                keys_to_unk.append(key)
        return keys_to_unk

    def normalize_model(self):
        counts = {}
        total = []
        num_bins = 0

        # Doing iteration twice, but whatever
        keys_to_unk = self.get_unk_keys()

        # Prior to UNK normalization
        for key in self.word_map:
            times_seen = self.word_map[key]

            if times_seen not in counts:
                counts[times_seen] = 0
            counts[times_seen] += 1
            total.append(times_seen)

        if DRAW_DISTRIBUTION:
            num_bins = len(list(counts.keys()))

            n, bins, patches = plt.hist(total, num_bins, facecolor='blue', alpha=0.5)
            plt.show()

        # Set up unk
        self.word_map[special_chars.UNK] = 0
        for key in keys_to_unk:
            times_seen = self.word_map[key]
            self.word_map.pop(key)
            self.word_map[special_chars.UNK] += times_seen

        if DRAW_DISTRIBUTION:
            total = []
            for key in self.word_map:
                times_seen = self.word_map[key]
                total.append(times_seen)
            n, bins, patches = plt.hist(total, num_bins, facecolor='blue', alpha=0.5)
            plt.show()

    def calculate_probabilities(self):
        for word in self.word_map:
            numeric_probability = self.word_map[word] / self.total_words
            self.probabilities[word] = math.log(numeric_probability, 2)

            if ASSERT_SUM:
                self.numeric_probabilities[word] = numeric_probability

        if ASSERT_SUM:
            total_probability = sum(self.numeric_probabilities[key] for key in list(self.numeric_probabilities.keys()))
            print('Total probability sum: %s' % total_probability)

    def get_probability(self, sentence_components):
        log_probability = 0

        # Skip start char
        for i in range(1, len(sentence_components)):
            word = sentence_components[i]

            if word not in self.word_map:
                word = special_chars.UNK

            if word not in self.probabilities:
                # Doesn't exist, so probability is 0
                return -float('inf')

            log_probability += self.probabilities[word]

        return log_probability
