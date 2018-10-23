from tags import get_manual_tag, UNK, MANUAL_TAGS
import math
from collections import OrderedDict
import copy

UNK_THRESHOLD = 5

class BigramHMM:

    def __init__(self):
        # Tracks words mapped to type-counts: 'lol' -> {'!': {'count': 1000000}}
        self.emission_map = {}

        # Tracks the transition counts: 'P' -> {'^': {'count': 1000}, 'O': {'count': 1000}}
        self.transition_map = {}

        # Tracks how many words/tokens were seen
        self.total_words = 0

        # Tracks the MLE of any given tag, over whole set
        self.tag_probabilities = {}


    def add_sentence(self, sentence):
        previous_tag = None

        # iterate over every word
        for word_tuple in sentence:
            word = word_tuple[0]
            tag = word_tuple[1]

            # Want to be careful with START tag - don't count as a word
            if previous_tag:
                self.total_words += 1

                if previous_tag not in self.transition_map:
                    self.transition_map[previous_tag] = {'__TOTAL__': 0}
                self.transition_map[previous_tag]['__TOTAL__'] += 1

                # Update the count of the transition
                if tag not in self.transition_map[previous_tag]:
                    self.transition_map[previous_tag][tag] = {'count': 0}
                self.transition_map[previous_tag][tag]['count'] += 1

            # Current tag is now the previous tag
            previous_tag = tag

            # update the emission count
            if word not in self.emission_map:
                self.emission_map[word] = {'__TOTAL__': 0}

            if tag not in self.emission_map[word]:
                self.emission_map[word][tag] = {'count': 0}

            self.emission_map[word]['__TOTAL__'] += 1
            self.emission_map[word][tag]['count'] += 1

            # Tracks the total amount of tags seen
            if tag not in self.tag_probabilities:
                self.tag_probabilities[tag] = {'count': 0}
            self.tag_probabilities[tag]['count'] += 1

    # Create an emission frequency distribution for unknown words
    def _low_freq_to_unk(self):
        unk_obj = {'__TOTAL__': 0}

        # TODO: Do we need to remove the word itself, or can we just create a low-count frequency distribution?
        for word in self.emission_map:
            if self.emission_map[word]['__TOTAL__'] <= UNK_THRESHOLD:
                manual_word_type = get_manual_tag(word)

                # Only make it part of the distribution if we can't guess a type
                if not manual_word_type:
                    total = 0
                    for key in self.emission_map[word]:
                        if key == '__TOTAL__':
                            continue
                        else:
                            # MANUAL_TAGS have their own distribution, so we don't want the UNK distribution calculating them
                            # otherwise we would have invalid probability distributions
                            if key not in MANUAL_TAGS:
                                if key not in unk_obj:
                                    unk_obj[key] = {'count': 0}
                                unk_obj[key]['count'] += self.emission_map[word][key]['count']
                                total += self.emission_map[word][key]['count']

                    # Update total after we know how many words we counted
                    unk_obj['__TOTAL__'] += total

        self.emission_map[UNK] = unk_obj

    def _calculate_emission_mle(self):
        for word in self.emission_map:
            total_count = self.emission_map[word]['__TOTAL__']

            for tag in self.emission_map[word]:
                if tag == '__TOTAL__':
                    continue

                self.emission_map[word][tag]['probability'] = self.emission_map[word][tag]['count'] / total_count
                self.emission_map[word][tag]['log_probability'] = math.log(self.emission_map[word][tag]['probability'], 2)

    def _calculate_transition_mle(self):
        for tag in self.transition_map:
            total_count = self.transition_map[tag]['__TOTAL__']

            for transition in self.transition_map[tag]:
                if transition == '__TOTAL__':
                    continue

                self.transition_map[tag][transition]['probability'] = self.transition_map[tag][transition]['count'] / total_count
                self.transition_map[tag][transition]['log_probability'] = math.log(self.transition_map[tag][transition]['probability'], 2)

    def _calculate_tag_probabilities(self):
        # NOTE: START is getting a probability, but it's not real, and won't be used
        for tag in self.tag_probabilities:
            self.tag_probabilities[tag]['probability'] = self.tag_probabilities[tag]['count'] / self.total_words
            self.tag_probabilities[tag]['log_probability'] = math.log(self.tag_probabilities[tag]['probability'], 2)

    def finalize(self):
        # Create a distribution for UNK words
        self._low_freq_to_unk()

        # Calculate the emission probabilities
        self._calculate_emission_mle()

        # Calculate the transition probabilities
        self._calculate_transition_mle()

        # Calculate overall probability of any given tag
        self._calculate_tag_probabilities()

    def get_emission_probabilities(self, word):
        # We have the word
        if word in self.emission_map:
            ret_val = copy.deepcopy(self.emission_map[word])

        else:
            manual_word_type = get_manual_tag(word)
            if manual_word_type:
                ret_val = {manual_word_type: {'probability': 1.0, 'log_probability': 0}}
            else:
                ret_val = copy.deepcopy(self.emission_map[UNK])

        # Sanitize, so we don't have this as a key
        if '__TOTAL__' in ret_val:
            ret_val.pop('__TOTAL__')
        return ret_val

    def get_transition_probability(self, tag, next_tag):
        if tag not in self.transition_map:
            # This is very unlikely, but can happen
            print('tag not in transition map!')
            return float('-inf')

        next_tag_probabilities = self.transition_map[tag]

        # If it doesn't exist, we can't calculate this
        if next_tag not in next_tag_probabilities:
            return float('-inf')

        return next_tag_probabilities[next_tag]['log_probability']

    def get_tag_probability(self, tag):
        return self.tag_probabilities[tag]['log_probability']


