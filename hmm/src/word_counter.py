
UNK_THRESHOLD = 5

class WordCounter:

    def __init__(self):
        self.word_counts = {}
        self.high_freq = {}

    def add_word(self, word):
        if word not in self.word_counts:
            self.word_counts[word] = 0
        self.word_counts[word] += 1

    def finalize(self):
        for word in self.word_counts:
            if self.word_counts[word] >= UNK_THRESHOLD:
                self.high_freq[word] = self.word_counts[word]
        # TODO: remove word_counts table

    def _get_class(self, word):
        pass

    def get_class_or_word(self, word):
        if word in self.high_freq:
            return word
        else:
            return self._get_class(word)
