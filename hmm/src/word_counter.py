import re

UNK_THRESHOLD = 5

# Word classes
UNK = '__UNK__'
UNK_POS = '__UNK_POS__' # UNK, but possessive
EMOJI = '__EMOJI__'
EMOJI_ALPHA = '__EMOJI_ALPHA__'
EMOJI_NUM = '__EMOJI_NUM__'
HYPHENATED = '__HYPHENATED__'
SLASHED = '__SLASHED__'
HASHTAG = '__HASHTAG__'
MENTION = '__MENTION__'
URL = '__URL__'
NUMBER = '__NUMBER__'
SYMBOL_NUMBER = '__SYM_NUMBER__'
SYMBOL_ALPHA = '__SYM_ALPHA__'
SYMBOL_ALPHANUM = '__SYM_ALPHANUM__'
SYMBOLS = '__SYMBOLS__'
BAD_SPLIT = '__BADSPLIT__'
LONG_WORD = '__LONGWORD__'
UPPERCASE = '__UPPERCASE__'
LOWERCASE = '__LOWERCASE__'
PROPER = '__PROPER__'
PROPER_POS = '__PROPER_POS__'
MULTI_CAP_PROPER = '__MULTI_PROPER__'
POSSESSIVE = '__POSSESSIVE__'
ALPHANUM = '__ALPHANUM__'
PERCENT = '__PERCENT__'
COLON_NUM = '__COLON_NUM__'
DASH_NUM = '__DASH_NUM__'
DASH_ALPHA = '__DASH_ALPHA__'
SLASH_NUM = '__SLASH_NUM__'
SLASH_ALPHA = '__SLASH_ALPHA__'
DOT_NUM = '__DOT_NUM__'
MONEY = '__MONEY__'
TICKER = '__TICKER__'
HAS_UNDERSCORE = '__UNDERSCORE__'



def get_manual_tag(word):
    if word.startswith('#'):
        return HASHTAG # hashtag
    elif word.startswith('@'):
        return MENTION  # mention
    elif word.startswith('http:') or word.startswith('https:') or word.startswith('www.') or '.co' in word:
        return URL  # url

    # maybe remove more chars found in words?
    clean_word = re.sub('[^0-9a-zA-Z ]+', '', word)
    if not clean_word.isalnum():
        return SYMBOLS # punctuation/unknown
    if not clean_word.isalpha():
        return NUMBER  # numeric

    # Don't have a solid rule for this one
    return None

class WordCounter:

    def __init__(self):
        self.word_counts = {}
        self.high_freq = {}
        self.unclassified = {}

    def add_word(self, word):
        if word not in self.word_counts:
            self.word_counts[word] = 0
        self.word_counts[word] += 1

    def finalize(self):
        for word in self.word_counts:
            if self.word_counts[word] >= UNK_THRESHOLD:
                self.high_freq[word] = self.word_counts[word]

        for word in self.high_freq:
            self.word_counts.pop(word)

        # Try to classify
        for word in self.word_counts:
            word_class = self._get_class(word, -1)

            if word_class == UNK:
                self.unclassified[word] = self.word_counts[word]

    @classmethod
    def _remove_non_ascii(cls, s):
        return "".join(i for i in s if ord(i) < 128)

    @classmethod
    def _get_possessive(cls, word):
        # People are idiots and use the wrong apostrophe char
        lower_word = word.lower()
        if len(word) > 2 and \
                (lower_word.endswith("'s") or lower_word.endswith("s'") or lower_word.endswith("’s") or lower_word.endswith("s’")):
            # Possessive
            if word[0].isupper() and word[1:].islower():
                # Proper
                return PROPER_POS
            return POSSESSIVE
        return None

    @classmethod
    def _get_split_type(cls, word):
        clean_word = word.replace('.', '').replace(',', '')
        if word[-1] == '%' and clean_word[:-1].isnumeric():
            return PERCENT

        if clean_word.isnumeric():
            return NUMBER

        all_num = True
        all_alpha = True

        if ':' in word:
            colon_split = word.split(':')
            if len(colon_split) > 1:
                for part in colon_split:
                    if part and not part.isnumeric():
                        all_num = False
                return COLON_NUM if all_num else None

        if '/' in word:
            slash_split = word.split('/')
            if len(slash_split) > 1:
                for part in slash_split:
                    if part and not part.isnumeric():
                        all_num = False
                    if part and not part.isalpha():
                        all_alpha = False
                if all_num:
                    return SLASH_NUM
                if all_alpha:
                    return SLASH_ALPHA

        if '-' in word:
            dash_split = word.split('-')
            if len(dash_split) > 1:
                for part in dash_split:
                    if part and not part.isnumeric():
                        all_num = False
                    if part and not part.isalpha():
                        all_alpha = False
                if all_num:
                    return DASH_NUM
                if all_alpha:
                    return DASH_ALPHA
                return HYPHENATED

        if '.' in word:
            dot_split = word.split('.')
            if len(dot_split) > 1:
                for part in dot_split:
                    if part and not part.isnumeric():
                        all_num = False
                return DOT_NUM if all_num else None

        return None

    @classmethod
    def _is_money(cls, word):
        if word.startswith('$') or word.endswith('$') or word.startswith('£') or word.endswith('£'):
            clean = word.replace('$', '').replace('£','').replace('.', '').replace(',', '')

            if len(clean) > 1 and len(clean) < 6 and clean.isalpha():
                return TICKER
            if clean and clean.isalnum():
                return MONEY
        return None

    @classmethod
    def _is_bad_split(cls, word):
        if '(' in word or ')' in word:
            clean_word = word.replace('(', '').replace(')', '')
            if clean_word.isalpha() or clean_word.isnumeric():
                return BAD_SPLIT
        return None

    @classmethod
    def _is_multi_cap_proper(cls, word):
        if word[0].isupper():
            multi_upper = False
            for i in range(1, len(word)):
                if not word[i].isalpha():
                    multi_upper = False
                    break
                elif word[i].isupper():
                    multi_upper = True

            if multi_upper:
                return MULTI_CAP_PROPER
        return None

    def _get_class(self, word, index):
        if word.startswith('#'):
            return HASHTAG  # hashtag
        elif word.startswith('@'):
            return MENTION  # mention
        elif word.startswith('http:') or word.startswith('https:') or word.startswith('www.') or '.co' in word:
            return URL  # url

        if word.isalpha():
            # No special chars
            if word.isupper():
                lower_word = word.lower()
                if lower_word in self.high_freq:
                    # TODO: Not sure if this should be returned... maybe a threshold?
                    return lower_word
                return UPPERCASE
            elif word.islower():
                return LOWERCASE
            elif word[0].isupper() and word[1:].islower():
                return PROPER
        elif word.isnumeric():
            # All numbers
            return NUMBER
        elif word.isalnum():
            # Letters and numbers
            return ALPHANUM

        # Special chars exist
        possessive = self._get_possessive(word)
        if possessive:
            return possessive

        # Numeric types
        split_type = self._get_split_type(word)
        if split_type:
            return split_type

        # Try removing unicode chars
        non_ascii_removed = self._remove_non_ascii(word)
        if not non_ascii_removed:
            # All were unicode chars - assume emoji
            return EMOJI

        if non_ascii_removed.isalpha() and len(non_ascii_removed) < len(word):
            return EMOJI_ALPHA

        if non_ascii_removed.isnumeric() and len(non_ascii_removed) < len(word):
            return EMOJI_NUM

        money = self._is_money(word)
        if money:
            return MONEY

        # Some words split by parents in wrong places
        bad_split = self._is_bad_split(non_ascii_removed)
        if bad_split:
            return bad_split

        alpha_num_word = re.sub('[^0-9a-zA-Z ]+', '', word)
        if not alpha_num_word:
            return SYMBOLS

        # multiple cap proper (ex: TheVerge)
        multi_proper = self._is_multi_cap_proper(alpha_num_word)
        if multi_proper:
            return multi_proper

        if '_' in word:
            return HAS_UNDERSCORE

        if alpha_num_word.isnumeric():
            return SYMBOL_NUMBER

        if alpha_num_word.isalpha():
            return SYMBOL_ALPHA

        # if alpha_num_word.isalnum():
        #     return SYMBOL_ALPHANUM

        return UNK

    def _get_class_or_word(self, word, index):
        if word in self.high_freq:
            return word
        else:
            return self._get_class(word, index)

    def classify_sentence(self, sentence):
        # Skip the START and STOP words
        for i in range(1, len(sentence) - 1):
            word_tag = sentence[i]
            new_word = self._get_class_or_word(word_tag[0], i - 1)
            word_tag[0] = new_word