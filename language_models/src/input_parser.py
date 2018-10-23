import os
import logging
import re
import special_chars

numeric_indicators = ['d', 'st', 'nd', 'rd', 'th', 's']

class InputParser:

    def __init__(self, filepath):
        if not filepath:
            raise Exception('path cant be empty')

        if not os.path.isfile(filepath):
            raise Exception('cant find file')

        self.filepath = filepath

    @classmethod
    def is_date_partial(cls, word):
        if word.lower() in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']:
            return True
        if word.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            return True
        return False

    @classmethod
    def sanitize_word(cls, word, first_word = False):
        if not word:
            return None

        potential_numeric = False
        for indicator in numeric_indicators:
            if word.lower().endswith(indicator):
                numeric = word.replace(indicator, '')
                if numeric.isnumeric():
                    potential_numeric = True
                    break

        if potential_numeric:
            return special_chars.NUM

        word = word.replace('\n', '')
        word = word.replace(' ', '')

        if cls.is_date_partial(word):
            return special_chars.DATE_PARTIAL

        if not first_word and word[0].upper() == word[0]:
            if word.endswith("'s") or word.endswith("'"):
                return special_chars.PROPER_POSSESSIVE
            else:
                return special_chars.PROPER
        return word.lower()

    @classmethod
    def parse_line(cls, line):
        words = line.split(' ')
        clean_words = []
        first_word = True

        clean_words.append(special_chars.START)
        for word in words:
            clean_word = re.sub('[^0-9a-zA-Z ]+', '', word)
            if not clean_word.isalnum():
                if word in [',', ';', '--', '(', ')']:
                    # Add a delimited
                    clean_words.append(special_chars.DELIM)
            elif clean_word.isnumeric():
                clean_word = special_chars.NUM
                clean_words.append(clean_word)

                # Capitalization rules/possessive calculation relies on first word
                first_word = False
            else:
                # If it's a word that's shortened such as "Feb." or "who's" we want the special char
                # but hyphenated words should be separate
                hyphenated_components = word.split('-')
                for hyp_word in hyphenated_components:
                    clean_word = cls.sanitize_word(hyp_word, first_word)
                    if clean_word:
                        clean_words.append(clean_word)

                # Capitalization rules/possessive calculation relies on first word
                first_word = False


        # Add stop char
        clean_words.append(special_chars.STOP)

        return clean_words

    def get_tokenized_sentences(self):
        num_lines = sum(1 for line in open(self.filepath))
        with open(self.filepath, 'r') as data_file:
            curr_line = 1
            line = data_file.readline()
            while line:
                logging.debug('Read input: %s', line)
                #print("Processing line %s/%s" % (curr_line, num_lines))
                line_components = self.parse_line(line)
                curr_line += 1
                line = data_file.readline()
                yield line_components
