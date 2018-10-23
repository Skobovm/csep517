import os
import logging
import re
import tags
import json

class InputParser:

    def __init__(self, filepath):
        if not filepath:
            raise Exception('path cant be empty')

        if not os.path.isfile(filepath):
            raise Exception('cant find file')

        self.filepath = filepath

    @classmethod
    def parse_line(cls, line):
        result = [[tags.START, tags.START]]
        line = json.loads(line)
        result.extend(line)
        result.append([tags.STOP, tags.STOP])

        return result

    def get_tokenized_sentences(self):
        num_lines = sum(1 for line in open(self.filepath))
        with open(self.filepath, 'r') as data_file:
            curr_line = 1
            line = data_file.readline()
            while line:
                logging.debug('Read input: %s', line)
                print("Processing line %s/%s" % (curr_line, num_lines))
                line_components = self.parse_line(line)
                curr_line += 1
                line = data_file.readline()
                yield line_components
