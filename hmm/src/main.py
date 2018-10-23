from input_parser import InputParser
from bigram_hmm import BigramHMM
from lattice import Lattice
import logging
import math


def get_num_correct(sentence, pos_list):
    num_correct = 0
    for i in range(len(pos_list)):
        if pos_list[i] == sentence[i + 1][1]:
            num_correct += 1

    return num_correct

def main():
    logging.info("Starting...")

    training_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.train.json')
    dev_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.dev.json')
    test_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.test.json')

    # Initialize the models
    bigram = BigramHMM()

    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            bigram.add_sentence(parsed_sentence)

    # Models have been initialized at this point, finalize the distributions
    bigram.finalize()

    # PICK THE PARSER HERE
    parser = dev_parser

    # Iterate over data and try to predict
    num_correct = 0
    total_words = 0
    for parsed_sentence in parser.get_tokenized_sentences():
        if parsed_sentence:
            lattice = Lattice(bigram, parsed_sentence)
            pos_list = lattice.get_pos()
            num_correct += get_num_correct(parsed_sentence, pos_list)
            total_words += len(pos_list)
        else:
            print('ERROR! Couldnt parse sentence')

    print("Accuracy: %s/%s - %s" % (num_correct, total_words, (num_correct/total_words)))


if __name__ == '__main__':
    main()