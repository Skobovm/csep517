from input_parser import InputParser
from bigram_hmm import BigramHMM
from trigram_hmm import TrigramHMM
from lattice import Lattice, TrigramLattice
import logging
import math


def get_num_correct(sentence, pos_list, model):
    num_correct = 0
    for i in range(len(pos_list)):
        correct_tag = sentence[i + 1][1]
        if pos_list[i] == correct_tag:
            num_correct += 1
        else:
            column = model.columns[i + 1]
            tag_found = False
            for node in column.nodes:
                if node == correct_tag:
                    tag_found = True
                    break
            if not tag_found:
                print('Pruned too hard!')

    return num_correct

def main():
    logging.info("Starting...")

    training_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.train.json')
    dev_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.dev.json')
    test_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.test.json')

    # Initialize the models
    bigram = BigramHMM()
    trigram = TrigramHMM()

    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            bigram.add_sentence(parsed_sentence)
            trigram.add_sentence(parsed_sentence)

    # Models have been initialized at this point, finalize the distributions
    bigram.finalize()
    trigram.finalize()

    # PICK THE PARSER HERE
    parser = dev_parser

    # Iterate over data and try to predict
    num_correct_bigram = 0
    num_correct_trigram = 0
    total_words = 0
    for parsed_sentence in parser.get_tokenized_sentences():
        if parsed_sentence:
            # Bigram lattice
            lattice = Lattice(bigram, parsed_sentence)

            # Trigram lattice
            tri_lattice = TrigramLattice(trigram, parsed_sentence)

            # Calculate best POS using viterbi
            pos_list_bigram = lattice.get_pos()
            pos_list_trigram = tri_lattice.get_pos()

            # Determine how many were correct
            num_correct_bigram += get_num_correct(parsed_sentence, pos_list_bigram, lattice)
            num_correct_trigram += get_num_correct(parsed_sentence, pos_list_trigram, lattice)
            total_words += len(pos_list_bigram)
        else:
            print('ERROR! Couldnt parse sentence')

    print("Bigram HMM Accuracy: %s/%s - %s" % (num_correct_bigram, total_words, (num_correct_bigram / total_words)))
    print("Trigram HMM Accuracy: %s/%s - %s" % (num_correct_trigram, total_words, (num_correct_trigram / total_words)))


if __name__ == '__main__':
    main()