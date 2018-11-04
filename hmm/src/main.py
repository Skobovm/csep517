from input_parser import InputParser
from bigram_hmm import BigramHMM
from trigram_hmm import TrigramHMM
from lattice import Lattice, TrigramLattice
from word_counter import WordCounter
import logging
import math
import copy

def get_num_correct(sentence, pos_list, model=None, original_sentence=None, counter=None):
    num_correct = 0
    for i in range(len(pos_list)):
        correct_tag = sentence[i + 1][1]
        guessed_tag = pos_list[i]
        if guessed_tag == correct_tag:
            num_correct += 1
        else:
            # if model:
            #     column = model.columns[i + 1]
            #     tag_found = False
            #     for node in column.nodes:
            #         if node == correct_tag:
            #             tag_found = True
            #             break
            #     if not tag_found:
            #         print('Pruned too hard!')
            # if model and original_sentence and counter:
            #     original_word = original_sentence[i + 1][0]
            #     converted_word = sentence[i + 1][0]
            #     word_hf = original_word in counter.high_freq
            #     word_seen = original_word in counter.word_counts or word_hf
            #
            #     # The plus 2 is trigram specific!
            #     correct_emission = model.columns[i + 2].nodes[correct_tag].emission if correct_tag in model.columns[i + 2].nodes else float('-inf')
            #     incorrect_emission = model.columns[i + 2].nodes[guessed_tag].emission if guessed_tag in model.columns[i + 2].nodes else float('-inf')
            #
            #     print('Word: %s -> %s' % (original_word, converted_word))
            #     print('Correct Emission: %s -> %s' % (correct_tag, correct_emission))
            #     print('Incorrect Emission: %s -> %s' % (guessed_tag, incorrect_emission))
            pass

    return num_correct

def main():
    logging.info("Starting...")

    training_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.train.json')
    dev_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.dev.json')
    test_parser = InputParser('/Users/skobovm/repos/csep517/hmm/data/twt.test.json')

    # First, count the words!
    counter = WordCounter()
    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            for i in range(1, len(parsed_sentence) - 1):
                counter.add_word(parsed_sentence[i][0])

    # Finalize counter and separate high frequency from low frequency
    counter.finalize()

    # Initialize the models
    bigram = BigramHMM()
    trigram = TrigramHMM()

    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            # Convert the low frequency words to classes
            counter.classify_sentence(parsed_sentence)

            bigram.add_sentence(parsed_sentence)
            trigram.add_sentence(parsed_sentence)

    # Models have been initialized at this point, finalize the distributions
    #bigram.finalize()
    trigram.finalize()

    # PICK THE PARSER HERE
    parser = dev_parser

    # Iterate over data and try to predict
    num_correct_bigram = 0
    num_correct_trigram = 0
    total_words = 0
    for parsed_sentence in parser.get_tokenized_sentences():
        if parsed_sentence:
            original_sentence = copy.deepcopy(parsed_sentence)

            # Convert the low frequency words to classes
            counter.classify_sentence(parsed_sentence)

            # Bigram lattice
            #lattice = Lattice(bigram, parsed_sentence)

            # Trigram lattice
            tri_lattice = TrigramLattice(trigram, parsed_sentence)

            # Calculate best POS using viterbi
            #pos_list_bigram = lattice.get_pos()
            pos_list_trigram = tri_lattice.get_pos()

            # Determine how many were correct
            #num_correct_bigram += get_num_correct(parsed_sentence, pos_list_bigram, lattice)
            num_correct_trigram += get_num_correct(parsed_sentence, pos_list_trigram, tri_lattice, original_sentence, counter)

            # Remove the START and STOP chars
            total_words += (len(parsed_sentence) - 2)

            print("Accuracy: %s" % (num_correct_trigram/total_words))
        else:
            print('ERROR! Couldnt parse sentence')

    print("Bigram HMM Accuracy: %s/%s - %s" % (num_correct_bigram, total_words, (num_correct_bigram / total_words)))
    print("Trigram HMM Accuracy: %s/%s - %s" % (num_correct_trigram, total_words, (num_correct_trigram / total_words)))


if __name__ == '__main__':
    main()