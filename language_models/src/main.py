from input_parser import InputParser
from unigram_model import UnigramModel
from bigram_model import BigramModel
from trigram_model import TrigramModel
import logging
import math

def main():
    logging.info("Starting...")

    training_parser = InputParser('/Users/skobovm/repos/csep517/language_models/data/prob1_brown_full/brown.train.txt')
    dev_parser = InputParser('/Users/skobovm/repos/csep517/language_models/data/prob1_brown_full/brown.dev.txt')
    test_parser = InputParser('/Users/skobovm/repos/csep517/language_models/data/prob1_brown_full/brown.test.txt')
    unigram = UnigramModel()

    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            unigram.add_sentence(parsed_sentence)

    # Normalize the model
    unigram.normalize_model()
    unigram.calculate_probabilities()

    bigram = BigramModel(unigram)
    trigram = TrigramModel(unigram)
    for parsed_sentence in training_parser.get_tokenized_sentences():
        if parsed_sentence:
            bigram.add_sentence(parsed_sentence)
            trigram.add_sentence(parsed_sentence)

    bigram.calculate_probabilities()
    trigram.calculate_probabilities()

    # Get dev probabilities
    unigram_probabilities = []
    bigram_probabilities = []
    trigram_probabilities = []
    total_corpus = 0

    num_bigrams_dropped = 0
    num_trigrams_dropped = 0
    total_sentences = 0

    for parsed_sentence in dev_parser.get_tokenized_sentences():
        if parsed_sentence:
            total_sentences += 1

            # Subtract 1 to account for START
            total_corpus += len(parsed_sentence) - 1

            unigram_probability = unigram.get_probability(parsed_sentence)
            bigram_probability = bigram.get_probability(parsed_sentence)
            trigram_probability = trigram.get_probability(parsed_sentence)

            if unigram_probability == -float('inf'):
                # This should NOT happen with UNKs...
                print('dropping 0 probability')
            else:
                unigram_probabilities.append(unigram_probability)

            if bigram_probability == -float('inf'):
                num_bigrams_dropped += 1
            else:
                bigram_probabilities.append(bigram_probability)

            if trigram_probability == -float('inf'):
                num_trigrams_dropped += 1
            else:
                trigram_probabilities.append(trigram_probability)

    # Calculate perplexities
    word_count = total_corpus

    # unigram
    unigram_prob_sum = sum(unigram_probabilities)
    unigram_l = unigram_prob_sum / word_count
    unigram_perplexity = math.pow(2, -unigram_l)
    print('unigram perplexity: %s' % unigram_perplexity)

    # bigram
    bigram_prob_sum = sum(bigram_probabilities)
    bigram_l = bigram_prob_sum / word_count
    bigram_perplexity = math.pow(2, -bigram_l)
    print('bigram perplexity: %s' % bigram_perplexity)

    # trigram
    trigram_prob_sum = sum(trigram_probabilities)
    trigram_l = trigram_prob_sum / word_count
    trigram_perplexity = math.pow(2, -trigram_l)
    print('bigram perplexity: %s' % trigram_perplexity)

if __name__ == '__main__':
    main()