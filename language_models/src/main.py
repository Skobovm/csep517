from input_parser import InputParser
from unigram_model import UnigramModel
from bigram_model import BigramModel
from trigram_model import TrigramModel
from linear_interpolation import LinearInterpolator
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

    # Set up the appropriate input parser
    parser = test_parser
    k_vals = [.0026] # [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    for k in k_vals:
        # Get dev probabilities
        unigram_probabilities = []
        bigram_probabilities = []
        trigram_probabilities = []
        total_corpus = 0

        num_bigrams_dropped = 0
        num_trigrams_dropped = 0
        total_sentences = 0

        for parsed_sentence in parser.get_tokenized_sentences():
            if parsed_sentence:
                total_sentences += 1

                # Subtract 1 to account for START
                total_corpus += len(parsed_sentence) - 1

                unigram_probability = unigram.get_probability(parsed_sentence)
                bigram_probability = bigram.get_probability(parsed_sentence, k_num=k)
                trigram_probability = trigram.get_probability(parsed_sentence, k_num=k)

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

        print('K: %s' % k)
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
        print('trigram perplexity: %s' % trigram_perplexity)

    # lambdas = [
    #     (1/3, 1/3, 1/3), # Even
    #     (.7, .15, .15),  # Trigram-heavy
    #     (.15, .7, .15), # Bigram-heavy
    #     (.15, .15, .7), # Unigram-heavy
    #     (.6, .3, .1) # tri > bi > uni
    # ]
    lambdas = [
        (.1, .55, .35),  # Trigram-heavy
        (.05, .6, .35),  # Unigram-heavy
        (.1, .6, .3)  # tri > bi > uni
    ]
    lirp = LinearInterpolator(unigram, bigram, trigram)
    for lambda_set in lambdas:
        probabilities = []
        total_corpus = 0

        for parsed_sentence in parser.get_tokenized_sentences():
            if parsed_sentence:
                # Subtract 1 to account for START
                total_corpus += len(parsed_sentence) - 1

                probabilities.append(lirp.get_probability(parsed_sentence, l1=lambda_set[0], l2=lambda_set[1], l3=lambda_set[2]))

        print('l1: %s, l2: %s, l3: %s' % lambda_set)
        # unigram
        unigram_prob_sum = sum(probabilities)
        unigram_l = unigram_prob_sum / total_corpus
        unigram_perplexity = math.pow(2, -unigram_l)
        print('perplexity: %s' % unigram_perplexity)

if __name__ == '__main__':
    main()