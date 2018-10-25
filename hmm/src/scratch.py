from tags import TAGS, START, STOP
from collections import OrderedDict
import math
import copy

# Linear interpolation args
L1 = .90 # Transition prob
L2 = .1 # Tag prob

class Lattice:
    class LatticeNode:

        def __init__(self, tag):
            self.transition = float('-inf')
            self.emission = float('-inf')
            self.word_probability = float('-inf')
            self.tag = tag
            self.previous_node = None
            self.max_val = None

    class LatticeColumn:

        def __init__(self, label):
            self.label = label
            self.nodes = {}

            if label == START:
                self.nodes[START] = Lattice.LatticeNode(START)
                self.nodes[START].emission = 0
            elif label == STOP:
                self.nodes[STOP] = Lattice.LatticeNode(STOP)
                self.nodes[STOP].emission = 0
            else:
                for tag in TAGS:
                    if tag not in [START, STOP]:
                        self.nodes[tag] = Lattice.LatticeNode(tag)

        def set_emission_probabilities(self, model):
            tags_to_remove = []
            for tag in self.nodes:
                emission = model.get_emission_probability(tag, self.label)

                if emission == float('-inf'):
                    tags_to_remove.append(tag)
                self.nodes[tag].emission = emission

            # Remove tags with 0 probability, if we have enough, otherwise leave alone
            # if len(self.nodes) > len(tags_to_remove):
            #     for tag in tags_to_remove:
            #         self.nodes.pop(tag)
            # else:
            #     print('Not enough nodes - not pruning')

        def set_tag_probabilities(self, probabilities):
            # Sanitize, so we don't have this as a key
            total = 0
            if '__TOTAL__' in probabilities:
                total = probabilities['__TOTAL__']
                probabilities.pop('__TOTAL__')

            # Prune if we have at least this many examples of the word
            if total >= 5:
                for tag in probabilities:
                    node = self.nodes[tag]
                    node.word_probability = probabilities[tag]['log_probability']

                # This will make it faster, but do we want to necessarily get rid of this?
                for tag in TAGS:
                    if tag in self.nodes and self.nodes[tag].word_probability == float('-inf'):
                        self.nodes.pop(tag)

    def __init__(self, model, sentence):
        # Keep the columns in inserted order
        self.columns = []

        # Keep a reference to the model, so it can invoke emission/transition getters
        self.model = model

        # Use to memoize pi function values
        self.pi_memo = {(0, START): 0}

        # Initialize all the columns - this will contains START and STOP
        for word_tag in sentence:
            column = Lattice.LatticeColumn(word_tag[0])
            self.columns.append(column)

    def _calculate_emissions(self):
        # Skip the START column
        for i in range(1, len(self.columns) - 1):
            word = self.columns[i].label

            # Get emissions from the model
            tag_probabilities_for_word = self.model.get_tag_possibilities(word)

            if tag_probabilities_for_word:
                # Prune the nodes in the column
                self.columns[i].set_tag_probabilities(tag_probabilities_for_word)

            # Get all the emission values for leftover nodes
            self.columns[i].set_emission_probabilities(self.model)

    # Linear interpolation happens here
    def _get_transition_probability(self, node, next_node):
        transition_probability = self.model.get_transition_probability(node.tag, next_node.tag)
        tag_probability = self.model.get_tag_probability(next_node.tag)

        transition_numeric_prob = math.pow(2, transition_probability) if transition_probability != float('-inf') else 0
        tag_numeric_prob = math.pow(2, tag_probability) if tag_probability != float('-inf') else 0

        numeric_prob = (transition_numeric_prob * L1) + (tag_numeric_prob * L2)

        if numeric_prob == 0:
            return float('-inf')
        return math.log(numeric_prob, 2)

    def _get_pi(self, i, tag):
        return self.pi_memo[(i, tag)]

    def _set_pi(self, i, tag, probability):
        key = (i, tag)
        if key not in self.pi_memo:
            self.pi_memo[key] = probability

            # Return True if we set it
            return True

        # New probability is greater than old
        if probability > self.pi_memo[key]:
            self.pi_memo[key] = probability
            return True

        return False

    def _calculate_transitions(self):
        for i in range(1, len(self.columns)):
            column = self.columns[i - 1]
            next_column = self.columns[i]

            for next_tag in next_column.nodes:
                next_node = next_column.nodes[next_tag]
                max_prob = float('-inf')
                max_prob_node = None

                # Loop over all "previous" nodes, to get the max for the "current" node, next_node
                for tag in column.nodes:
                    current_node = column.nodes[tag]

                    # q(y_i | y_i-1)
                    transition_probability = self._get_transition_probability(current_node, next_node)

                    # e(x_i | y_i) - this is pre-calculated
                    emission = next_node.emission

                    # pi(i - 1, y_i-1)
                    prev_pi = self._get_pi(i - 1, tag)

                    # Get new pi value
                    log_prob = transition_probability + emission + prev_pi

                    is_new_max = self._set_pi(i, next_tag, log_prob)
                    if is_new_max:
                        max_prob = log_prob
                        max_prob_node = current_node

                # Set the transition maxima for the current node
                next_node.max_val = max_prob
                next_node.previous_node = max_prob_node

    def get_pos(self):
        # Get the emissions for all the nodes
        self._calculate_emissions()

        # Calculate the transitions for all remaining nodes
        self._calculate_transitions()

        # Get the transitions (saved as linked list)
        result = []
        path_node = self.columns[-1].nodes[STOP]
        while path_node:
            result.insert(0, path_node.tag)
            path_node = path_node.previous_node

        return result[1:-1]
