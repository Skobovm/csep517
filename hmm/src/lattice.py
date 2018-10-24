from tags import TAGS, START, STOP
from collections import OrderedDict
import math
import copy

# Linear interpolation args
L1 = .99 # Transition prob
L2 = .01 # Tag prob

# Trigram interpolation args
TL1 = .95
TL2 = .04
TL3 = .01


class Lattice:
    class LatticeNode:

        def __init__(self, tag):
            self.transition = float('-inf')
            self.emission = float('-inf')
            self.tag = tag
            self.previous_node = None
            self.max_val = None

    class LatticeColumn:

        def __init__(self, label):
            self.label = label
            self.nodes = {}

            if label == START:
                self.nodes[START] = Lattice.LatticeNode(START)
            else:
                for tag in TAGS:
                    self.nodes[tag] = Lattice.LatticeNode(tag)

        def set_emission_probabilities(self, probabilities):
            for tag in probabilities:
                node = self.nodes[tag]
                node.emission = probabilities[tag]['log_probability']

            # This will make it faster, but do we want to necessarily get rid of this?
            for tag in TAGS:
                if self.nodes[tag].emission == float('-inf'):
                    self.nodes.pop(tag)

    def __init__(self, model, sentence):
        # Keep the columns in inserted order
        self.columns = []

        # Keep a reference to the model, so it can invoke emission/transition getters
        self.model = model

        # Initialize all the columns - this will contains START and STOP
        for word_tag in sentence:
            column = Lattice.LatticeColumn(word_tag[0])
            self.columns.append(column)

    def _calculate_emissions(self):
        # Skip the START column
        for i in range(1, len(self.columns)):
            word = self.columns[i].label

            # Get emissions from the model
            emission_probabilities = self.model.get_emission_probabilities(word)

            # Set emissions for the column
            self.columns[i].set_emission_probabilities(emission_probabilities)

    # Linear interpolation happens here
    def _get_transition_probability(self, node, next_node):
        transition_probability = self.model.get_transition_probability(node.tag, next_node.tag)
        tag_probability = self.model.get_tag_probability(next_node.tag)

        if transition_probability == float('-inf'):
            return tag_probability
        else:
            # TODO: probably faster to just return the right one...
            transition_numeric_prob = math.pow(2, transition_probability)
            tag_numeric_prob = math.pow(2, tag_probability)

            numeric_prob = (transition_numeric_prob * L1) + (tag_numeric_prob * L2)
            return math.log(numeric_prob, 2)


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
                    transition_probability = self._get_transition_probability(current_node, next_node)

                    # If a max exists, this is like multiplying by the prior
                    if current_node.max_val != None:
                        transition_probability += current_node.max_val

                    # Add the emission probability
                    transition_probability += next_node.emission

                    if transition_probability > max_prob:
                        max_prob = transition_probability
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

class TrigramLattice:
    class LatticeNode:

        def __init__(self, tag):
            self.emission = float('-inf')
            self.tag = tag
            self.previous_max_nodes = {}
            self.max_vals = {}

    class LatticeColumn:

        def __init__(self, label):
            self.label = label
            self.nodes = {}

            if label == START:
                self.nodes[START] = TrigramLattice.LatticeNode(START)
            else:
                for tag in TAGS:
                    self.nodes[tag] = TrigramLattice.LatticeNode(tag)

        def set_emission_probabilities(self, probabilities):
            for tag in probabilities:
                node = self.nodes[tag]
                node.emission = probabilities[tag]['log_probability']

            # This will make it faster, but do we want to necessarily get rid of this?
            for tag in TAGS:
                if self.nodes[tag].emission == float('-inf'):
                    self.nodes.pop(tag)

    def __init__(self, model, sentence):
        sentence_copy = copy.deepcopy(sentence)

        # Make sentence friendly to trigrams
        sentence_copy.insert(0, [START, START])

        # Keep the columns in inserted order
        self.columns = []

        # Keep a reference to the model, so it can invoke emission/transition getters
        self.model = model

        # Initialize all the columns - this will contains START and STOP
        for word_tag in sentence_copy:
            column = TrigramLattice.LatticeColumn(word_tag[0])
            self.columns.append(column)

    def _calculate_emissions(self):
        # Skip the START columns
        for i in range(2, len(self.columns)):
            word = self.columns[i].label

            # Get emissions from the model
            emission_probabilities = self.model.get_emission_probabilities(word)

            # Set emissions for the column
            self.columns[i].set_emission_probabilities(emission_probabilities)

    # Linear interpolation happens here
    def _get_transition_probability(self, node1, node2, next_node):
        # This will come from the trigram model
        trigram_transition_probability = self.model.get_transition_probability(node1.tag, node2.tag, next_node.tag)

        bigram_transition_probability = self.model.get_bigram_transition_probability(node2.tag, next_node.tag)

        # This comes from the bigram model
        tag_probability = self.model.get_tag_probability(next_node.tag)

        trigram_transition_numeric_prob = math.pow(2, trigram_transition_probability) if trigram_transition_probability != float('-inf') else 0
        bigram_transition_numeric_prob = math.pow(2, bigram_transition_probability) if bigram_transition_probability != float('-inf') else 0
        tag_numeric_prob = math.pow(2, tag_probability)

        # Use the bigram lambdas
        numeric_prob = (trigram_transition_numeric_prob * TL1) + (bigram_transition_numeric_prob * TL2) + (
                    tag_numeric_prob * TL3)
        return math.log(numeric_prob, 2)


    def _calculate_transitions(self):
        for i in range(2, len(self.columns)):
            column1 = self.columns[i - 2]
            column2 = self.columns[i - 1]
            next_column = self.columns[i]

            for next_tag in next_column.nodes:
                next_node = next_column.nodes[next_tag]

                # Loop over all "previous" nodes, to get the max for the "current" node, next_node
                for tag2 in column2.nodes:
                    max_prob = float('-inf')
                    max_prob_node = None

                    node2 = column2.nodes[tag2]

                    for tag1 in column1.nodes:
                        node1 = column1.nodes[tag1]
                        transition_probability = self._get_transition_probability(node1, node2, next_node)

                        # If a max exists, this is like multiplying by the prior
                        if node2.max_vals:
                            transition_probability += node2.max_vals[node1.tag]

                        # Add the emission probability
                        transition_probability += next_node.emission

                        if transition_probability > max_prob:
                            max_prob = transition_probability
                            max_prob_node = node2

                    # Set the transition maxima for the current node
                    next_node.max_vals[tag2] = max_prob
                    next_node.previous_max_nodes[tag2] = max_prob_node

    def get_pos(self):
        # Get the emissions for all the nodes
        self._calculate_emissions()

        # Calculate the transitions for all remaining nodes
        self._calculate_transitions()

        # Get the transitions (saved as linked list)
        result = []
        path_node = self.columns[-1].nodes[STOP]
        while path_node and path_node.tag != START:
            result.insert(0, path_node.tag)

            max_val = float('-inf')
            max_tag = None
            for tag in path_node.max_vals:
                val = path_node.max_vals[tag]
                if val > max_val:
                    max_val = val
                    max_tag = tag
            path_node = path_node.previous_max_nodes[max_tag]

        return result[:-1]

