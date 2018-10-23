from tags import TAGS, START, STOP
from collections import OrderedDict
import math

# Linear interpolation args
L1 = .99 # Transition prob
L2 = .01 # Tag prob


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
            self.nodes[START] = LatticeNode(START)
        else:
            for tag in TAGS:
                self.nodes[tag] = LatticeNode(tag)

    def set_emission_probabilities(self, probabilities):
        for tag in probabilities:
            node = self.nodes[tag]
            node.emission = probabilities[tag]['log_probability']

        # This will make it faster, but do we want to necessarily get rid of this?
        for tag in TAGS:
            if self.nodes[tag].emission == float('-inf'):
                self.nodes.pop(tag)

class Lattice:

    def __init__(self, model, sentence):
        # Keep the columns in inserted order
        self.columns = []

        # Keep a reference to the model, so it can invoke emission/transition getters
        self.model = model

        # Initialize all the columns - this will contains START and STOP
        for word_tag in sentence:
            column = LatticeColumn(word_tag[0])
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


