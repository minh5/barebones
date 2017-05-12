import random
from math import exp

class NeuralNetwork:


    def __init__(self, x_train, y_train, n_hidden_layers=1):
        self.x_train = x_train
        self.y_train = y_train
        self.n_hidden_layers = n_hidden_layers

    def initialize_network(self):
        network = dict()
        hidden_layer = [{'weights': [
             random.random() for i in range(self.x_train.shape[1])
             ]}
             for n in range(self.n_hidden_layers)
        ] + [{'bias': random.random()}]
        network['hidden'] = hidden_layer
        output_layer = [{'weights': [
             random.random() for i in range(self.n_hidden_layers)
             ]}
             for n in range(len(set(self.y_train)))
        ] + [{'bias': random.random()}]
        network['output'] = output_layer
        return network

    @staticmethod
    def activation_function(weights, bias, input):
        activation = bias
        for w, i in zip(weights, input):
            activation += w * i
        return activation

    @staticmethod
    def neuron_transfer(activation):
        return 1/(1 + exp(-activation))

    def forward_propogation(self, input):
        network = self.initialize_network()
        for key, value in network.items():
            new_inputs = []
            for neuron in value:
                if 'bias' in neuron.keys():
                    continue
                else:
                    weights = neuron['weights']
                    bias = [i for i in network['output'] if 'bias' in i.keys()][0]['bias']
                    activation = self.activation_function(weights, bias, input)
                    neuron['output'] = self.neuron_transfer(activation)
                    new_inputs.append(neuron['output'])
        return new_inputs
