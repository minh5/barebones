import random
from math import exp


class NeuralNetwork:

    def __init__(
        self, x_train, y_train, expected_values, n_hidden_layers=1,
        learning_rate=0.1, n_epochs=5
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.n_hidden_layers = n_hidden_layers
        self.expected_values = expected_values
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

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
        self.network = network
        return network

    @staticmethod
    def activation_function(weights, bias, values):
        activation = bias
        for w, i in zip(weights, values):
            activation += w * i
        return activation

    @staticmethod
    def neuron_transfer(activation):
        return 1 / (1 + exp(-activation))

    @staticmethod
    def get_derivative(output):
        return output * (1 - output)

    def forward_propogation(self, values):
        network = self.network
        for key, value in network.items():
            weights = []
            for neuron in value:
                if 'bias' in neuron.keys():
                    continue
                else:
                    weights.append(neuron['weights'])
                    bias = [i for i in network['output']
                            if 'bias' in i.keys()][0]['bias']
                    activation = self.activation_function(
                        weights, bias, values)
                    neuron['output'] = self.neuron_transfer(activation)
        return network

    def calculate_output_delta(self, row):
        network = self.forward_propogation(row)
        output_layer = network['output']
        layers = []
        for layer in output_layer:
            if 'bias' in layer.keys():
                output_layer = [layer]
            else:
                layers.append(layer)
        for e, l in zip(row, layers):
            l['error'] = e - l['output'] * self.get_derivative(l['output'])
            print('output_error:', l['error'])
        output_layer += layers
        self.network['output'] = output_layer

    def backpropogate_error(self, row):
        _output = self.network['output']
        output_layer = [o for o in _output if 'bias' not in o.keys()]
        hidden_layer = self.network['hidden']
        layers = []
        for layer in hidden_layer:
            if 'bias' in layer.keys():
                hidden_layer = [layer]
            else:
                layers.append(layer)
        for layer in layers:
            layer['error'] = sum(
                [e * l for e, l in zip(row, layer['weights'])])
        for h, o in zip(layers, output_layer):
            h['error'] = h['error'] * self.get_derivative(o['output'])
            print('hidden_error:', h['error'])
        hidden_layer += layers
        self.network['hidden'] = hidden_layer

    @staticmethod
    def _update_weights(weight, learning_rate, error, value):
        return weight + learning_rate * error * value

    def update_weights(self, row):
        output_error = [i['error']
                        for i in self.network['output'] if 'error' in i.keys()][0]
        hidden_error = [i['error']
                        for i in self.network['hidden'] if 'error' in i.keys()][0]
        weights = []
        for o in self.network['output']:
            if 'bias' in o.keys():
                new_o_bias = self._update_weights(
                    o['bias'], self.learning_rate, output_error, 1)
            else:
                weights.append(o['weights'])
        new_o_weights = [self._update_weights(
            w[0], self.learning_rate, output_error, i) for w, i in zip(weights, row)]
        for h in self.network['hidden']:
            if 'bias' in h.keys():
                new_h_bias = self._update_weights(
                    h['bias'], self.learning_rate, hidden_error, 1)
            else:
                new_h_weights = [self._update_weights(
                    w, self.learning_rate, hidden_error, i) for w, i in zip(h['weights'], row)]
        new_network, new_network['hidden'], new_network['output'] = {}, [], []
        new_network['hidden'].append({'bias': new_h_bias})
        new_network['hidden'].append({'weights': new_h_weights})
        new_network['output'].append({'bias': new_o_bias})
        for o in new_o_weights:
            new_network['output'].append({'weights': o})
        self.network = new_network

    def run(self):
        self.initialize_network()
        for n in range(self.n_epochs):
            for row in self.x_train:
                self.calculate_output_delta(row)
                self.backpropogate_error(row)
                self.update_weights(row)
