import operator

import numpy as np


class LDA:

    def __init__(self, x_set, y_set):
        self.x = x_set
        self.y = y_set

    @property
    def classes(self):
        return set(self.y)

    @property
    def class_mean(self):
        means = dict()
        for item in self.classes:
            subset = self.x[self.y == item]
            means[item] = sum(subset)/len(subset)

    @property
    def class_prob(self):
        class_prob = dict()
        for item in self.classes:
            subset = self.x[self.y == item]
            class_prob[item] = len(subset)/len(subset)

    @property
    def variance(self):
        squared_diff = []
        for item in self.classes:
            subset = self.x[self.y == item]
            squared_diff.append(sum([(x - self.class_prob[item])**2 for x in subset]))
        return 1/(len(self.x) - len(self.classes)) * sum(squared_diff)

    def make_predictions(self, x):
        assert x in self.classes
        predicted_value = dict()
        for item in self.classes:
            value = x * (self.class_mean[item]/self.variance) - \
                (((self.class_mean[item])**2)/(2*self.variance)) + \
                np.log(self.class_mean[item])
            predicted_value[item] = value
        return max(predicted_value.iteritems(), key=operator.itemgetter(1))[0]
