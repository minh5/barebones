import operator

import numpy as np


class LDA:

    def __init__(self, x_set, y_set, x_test, y_test):
        self.x = x_set
        self.y = y_set
        self.test_x = x_test
        self.test_y = y_test

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

    def make_prediction(self, x):
        assert x in self.classes
        predicted_value = dict()
        for item in self.classes:
            value = x * (self.class_mean[item]/self.variance) - \
                (((self.class_mean[item])**2)/(2*self.variance)) + \
                np.log(self.class_mean[item])
            predicted_value[item] = value
        return max(predicted_value.iteritems(), key=operator.itemgetter(1))[0]

    def determine_accuracy(self, prediction, observed):
        return 1 if prediction == observed else 0

    def calculate_accuracy(self, list_of_predictions):
        return sum(list_of_predictions)/len(list_of_predictions)

    def run(self):
        accuracy_holder = []
        for x, y in zip(self.test_x, self.test_y):
            prediction = self.make_prediction(x)
            accuracy_holder.append(self.determine_accuracy(prediction, y))
        print('accuracy:', self.calculate_accuracy(accuracy_holder))
