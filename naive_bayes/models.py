from math import exp, pi
import operator


class NaiveBayes:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    @staticmethod
    def mean(x):
        return sum(x)/len(x)

    @staticmethod
    def gaussian_dist(x, mean, std_dev):
        exponent = exp(-((x - mean)**2 / (2*std_dev**2)))
        return 1/((2*pi)**(1/2) * std_dev) * exponent

    def std_dev(self, x):
        mean = self.mean(x)
        result = []
        for i in x:
            result.append((i - mean)**2)
        return (sum(result)/(len(result)-1))**(1/2)

    def class_summary(self, data):
        return [(self.mean(column), self.std_dev(column), len(column)) for column in zip(*data)]

    def separate_by_class(self):
        class_dict = dict()
        for x in set(self.y_train):
            subset = self.x_train[self.y_train == x]
            class_dict[x] = subset
        return class_dict

    def get_class_summary(self):
        separated = self.separate_by_class()
        summary = dict()
        for key, value in separated.items():
            summary[key] = self.class_summary(value)
        return summary

    def calculate_probs(self, row, summary):
        probs = dict()
        for y, x in summary.items():
            probs[y] = summary[y][0][2]/self.x_train.shape[0]
            for i in range(len(x)):
                mean, std_dev, count = x[i]
                probs[y] *= self.gaussian_dist(row[i], mean, std_dev)
        return probs

    def predict(self, probs_dict):
        return max(probs_dict.items(), key=operator.itemgetter(1))[0]
