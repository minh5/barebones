from itertools import cycle
from math import exp

import numpy as np


class LinearKernel:

    def __call__(self, x_set, y_set):
        return sum([x * y for x, y in zip(x_set, y_set)])


class PolyKernel:

    def __call__(self, x_set, y_set, degree):
        return sum([x * y for x, y in zip(x_set, y_set)]) ** degree


class RadianKernel:

    def __call__(self, x_set, y_set, gamma):
        exponent = sum([(x - y)**2 for x, y in zip(x_set, y_set)])
        return exp(-gamma * exponent)


class SVM:

    def __init__(
        self, x, y, n_iterations=10, lambda_rate=0.05, C=1,
        degrees=3, gamma=1, kernel_type='linear'
    ):
        self.x = x
        self.y = y
        self.n_iterations = n_iterations
        self.lambda_rate = lambda_rate
        self.C = C
        self.coefs = np.zeros(self.x.shape[1])
        if kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'poly':
            self.kernel = PolyKernel()
        elif kernel_type == 'radian':
            self.kernel = RadianKernel()
        else:
            raise BaseException('invalid kernel type')

    @staticmethod
    def calculate_output(row, y, coefs, kernel, **kwargs):
        return kernel(row, coefs, **kwargs) * y

    def update_coefs(self, output, x, y, coef, iteration):
        if output > 1:
            return (1 - (1/iteration)) * coef
        else:
            return (1 - (1/iteration)) * coef + \
                (1/(self.lambda_rate * iteration)) * (x * y)

    def run(self):
        counter = 0
        while counter < self.n_iterations:
            for i, row, y in zip(range(self.n_iterations), cycle(self.x), cycle(self.y)):
                output = self.calculate_output(row, y, self.coefs, self.kernel)
                new_coefs = []
                for x, coef in zip(row, self.coefs):
                    new_coef = self.update_coefs(output, x, y, coef, i)
                    new_coefs.append(new_coef)
                self.coefs = new_coefs
                counter += 1
