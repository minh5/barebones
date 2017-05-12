import numpy as np


class Perceptron:

    def __init__(
        self, x_train, y_train, bias=0, learning_rate=0.01,
        threshold=0, epoch=5
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.bias = bias
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epoch = epoch

    def predict(self, x, weights):
        activation = self.bias
        for i in range(len(x)):
            activation += weights[i] * x[i]
        return 1 if activation >= self.threshold else 0

    def run(self):
        weights = np.zeros(self.x_train.shape[1])
        for epoch in range(self.epoch):
            sum_error = 0
            for x, y in zip(self.x_train, self.y_train):
                prediction = self.predict(x, weights)
                error = y - prediction
                sum_error += error**2
                self.bias = self.bias + self.learning_rate * error
                for i in range(len(x)):
                    weights[i] = weights[i] + self.learning_rate * error * x[i]
                    print('>epoch=%d, lrate=%.3f, error=%.3f' %
                          (epoch, self.learning_rate, sum_error))
        print('weight:', weights)
        return weights
