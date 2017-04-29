
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
        return 1 if self.threshold >= 0 else 0

    def run(self):
        weights = [0 for i in range(len(self.x_train))]
        for epoch in range(self.epoch):
            sum_error = 0
        for x, y in zip(self.x_train, self.y_train):
            prediction = self.predict(x, weights)
            error = y - prediction
            sum_error += error**2
            self.bias = self.bias + self.learning_rate * error
        for i in range(len(x)):
            weights[i] = weights[i] + self.learning_rate * error * x[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learning_rate, sum_error))
        return weights
