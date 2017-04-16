
class GradientDescent:

    def __init__(self, x, y, iterations, weight=0, alpha=0.01):
        self.weight = weight
        self.bias = weight
        self.alpha = alpha
        self.iterations = iterations
        self.x = x
        self.y = y

    def run(self):
        pass


class LinearGradientDescent(GradientDescent):

    def run(self):
        epoch = len(self.y)
        for i in range(self.iterations):
            index = i % epoch
            observed = self.y[index]
            predicted = self.weight * observed + self.bias
            error = predicted - observed
            self.bias = self.bias - self.alpha * error
            self.weight = self.weight - self.alpha * error * observed
            print('current predicted:', predicted)
            print('current error:', error)
            print('current bias:', self.bias)
            print('current weight:', self.weight)
