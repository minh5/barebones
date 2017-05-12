import math


class Regression:

    def __init__(self, x, y, test_set):
        self.x = x
        self.y = y
        self.test_set = test_set

    @staticmethod
    def mean(x):
        return sum(x) / len(x)

    def std_dev(self, x):
        variance = sum([(i - self.mean(x))**2 for i in x]) / (len(x) - 1)
        return variance**1 / 2

    def calculate_rmse(self):
        predicted = self.make_predictions()
        assert len(self.y) == len(predicted)
        errors = [(p - y)**2 for p, y in zip(predicted, self.y)]
        return (sum(errors) / len(self.y))**(1 / 2)


class LinearRegression(Regression):

    def estimate_coefficient(self):
        x_terms = [i - self.mean(self.x) for i in self.x]
        y_terms = [i - self.mean(self.y) for i in self.y]
        numerator = sum([x * y for x, y in zip(x_terms, y_terms)])
        denominator = sum([(i - self.mean(self.x))**2 for i in self.x])
        return numerator / denominator

    def estimate_bias_term(self):
        coef = self.estimate_coefficient()
        return self.mean(self.y) - (coef * self.mean(self.x))

    def make_predictions(self):
        coef = self.estimate_coefficient()
        bias = self.estimate_bias_term()
        return [coef * i + bias for i in self.test_set]

    def pearsons_correlation(self, x, y, n=None):
        if not n:
            n = len(x)
        x_terms = [i - self.mean(x) for i in x]
        y_terms = [i - self.mean(y) for i in y]
        _covariance = [x * y for x, y in zip(x_terms, y_terms)]
        covariance = sum(_covariance) / (n - 1)
        x_stdev = self.std_dev(x)
        y_stdev = self.std_dev(y)
        return covariance / (x_stdev * y_stdev)

    def estimate_coefficient_alternative(self):
        corr = self.pearsons_correlation(self.x, self.y)
        return corr * (self.std_dev(self.x) / self.std_dev(self.y))
