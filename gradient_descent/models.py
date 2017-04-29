import math
import numpy as np


class GradientDescent:

    def __init__(self, x, y, iterations, weight=0, alpha=0.01):
        self.weight = weight
        self.bias = 0
        self.alpha = alpha
        self.iterations = iterations
        self.x = x
        self.y = y

    def make_prediction(self):
        pass

    def update_bias(self):
        pass

    def update_coefficients(self):
        pass

    def calculate_rmse(self, list_of_errors):
        sum_of_errors = sum(list_of_errors)
        return sum_of_errors/len(list_of_errors)

    def run(self):
        pass


class LinearGradientDescent(GradientDescent):

    def make_prediction(self, x):
        return self.bias + self.weight * x

    def update_bias(self, error):
        return self.bias - self.alpha * error

    def update_coefficients(self, error, observed):
        return self.weight - self.alpha * error * observed

    def run(self):
        epoch = len(self.y)
        for i in range(self.iterations):
            index = i % epoch
            x = self.x[index]
            observed = self.y[index]
            predicted = self.make_prediction(x)
            error = predicted - observed
            if index == 0:
                list_of_errors = [error**2]
            elif index == epoch-1:
                rmse = self.calculate_rmse(list_of_errors)
                print('CURRENT LINEAR RMSE:', rmse)
            else:
                list_of_errors.append(error**2)
            self.bias = self.update_bias(error)
            self.weight = self.update_coefficients(error, observed)


class MultiVariateGradientDescent(GradientDescent):

    def make_prediction(bias, x_values, coefficients):
        return sum([x * c for x, c in zip(x_values, coefficients)]) + bias

    @staticmethod
    def update_bias(bias, alpha, error):
        return bias - alpha * error

    @staticmethod
    def update_coefficients(coefficient, alpha, error, x):
        return coefficient - alpha * error * x

    def run(self, train_set, y_train, n_epochs):
        coefficients = [0 for i in range(len(train_set))]
        for e in range(n_epochs):
            sum_errors = 0
            for row, y in zip(train_set, y_train):
                yhat = self.make_prediction(self.bias, row, coefficients)
                error = (yhat - y)
                sum_errors += error**2
                self.bias = self.update(self.bias, self.alpha, error)
                for i in range(row):
                    coefficients[i] = self.update_coefficients(
                        coefficients[i], self.alpha, error, row[i]
                        )
        return coefficients


class LogisticGradientDescent(GradientDescent):

    def __init__(self, x, y, iterations, alpha=0.5):
        self.set_of_x = x
        self.y = y
        self.iterations = iterations
        self.set_of_coefficients = np.zeros(self.set_of_x.shape[0])
        self.alpha = alpha

    @staticmethod
    def make_prediction(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def make_binary_prediction(prediction, observed):
        binary = 1 if prediction > .5 else 0
        return 1 if binary == observed else 0

    def get_derivative(self, x):
        return self.get_logistic_function(x)*(1-self.get_logistic_function(x))

    def update_coef(self, coef, observed, prediction, x):
        return coef + self.alpha * (observed - prediction) * prediction * (1 - prediction) * x

    def calculate_accuracy(self, list_of_values):
        summed = sum(list_of_values)
        return summed/len(list_of_values)

    def run(self):
        epoch = len(self.y)
        for i in range(self.iterations):
            index = i % epoch
            set_of_x = [x[index] for x in self.set_of_x]
            observed = self.y[index]
            logit_value = sum([w * i for w, i in zip(self.set_of_coefficients, set_of_x)])
            prediction = self.make_prediction(logit_value)
            self.set_of_coefficients = [self.update_coef(coef, observed, prediction, x) for coef, x in zip(self.set_of_coefficients, set_of_x)]
            squared_error = (prediction - observed)**2
            binary_prediction = self.make_binary_prediction(prediction, observed)
            if index == 0:
                list_of_errors = [squared_error]
                list_of_binary = [binary_prediction]
            elif index == epoch-1:
                rmse = self.calculate_rmse(list_of_errors)
                accuracy = self.calculate_accuracy(list_of_binary)
                print('CURRENT LOGIT RMSE:', rmse)
                print('CURRENT LOGIT ACCURACY:', accuracy)
            else:
                list_of_errors.append(squared_error)
                list_of_binary.append(binary_prediction)
