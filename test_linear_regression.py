from linear_regression import *

x_set = [1, 2, 4, 3, 5]
y_set = [1, 3, 3, 2, 5]
x_test = [1, 2, 4, 3, 5]

print(mean(x_set))
print(mean(y_set))
print('coef:', estimate_coefficient(x_set, y_set))
print('alternative coef:', estimate_coefficient_alternative(x_set, y_set))

print('corr:', pearsons_correlation(x_set, y_set))
print('std x:', std_dev(x_set))
print('std y:', std_dev(y_set))
print('bias:', estimate_bias_term(x_set, y_set))

coef = estimate_coefficient(x_set, y_set)
bias = estimate_bias_term(x_set, y_set)
y_test = make_predictions(coef, bias, x_test)

print('new predicted values:', make_predictions(coef, bias, x_test))
print('RMSE:', calculate_rmse(y_set, y_test))
