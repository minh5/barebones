from models import LinearRegression

x_set = [1, 2, 4, 3, 5]
y_set = [1, 3, 3, 2, 5]
x_test = [1, 2, 4, 3, 5]

test_model = LinearRegression(x_set, y_set, x_test)

print(test_model.mean(test_model.x))
print(test_model.mean(test_model.y))
print('coef:', test_model.estimate_coefficient())
print('alternative coef:', test_model.estimate_coefficient_alternative())

print('corr:', test_model.pearsons_correlation(test_model.x, test_model.y))
print('std x:', test_model.std_dev(test_model.x))
print('std y:', test_model.std_dev(test_model.y))
print('bias:', test_model.estimate_bias_term())

coef = test_model.estimate_coefficient()
bias = test_model.estimate_bias_term()
y_test = test_model.make_predictions()

print('new predicted values:', test_model.make_predictions())
print('RMSE:', test_model.calculate_rmse())
