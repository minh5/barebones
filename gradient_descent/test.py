from models import LinearGradientDescent

x_set = [1, 2, 4, 3, 5]
y_set = [1, 3, 3, 2, 5]

test = LinearGradientDescent(x_set, y_set, 10)
test.run()
