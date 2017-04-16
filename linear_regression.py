

def mean(x):
    return sum(x)/len(x)


def estimate_coefficient(x, y):
    x_terms = [i - mean(x) for i in x]
    y_terms = [i - mean(y) for i in y]
    numerator = sum([x * y for x, y in zip(x_terms, y_terms)])
    denominator = sum([(i - mean(x))**2 for i in x])
    return numerator/denominator


def estimate_bias_term(x, y):
    coef = estimate_coefficient(x, y)
    return mean(y) - (coef * mean(x))


def make_predictions(coef, bias, x):
    return [coef * i + bias for i in x]


def calculate_rmse(observed, predicted):
    assert len(observed) == len(predicted)
    errors = [(p - y)**2 for p, y in zip(predicted, observed)]
    return (sum(errors)/len(observed))**(1/2)


def std_dev(x):
    variance = sum([(i - mean(x))**2 for i in x])/(len(x)-1)
    return variance**1/2


def pearsons_correlation(x, y, n=None):
    if not n:
        n = len(x)
    x_terms = [i - mean(x) for i in x]
    y_terms = [i - mean(y) for i in y]
    _covariance = [x * y for x, y in zip(x_terms, y_terms)]
    covariance = sum(_covariance)/(n-1)
    x_stdev = std_dev(x)
    y_stdev = std_dev(y)
    return covariance/(x_stdev*y_stdev)


def estimate_coefficient_alternative(x, y):
    return pearsons_correlation(x, y) * (std_dev(x)/std_dev(y))
