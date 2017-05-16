
def mean(x):
    return sum(x)/len(x)


def median(x):
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n // 2
    if n % 2 != 0:
        left = midpoint - 1
        right = midpoint
        return (sorted_x[left] + sorted_x[right]) / 2
    else:
        return sorted_x[midpoint]


def quantile(x, percentile):
    index = int(len(x) * percentile)
    return sorted(x)[index]


def mode(x):
    return max(set(x), key=x.count)


def range(x):
    return max(x) - min(x)


def variance(x):
    n = len(x)
    x_bar = mean(x)
    return sum([(i - x_bar)**2 for i in x])/(n - 1)


def std_dev(x):
    return variance(x) ** (1/2)


def interquartile_range(x):
    return quantile(x, .75) - quantile(x, .25)


def dot(x, y):
    assert len(x) == len(y)
    return sum([i * j for i, j in zip(x, y)])


def covariance(x, y):
    assert len(x) == len(y)
    n = len(x)
    x_bar = sum(x)/n
    y_bar = sum(y)/n
    _x = [i - x_bar for i in x]
    _y = [j - y_bar for j in y]
    return dot(_x, _y)/(n-1)


def correlation(x, y):
    assert len(x) == len(y)
    std_dev_x = std_dev(x)
    std_dev_y = std_dev(y)
    return covariance(x, y) / (std_dev_x * std_dev_y)
