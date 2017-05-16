from math import exp, pi, erf
import random


def joint_prob(prob1, prob2):
    return prob1 * prob2


def conditional_prob_dep(prob1, prob2):
    return joint_prob(prob1, prob2) * prob2


def bayes_thereom(x):
    pass


def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0


def uniform_cdf(x):
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1


def normal_dist(x, mu=0, sigma=1):
    first_part = 1/((2 * pi * sigma)**(1/2))
    second_part = exp(-((x - mu)**2)/(2 * sigma**2))
    return first_part * second_part


def normal_cdf(x, mu=0, sigma=1):
    return (1 + erf(x - mu) / (2**(1/2)) / sigma) / 2


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.0001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, low_p = -10, 0  # normal_cdf(-10) is (very close to) 0
    hi_z, hi_p = 10, 1  # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # consider the midpoint
        mid_p = normal_cdf(mid_z)  # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break
        return mid_z


def bernoulli_trial(p):
    return 1 if random.random() < p else 0


def binomial_dist(n, p):
    return sum([bernoulli_trial(p) for _ in range(n)])


# NOTE: the mean of Bernoulli(p) is p, the sigma is p(1-p)**(1/2).
# As sample size goes up, binomial dist will resemble normal with mean
# of np and sigma of np(1 -p)**(1/2)


def normal_from_binomial(n, p):
    mu = p * n
    sigma = (p * (1 - p) * n)**(1/2)
    return mu, sigma
