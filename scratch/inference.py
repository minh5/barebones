from math import gamma

from scratch import prob


def normal_cdf_above(lower, mu=0, sigma=1):
    return 1 - prob.normal_cdf(lower, mu, sigma)


def normal_cdf_between(lower, upper, mu=0, sigma=1):
    return prob.normal_cdf(upper, mu, sigma) - prob.normal_cdf(lower, mu, sigma)


def normal_cdf_tails(lower, upper, mu=0, sigma=1):
    return 1 - normal_cdf_between(lower, upper, mu, sigma)


def normal_upper_bound(probability, mu=0, sigma=1):
    return prob.inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability, mu=0, sigma=1):
    return 1 - prob.inverse_normal_cdf(probability, mu, sigma)


def normal_two_bounds(probability, mu=0, sigma=1):
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower = normal_upper_bound(tail_probability, mu, sigma)
    return lower, upper


def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        # if x is greater than the mean, the tail is what's greater than x
        return 2 * normal_cdf_above(x, mu, sigma)
    else:
        # if x is less than the mean, the tail is what's less than x
        return 2 * prob.normal_cdf(x, mu, sigma)


def B(alpha, beta):
    return gamma(alpha) * gamma(beta) / gamma(alpha + beta)


def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    else:
        return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
