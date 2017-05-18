from math import gamma
import scipy.stats

from scratch import base


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


def calculate_z_score(x):
    return [i - prob.mean(i)/prob.std_dev(i) for i in x]


def indepenedent_t_test(set1, set2):
    part1 = sum(set1)**2 / len(set1)
    part2 = sum(set2)**2 / len(set2)
    mean_1 = sum(set1)/len(set1)
    mean_2 = sum(set2)/len(set2)
    ss_1 = sum([s**2 for s in set1])
    ss_2 = sum([s**2 for s in set2])
    means_formula = ((ss_1 - part1) + (ss_2 - part2)) / (len(set1) + len(set2) - 2)
    inverse = (1 / len(set1)) + (1 / len(set2))
    tt = (mean_1 - mean_2) / (means_formula * inverse)**(1/2)
    return scipy.stats.t(tt, (len(set1) + len(set2) - 2))


def f_statistic(sample1, sample2, alpha=.05, tail="one"):
    # test to see if variances from two samples are statistically difference
    var_1 = base.variance(sample1)
    var_2 = base.variance(sample2)
    if var_1 > var_2:
        f_critical = var_1 / var_2
    else:
        f_critical = var_2 / var_1
    if tail == "two":
        alpha = alpha / 2
    p_value = scipy.stats.f.cdf(f_critical, len(sample1) - 1, len(sample2) - 2)
    return "p_value is %d" % p_value


def one_way_anova(*args):
    means = []
    stdev = []
    _n = []
    sst = 0
    p = len(args)
    for arg in args:
        _n.append(len(arg))
        means.append(sum(arg)/len(arg))
        stdev.append(base.std_dev(arg))
    x_bar = sum(means)/p
    for n, mean in zip(_n, means):
        sst += n*(mean - x_bar)**2
    mst = sst / p - 2
    sse = sum([(n - 1)(sd) for n, sd in zip(_n, stdev)])
    mse = sse / (sum(_n) - p)
    return mst / mse

def chi_squared_test(observed, expected, df=None):
    chi_stat = sum([(o - e)**2 / e for o, e in zip(observed, expected)])
    if df is None:
        df = len(observed) - 1
    return 1 - scipy.stats.chi2.cdf(x=chi_stat, df=df)
