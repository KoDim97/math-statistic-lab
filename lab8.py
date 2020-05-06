import numpy as np
import scipy.stats
import statistics

# params
samples = [20, 100]
gamma = 0.95


def ConfidenceIntervalMean(data, confidence=gamma):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1) / (n - 1) ** 0.5
    return 'mean', round(m - h, 3), round(m + h, 3)


def ConfidenceIntervalVariance(data, confidence=gamma):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    a = se * (n / (scipy.stats.chi2.ppf((1 + confidence) / 2, n - 1))) ** 0.5
    b = se * n ** 0.5 / (scipy.stats.chi2.ppf((1 - confidence) / 2, n - 1)) ** 0.5
    return 'variance', round(a, 3), round(b, 3)


def AsimptoticConfidenceIntervalMean(data, confidence=gamma):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    u = scipy.stats.norm.ppf((1 + confidence) / 2)
    h = se * u / n ** 0.5
    return 'mean asimptotic', round(m - h, 3), round(m + h, 3)


def AsimptoticConfidenceIntervalVariance(data, confidence=gamma):
    data = np.array(data)
    n = data.size
    m, se = np.mean(data), statistics.variance(data)
    m4 = scipy.stats.moment(data, 4)
    e = m4 / se ** 4 - 3
    u = scipy.stats.norm.ppf((1 + confidence) / 2)
    U = u * ((e + 2) / n) ** 0.5
    a = se * (1 + U) ** (-0.5)
    b = se * (1 - U) ** (-0.5)
    return 'variance asimptotic', round(a, 3), round(b, 3)


def main():
    for i in range(len(samples)):
        print(samples[i])
        data = scipy.stats.norm.rvs(size=samples[i])
        print(ConfidenceIntervalMean(data))
        print(ConfidenceIntervalVariance(data))
        print(AsimptoticConfidenceIntervalMean(data))
        print(AsimptoticConfidenceIntervalVariance(data))


if __name__ == '__main__':
    main()
