import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from scipy.special import factorial

# params
l_param = float(1/2**0.5)
u_param = float(3**0.5)
distributions = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
size = [20, 60, 100]


def GetSamples(d_name, num):
    if d_name == 'Normal':
        return np.random.normal(0, 1, num)
    elif d_name == 'Cauchy':
        return np.random.standard_cauchy(num)
    elif d_name == 'Laplace':
        return np.random.laplace(0, l_param, num)
    elif d_name == 'Poisson':
        return np.random.poisson(10, num)
    elif d_name == 'Uniform':
        return np.random.uniform(-u_param, u_param, num)
    return []


def GetCDFValues(x, d_name):
    if d_name == 'Normal':
        return stats.norm.cdf(x, 0, 1)
    elif d_name == 'Cauchy':
        return stats.cauchy.cdf(x, 0, 1)
    elif d_name == 'Laplace':
        return stats.laplace.cdf(x, 0, l_param)
    elif d_name == 'Poisson':
        return stats.poisson.cdf(x, 10)
    elif d_name == 'Uniform':
        return stats.uniform.cdf(x, -u_param, 2 * u_param)
    return []


def GetPDFValues(x, d_name):
    if d_name == 'Normal':
        return stats.norm.pdf(x, 0, 1)
    elif d_name == 'Cauchy':
        return stats.cauchy.pdf(x, 0, 1)
    elif d_name == 'Laplace':
        return stats.laplace.pdf(x, 0, l_param)
    elif d_name == 'Poisson':
        return np.exp(-10) * np.power(10, x)/factorial(x)
    elif d_name == 'Uniform':
        return stats.uniform.pdf(x, -u_param, 2 * u_param)
    return []


def CDEPlots():
    samples = [[], [], []]
    for dist_name in distributions:
        fig, ax = plt.subplots(1, 3)
        for i in range(len(size)):
            if dist_name == 'Poisson':
                r = (6, 14)
            else:
                r = (-4, 4)
            x = np.linspace(r[0], r[1], 1000)
            samples[i] = GetSamples(dist_name, size[i])
            ecdf = sm.distributions.ECDF(samples[i])
            y = ecdf(x)
            ax[i].plot(x, y, color='chocolate', label='Empirical distribution func')
            y = GetCDFValues(x, dist_name)
            ax[i].plot(x, y, color='seagreen', label='Distribution func')
            ax[i].set_title(dist_name + '\n n = ' + str(size[i]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0., prop={'size': 6})
        fig.savefig(dist_name + '_cde.png', dpi=200)

def main():
    CDEPlots()
    for dist_name in distributions:
        for i in range(len(size)):
            fig, ax = plt.subplots(1, 3)
            if dist_name == 'Poisson':
                r = (6, 14)
            else:
                r = (-4, 4)
            x = np.linspace(r[0], r[1], 1000)
            y = GetPDFValues(x, dist_name)
            samples = GetSamples(dist_name, size[i])
            samples = samples[samples <= r[1]]
            samples = samples[samples >= r[0]]
            kde = stats.gaussian_kde(samples)
            kde.set_bandwidth(bw_method='silverman')
            h_n = kde.factor
            sns.kdeplot(samples, ax=ax[0], bw=h_n/2, color='chocolate')
            ax[0].set_title(r'$h = \frac{h_n}{2}$')
            ax[0].plot(x, y, color='seagreen')
            ax[1].plot(x, y, color='seagreen')
            ax[2].plot(x, y, color='seagreen', label='Real density function')
            sns.kdeplot(samples, ax=ax[1], bw=h_n, color='chocolate')
            ax[1].set_title(r'$h = h_n$')
            sns.kdeplot(samples, ax=ax[2], bw=2*h_n, color='chocolate', label='Kernel density esimation')
            ax[2].set_title(r'$h = 2 * h_n$')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
            plt.show()
            fig.savefig(dist_name + str(size[i]) + '_kde.png', dpi=200)

if __name__ == "__main__":
    main()