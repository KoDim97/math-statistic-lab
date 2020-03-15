import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def createPlot(fig, data, x_data, y_data, bin, name):
    fig.set_title(name)
    fig.plot(x_data, y_data, label='PDF')
    fig.hist(data, density=True, label='HIST')
    fig.set_ylabel('Density')
    fig.set_xlabel('x')
    fig.legend()
    fig.grid()


def main():
    # params
    size = [10, 50, 1000]
    x_axis1 = np.arange(-3, 3, 0.01)
    x_axis2 = np.arange(-5, 5, 0.01)
    x_axis3 = np.arange(0, 20, 1)
    l_param = float(1/2**0.5)
    u_param = float(3**0.5)

    #results
    fig_norm, subs_norm = plt.subplots(1, 3, figsize=(14,8))
    fig_norm.suptitle('Normal distribution')
    fig_cauchy, subs_cauchy = plt.subplots(1, 3, figsize=(14, 8))
    fig_cauchy.suptitle('Сauchy distribution')
    fig_laplace, subs_laplace = plt.subplots(1, 3, figsize=(14, 8))
    fig_laplace.suptitle('Laplace distribution')
    fig_poisson, subs_poisson = plt.subplots(1, 3, figsize=(14, 9))
    fig_poisson.suptitle('Poisson distribution')
    fig_uniform, subs_uniform = plt.subplots(1, 3, figsize=(14, 8))
    fig_uniform.suptitle('Uniform distribution')
    j = int(0)
    for i in size:
        createPlot(subs_norm[j], stats.norm.rvs(size=i), x_axis1, stats.norm.pdf(x_axis1, 0, 1), i**(1/3), "Sample = " + str(i))
        createPlot(subs_cauchy[j], stats.cauchy.rvs(size=i), x_axis2, stats.cauchy.pdf(x_axis2), i**(1/3), "Cauchy distribution. Sample = " + str(i))
        createPlot(subs_laplace[j], stats.laplace.rvs(0, l_param, size=i), x_axis2, stats.laplace.pdf(x_axis2, 0, l_param), i**(1/3), "Laplace distribution. Sample = " + str(i))
        createPlot(subs_poisson[j], stats.poisson.rvs(10, size=i), x_axis3, stats.poisson.pmf(x_axis3, 10), i**(1/3), "Poisson distribution. Sample = " + str(i))
        createPlot(subs_uniform[j], stats.uniform.rvs(-u_param, 2*u_param, size=i), [-u_param, u_param], stats.uniform.pdf([-u_param, u_param], -u_param, 2*u_param), i**(1/3), "Uniform distribution. Sample = " + str(i))
        j += 1
    fig_norm.savefig('Norm'+'.png')
    fig_cauchy.savefig('Cauchy' + '.png')
    fig_laplace.savefig('Laplace' + '.png')
    fig_poisson.savefig('Poisson' + '.png')
    fig_uniform.savefig('Uniform' + '.png')

if __name__ == "__main__":
    main()

