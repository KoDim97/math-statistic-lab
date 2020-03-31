import numpy as np
import matplotlib.pyplot as plt

# params
l_param = float(1/2**0.5)
u_param = float(3**0.5)
distributions = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']

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


def main():
    for dist_name in distributions:
        data1_more_sum = 0;
        data1_less_sum = 0;
        data2_less_sum = 0;
        data2_more_sum = 0;
        for i in range(1000):
            data1 = GetSamples(dist_name, 20)
            LQ_20 = np.quantile(data1, 0.25)
            UQ_20 = np.quantile(data1, 0.75)
            x_20_1 = LQ_20 - 1.5 * (UQ_20 - LQ_20)
            x_20_2 = UQ_20 + 1.5 * (UQ_20 - LQ_20)
            data1_more = data1[data1 > x_20_2]
            data1_less = data1[data1 < x_20_1]
            data1_less_sum += len(data1_less)
            data1_more_sum += len(data1_more)

            data2 = GetSamples(dist_name, 100)
            LQ_100 = np.quantile(data2, 0.25)
            UQ_100 = np.quantile(data2, 0.75)
            x_100_1 = LQ_100 - 1.5 * (UQ_100 - LQ_100)
            x_100_2 = UQ_100 + 1.5 * (UQ_100 - LQ_100)
            data2_more = data2[data2 > x_100_2]
            data2_less = data2[data2 < x_100_1]
            data2_less_sum += len(data2_less)
            data2_more_sum += len(data2_more)

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.set_title(dist_name + ' distribution. 20 samples')
        # ax1.boxplot(data1)
        # ax2.set_title(dist_name + ' distribution. 100 samples')
        # ax2.boxplot(data2)
        # plt.savefig(dist_name + '.png')
        print(f"{dist_name}_20:\n"
              f"{(data1_less_sum + data1_more_sum) / 20000}\n"
              f"{dist_name}_100:\n"
              f"{(data2_less_sum + data2_more_sum) / 100000}\n")

if __name__ == "__main__":
    main()