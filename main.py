import numpy as np
import scipy.stats as stats
import csv

from prettytable import PrettyTable

# params
size = [10, 100, 1000]
l_param = float(1/2**0.5)
u_param = float(3**0.5)
distribution = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']

def getQuartile(distr, p):
    n = len(distr)
    sorted = np.sort(distr)
    return sorted[int(np.floor(n * p) + np.ceil((n * p) - int(n * p)))]

def getDistrSamples(d_name, num):
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
    for distr in distribution:
        print(f"Distribution {distr}")
        resTable = PrettyTable()
        resTable.float_format = "2.2"
        resTable.field_names = ["", "$\\bar{x}$", "$medx$", "$Z_r$", "$Z_q$", "$Z_t_r$"]
        for d_num in size:
            mean = []
            med = []
            z_r = []
            z_q = []
            z_tr = []
            for it in range(1000):
                sample_d = getDistrSamples(distr, d_num)
                sample_d_sorted = np.sort(sample_d)
                mean.append(np.mean(sample_d))
                med.append(np.median(sample_d))
                z_r.append((sample_d_sorted[0] + sample_d_sorted[-1]) / 2)
                z_q.append((getQuartile(sample_d, 0.25) + getQuartile(sample_d, 0.75)) / 2)
                z_tr.append(stats.trim_mean(sample_d, 0.25))
            resTable.add_row([distr + ' n = ' + str(d_num), '','','','',''])
            resTable.add_row(['', '', '', '', '', ''])
            resTable.add_row(["$E(z)$",
                              np.around(np.mean(mean), decimals=4),
                              np.around(np.mean(med), decimals=4),
                              np.around(np.mean(z_r), decimals=4),
                              np.around(np.mean(z_q), decimals=4),
                             np.around(np.mean(z_tr), decimals=4)])
            resTable.add_row(["$D(z)$",
                              np.around(np.std(mean) * np.std(mean), decimals=4),
                              np.around(np.std(med) * np.std(med), decimals=4),
                              np.around(np.std(z_r) * np.std(z_r), decimals=4),
                              np.around(np.std(z_q) * np.std(z_q), decimals=4),
                              np.around(np.std(z_tr) * np.std(z_tr), decimals=4)])
            resTable.add_row(['', '', '', '', '', ''])

        print(resTable)
        data = resTable.get_string()
        result = [tuple(filter(None, map(str.strip, splitline))) for line in data.splitlines() for splitline in
                  [line.split("|")] if len(splitline) > 1]
        with open(distr + '.csv', 'w') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result)

if __name__ == "__main__":
    main()
