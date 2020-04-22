import csv

import numpy as np
from prettytable import PrettyTable
from scipy.stats import chi2
import scipy.stats as stats



#params
alpha = 0.05
size = 100
p = 1 - alpha
k = 6



def main():
    samples = np.random.normal(0, 1, size=size)
    mu = np.mean(samples)
    print(mu)
    sigma = np.std(samples)
    print(sigma)
    borders = np.linspace(mu - 1, mu + 1, num=(k - 1))
    real = chi2.ppf(p, k - 1)

    p_arr = np.array([stats.norm.cdf(borders[0])])
    for i in range(len(borders) - 1):
        val = stats.norm.cdf(borders[i + 1]) - stats.norm.cdf(borders[i])
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - stats.norm.cdf(borders[-1]))
    print(f"Промежутки: {borders} \n"
          f"p_i: \n {p_arr} \n"
          f"n * p_i: \n {p_arr * 100} \n"
          f"Сумма: {np.sum(p_arr)}")
    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
    res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)
    print(f"n_i: \n {n_arr} \n"
          f"Сумма: {np.sum(n_arr)}\n"
          f"n_i  - n*p_i: {n_arr - p_arr * 100}\n"
          f"res: {res_arr}\n"
          f"res_sum = {np.sum(res_arr)}\n")

    borders = np.append(borders, np.math.inf)
    borders = np.append(borders, -np.math.inf)
    resTable = PrettyTable()
    resTable.field_names = ["i", "Границы", "$n_i$", "$p_i$", "$np_i$", "$n_i - np_i$", "$\\frac{(n_i -np_i)^2}{np_i}$ "]
    for i in range(k):
        resTable.add_row([i+1,
                          '(' + str(borders[i-1])+ '; ' + str(borders[i]) + ']',
                          n_arr[i],
                          p_arr[i],
                          p_arr[i] * size,
                          n_arr[i] - p_arr[i] * size,
                          res_arr[i]])
    resTable.add_row(["$\sum$",
                      "--",
                      np.sum(n_arr),
                      np.sum(p_arr),
                      np.sum(p_arr) * size,
                      "0.00",
                      np.sum(res_arr)])
    print(resTable)
    data = resTable.get_string()
    result = [tuple(filter(None, map(str.strip, splitline))) for line in data.splitlines() for splitline in
              [line.split("|")] if len(splitline) > 1]
    with open('table.csv', 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)


if __name__ == '__main__':
    main()