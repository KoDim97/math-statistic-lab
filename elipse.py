import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from prettytable import PrettyTable
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


#params
coef_list = [0, 0.5, 0.9]
size_list = [20, 60, 100]


def quadrantr(x, y):
    x_0 = x - np.mean(x)
    y_0 = y - np.mean(y)
    r_q = np.sum(np.multiply(np.sign(x_0), np.sign(y_0))) / x_0.shape
    return r_q

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x*1.5 , height=ell_radius_y *1.5,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def draw_plot_scatter(n):
    po = [0, 0.5, 0.9]
    rv_mean = [0, 0]
    fig, ax = plt.subplots(1, 3)
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    for i in range(3):
        rv_cov = [[1.0, po[i]], [po[i], 1.0]]
        rv = multivariate_normal.rvs(rv_mean, rv_cov, size=n)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[i].scatter(x, y, s=3)
        confidence_ellipse(x, y, ax[i], edgecolor='black')
        ax[i].set_title(titles[i])
        ax[i].scatter(np.mean(x), np.mean(y), c='black', s=3)
    plt.show()

def doFirstTask():
    for size in size_list:
        resTable = PrettyTable()
        resTable.float_format = "2.2"
        resTable.field_names = ["", "$r$", "$r_S$", "$r_Q$"]
        for coef in coef_list:
            resTable.add_row(["$\\rho = " + str(coef) + "$", "", "", ""])
            pearson = []
            spearman = []
            quad = []
            rv_mean = [0, 0]
            rv_cov = [[1.0, coef], [coef, 1.0]]
            for it in range(1000):
                rv = multivariate_normal.rvs(rv_mean, rv_cov, size=size)
                x = rv[:, 0]
                y = rv[:, 1]
                ans, t = pearsonr(x, y)
                pearson.append(ans)
                ans, t = spearmanr(x, y)
                spearman.append(ans)
                quad.append(quadrantr(x, y))
            resTable.add_row(["$E(z)$",
                              np.around(np.mean(pearson), decimals=6),
                              np.around(np.mean(spearman), decimals=6),
                              np.around(np.mean(quad), decimals=6)])
            resTable.add_row(["$E(z^2)$",
                              np.around(np.mean(np.square(pearson)), decimals=6),
                              np.around(np.mean(np.square(spearman)), decimals=6),
                              np.around(np.mean(np.square(quad)), decimals=6)])

            resTable.add_row(["$D(z)$",
                              np.around(np.std(pearson) * np.std(pearson), decimals=6),
                              np.around(np.std(spearman) * np.std(spearman), decimals=6),
                              np.around(np.std(quad) * np.std(quad), decimals=6)])
            resTable.add_row(['', '', '', ''])
        print(resTable)
        data = resTable.get_string()
        result = [tuple(filter(None, map(str.strip, splitline))) for line in data.splitlines() for splitline in
                  [line.split("|")] if len(splitline) > 1]
        with open(str(size) + "_" + str(coef) + '.csv', 'w') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerows(result)

def doSecondTask():
    resTable = PrettyTable()
    resTable.float_format = "2.2"
    resTable.field_names = ["", "$r$", "$r_S$", "$r_Q$"]
    for size in size_list:
        pearson = []
        spearman = []
        quad = []
        resTable.add_row(['$n = $' + str(size), '', '', ''])
        for it in range(1000):
            rv = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) \
                 + 0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], size=size)
            x = rv[:, 0]
            y = rv[:, 1]
            ans, t = pearsonr(x, y)
            pearson.append(ans)
            ans, t = spearmanr(x, y)
            spearman.append(ans)
            quad.append(quadrantr(x, y))
        resTable.add_row(["$E(z)$" ,
                          np.around(np.mean(pearson), decimals=6),
                          np.around(np.mean(spearman), decimals=6),
                          np.around(np.mean(quad), decimals=6)])
        resTable.add_row(["$E(z^2)$" ,
                          np.around(np.mean(np.square(pearson)), decimals=6),
                          np.around(np.mean(np.square(spearman)), decimals=6),
                          np.around(np.mean(np.square(quad)), decimals=6)])

        resTable.add_row(["$D(z)$",
                          np.around(np.std(pearson) * np.std(pearson), decimals=6),
                          np.around(np.std(spearman) * np.std(spearman), decimals=6),
                          np.around(np.std(quad) * np.std(quad), decimals=6)])
        resTable.add_row(['', '', '', ''])
    print(resTable)
    data = resTable.get_string()
    result = [tuple(filter(None, map(str.strip, splitline))) for line in data.splitlines() for splitline in
              [line.split("|")] if len(splitline) > 1]
    with open(str(size) + "Comb" + '.csv', 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)

def doThirdTask():
    for i in range(3):
        draw_plot_scatter(size_list[i])

def main():
    doFirstTask()
    doSecondTask()
    doThirdTask()


if __name__ == "__main__":
    main()