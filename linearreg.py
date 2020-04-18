import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def LsmLinreg(x, y):
    betta1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    betta0 = np.mean(y) - betta1 * np.mean(x)
    return betta1, betta0


def CostFunction(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_lam(x, y):
    init_b = np.array([0, 1])
    res = minimize(CostFunction, init_b, args=(x, y), method='COBYLA')
    return res.x


def draw_res(lsm_0, lsm_1, lam_0, lam_1, x, y, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='forestgreen', s=6, label='Sample')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='forestgreen', label='LSA')
    ax.plot(x, y_lam, color='indigo', label='LMA')
    ax.plot(x, y_real, color='tomato', label='Modal')
    ax.set(xlabel='X', ylabel='Y',
       title=title)
    ax.legend()
    ax.grid()
    fig.savefig(title + '.png',dpi=200)
    plt.show()

def main():
    x = np.arange(-1.8, 2.1, 0.2)
    y = np.add(np.add(np.full(20, 2), x * 2), np.random.normal(0, 1, size=20))
    y2 = np.add(np.add(np.full(20, 2), x * 2), np.random.normal(0, 1, size=20))
    y2[0] += 10
    y2[-1] -= 10

    lsm_0, lsm_1 = LsmLinreg(x, y)
    print(f" МНК, без возмущений: {lsm_0}, {lsm_1}")
    lam_0, lam_1 = get_lam(x, y)
    print(f" МНM, без возмущений: {lam_0}, {lam_1}")

    lsm_02, lsm_12 = LsmLinreg(x, y2)
    print(f" МНК, с возмущениями: {lsm_02}, {lsm_12}")
    lam_02, lam_12 = get_lam(x, y2)
    print(f" МНM, с возмущениями: {lam_02}, {lam_12}")

    draw_res(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Sample without disturbances')
    draw_res(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Sample with disturbances')

if __name__ == "__main__":
    main()