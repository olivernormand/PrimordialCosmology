import numpy as np
import matplotlib.pyplot as plt
from fgivenx import plot_contours, samples_from_getdist_chains


def plf(x, theta):
    nDims = len(theta)
    node_x = np.concatenate(
        [np.array([0]), theta[:nDims//2 - 1], np.array([1])])
    node_y = theta[nDims//2 - 1:]

    N = len(node_x)

    cond_list = []
    func_list = []

    # Generate the condition list
    for i in range(N-1):
        cond_list.append(np.logical_and(
            np.where(x > node_x[i], True, False), np.where(x < node_x[i+1], True, False)))

    # Generate the function list
    for i in range(N-1):
        func_list.append(return_linear_function(
            node_x[i], node_x[i+1], node_y[i], node_y[i+1]))

    return np.piecewise(x, cond_list, func_list)
def return_linear_function(xi, xi1, yi, yi1):
    def func(x):
        return (yi * (xi1 - x) + yi1 * (x - xi)) / (xi1 - xi)
    return func
def line(x):
    """ Returns a line(2 pi x) function """

    x_mod = np.mod(x + 0.25, 1)

    y1 = 4 * x_mod - 1
    y2 = 3 - 4 * x_mod

    return np.where(x_mod < 0.5, y1, y2)
def sin2pi(x):
    return np.sin(2 * np.pi * x)


file_root = 'chains/model_comparison'
nDims = 6

samples, weights = samples_from_getdist_chains(
    ['x1', 'x2', 'x3',  'y0', 'y1', 'y2', 'y3', 'y4'], file_root)
# samples, weights = samples_from_getdist_chains(['p%i' % i for i in range(nDims)], file_root)

x = np.linspace(0, 1, 100)

fig, axs = plt.subplots()
cbar = plot_contours(plf, x, samples, axs, weights=weights)
cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

# if title: axs.set_title(title)
axs.set_ylim([-2, 2])
axs.set_ylabel('y')
axs.set_xlabel('x')

# if plot_function: axs.plot(x, f(x), 'g')
# if plot_data: axs.plot(plot_data[0], plot_data[1], 'b.', markersize = 2)

fig.tight_layout()
plt.savefig('output_plot.png')
