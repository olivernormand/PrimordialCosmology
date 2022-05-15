import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours, samples_from_getdist_chains

from lineorsin.theory import get_params_from_nDims
from lineorsin.priors import UniformPrior, SortedUniformPrior, hypercube_to_theta


def generate_plot(file_root, nDims, xlim, ylim, title=None, plot_function=None):
    params_list, _ = get_params_from_nDims(nDims)

    samples, weights = samples_from_getdist_chains(params_list, file_root)

    x = np.linspace(xlim[0], xlim[1], 200)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        print(theta)
        return plf(x, theta, xlim)

    fig, axs = plt.subplots()
    cbar = plot_contours(plf_adjusted, x, samples, axs, weights=weights, ny = 500)
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    if title:
        axs.set_title(title)
    axs.set_ylim(ylim)
    axs.set_ylabel('y')
    axs.set_xlabel('x')

    if plot_function:
        axs.plot(x, plot_function(x), 'g')

    fig.tight_layout()
    plt.savefig('output.png')

def plf(x, theta, xlim = [-4, -0.3]):
    """
        Return the piecewise linear function defined by a set of parameters theta

        theta represents the x coordinates of all but the edge points, and the y coordinates of those same points. 
        The x coordinates of the edge points do not need to be included. 

        x represents the values over which you wish the function to be evaluated. 

    """

    x_lower, x_upper = xlim
    nDims = len(theta)
    node_x = np.concatenate([np.array([x_lower]), theta[:nDims//2 - 1], np.array([x_upper])])
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

