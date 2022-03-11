import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours, samples_from_getdist_chains


from theory import get_params_from_nDims, plf


def generate_plot(file_root, nDims, xlim, ylim, title=None, plot_function=None):
    params_list, _ = get_params_from_nDims(nDims)

    samples, weights = samples_from_getdist_chains(params_list, file_root)

    x = np.linspace(xlim[0], xlim[1], 200)

    fig, axs = plt.subplots()
    cbar = plot_contours(plf, x, samples, axs, weights=weights, ny = 500)
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

def generate_plot_overlay(file_roots, nDims, xlim, ylim):
    params_list_0, _ = get_params_from_nDims(nDims[0])
    params_list_1, _ = get_params_from_nDims(nDims[1])

    samples_0, weights_0 = samples_from_getdist_chains(params_list_0, file_roots[0])
    samples_1, weights_1 = samples_from_getdist_chains(params_list_1, file_roots[1])

    x = np.linspace(xlim[0], xlim[1], 200)

    fig, axs = plt.subplots()
    cbar = plot_contours(plf, x, samples_0, axs, weights = weights_0, ny = 500)
    cbar = plot_contours(plf, x, samples_1, axs, colors=plt.cm.Blues_r, weights = weights_1, ny = 500)

    cbar = plt.colorbar(cbar, ticks = [0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    axs.set_ylim(ylim)
    axs.set_ylabel('y')
    axs.set_xlabel('x')

    fig.tight_layout()
    plt.savefig('output.png')




